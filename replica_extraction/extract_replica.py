#!/usr/bin/env python3
"""
Interactive RGBD + semantic extractor for the Replica dataset.

Drive the camera manually; frames are auto-saved every --save-every moves.

Controls:
  W / S        move forward / back
  A / D        strafe left / right
  Q / E        rotate yaw (look left / right)
  R / F        tilt pitch (look up / down)
  T            toggle RGB ↔ semantic colour view
  Space        save current frame immediately
  Esc          quit and write metadata files

Install:
  conda install habitat-sim=0.3.0 headless -c aihabitat -c conda-forge

Run:
  python replica_extraction/extract_replica.py \\
      /path/to/replica_data/apartment_0 ./output/apartment_0

Output per saved frame in <output_dir>/:
  {i:06d}_rgb.png            RGB image        (H x W x 3, uint8, BGR)
  {i:06d}_depth.npy          depth map        (H x W, float32, metres)
  {i:06d}_pose.npz           RT_camera=eye(4), RT_base=T_world_cam (4x4)
  {i:06d}_semantic.png       instance-ID map  (H x W, uint16, lossless PNG)
  {i:06d}_semantic_color.png category colours (H x W x 3, uint8, BGR)
  semantic_instances.json    {semantic_id: category_name} for the whole scene
  intrinsics.json            fx, fy, cx, cy, width, height
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np

try:
    import habitat_sim
    import quaternion
except ImportError:
    sys.exit(
        "habitat-sim not found.\n"
        "Install with:\n"
        "  conda install habitat-sim=0.3.0 headless -c aihabitat -c conda-forge"
    )

# ── Camera settings ─────────────────────────────────────────────────────────────
# 90° HFOV matches the original Replica C++ renderer (render.cpp).
# fx = (W/2) / tan(HFOV/2) = 320 / tan(45°) = 320.0 at 640×480.
WIDTH, HEIGHT = 640, 480
HFOV_DEG = 90.0
# ────────────────────────────────────────────────────────────────────────────────


# ── Simulator setup ─────────────────────────────────────────────────────────────

def make_simulator(scene_dir: str) -> habitat_sim.Simulator:
    scene_config  = os.path.join(scene_dir, "habitat", "replica_stage.stage_config.json")
    dataset_config = os.path.join(scene_dir, "..", "replica.scene_dataset_config.json")
    assert os.path.exists(scene_config),   f"Not found: {scene_config}"
    assert os.path.exists(dataset_config), f"Not found: {dataset_config}"

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = os.path.realpath(dataset_config)
    sim_cfg.scene_id                  = os.path.realpath(scene_config)
    sim_cfg.enable_physics            = False

    def _cam_spec(uuid, sensor_type):
        spec             = habitat_sim.CameraSensorSpec()
        spec.uuid        = uuid
        spec.sensor_type = sensor_type
        spec.resolution  = [HEIGHT, WIDTH]
        spec.hfov        = HFOV_DEG
        spec.position    = np.array([0.0, 0.0, 0.0])
        return spec

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [
        _cam_spec("rgb",      habitat_sim.SensorType.COLOR),
        _cam_spec("depth",    habitat_sim.SensorType.DEPTH),
        _cam_spec("semantic", habitat_sim.SensorType.SEMANTIC),
    ]

    return habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))


def sensor_T_world(agent_state) -> np.ndarray:
    """4×4 camera-to-world transform (RT_base) from the rgb sensor state."""
    ss = agent_state.sensor_states["rgb"]
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = quaternion.as_rotation_matrix(ss.rotation)
    T[:3, 3]  = ss.position
    return T


# ── Semantic helpers ─────────────────────────────────────────────────────────────

def build_semantic_metadata(sim: habitat_sim.Simulator) -> tuple:
    """
    Returns:
        id_to_category : {semantic_id (int) -> category_name (str)}
        cat_to_color   : {category_name (str) -> BGR tuple}

    Uses obj.semantic_id (int) — matches obs["semantic"] pixel values.
    obj.id is a string handle like "_1" and must NOT be used as the key.
    """
    id_to_category = {}
    for obj in sim.semantic_scene.objects:
        if obj is None:
            continue
        cat = obj.category.name() if obj.category else "unknown"
        id_to_category[int(obj.semantic_id)] = cat

    cat_to_color = {}
    for name in sorted(set(id_to_category.values())):
        h = abs(hash(name)) & 0xFFFFFF
        cat_to_color[name] = (h & 0xFF, (h >> 8) & 0xFF, (h >> 16) & 0xFF)

    return id_to_category, cat_to_color


def colorize_semantic(semantic: np.ndarray,
                      id_to_category: dict,
                      cat_to_color: dict) -> np.ndarray:
    """Instance-ID image → per-category BGR colour image."""
    color_img = np.zeros((*semantic.shape, 3), dtype=np.uint8)
    for inst_id, cat_name in id_to_category.items():
        color_img[semantic == inst_id] = cat_to_color.get(cat_name, (128, 128, 128))
    return color_img


# ── Interactive session ──────────────────────────────────────────────────────────

class InteractiveSession:
    MOVE_STEP  = 0.10   # metres per key press
    TURN_STEP  = 5.0    # degrees per key press (yaw)
    PITCH_STEP = 3.0    # degrees per key press (pitch)
    PITCH_MAX  = 60.0   # degrees

    def __init__(self, sim: habitat_sim.Simulator,
                 agent, output_dir: str,
                 save_every: int,
                 id_to_category: dict,
                 cat_to_color: dict):
        self.sim           = sim
        self.agent         = agent
        self.output_dir    = output_dir
        self.save_every    = save_every
        self.id_to_category = id_to_category
        self.cat_to_color  = cat_to_color
        self.RT_camera     = np.eye(4, dtype=np.float64)

        # Start at a random navigable point
        self.pos   = np.array(sim.pathfinder.get_random_navigable_point())
        self.yaw   = 0.0    # degrees, world-Y axis
        self.pitch = 0.0    # degrees, local-X axis

        self.move_count  = 0   # actions that changed agent state
        self.saved_count = 0
        self.show_semantic = False
        self._pending_save = False  # save at the top of the next render tick

        self._apply_pose()

    # ── Pose helpers ───────────────────────────────────────────────────────────

    def _make_rotation(self) -> np.quaternion:
        q_yaw   = quaternion.from_rotation_vector(
            np.array([0.0, np.deg2rad(self.yaw), 0.0]))
        q_pitch = quaternion.from_rotation_vector(
            np.array([np.deg2rad(self.pitch), 0.0, 0.0]))
        return q_yaw * q_pitch

    def _forward(self) -> np.ndarray:
        yr = np.deg2rad(self.yaw)
        return np.array([-np.sin(yr), 0.0, -np.cos(yr)])

    def _right(self) -> np.ndarray:
        yr = np.deg2rad(self.yaw)
        return np.array([np.cos(yr), 0.0, -np.sin(yr)])

    def _apply_pose(self) -> None:
        state          = habitat_sim.AgentState()
        state.position = self.pos
        state.rotation = self._make_rotation()
        self.agent.set_state(state)

    def _try_move(self, delta: np.ndarray) -> bool:
        """Move by delta if target is navigable; return True if moved."""
        candidate = self.pos + delta
        snapped   = self.sim.pathfinder.snap_point(candidate)
        if np.any(np.isnan(snapped)):
            return False
        self.pos = snapped
        self._apply_pose()
        return True

    # ── Frame saving ───────────────────────────────────────────────────────────

    def _save_frame(self, obs: dict) -> None:
        prefix = os.path.join(self.output_dir, f"{self.saved_count:06d}")
        bgr    = cv2.cvtColor(obs["rgb"][:, :, :3], cv2.COLOR_RGB2BGR)
        depth  = obs["depth"].astype(np.float32)
        sem    = obs["semantic"].astype(np.uint16)
        sem_c  = colorize_semantic(obs["semantic"], self.id_to_category, self.cat_to_color)
        RT_base = sensor_T_world(self.agent.get_state())

        cv2.imwrite(f"{prefix}_rgb.png", bgr)
        np.save(f"{prefix}_depth.npy", depth)
        np.savez(f"{prefix}_pose.npz", RT_camera=self.RT_camera, RT_base=RT_base)
        cv2.imwrite(f"{prefix}_semantic.png", sem)
        cv2.imwrite(f"{prefix}_semantic_color.png", sem_c)

        self.saved_count += 1

    # ── Display ────────────────────────────────────────────────────────────────

    def _build_display(self, obs: dict) -> np.ndarray:
        bgr  = cv2.cvtColor(obs["rgb"][:, :, :3], cv2.COLOR_RGB2BGR)
        sem_c = colorize_semantic(obs["semantic"], self.id_to_category, self.cat_to_color)
        # Side-by-side: RGB left, semantic right
        display = np.concatenate([bgr, sem_c], axis=1)

        hud = [
            "WASD: move  Q/E: yaw  R/F: pitch",
            "T: toggle view  Space: save now  Esc: quit",
            f"Saved: {self.saved_count}  |  Moves: {self.move_count}  "
            f"|  Save every: {self.save_every}",
            f"Yaw: {self.yaw:.1f}  Pitch: {self.pitch:.1f}  "
            f"Pos: ({self.pos[0]:.2f}, {self.pos[2]:.2f})",
        ]
        for row, text in enumerate(hud):
            cv2.putText(display, text, (8, 18 + row * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        return display

    # ── Main loop ──────────────────────────────────────────────────────────────

    def run(self) -> None:
        print("Controls: WASD move | Q/E yaw | R/F pitch | T toggle | Space save | Esc quit")
        print(f"Auto-saving every {self.save_every} moves → {self.output_dir}")

        while True:
            obs = self.sim.get_sensor_observations()

            if self._pending_save:
                self._save_frame(obs)
                self._pending_save = False

            display = self._build_display(obs)
            cv2.imshow("Replica — RGB | Semantic", display)
            key = cv2.waitKey(30) & 0xFF

            moved = False

            if key == ord('w'):
                moved = self._try_move(self._forward() * self.MOVE_STEP)
            elif key == ord('s'):
                moved = self._try_move(-self._forward() * self.MOVE_STEP)
            elif key == ord('a'):
                moved = self._try_move(-self._right() * self.MOVE_STEP)
            elif key == ord('d'):
                moved = self._try_move(self._right() * self.MOVE_STEP)
            elif key == ord('q'):
                self.yaw -= self.TURN_STEP
                self._apply_pose()
                moved = True
            elif key == ord('e'):
                self.yaw += self.TURN_STEP
                self._apply_pose()
                moved = True
            elif key == ord('r'):
                self.pitch = max(-self.PITCH_MAX, self.pitch - self.PITCH_STEP)
                self._apply_pose()
                moved = True
            elif key == ord('f'):
                self.pitch = min(self.PITCH_MAX, self.pitch + self.PITCH_STEP)
                self._apply_pose()
                moved = True
            elif key == ord('t'):
                self.show_semantic = not self.show_semantic
            elif key == ord(' '):
                self._save_frame(obs)
                print(f"  [manual] saved frame {self.saved_count}")
            elif key == 27:  # Esc
                break

            if moved:
                self.move_count += 1
                if self.move_count % self.save_every == 0:
                    self._pending_save = True  # save on next render (post-move view)

        cv2.destroyAllWindows()
        print(f"\nDone. {self.saved_count} frames saved → {self.output_dir}")


# ── Entry point ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively extract RGBD + semantic frames from a Replica scene."
    )
    parser.add_argument("scene_dir",
                        help="Replica scene directory, e.g. .../replica_data/apartment_0")
    parser.add_argument("output_dir",
                        help="Where to write frames and metadata.")
    parser.add_argument("--save-every", type=int, default=5,
                        help="Auto-save one frame every N moves (default: 5).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    sim   = make_simulator(args.scene_dir)
    agent = sim.initialize_agent(0)

    # Intrinsics
    fx = fy = (WIDTH / 2.0) / np.tan(np.deg2rad(HFOV_DEG / 2.0))
    cx, cy  = (WIDTH - 1) / 2.0, (HEIGHT - 1) / 2.0
    intrinsics = {"width": WIDTH, "height": HEIGHT,
                  "fx": fx, "fy": fy, "cx": cx, "cy": cy, "hfov_deg": HFOV_DEG}
    with open(os.path.join(args.output_dir, "intrinsics.json"), "w") as f:
        json.dump(intrinsics, f, indent=2)

    # Semantic metadata
    id_to_category, cat_to_color = build_semantic_metadata(sim)
    with open(os.path.join(args.output_dir, "semantic_instances.json"), "w") as f:
        json.dump({str(k): v for k, v in id_to_category.items()}, f, indent=2)
    print(f"Scene: {len(id_to_category)} instances, {len(cat_to_color)} categories.")

    session = InteractiveSession(sim, agent, args.output_dir,
                                  args.save_every, id_to_category, cat_to_color)
    session.run()
    sim.close()


if __name__ == "__main__":
    main()
