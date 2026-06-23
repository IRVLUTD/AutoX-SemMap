#!/usr/bin/env python3
"""
Semantic map construction and update — Algorithm 1, offline (no ROS).

Reads pre-extracted Replica frames produced by extract_replica.py and runs
the same 5-step semantic mapping loop as robokit/semantic.py.

Run:
    python replica_extraction/semantic_offline.py \
        ./output/apartment_0 \
        --out-dir ./results/apartment_0

Output:
    graph.json              final scene graph
    <out-dir>/{i:06d}_annotated.png   detection overlay per frame (optional)
"""

import argparse
import os
import sys

import cv2
import numpy as np
from networkx import Graph
from PIL import Image as PILImg
from shapely.geometry import Point, Polygon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from robokit.utils import annotate, overlay_masks, filter_large_boxes, filter
from scripts.utils import (
    get_fov_points_in_map,
    pose_in_map_frame,
    read_graph_json,
    save_graph_json,
)

GRAPH_FILE = "graph.json"


# ── Frame reader ───────────────────────────────────────────────────────────────

class ReplicaReader:
    """Iterates over pre-extracted frames from extract_replica.py."""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        indices = sorted(
            int(f[:6])
            for f in os.listdir(data_dir)
            if len(f) >= 10 and f[:6].isdigit() and f.endswith("_rgb.png")
        )
        if not indices:
            sys.exit(f"No frames found in {data_dir}")
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, pos: int) -> dict:
        i = self.indices[pos]
        prefix = os.path.join(self.data_dir, f"{i:06d}")
        rgb_bgr = cv2.imread(f"{prefix}_rgb.png")
        depth   = np.load(f"{prefix}_depth.npy")
        pose    = np.load(f"{prefix}_pose.npz")
        return {
            "rgb_bgr":   rgb_bgr,
            "depth":     depth,
            "RT_camera": pose["RT_camera"],
            "RT_base":   pose["RT_base"],
        }


# ── Mapper ─────────────────────────────────────────────────────────────────────

class SemanticMapper:

    def __init__(self, data_dir: str, out_dir: str,
                 graph_file: str = GRAPH_FILE, save_vis: bool = True):
        self.reader    = ReplicaReader(data_dir)
        self.out_dir   = out_dir
        self.graph_file = graph_file
        self.save_vis  = save_vis
        os.makedirs(out_dir, exist_ok=True)

        self.text_prompt = "table . door . chair ."
        self.gdino = GroundingDINOObjectPredictor()
        self.SAM   = SegmentAnythingPredictor()
        self.threshold = {"table": 2.0, "chair": 0.6, "door": 2.0}

        # Algorithm init: no prior map → empty; prior map → load + max id
        if os.path.exists(graph_file):
            self.graph = read_graph_json(graph_file)
            int_ids = [
                d["id"] for _, d in self.graph.nodes(data=True)
                if isinstance(d.get("id"), int)
            ]
            self.next_id = max(int_ids) + 1 if int_ids else 0
            print(f"Loaded prior graph: {self.graph.number_of_nodes()} nodes, "
                  f"next_id={self.next_id}")
        else:
            self.graph   = Graph()
            self.next_id = 0

    def _nodes_in_fov(self, fov: Polygon) -> list:
        return [
            (node, data)
            for node, data in list(self.graph.nodes(data=True))
            if fov.contains(Point(data["pose"][0], data["pose"][1]))
        ]

    def process_frame(self, frame: dict, frame_idx: int) -> None:
        rgb_bgr   = frame["rgb_bgr"]
        depth_img = frame["depth"]
        RT_camera = frame["RT_camera"]
        RT_base   = frame["RT_base"]

        # GDino expects RGB PIL
        im_rgb  = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
        img_pil = PILImg.fromarray(im_rgb)

        bboxes, phrases, gdino_conf = self.gdino.predict(
            img_pil, self.text_prompt, 0.55, 0.55
        )
        bboxes, gdino_conf, phrases, flag = filter(
            bboxes, gdino_conf, phrases, 1, 0.8, 0.8, 0.8, 0.01, True
        )

        # ── Step 1: O_current ─────────────────────────────────────────────────
        detected_objects = []
        annotated_image  = None

        if not flag and len(phrases) > 0:
            h, w = rgb_bgr.shape[:2]
            image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)
            image_pil_bboxes, masks = self.SAM.predict(img_pil, image_pil_bboxes)
            image_pil_bboxes, keep_index = filter_large_boxes(
                image_pil_bboxes, w, h, threshold=0.5
            )
            masks        = masks[keep_index]
            kept_indices = np.where(keep_index)[0]

            for i, mask in enumerate(masks.cpu().numpy()):
                pose = pose_in_map_frame(RT_camera, RT_base, depth_img.copy(),
                                         segment=mask[0])
                if pose is None:
                    continue
                orig_idx = kept_indices[i]
                detected_objects.append({
                    "pose":       pose,
                    "category":   phrases[orig_idx],
                    "confidence": float(gdino_conf[orig_idx]),
                })

            if self.save_vis:
                phrases_kept = [phrases[i] for i in kept_indices]
                annotated_image = np.array(annotate(
                    overlay_masks(img_pil, masks),
                    image_pil_bboxes,
                    gdino_conf[keep_index],
                    phrases_kept,
                ))

        # ── Step 2: V_expected ────────────────────────────────────────────────
        fov_points   = get_fov_points_in_map(depth_img.copy(), RT_camera, RT_base)
        fov          = Polygon(fov_points)
        nodes_in_fov = self._nodes_in_fov(fov)

        # ── Step 3: O_unmatched ← O_current ──────────────────────────────────
        unmatched = list(range(len(detected_objects)))

        # ── Step 4: for v in V_expected ───────────────────────────────────────
        nodes_to_remove = []
        for node, data in nodes_in_fov:
            v_pos = np.array(data["pose"][:2])
            v_cat = data["category"]
            matched_idx = None
            for idx in unmatched:
                o = detected_objects[idx]
                if o["category"] == v_cat:
                    if np.linalg.norm(np.array(o["pose"][:2]) - v_pos) <= self.threshold[v_cat]:
                        matched_idx = idx
                        break
            if matched_idx is None:
                nodes_to_remove.append(node)
            else:
                unmatched.remove(matched_idx)

        for node in nodes_to_remove:
            print(f"  frame {frame_idx}: removing node {node}")
            self.graph.remove_node(node)

        # ── Step 5: for o in O_unmatched ─────────────────────────────────────
        for idx in unmatched:
            o   = detected_objects[idx]
            V_c = [d for _, d in self.graph.nodes(data=True)
                   if d.get("category") == o["category"]]
            d_min = min(
                (np.linalg.norm(np.array(o["pose"][:2]) - np.array(d["pose"][:2]))
                 for d in V_c),
                default=float("inf"),
            )
            if d_min > self.threshold[o["category"]]:
                self.graph.add_node(
                    self.next_id,
                    id=self.next_id,
                    pose=o["pose"],
                    category=o["category"],
                    confidence=o["confidence"],
                )
                print(f"  frame {frame_idx}: added node {self.next_id} "
                      f"({o['category']})")
                self.next_id += 1

        # ── Save visualisation ────────────────────────────────────────────────
        if self.save_vis and annotated_image is not None:
            out_path = os.path.join(self.out_dir, f"{frame_idx:06d}_annotated.png")
            cv2.imwrite(out_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

    def run(self) -> None:
        n = len(self.reader)
        print(f"Processing {n} frames ...")
        for pos in range(n):
            frame = self.reader[pos]
            self.process_frame(frame, pos)
            print(f"  [{pos+1}/{n}] nodes in graph: {self.graph.number_of_nodes()}")

        save_graph_json(self.graph, self.graph_file)
        print(f"\nDone. Graph saved → {self.graph_file} "
              f"({self.graph.number_of_nodes()} nodes)")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline semantic mapping from pre-extracted Replica frames."
    )
    parser.add_argument("data_dir",
                        help="Directory with extracted frames (from extract_replica.py).")
    parser.add_argument("--out-dir", default="./results",
                        help="Where to write annotated images (default: ./results).")
    parser.add_argument("--graph-file", default=GRAPH_FILE,
                        help=f"Graph JSON path (default: {GRAPH_FILE}).")
    parser.add_argument("--no-vis", action="store_true",
                        help="Skip saving annotated images (faster).")
    args = parser.parse_args()

    mapper = SemanticMapper(
        data_dir   = args.data_dir,
        out_dir    = args.out_dir,
        graph_file = args.graph_file,
        save_vis   = not args.no_vis,
    )
    mapper.run()
