# Replica Dataset — RGBD + Pose Extraction

## 1. Install habitat-sim

```bash
conda activate replica
conda install habitat-sim=0.3.0 headless -c aihabitat -c conda-forge
```

> Drop `headless` if you have a display and want GPU rendering.

## 2. Run the extractor

```bash
conda activate replica

python replica_extraction/extract_replica.py \
    /home/haneesh/repos/datasets/replica_data/apartment_0 \
    ./output/apartment_0 \
    --n-frames 300 \
    --n-waypoints 15 \
    --seed 42
```

Change `apartment_0` to any scene:

```
apartment_0  apartment_1  apartment_2
frl_apartment_0 .. frl_apartment_5
hotel_0
office_0 .. office_4
room_0  room_1  room_2
```

## 3. Output format (per frame)

| File | Shape / Type | Notes |
|------|-------------|-------|
| `000000_rgb.png` | H×W×3 uint8 | BGR (OpenCV order) |
| `000000_depth.npy` | H×W float32 | metres; 0 = invalid |
| `000000_pose.npz` | — | `RT_camera` (4×4), `RT_base` (4×4) |
| `000000_semantic.png` | H×W uint16 | per-pixel instance ID (lossless PNG) |
| `000000_semantic_color.png` | H×W×3 uint8 | category-coloured vis (BGR) |
| `semantic_instances.json` | — | `{"instance_id": "category_name", ...}` for whole scene |
| `intrinsics.json` | — | fx, fy, cx, cy, width, height |

### Load a frame in Python

```python
import cv2, json, numpy as np

rgb      = cv2.imread("output/apartment_0/000000_rgb.png")
depth    = np.load("output/apartment_0/000000_depth.npy")
pose     = np.load("output/apartment_0/000000_pose.npz")
semantic = cv2.imread("output/apartment_0/000000_semantic.png",
                      cv2.IMREAD_UNCHANGED)   # uint16, instance IDs

RT_camera = pose["RT_camera"]   # eye(4) — camera IS the agent
RT_base   = pose["RT_base"]     # T_world_cam (camera → world)

with open("output/apartment_0/semantic_instances.json") as f:
    id_to_cat = json.load(f)    # {"0": "background", "3": "chair", ...}

# Boolean mask for all chair pixels in this frame
chair_ids = {k for k, v in id_to_cat.items() if v == "chair"}
chair_mask = np.isin(semantic, [int(i) for i in chair_ids])
```

## 4. Camera intrinsics

HFOV = 58.4° at 640×480 gives:

```
fx = fy ≈ 574.05
cx = 319.5,  cy = 239.5
```

These match the hardcoded values in `robokit/utils.py`.

## 5. Extract all scenes in a loop

```bash
SCENES_ROOT=/home/haneesh/repos/datasets/replica_data
OUTPUT_ROOT=./output

for scene in apartment_0 apartment_1 apartment_2 room_0 room_1 room_2; do
    python replica_extraction/extract_replica.py \
        "$SCENES_ROOT/$scene" \
        "$OUTPUT_ROOT/$scene" \
        --n-frames 300 \
        --seed 42
done
```
