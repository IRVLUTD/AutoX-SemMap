# Semantic Map — Construction & Update

## Algorithm 1 summary

```
Init:
  if no graph.json → V' = ∅, id = 0
  else             → load graph, id = max(v.id) + 1

Per frame:
  1. O_current   ← GDino + SAM detections with valid depth poses
  2. V_expected  ← graph nodes whose (x,y) falls inside camera FoV
  3. O_unmatched ← O_current
  4. for v in V_expected:
       if no o in O_unmatched with same category and dist ≤ δ_c → remove v
       else → O_unmatched \ {o}
  5. for o in O_unmatched:
       if min dist to same-category nodes > δ_c → add new node (id, cat, pose, conf); id++
```

Distance thresholds (`δ_c`):

| Category | δ_c |
|----------|-----|
| table    | 2.0 m |
| chair    | 0.6 m |
| door     | 2.0 m |

## Run (with ROS / Fetch robot)

```bash
# Terminal 1 — ROS core
roscore

# Terminal 2 — semantic mapping
conda activate autox          # or whichever env has robokit
cd /home/haneesh/repos/AutoX-SemMap
python robokit/semantic.py
```

The node publishes:
- `/seg_image`   — annotated RGB with detections
- `/graph_nodes` — MarkerArray for RViz

On shutdown, `graph.json` is written to the working directory.

## Resume from a saved graph

Just keep `graph.json` in the working directory before re-running.
The script detects it at startup:

```bash
ls graph.json      # must exist
python robokit/semantic.py
```

## Visualise graph on the occupancy map

```bash
python robokit/utils.py
# uses map.png + map.yaml in the working directory
```

## Key files

| File | Purpose |
|------|---------|
| `robokit/semantic.py` | Main node — Algorithm 1 |
| `robokit/utils.py` | `pose_in_map_frame`, `get_fov_points_in_map`, graph I/O |
| `graph.json` | Persisted scene graph (node-link JSON) |
