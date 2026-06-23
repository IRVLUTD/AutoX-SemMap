#!/usr/bin/env python

"""Semantic map construction and update — Algorithm 1."""

import os
import threading
import numpy as np
import rospy
from PIL import Image as PILImg

import ros_numpy
from networkx import Graph
from shapely.geometry import Point, Polygon

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray

from robokit.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor
from robokit.utils import annotate, overlay_masks, filter_large_boxes, filter

lock = threading.Lock()
from listener import ImageListener
import time
from utils import (
    pose_in_map_frame,
    read_graph_json,
    save_graph_json,
    get_fov_points_in_map,
)

GRAPH_FILE = "graph.json"


class robokitRealtime:

    def __init__(self):
        rospy.init_node("seg_rgb")
        self.listener = ImageListener(camera="Fetch")

        self.text_prompt = "table . door . chair ."
        self.gdino = GroundingDINOObjectPredictor()
        self.SAM = SegmentAnythingPredictor()
        self.threshold = {"table": 2.0, "chair": 0.6, "door": 2.0}

        self.image_pub = rospy.Publisher("seg_image", Image, queue_size=10)
        self.marker_pub = rospy.Publisher("graph_nodes", MarkerArray, queue_size=10)

        # If no prior map exists → init empty graph, id=0
        # Else → load graph, id = max(v.id) + 1
        if os.path.exists(GRAPH_FILE):
            self.graph = read_graph_json(GRAPH_FILE)
            int_ids = [
                d["id"] for _, d in self.graph.nodes(data=True)
                if isinstance(d.get("id"), int)
            ]
            self.next_id = max(int_ids) + 1 if int_ids else 0
        else:
            self.graph = Graph()
            self.next_id = 0

        time.sleep(5)

    def create_marker(self, pose, category, node_id):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = rospy.Time.now()
        marker.ns = category
        marker.id = node_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position.x = pose[0]
        marker.pose.position.y = pose[1]
        marker.pose.position.z = 0.0
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.color.a = 1.0
        if category == "table":
            marker.color.r, marker.color.g, marker.color.b = 0.0, 0.0, 1.0
        elif category == "chair":
            marker.color.r, marker.color.g, marker.color.b = 0.0, 1.0, 0.0
        elif category == "door":
            marker.color.r, marker.color.g, marker.color.b = 1.0, 0.0, 0.0
        return marker

    def publish_graph_to_rviz(self):
        marker_array = MarkerArray()
        for rviz_id, (_, data) in enumerate(self.graph.nodes(data=True)):
            marker_array.markers.append(
                self.create_marker(data["pose"], data["category"], rviz_id)
            )
        self.marker_pub.publish(marker_array)

    def _nodes_in_fov(self, fov):
        """Return (node_key, data) for all graph nodes whose pose falls within fov."""
        return [
            (node, data)
            for node, data in list(self.graph.nodes(data=True))
            if fov.contains(Point(data["pose"][0], data["pose"][1]))
        ]

    def run_network(self):
        while not rospy.is_shutdown():
            with lock:
                if self.listener.im is None:
                    continue
                im_color = self.listener.im.copy()
                depth_img = self.listener.depth.copy()
                rgb_frame_id = self.listener.rgb_frame_id
                rgb_frame_stamp = self.listener.rgb_frame_stamp
                RT_camera, RT_base = self.listener.RT_camera, self.listener.RT_base

            print("===========================================")
            im = im_color.astype(np.uint8)[:, :, (2, 1, 0)]
            img_pil = PILImg.fromarray(im)

            bboxes, phrases, gdino_conf = self.gdino.predict(
                img_pil, self.text_prompt, 0.55, 0.55
            )
            bboxes, gdino_conf, phrases, flag = filter(
                bboxes, gdino_conf, phrases, 1, 0.8, 0.8, 0.8, 0.01, True
            )

            # --- Step 1: O_current — objects detected in FoV ---
            # Each entry: {'pose': [x,y,z], 'category': str, 'confidence': float}
            detected_objects = []
            annotated_image = None

            if not flag and len(phrases) > 0:
                w, h = im.shape[1], im.shape[0]
                image_pil_bboxes = self.gdino.bbox_to_scaled_xyxy(bboxes, w, h)
                image_pil_bboxes, masks = self.SAM.predict(img_pil, image_pil_bboxes)
                print(masks.shape)

                image_pil_bboxes, keep_index = filter_large_boxes(
                    image_pil_bboxes, w, h, threshold=0.5
                )
                masks = masks[keep_index]
                # kept_indices maps post-filter position i → pre-filter phrase/conf index
                kept_indices = np.where(keep_index)[0]

                mask_array = masks.cpu().numpy()
                for i, mask in enumerate(mask_array):
                    pose = pose_in_map_frame(RT_camera, RT_base, depth_img, segment=mask[0])
                    if pose is None:
                        continue
                    orig_idx = kept_indices[i]
                    detected_objects.append({
                        "pose": pose,
                        "category": phrases[orig_idx],
                        "confidence": float(gdino_conf[orig_idx]),
                    })

                phrases_kept = [phrases[i] for i in kept_indices]
                annotated_image = np.array(annotate(
                    overlay_masks(img_pil, masks),
                    image_pil_bboxes,
                    gdino_conf[keep_index],
                    phrases_kept,
                ))

            # --- Step 2: V_expected — graph nodes within current FoV ---
            fov_points = get_fov_points_in_map(depth_img, RT_camera, RT_base)
            fov = Polygon(fov_points)
            nodes_in_fov = self._nodes_in_fov(fov)

            # --- Step 3: O_unmatched ← O_current ---
            unmatched = list(range(len(detected_objects)))

            # --- Step 4: for v in V_expected ---
            # Find a matching detection (same category, dist ≤ δ_c).
            # No match → remove node from V'.
            # Match found → remove that detection from O_unmatched.
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
                    unmatched.remove(matched_idx)  # O_unmatched \ {o}

            for node in nodes_to_remove:
                print(f"removing node {node}")
                self.graph.remove_node(node)

            # --- Step 5: for o in O_unmatched ---
            # Add new node only if no existing same-category node is within δ_c.
            for idx in unmatched:
                o = detected_objects[idx]
                V_c = [
                    d for _, d in self.graph.nodes(data=True)
                    if d.get("category") == o["category"]
                ]
                d_min = min(
                    (np.linalg.norm(np.array(o["pose"][:2]) - np.array(d["pose"][:2])) for d in V_c),
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
                    print(f"adding node id={self.next_id} category={o['category']}")
                    self.next_id += 1

            if annotated_image is not None:
                rgb_msg = ros_numpy.msgify(Image, annotated_image, "rgb8")
                rgb_msg.header.stamp = rgb_frame_stamp
                rgb_msg.header.frame_id = rgb_frame_id
                self.image_pub.publish(rgb_msg)

            self.publish_graph_to_rviz()


if __name__ == "__main__":
    robokit_instance = robokitRealtime()
    robokit_instance.run_network()
    print("closing script! saving graph")
    save_graph_json(robokit_instance.graph, file=GRAPH_FILE)
