from networkx import Graph

import sys
import os
from os.path import join

import numpy as np
from utils import save_graph_json
class ExtractTrajectory:
    def __init__(self, root_dir) -> None:
        self.root_dir = root_dir
        self.pose_dir = join(root_dir, "pose")
        self.pose_files = os.listdir(self.pose_dir)
        self.pose_files.sort()
        self.trajectory_graph = Graph()

    def save_trajectory_json(self, file_name="robot_trajectory.json"):
        for id, pose_data_file in enumerate(self.pose_files):
            pose_data = np.load(join(self.pose_dir, pose_data_file))
            self.trajectory_graph.add_node(
                id,
                pose = pose_data["RT_base"][:3,3].tolist()
            )
        save_graph_json(self.trajectory_graph, file=file_name)

if __name__=="__main__":
    extract_traj = ExtractTrajectory(sys.argv[1])
    extract_traj.save_trajectory_json()