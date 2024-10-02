import sys
import networkx as nx
from numpy.linalg import norm
from utils import read_graph_json,  read_map_image, read_map_metadata, pose_to_map_pixel
import cv2
import numpy as np

MIN_SEPARATION_TSP = 1.5

def tsp_greedy_solution(G):
    nodes = sorted(list(G.nodes))
    path = [nodes[0]]  # Start at the first node
    visited = set(path)

    while len(path) < len(nodes):
        last = path[-1]
        next_node = min(
            (n for n in nodes if n not in visited),
            key=lambda n: G[last][n]['weight']
        )
        path.append(next_node)
        visited.add(next_node)

    path.append(path[0])  # Return to the starting node
    return path

def preprocess_trajectory_graph(trajectory_graph):
    deleted_noodes = []
    trajectory_graph_ = trajectory_graph.copy()
    for id, (node, data) in enumerate(trajectory_graph.nodes(data=True)):

        if node in deleted_noodes:
                continue
        for id_, (node_, data_) in enumerate(trajectory_graph.nodes(data=True)):
            if node_ in deleted_noodes:
                continue

            if node == node_ or (id_ < id):
                continue
            pose1 = data["pose"]
            pose2 = data_["pose"]
            weight=norm((pose1[0] - pose2[0], pose1[1] - pose2[1]))
            if weight < MIN_SEPARATION_TSP:
                trajectory_graph_.remove_node(node_)
                deleted_noodes.append(node_)
                
    for id, (node, data) in enumerate(trajectory_graph_.nodes(data=True)):
        for id_, (node_, data_) in enumerate(trajectory_graph_.nodes(data=True)):
            if id < id_:
                pose1 = data["pose"]
                pose2 = data_["pose"]
                weight=norm((pose1[0] - pose2[0], pose1[1] - pose2[1]))
                trajectory_graph_.add_edge(node, node_, weight=weight)
    return trajectory_graph_



if __name__=="__main__":

    map_image = read_map_image("map.pgm")
    map_metadata = read_map_metadata("map.yaml")
    map_image_dimensions = (map_image.shape[0], map_image.shape[1])
    robot_trajectory = read_graph_json(sys.argv[1])
    pruned_trajectory = preprocess_trajectory_graph(robot_trajectory)
    surveillance_path = tsp_greedy_solution(pruned_trajectory)

    surveillance_traj = []
    for node in surveillance_path:
        pose = pruned_trajectory.nodes[node]["pose"]
        surveillance_traj.append(pose)
    np.savez("surveillance_traj", traj=np.array(surveillance_traj))

    cv2.namedWindow("map_image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("map_image", map_image_dimensions) 

    for i, node in enumerate(surveillance_path):
        pose = pruned_trajectory.nodes[node]["pose"]
        x,y = pose_to_map_pixel(map_metadata, pose)
        map_image[
            -y-10// 2 :-y+10// 2,
                    x-10 // 2 : x+10 // 2,
                    
                        
                        :,
                    ] = [0,0,255]
        
        cv2.imshow("map_image", map_image)
        cv2.waitKey(0)
