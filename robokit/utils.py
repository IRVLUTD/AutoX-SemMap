import numpy as np
import os
import cv2
import yaml
from numpy.linalg import norm
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import json

intrinsics = [
    [574.0527954101562, 0.0, 319.5],
    [0.0, 574.0527954101562, 239.5],
    [0.0, 0.0, 1.0],
]
# intrinsics = [[554.254691191187, 0.0, 320.5],[0.0, 554.254691191187, 240.5], [0.0, 0.0, 1.0]]
fx = intrinsics[0][0]
fy = intrinsics[1][1]
px = intrinsics[0][2]
py = intrinsics[1][2]


def compute_xyz(depth_img, fx, fy, px, py, height, width):
    indices = np.indices((height, width), dtype=np.float32).transpose(1, 2, 0)
    z_e = depth_img
    x_e = (indices[..., 1] - px) * z_e / fx
    y_e = (indices[..., 0] - py) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1)  # Shape: [H x W x 3]
    return xyz_img


def pose_to_map_pixel(map_metadata, pose):
    pose_x = pose[0]
    pose_y = pose[1]

    map_pixel_x = int((pose_x - map_metadata["origin"][0]) / map_metadata["resolution"])
    map_pixel_y = int((pose_y - map_metadata["origin"][1]) / map_metadata["resolution"])

    return [map_pixel_x, map_pixel_y]


def pose_along_line(pose1, pose2, distance=2):
    '''
    creates a new pose that is at the specified distance from pose1
    along the line from pose1 to pose2
    '''
    pose2 = pose2[0:3,3]
    difference_vector = pose2 - pose1
    unit_vector = difference_vector / norm(difference_vector)
    new_pose = pose1 + unit_vector * distance

    return new_pose


def read_map_image(map_file_path):
    assert os.path.exists(map_file_path)
    if map_file_path.endswith(".pgm"):
        map_image = cv2.imread(map_file_path)
    else:
        map_image = cv2.imread(map_file_path)

    return map_image


def read_map_metadata(metadata_file_path):
    assert os.path.exists(metadata_file_path)
    assert metadata_file_path.endswith(".yaml")
    with open(metadata_file_path, "r") as file:
        metadata = yaml.safe_load(file)
    file.close()
    return metadata


def display_map_image(map_image, write=False):
    width, height, _ = map_image.shape
    cv2.namedWindow("Map Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Map Image", width, height)
    if write:
        cv2.imwrite("map_image.png", map_image)
    cv2.imshow("Map Image", map_image)
    cv2.waitKey(0)


def is_nearby(pose1, pose2, threshold=0.5):
    if norm((pose1[0] - pose2[0], pose1[1] - pose2[1])) < threshold:
        return True


def normalize_depth_image(depth_array, max_depth):
    depth_image = (max_depth - depth_array) / max_depth
    depth_image = depth_image * 255
    return depth_image.astype(np.uint8)


def denormalize_depth_image(depth_image, max_depth):

    depth_array = max_depth * (1 - (depth_image / 255))
    # print(f"max {depth_array.max()}")
    return depth_array.astype(np.float32)

def get_fov_points_in_baselink(depth_array, RT_camera):
        mask1 = np.isnan(depth_array)
        depth_array[mask1] = 0.0
        xyz_array = compute_xyz(
            depth_array, fx, fy, px, py, depth_array.shape[0], depth_array.shape[1]
        )
        xyz_array = xyz_array.reshape((-1, 3))

        mask = ~(np.all(xyz_array == [0.0, 0.0, 0.0], axis=1))
        xyz_array = xyz_array[mask]

        xyz_base = np.dot(RT_camera[:3, :3], xyz_array.T).T
        xyz_base += RT_camera[:3, 3]

        min_x = np.min(xyz_base[:,0])
        max_x = np.max(xyz_base[:,0])
        min_y = np.min(xyz_base[:,1])
        max_y = np.max(xyz_base[:,1])

        return [[0,0,0],[max_x,min_y,0], [max_x, max_y,0]]

def get_fov_points_in_map(depth_array, RT_camera, RT_base):
        mask1 = np.isnan(depth_array)
        depth_array[mask1] = 0.0
        xyz_array = compute_xyz(
            depth_array, fx, fy, px, py, depth_array.shape[0], depth_array.shape[1]
        )
        xyz_array = xyz_array.reshape((-1, 3))

        mask = ~(np.all(xyz_array == [0.0, 0.0, 0.0], axis=1))
        xyz_array = xyz_array[mask]

        xyz_base = np.dot(RT_camera[:3, :3], xyz_array.T).T
        xyz_base += RT_camera[:3, 3]

        min_x = np.min(xyz_base[:,0])
        max_x = np.max(xyz_base[:,0])
        min_y = np.min(xyz_base[:,1])
        max_y = np.max(xyz_base[:,1])

        points_baselink = [[0,0,0],[max_x,min_y,0], [max_x, max_y,0]]
        points_map = np.dot(RT_base[:3,:3], np.array(points_baselink).T).T + RT_base[:3,3]

        return points_map.tolist()

def pose_in_map_frame(RT_camera, RT_base, depth_array, segment=None):
    if segment is not None:
        print(depth_array.max())
        depth_array = depth_array * (segment / 1)

    #TODO: if depth is not normalized, then we need to remoev nans in the read image 
    # depth_array[np.isnan(depth_array)] = 0.0
    mask1 = np.isnan(depth_array)
    depth_array[mask1] = 0.0
    

    if depth_array.max() == 0.0:
        return None
    else:
        xyz_array = compute_xyz(
            depth_array, fx, fy, px, py, depth_array.shape[0], depth_array.shape[1]
        )
        xyz_array = xyz_array.reshape((-1, 3))

        mask = ~(np.all(xyz_array == [0.0, 0.0, 0.0], axis=1))
        xyz_array = xyz_array[mask]
        print(f"mean pose cam link {np.mean(xyz_array, axis=0)}")

        xyz_base = np.dot(RT_camera[:3, :3], xyz_array.T).T
        xyz_base += RT_camera[:3, 3]
        print(f"mean pose base link {np.mean(xyz_base, axis=0)}")

        xyz_map = np.dot(RT_base[:3, :3], xyz_base.T).T
        xyz_map += RT_base[:3, 3]

        mean_pose = np.mean(xyz_map, axis=0)
        # mean_pose = pose_along_line( mean_pose, RT_base)
        return mean_pose.tolist()


def is_nearby_in_map(pose_list, node_pose, threshold=0.5):
    if len(pose_list) == 0:
        return pose_list, False
    pose_array = np.array(pose_list)
    node_pose_array = np.array([node_pose])
    distances = np.linalg.norm((pose_array[:, 0:2] - node_pose_array[:, 0:2]), axis=1)
    if np.any(distances < threshold):
        # print("not a new object")
        return pose_list, True
    else:
        # print("new node added")
        pose_list.append(node_pose)
        # print(f"pose list after {pose_list}")
        return pose_list, False


def save_graph_json(graph, file="graph.json"):
    '''
    input graph \n
    save graph to graph.json
    '''
    file = file
    data_to_save = json_graph.node_link_data(graph)
    with open(file, "w") as file:
        json.dump(data_to_save, file, indent=4)
        file.close()
    print(f"-=---------------------")


def read_graph_json(file="graph.json"):
    with open(file, "r") as file:
        data = json.load(file)
        file.close()
    # print(data)
    graph = json_graph.node_link_graph(data)
    return graph


def read_and_visualize_graph(map_file_path, map_metadata_filepath, on_map=False, catgeories=[], graph=None):
    if graph is None:
        graph = read_graph_json()
    else:
        graph = graph
    color_palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    if not on_map:
        pos = nx.spring_layout(graph)
        nx.draw(graph, pos, with_labels=True)
        plt.show()
    else:
        # c ncj
        map_image = read_map_image(map_file_path)
        map_metadata = read_map_metadata(map_metadata_filepath)
        for node, data in graph.nodes(data=True):
            if data["category"] in catgeories:
                x, y = pose_to_map_pixel(map_metadata, data["pose"])
                map_image[
                    y - 10 // 2 : y + 10 // 2,
                    x - 10 // 2 : x + 10 // 2,
                    :,
                ] = color_palette[catgeories.index(data["category"])]
        display_map_image(map_image, write=True)

def plot_point_on_map(map_file_path, map_metadata_filepath, position):
    map_image = read_map_image(map_file_path)
    map_metadata = read_map_metadata(map_metadata_filepath)
    x, y = pose_to_map_pixel(map_metadata, position)
    map_image[
        y - 10 // 2 : y + 10 // 2,
        x - 10 // 2 : x + 10 // 2,
        :,
    ] = [0,0,255]
    display_map_image(map_image, write=False)

def visualize_graph(graph):
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True)
    plt.show()


if __name__ == "__main__":
    # a=compute_xyz(np.array([[0,0,0],[0,0,0],[0,0,0]]), fx,fy,px,py, 3,3)
    # a=a.reshape((-1,3))
    # print(a)
    # save_graph_json()
    graph = read_graph_json()
    read_and_visualize_graph("map.png","map.yaml", on_map=True, catgeories=["table", "chair", "door"], graph=graph)
