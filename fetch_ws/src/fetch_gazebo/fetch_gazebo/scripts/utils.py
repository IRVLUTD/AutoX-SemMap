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


def display_map_image(map_image, write=False, key=0):
    width, height, _ = map_image.shape
    cv2.namedWindow("Map Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Map Image", width, height)
    if write:
        cv2.imwrite("map_image.png", map_image)
    cv2.imshow("Map Image", map_image)
    cv2.waitKey(key)


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




    

def pose_in_map_frame(RT_camera, RT_base, depth_array, segment=None):
    if segment is not None:
        depth_array = depth_array * (segment / 255)

    #TODO: if depth is not normalized, then we need to remoev nans in the read image 
    # depth_array[np.isnan(depth_array)] = 0.0

    if depth_array.max() == 0.0:
        return None
    else:
        xyz_array = compute_xyz(
            depth_array, fx, fy, px, py, depth_array.shape[0], depth_array.shape[1]
        )
        xyz_array = xyz_array.reshape((-1, 3))

        mask = ~(np.all(xyz_array == [0.0, 0.0, 0.0], axis=1))
        xyz_array = xyz_array[mask]

        xyz_base = np.dot(RT_camera[:3, :3], xyz_array.T).T
        xyz_base += RT_camera[:3, 3]

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
                    y - 5//2 : y + 5//2,
                    x - 5//2 : x + 5//2,
                    :,
                ] = color_palette[catgeories.index(data["category"])]
        display_map_image(map_image, write=True)

def plot_point_on_map(map_file_path, map_metadata_filepath, position):
    map_image = read_map_image(map_file_path)
    map_metadata = read_map_metadata(map_metadata_filepath)
    x, y = pose_to_map_pixel(map_metadata, position)
    map_image[
        y - 5//2 : y + 5//2,
        x - 5//2 : x + 5//2,
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
    
    # g = read_graph_json("graph_ecss4.json")
    # rpose = [3,2,0]
    # import time
    # t=time.time()
    # for i in range(100):
    #     for node, data in g.nodes(data=True):
    #         if np.linalg.norm(np.array(data["pose"])-np.array(rpose)) < 3:
    #             continue
    #         else:
    #             continue
    # endt = time.time()-t
    # print(endt)
    
    # graph=read_graph_json("graph.json")
    # if graph is None:
    #     graph = read_graph_json()
    # else:
    #     graph = graph
    color_palette = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]

    map_image = read_map_image("thickened_black_pixels.png")
    map_metadata = read_map_metadata("map.yaml")

    # for node, data in graph.nodes(data=True):
        # 
        # x, y = pose_to_map_pixel(map_metadata, data["pose"])
        # map_image[
            # y - 5//2 : y + 5//2,
            # x - 5//2 : x + 5//2,
            # :,
        # ] = color_palette[0]
        # display_map_image(map_image, write=True)
        # read_and_visualize_graph("map.png", "map.yaml", catgeories=["table", "chair", "door"], on_map=True,)
        # save_graph_json()
    # 

    # with np.load("surveillance_traj.npz") as traj_file:
    #     surveillance_traj = traj_file["traj"]
    
    # for node in  surveillance_traj:
        
    #     x, y = pose_to_map_pixel(map_metadata, node)
    #     map_image[
    #         y - 5//2 : y + 5//2,
    #         x - 5//2 : x + 5//2,
    #         :,
    #     ] = color_palette[0]
    # display_map_image(map_image, write=True)
    # read_and_visualize_graph("map.png", "map.yaml", catgeories=["table", "chair", "door"], on_map=True,)


    import cv2
    import numpy as np

    def draw_arrow(map_image, start, end,  i, stop_distance=30):
        """Draw an arrow between two points."""
        # Convert start and end to numpy arrays for easy manipulation
        start = np.array(start)
        end = np.array(end)

        # Calculate the vector and its length
        vector = end - start
        length = norm(vector)
        
        # Shorten the arrow length by 30 cm (0.3 meters)
        # stop_distance = 0.3
        if length > stop_distance:
            unit_vector = vector / length
            shortened_end = end - stop_distance * unit_vector
        else:
            shortened_end = end
    
        # Draw the arrow
        if i <50:
            arrow_color = (0,204,0)
            # arrow_color = (255,153,51)
        else:
            arrow_color = (51, 51,255)
            # arrow_color = (255,153,51)
        cv2.arrowedLine(map_image, tuple(start.astype(int)), tuple(shortened_end.astype(int)), arrow_color, 4, tipLength=0.4)



    # Load the surveillance trajectory
    with np.load("surveillance_traj.npz") as traj_file:
        surveillance_traj = traj_file["traj"]

    # Iterate over the waypoints to plot each point and connect them with arrows
    for i in range(len(surveillance_traj) - 1):
        if i%2 ==1:
            continue
        # Current node
        current_node = surveillance_traj[i]
        next_node = surveillance_traj[i + 2]
        
        # Convert pose to pixel coordinates
        x1, y1 = pose_to_map_pixel(map_metadata, current_node)
        x2, y2 = pose_to_map_pixel(map_metadata, next_node)
        
        distance = norm(np.array([current_node[0], current_node[1]]) - np.array([next_node[0], next_node[1]]))
        
        if distance > 10:
            continue
        # waypoint_arrow_color = (0, 199, 255)
        waypoint_arrow_color = ( 98,208,245)
        # Mark current node
        map_image[
            y1 - 5//2 : y1 + 5//2,
            x1 - 5//2 : x1 + 5//2,
            :
        ] = waypoint_arrow_color
        
        # Draw an arrow between the two waypoints temporarily
        # temp_image = map_image.copy()
        if distance > 10:
            continue
        else:
            draw_arrow(map_image=map_image, start=(x1, y1),end= (x2, y2),i=i)
        # display_map_image(map_image, write=True)  # Show the temporary arrow



    # Display and optionally save the updated map image
    display_map_image(map_image, write=True)



    # save_graph_json()
    # read_and_visualize_graph(on_map=True, catgeories=["table", "chair", "door"])
