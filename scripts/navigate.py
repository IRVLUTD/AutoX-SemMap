import math
import rospy
import actionlib
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
import tf
from tf.transformations import quaternion_from_euler
from utils import read_graph_json
import numpy as np
from std_msgs.msg import Int32

class Navigate:
    def __init__(self) -> None:
        self.move_base_client = actionlib.SimpleActionClient(
            "move_base", MoveBaseAction
        )
        self.move_base_client.wait_for_server()
        self.tf_listener = tf.TransformListener()
        self.goal = MoveBaseGoal()
        self.base_position = [0,0,0]
        self.pause = 0
        rospy.Subscriber("/yes_no", Int32, self.pause_callback)
    
    def pause_callback(self, data):
        self.pause = data.data

    def clear_goal(self):
        self.goal = MoveBaseGoal()

    def set_goal(self, pose, qt=None):
        self.goal.target_pose.header.frame_id = "map"
        
        self.goal.target_pose.pose.position.x = pose[0]
        self.goal.target_pose.pose.position.y = pose[1]

        if qt is not None:
            self.goal.target_pose.pose.orientation.x = qt[0]
            self.goal.target_pose.pose.orientation.y = qt[1]
            self.goal.target_pose.pose.orientation.z = qt[2]
            self.goal.target_pose.pose.orientation.w = qt[3]

    def get_base_position(self):
        while True:
            try:
                self.tf_listener.waitForTransform("map", "base_link", rospy.Time(), rospy.Duration(4.0))
                (trans, rot) = self.tf_listener.lookupTransform("map", "base_link", rospy.Time(0))
                self.base_position = trans
                break
            except tf.ExtrapolationException:
                self.base_position = [0,0,0]
                break

    def compute_orientation(self, pose1, pose2):
        yaw = math.atan2(pose2[1] - pose1[1], pose2[0] - pose1[0])
        qt = quaternion_from_euler(0, 0, yaw)
        return qt

    def navigate_to(
        self,
        pose,
        wait_until=False,
        set_orientation=False,
        orientation_qt=None,
        # move_head=False,
    ):
        if set_orientation:
            self.set_goal(pose, orientation_qt)
        else:
            self.get_base_position()
            if self.base_position == [0,0,0]:
                return None
            else:
                orientation_qt = self.compute_orientation( self.base_position, pose)
            self.set_goal(pose, orientation_qt)
            self.move_base_client.send_goal_and_wait(self.goal)
            wait = self.move_base_client.wait_for_result()
            print(wait)

    def track_trajectory(self, waypoints=[]):
        for waypoint in waypoints:
                if self.pause == 0:
                    self.navigate_to(waypoint)
                else:
                    input("conitue?")

    def navigate_to_object_class(self, graph_file, _class="door", nearest=False):
        graph = read_graph_json(file=graph_file)
        if nearest == False:
            for node, data in graph.nodes(data=True):
                if _class in data["category"]:
                    self.navigate_to(data["pose"]) 
                    break
        else:
            self.get_base_position()
            class_nodes_list = [(data["pose"], node) for node, data in graph.nodes(data=True) if data["category"] == _class]
            class_node_positions = np.array([pose for pose, _ in class_nodes_list])
            print(class_node_positions,"\n")
            min_dist_index = np.argmin(np.linalg.norm((class_node_positions[:, 0:2] - np.array([self.base_position])[:, 0:2]), axis=1))
            nearest_class_position = class_nodes_list[min_dist_index][0]
            print(nearest_class_position)
            self.navigate_to(nearest_class_position)


if __name__=="__main__":
    rospy.init_node('movebase_client_py')
    nav  = Navigate()
    with np.load("surveillance_traj.npz") as traj_file:
        surveillance_traj = traj_file["traj"]  
    print(surveillance_traj)
    nav.track_trajectory(waypoints=surveillance_traj)
    rospy.spin()