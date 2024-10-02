# Point the head using controller
from control_msgs.msg import PointHeadAction, PointHeadGoal
import rospy
import actionlib

class PointHeadClient(object):
    def __init__(self):
        self.client = actionlib.SimpleActionClient(
            "head_controller/point_head", PointHeadAction
        )
        rospy.loginfo("Waiting for head_controller...")
        self.success = self.client.wait_for_server(timeout = rospy.Duration(3.0))
        if (self.success is False):
            rospy.loginfo("no point head controller available")
        else:
            rospy.loginfo("Use head_controller/point_head")

    def look_at(self, x, y, z, frame, duration=1.0):
        """
        Turning head to look at x,y,z
        :param x: x location
        :param y: y location
        :param z: z location
        :param frame: the frame of reference
        :param duration: given time for operation to calcualte the motion plan
        :return:
        """
        goal = PointHeadGoal()
        goal.target.header.stamp = rospy.Time.now()
        goal.target.header.frame_id = frame
        goal.target.point.x = x
        goal.target.point.y = y
        goal.target.point.z = z
        goal.min_duration = rospy.Duration(duration)
        self.client.send_goal(goal)
        self.client.wait_for_result()

if __name__=="__main__":
    rospy.init_node("set_head")
    head_action = PointHeadClient()
    for _ in range(5):
        head_action.look_at(0.75, 0, 0.75, "base_link")
