#!/usr/bin/env python3
import rospy
from std_msgs.msg import Bool
import os

class saveMap():
    def __init__(self):
        self.is_terminate = False
        rospy.init_node("savemap")
        rospy.Subscriber("explore/exploration_termination", Bool, self.callback_termination)
        #print(f"node initialised successfully")
        rospy.spin()


    def callback_termination(self, is_terminated):
        #print(f"termination condition received")
        assert is_terminated.data == True
        os.system("rosrun map_server map_saver -f ~/map")


if __name__=="__main__":
    saveMap()
    

