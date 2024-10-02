#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseWithCovarianceStamped
import numpy as np

def publish_initial_pose():
    rospy.init_node('initial_pose_publisher')

    pub = rospy.Publisher('/initialpose', PoseWithCovarianceStamped, queue_size=10)
    rospy.sleep(5) 

    # Covariance matrix for (x, y, θ)
    cov_x = 1.0  # 1 meter s.d in x
    cov_y = 1.0  # 1 meter sd. in y
    cov_theta = np.deg2rad(10) ** 2  # 10 degree s.d in θ, (radians)

    # Full 6x6 covariance matrix 
    covariance_matrix = [cov_x, 0, 0, 0, 0, 0,
                        0, cov_y, 0, 0, 0, 0,
                        0, 0, 1e-9, 0, 0, 0,  
                        0, 0, 0, 1e-9, 0, 0,  
                        0, 0, 0, 0, 1e-9, 0,  
                        0, 0, 0, 0, 0, cov_theta]

    # Applying this covariance to the initial pose
    initial_pose = PoseWithCovarianceStamped()
    initial_pose.header.frame_id = "map"

    initial_pose.pose.pose.position.x = 0.0
    initial_pose.pose.pose.position.y = 0.0
    initial_pose.pose.pose.position.z = 0.0
    initial_pose.pose.pose.orientation.x = 0.0
    initial_pose.pose.pose.orientation.y = 0.0
    initial_pose.pose.pose.orientation.z = 0.0
    initial_pose.pose.pose.orientation.w = 1.0


    initial_pose.pose.covariance = covariance_matrix

    pub.publish(initial_pose)
    rospy.loginfo("Published initial pose with zero position and orientation.")

if __name__ == '__main__':
    try:
        publish_initial_pose()
    except rospy.ROSInterruptException:
        pass
