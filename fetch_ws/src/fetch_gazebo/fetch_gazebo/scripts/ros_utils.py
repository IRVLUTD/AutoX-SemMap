import tf
import numpy as np


def ros_qt_to_rt(quat, posn):
    """
    Converts ROS quaternion and position to a 4x4 transformation matrix.
    :param quat: List of quaternion [x, y, z, w]
    :param posn: List of position [x, y, z]
    :return: 4x4 numpy array
    """
    mat = tf.transformations.quaternion_matrix(quat)
    mat[:3, 3] = np.array(posn)
    return mat


def ros_pose_to_rt(pose):
    """
    Converts ROS Pose to a 4x4 transformation matrix.
    :param pose: ROS Pose message
    :return: 4x4 numpy array
    """
    quat = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    posn = [pose.position.x, pose.position.y, pose.position.z]
    return ros_qt_to_rt(quat, posn)


def rt_to_ros_pose(mat):
    """
    Converts a 4x4 transformation matrix to ROS Pose.
    :param mat: 4x4 numpy array
    :return: ROS Pose message
    """
    quat = tf.transformations.quaternion_from_matrix(mat)
    posn = mat[:3, 3]
    pose = tf.Pose()
    pose.orientation.x = quat[0]
    pose.orientation.y = quat[1]
    pose.orientation.z = quat[2]
    pose.orientation.w = quat[3]
    pose.position.x = posn[0]
    pose.position.y = posn[1]
    pose.position.z = posn[2]
    return pose
