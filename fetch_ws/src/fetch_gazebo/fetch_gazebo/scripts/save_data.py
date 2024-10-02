from listener import ImageListener
import rospy
import numpy as np
import cv2
import os
import datetime
import time
import sys

class SaveData:
    def __init__(self, time_interval):
        rospy.init_node("img_listen_n_infer")
        self.listener = ImageListener("Fetch")
        time.sleep(5)
        self.time_delay = time_interval
        self.create_directory()

    def create_directory(self):
        # Create a directory named as the current date and time in the current working directory
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.main_dir_name = os.path.join(os.getcwd(), current_time)
        self.pose_dir_name = os.path.join(self.main_dir_name, "pose")
        os.makedirs(self.main_dir_name)
        os.makedirs(self.pose_dir_name)

    def save_data(self):
        data_count = 0
        while not rospy.is_shutdown():
            RT_camera, RT_base = self.listener.get_data_to_save()
            np.savez("{}_pose.npz".format(os.path.join(self.pose_dir_name, "{:06d}".format(data_count))), RT_camera=RT_camera, RT_base=RT_base)
            rospy.sleep(self.time_delay)
            data_count += 1

if __name__ == "__main__":
    saver = SaveData(float(sys.argv[1]))
    saver.save_data()