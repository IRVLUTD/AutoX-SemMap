#!/bin/bash
set -e

# ROS Noetic Environment Setup
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source /opt/ros/noetic/setup.bash

echo "==== AutoX-SemMap Docker Ready ===="

exec bash