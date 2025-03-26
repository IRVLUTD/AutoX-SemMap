FROM osrf/ros:noetic-desktop-full

## Install requried libraries
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake wget curl git unzip pkg-config && \
    rm -rf /var/lib/apt/lists/*

## Install Python3.9
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.9 \
        python3.9-dev \
        python3-distutils \
        python3-pip \
        python3-wheel \
        libeigen3-dev \
        libpcl-dev && \
        rm -rf /var/lib/apt/lists/* && \
    cd /usr/local/bin && \
    ln -s /usr/bin/python3.9 /usr/bin/python && \
    ln -s /usr/bin/pip3 pip

## Install PyPI packages
RUN pip install --upgrade pip setuptools numpy
RUN pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements.txt and install Python packages
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install -r /tmp/requirements.txt && rm -f /tmp/requirements.txt

# Install additional ROS Noetic packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-tf \
    ros-noetic-cv-bridge \
    ros-noetic-pcl-conversions \
    ros-noetic-xacro \
    ros-noetic-robot-state-publisher \
    ros-noetic-rviz \
    ros-noetic-image-transport \
    ros-noetic-image-transport-plugins \
    ros-noetic-pcl-ros \
    ros-noetic-eigen-conversions \
    ros-noetic-tf-conversions \
    ros-noetic-catkin \
    ros-noetic-tf2-sensor-msgs \
    ros-noetic-gazebo-ros-pkgs \
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    ros-noetic-navigation \
    ros-noetic-move-base \
    ros-noetic-slam-gmapping \
    ros-noetic-grid-map-costmap-2d \
    ros-noetic-teleop-twist-joy \
    ros-noetic-pointcloud-to-laserscan \
    ros-noetic-rosbridge-server \
    ros-noetic-robot-controllers \
    ros-noetic-gazebo-ros-control \
    ros-noetic-rgbd-launch \
    ros-noetic-moveit \
    ros-noetic-trac-ik \
    ros-noetic-moveit-python \
    && rm -rf /var/lib/apt/lists/*

# Clone AutoX-SemMap repository
RUN git clone https://github.com/IRVLUTD/AutoX-SemMap.git /home

# Load ROS environment at each run
COPY ./entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]

CMD ["bash"]