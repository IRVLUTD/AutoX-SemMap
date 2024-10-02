# Autonomous Exploration and Semantic Updating of Large-Scale Indoor Environments with Mobile Robots​​

<p style="text-align:center;">Sai Haneesh Allu, Itay Kadosh, Tyler Summers, Yu Xiang</p>

<center>

[arXiv](https://arxiv.org/abs/2409.15493)  **|** [Project WebPage](https://irvlutd.github.io/SemanticMapping/)  **|** [Video](https://youtu.be/q3bfSFYbX08)

</center>


Code base for autonomous exploration, construction and update of semantic map in real-time. 

# Index

1. [Installation](#installation)
2. [Initialization](Initialization)
3. [Mapping and Exploration](Mapping-and-Exploration)
4. [Environment Traversal planning](Environment-Traversal-planning)
5. [Semantic Map Construction and Update](Semantic-Map-Construction-and-Update)

<br/>
<br/>
<br/>

# Installation
The following subsections provides detailed installation guidelines related to workspace setup, dependencies and other requirements to test this work effectively. 

## A.  Install ROS and Gazebo
This code is tested on ros noetic version. Detailed installation instructions are found [here](http://wiki.ros.org/noetic/Installation/Ubuntu).To install ROS Noetic, execute the following commands in your terminal:
```
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

sudo apt install -y curl

curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -

sudo apt update

sudo apt install -y ros-noetic-desktop-full

echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

source ~/.bashrc
```
For compatibility with ROS Noetic, Gazebo 11 is recommended. Detailed installation instructions are found [here](https://classic.gazebosim.org/tutorials?tut=install_ubuntu&cat=install#Defaultinstallation:one-liner).

```
sudo sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list'

wget https://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -

sudo apt-get update

sudo apt-get install -y gazebo11
```
## B. Create Conda Environment
We strongly recommend using a virtual environment for this work, preferably Anaconda or Miniconda. Create a new environment as follows:
```
conda create -n sem-map python==3.9
conda activate sem-map
```

## C. Install dependencies
This script will install the ros dependencies required for this work.
```
./install_ros_dependencies.sh
```
Next, install the python modules required.
```
pip install -r requirements.txt
```

## D. Compiling workspace
Compile and source the ROS workspace using the following commands:
```
cd fetch_ws
catkin_make
source devel/setup.bash
```

> If the compilation doesn't conisder python3 by default, compile with the following command. Make sure to use correct PYTHONPATH.
```
catkin_make -DPYTHON_EXECUTABLE=/usr/bin/python3
source devel/setp.bash
```

## E. Install Robokit
Please refer to the instructions [here](robokit/README.md) to install the robokit module. Robokit is a stand alone module and is not related to the ros workspace here. Therefore, ***do not source the workspace during instalaltion or while running Robokit***. 

<br/>
<br/>
<br/>

# Initialization
This section covers the steps to start the simulation environment and spawn the robot in the scene. 
## Launch environment and spawn the robot
```
roslaunch aws_robomaker_small_house_world small_house.launch gui:=true
roslaunch fetch_gazebo spawn_robot.launch
```

<br/>
<br/>
<br/>

# Mapping and Exploration
Once system has been initialized, we proceed to explore and map the environment, while also recording robot's camera pose and base_link pose. To achieve this, follow the steps below in this sequence. 

## A. Mapping
Starts the GMapping ROS node. 
```
roslaunch fetch_navigation fetch_mapping.launch
```
## B. Record robot trajectory
This script first creates a data-folder of format <Year-month-date_Hour-Minute-Seconds>. Then saves the data points in .npz format. Specify the time-interval between consecutive data points as the argument.
```
cd scripts
python save_data.py <time-interval>
```
## C. Exploration
This command launches the explroation node. When the exploration ends, it saves  ***map.pgm*** and ***map.yaml*** files in the user's HOME directory. 
```
roslaunch explore_lite explore_n_save.launch
```

<br/>
<br/>
<br/>

# Environment Traversal planning
This section describes how to plan the robot's traversal through the environment.
```
cd scripts
```

## A. Extract the robot exploration trajectory points
From the saved data-folder at the end of exploration, first get the recorded robot poses. These poses are saved by default in ***robot_trajectory.json***.
```
python extract_robot_trajectory.py <data-folder>
```
## B. Generate traversal trajectory - Travelling Salesman Problem 
Next, sample the poses and plan the sequence to vsit the sampled points at low cost, using a Traveling Salesman Problem fomrulation.  
```
python tsp_surveillance_trajectory.py robot_trajectory.json
```
This saves the sequence of sampled points as ***surveillance_traj.npz*** .

<br/>
<br/>
<br/>

# Semantic Map Construction and Update
To construct or update the semantic map, the robot first needs to localize itself in the built map and traverse the environment to see the things. For this, either move the robot to initial position (x=0,y=0.yaw=0) in gazebo or delete the robot and spawn it again. 

## A. Localization
```
source feth_ws/devel/setup.bash
roslaunch fetch navigation fetch_localize.launch map_file:=<absolute-path-of-map.yaml>
```
In another termianl publish the initial pose of the robot. This 
```
rosrun fetch_navigation pub_initial_pose.py
```

## B. Construction
To construct the semantic map, start the object detection and segmentation module, and perform object association. Run the following scripts simultaneously in two terminals to construct the semnantic map while traversing the environment.  
```
cd robokit
python semanticmap_construction.py
```

```
cd scripts
python navigate.py
```
Once the traversal is completed, close the scripts and the semantic map is stored as ***graph.json***
## C. Update
Similarly run the following scripts simultaneously in two terminals to construct the semnantic map while traversing the environment.  
```
cd robokit
python semanticmap_update.py
```

```
cd scripts
python navigate.py
```
Once the traversal is completed, close the scripts and the updated semantic map is stored as ***graph_updated.json***

<br/>
<br/>
<br/>
<br/>


# Citation
Please cite this work if it helps in your research
```
@inproceedings{allu2024semanticmapping,
      title={Autonomous Exploration and Semantic Updating of Large-Scale Indoor Environments with Mobile Robots},
      author={Allu, Sai Haneesh and Kadosh, Itay and Summers, Tyler and Xiang, Yu},
      journal={arXiv preprint arXiv:2409.15493},
      year={2024}
    }
```