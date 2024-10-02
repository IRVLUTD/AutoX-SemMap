# dynamic-window-frontier-exploration
This package is developed on top of the existing [explore_lite](https://wiki.ros.org/explore_lite) package.  It leverages a dynamic window approach to efficiently explore unknown environments by identifying and navigating towards frontiers, while switching between a local and the global windows for improved realworld explolration. It also saves the explored map automatically once the exploration stops.

The following parameters can be configured in the launch file to control the behavior of the exploration algorithm:

* _`local_frontier_filter_radius:`_ Filters frontiers within this radius from the robot. If the number of filtered frontiers is less than _min_local_frontiers_, the filter radius is expanded to _global_frontier_filter_radius_ to search through frontiers in a larger area.
  
* _`min_local_frontiers:`_ The minimum number of frontiers required within the local_frontier_filter_radius to avoid expanding the search radius.

* _`min_global_frontiers:`_ The minimum number of frontiers required within the _global_frontier_filter_radius_ to continue exploration. If the number of filterede frontiers falls below this threshold, the exploration stops.

* _`global_frontier_filter_radius:`_ The radius used to filter frontiers when the number of local frontiers is below _min_local_frontiers_.

* _`min_frontier_spacing:`_ The minimum distance required between new frontier and the existing frontiers to consider adding it to the frontiers list.


# m-explore

[![Build Status](http://build.ros.org/job/Kdev__m_explore__ubuntu_xenial_amd64/badge/icon)](http://build.ros.org/job/Kdev__m_explore__ubuntu_xenial_amd64)

ROS packages for multi robot exploration.

Installing
----------

Packages are released for ROS Kinetic and ROS Lunar.

```
	sudo apt install ros-${ROS_DISTRO}-multirobot-map-merge ros-${ROS_DISTRO}-explore-lite
```

Building
--------

Build as standard catkin packages. There are no special dependencies needed
(use rosdep to resolve dependencies in ROS). You should use brach specific for
your release i.e. `kinetic-devel` for kinetic. Master branch is for latest ROS.

WIKI
----

Packages are documented at ROS wiki.
* [explore_lite](http://wiki.ros.org/explore_lite)
* [multirobot_map_merge](http://wiki.ros.org/multirobot_map_merge)

COPYRIGHT
---------

Packages are licensed under BSD license. See respective files for details.
