cmake_minimum_required(VERSION 2.8.3)
project(master_rad)

## Find catkin and any catkin packages
find_package(catkin REQUIRED COMPONENTS rospy std_msgs )

## Declare a catkin package
catkin_package()

# Install python scripts
catkin_install_python(PROGRAMS scripts/control_node.py
				scripts/Control.py
				scripts/feedback_control_node.py
				scripts/learning_node.py
				scripts/Lidar.py
				scripts/Plots.py
				scripts/Qlearning.py
				scripts/scan_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

## Build talker and listener
include_directories(include ${catkin_INCLUDE_DIRS})
