# Autonomous-Driving-Turtlebot-with-Reinforcement-Learning
Implementation of Q-learning algorithm and Feedback control on turtlebot3_burger in ROS

Content:

Log_learning -> folder containing data and parameters from the learning phase, as well as the Q-table 

Control.py -> functions for robot control, Odometry message processing and setting robot's initial position

Lidar.py -> functions for Lidar message processing and discretization

Qlearning.py -> functions for Q-learning algorithm

Plots.py -> plotting the data from learning phase and Q-table

scan_node.py -> initializing the node for displaying the Lidar measurements and the current state of the agent

learning_node.py -> initializing the node for learning session

control_node.py -> initializing the node for applying the Q-learning algorithm combined with Feedback control

