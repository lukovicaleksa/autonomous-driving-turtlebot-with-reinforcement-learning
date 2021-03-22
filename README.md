# autonomous-driving-turtlebot-with-reinforcement-learning
Implementation of Q-learning algorithm and Feedback control for the mobile robot (turtlebot3_burger) in ROS.

 * Algorithm is implemented from scratch.
 * To run the code in gazebo simulator, export the model (burger) and roslaunch turtlebot3_world.launch file.
 * To run the code live on a physical TurtleBot, the ROS_MASTER_URI and ROS_HOSTNAME need to be set via the terminal by editing the ~/.bashrc script.
 * After setting up the environment, rosrun the desired nodes!
 * Link to video: https://www.youtube.com/watch?v=zw1BCfku1Dc&t=3s

![radno okruzenje 3](https://user-images.githubusercontent.com/72970001/98345474-02e69500-2015-11eb-8552-9949ddae6cab.jpeg)

# Content
* [Log_feedback_1, 2, 3](Log_feedback_1) -> folders containing data and parameters from Feedback Control algorithm testing
* [Log_learning](Log_learning) -> folder containing data and parameters from the learning phase, as well as the Q-table 
* [rqt_graphs](rqt_graphs) -> folder containing rqt graphs in ROS
* [Control.py](Control.py) -> functions for robot control, Odometry message processing and setting robot's initial position
* [Lidar.py](Lidar.py) -> functions for Lidar message processing and discretization
* [Qlearning.py](Qlearning.py) -> functions for Q-learning algorithm
* [Plots.py](Plots.py) -> plotting the data from learning phase and Q-table
* [scan_node.py](scan_node.py) -> initializing the node for displaying the Lidar measurements and the current state of the agent
* [learning_node.py](learning_node.py) -> initializing the node for learning session
* [feedback_control_node.py](feedback_control_node.py) -> initializing the node for applying Feedback Control algorithm
* [control_node.py](control_node.py) -> initializing the node for applying the Q-learning algorithm combined with Feedback control
* [learning phase flow chart](learning_phase_flow_chart.png) -> flow chart of the learning phase of the algorithm
    

