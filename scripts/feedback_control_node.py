#! /usr/bin/env python

import rospy
from time import time
from time import sleep
from datetime import datetime
import matplotlib.pyplot as plt

import sys
DATA_PATH = '/home/maestro/catkin_ws/src/master_rad/Data'
MODULES_PATH = '/home/maestro/catkin_ws/src/master_rad/scripts'
sys.path.insert(0, MODULES_PATH)

from Control import *

X_INIT = 0.0
Y_INIT = 0.0
THETA_INIT = 0.0
X_GOAL = 3
Y_GOAL = 2
THETA_GOAL = 15

# init trajectory
X_traj = np.array([])
Y_traj = np.array([])
THETA_traj = np.array([])
X_goal = np.array([])
Y_goal = np.array([])
THETA_goal = np.array([])

# log directory
LOG_DIR = DATA_PATH + '/Log_feedback'

if __name__ == '__main__':
    try:
        # init nodes
        rospy.init_node('feedback_control_node', anonymous = False)
        rate = rospy.Rate(10)

        # init topics
        setPosPub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size = 10)
        velPub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)

        # init log files
        log_sim_params = open(LOG_DIR+'/LogSimParams.txt','w+')

        # log simulation params
        text = 'Simulation parameters: \r\n'
        text = text + 'k_rho = %.3f \r\n' % K_RO
        text = text + 'k_alpha = %.3f \r\n' % K_ALPHA
        text = text + 'k_beta = %.3f \r\n' % K_BETA
        text = text + 'v_const = %.3f \r\n' % V_CONST
        log_sim_params.write(text)

        # close log files
        log_sim_params.close()

        print('\r\n' + text)

        # check stability
        stab_dict = { True : 'Satisfied!', False : 'Not Satisfied!'}

        print('Stability Condition: ' + stab_dict[check_stability(K_RO, K_ALPHA, K_BETA)])
        print('Strong Stability Condition: ' + stab_dict[check_strong_stability(K_RO, K_ALPHA, K_BETA)])

        # because of the video recording
        sleep(5)

        # main loop
        while not rospy.is_shutdown():

            # Wait for odometry message
            odomMsg = rospy.wait_for_message('/odom', Odometry)

            # Get robot position and orientation
            ( x , y ) = getPosition(odomMsg)
            theta = getRotation(odomMsg)

            # Update trajectory
            X_traj = np.append(X_traj, x)
            Y_traj = np.append(Y_traj, y)
            THETA_traj = np.append(THETA_traj, degrees(theta))
            X_goal = np.append(X_goal, X_GOAL)
            Y_goal = np.append(Y_goal, Y_GOAL)
            THETA_goal = np.append(THETA_goal, THETA_GOAL)

            status = robotFeedbackControl(velPub, x, y, theta, X_GOAL, Y_GOAL, radians(THETA_GOAL))

            text = '\r\n'
            text = text + '\r\nx :       %.2f -> %.2f [m]' % (x, X_GOAL)
            text = text + '\r\ny :       %.2f -> %.2f [m]' % (y, Y_GOAL)
            text = text + '\r\ntheta :   %.2f -> %.2f [degrees]' % (degrees(theta), THETA_GOAL)

            if status == 'Goal position reached!':
                # stop the robot
                robotStop(velPub)

                # log trajectory
                np.savetxt(LOG_DIR+'/X_traj.csv', X_traj, delimiter = ' , ')
                np.savetxt(LOG_DIR+'/Y_traj.csv', Y_traj, delimiter = ' , ')
                np.savetxt(LOG_DIR+'/THETA_traj.csv', THETA_traj, delimiter = ' , ')
                np.savetxt(LOG_DIR+'/X_goal.csv', X_goal, delimiter = ' , ')
                np.savetxt(LOG_DIR+'/Y_goal.csv', Y_goal, delimiter = ' , ')
                np.savetxt(LOG_DIR+'/THETA_goal.csv', THETA_goal, delimiter = ' , ')


                rospy.signal_shutdown('Goal position reached! End of simulation!')
                text = text + '\r\n\r\nGoal position reached! End of simulation!'

            print(text)

    except rospy.ROSInterruptException:
        robotStop(velPub)
        print('Simulation terminated!')
        pass
