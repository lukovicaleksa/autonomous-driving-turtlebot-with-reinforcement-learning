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

from Qlearning import *
from Lidar import *
from Control import *

# Episode parameters
MAX_EPISODES = 50
MAX_STEPS_PER_EPISODE = 50
MIN_TIME_BETWEEN_ACTIONS = 0.0

# Learning parameters
ALPHA = 0.5
GAMMA = 0.9

T_INIT = 25
T_GRAD = 0.95
T_MIN = 0.001

EPSILON_INIT = 0.9
EPSILON_GRAD = 0.96
EPSILON_MIN = 0.05

# 1 - Softmax , 2 - Epsilon greedy
EXPLORATION_FUNCTION = 1

# Initial position
X_INIT = -0.4
Y_INIT = -0.4
THETA_INIT = 45.0

RANDOM_INIT_POS = False

# Log file directory
LOG_FILE_DIR = DATA_PATH + '/Log_learning'

# Q table source file
Q_SOURCE_DIR = ''

def initLearning():
    global actions, state_space, Q_table
    actions = createActions()
    state_space = createStateSpace()
    if Q_SOURCE_DIR != '':
        Q_table = readQTable(Q_SOURCE_DIR+'/Qtable.csv')
    else:
        Q_table = createQTable(len(state_space),len(actions))
    print('Initial Q-table:')
    print(Q_table)

def initParams():
    global T, EPSILON, alpha, gamma
    global ep_steps, ep_reward, episode, steps_per_episode, reward_per_episode, T_per_episode, EPSILON_per_episode
    global ep_reward_arr, reward_max_per_episode, reward_min_per_episode, reward_avg_per_episode
    global crash, t_ep, t_per_episode, t_sim_start, t_step
    global log_sim_info, log_sim_params
    global now_start, now_stop
    global robot_in_pos, first_action_taken

    # Init log files
    log_sim_info = open(LOG_FILE_DIR+'/LogInfo.txt','w+')
    log_sim_params = open(LOG_FILE_DIR+'/LogParams.txt','w+')

    # Learning parameters
    T = T_INIT
    EPSILON = EPSILON_INIT
    alpha = ALPHA
    gamma = GAMMA

    # Episodes, steps, rewards
    ep_steps = 0
    ep_reward = 0
    episode = 1
    crash = 0
    reward_max_per_episode = np.array([])
    reward_min_per_episode = np.array([])
    reward_avg_per_episode = np.array([])
    ep_reward_arr = np.array([])
    steps_per_episode = np.array([])
    reward_per_episode = np.array([])

    # initial position
    robot_in_pos = False
    first_action_taken = False

    # init time
    t_0 = rospy.Time.now()
    t_start = rospy.Time.now()

    # init timer
    while not (t_start > t_0):
        t_start = rospy.Time.now()

    t_ep = t_start
    t_sim_start = t_start
    t_step = t_start

    T_per_episode = np.array([])
    EPSILON_per_episode = np.array([])
    t_per_episode = np.array([])

    # Date
    now_start = datetime.now()
    dt_string_start = now_start.strftime("%d/%m/%Y %H:%M:%S")

    # Log date to files
    text = '\r\n' + 'SIMULATION START ==> ' + dt_string_start + '\r\n\r\n'
    print(text)
    log_sim_info.write(text)
    log_sim_params.write(text)

    # Log simulation parameters
    text = '\r\nSimulation parameters: \r\n'
    text = text + '--------------------------------------- \r\n'
    if RANDOM_INIT_POS:
        text = text + 'INITIAL POSITION = RANDOM \r\n'
    else:
        text = text + 'INITIAL POSITION = ( %.2f , %.2f , %.2f ) \r\n' % (X_INIT,Y_INIT,THETA_INIT)
    text = text + '--------------------------------------- \r\n'
    text = text + 'MAX_EPISODES = %d \r\n' % MAX_EPISODES
    text = text + 'MAX_STEPS_PER_EPISODE = %d \r\n' % MAX_STEPS_PER_EPISODE
    text = text + 'MIN_TIME_BETWEEN_ACTIONS = %.2f s \r\n' % MIN_TIME_BETWEEN_ACTIONS
    text = text + '--------------------------------------- \r\n'
    text = text + 'ALPHA = %.2f \r\n' % ALPHA
    text = text + 'GAMMA = %.2f \r\n' % GAMMA
    if EXPLORATION_FUNCTION == 1:
        text = text + 'T_INIT = %.3f \r\n' % T_INIT
        text = text + 'T_GRAD = %.3f \r\n' % T_GRAD
        text = text + 'T_MIN = %.3f \r\n' % T_MIN
    else:
        text = text + 'EPSILON_INIT = %.3f \r\n' % EPSILON_INIT
        text = text + 'EPSILON_GRAD = %.3f \r\n' % EPSILON_GRAD
        text = text + 'EPSILON_MIN = %.3f \r\n' % EPSILON_MIN
    text = text + '--------------------------------------- \r\n'
    text = text + 'MAX_LIDAR_DISTANCE = %.2f \r\n' % MAX_LIDAR_DISTANCE
    text = text + 'COLLISION_DISTANCE = %.2f \r\n' % COLLISION_DISTANCE
    text = text + 'ZONE_0_LENGTH = %.2f \r\n' % ZONE_0_LENGTH
    text = text + 'ZONE_1_LENGTH = %.2f \r\n' % ZONE_1_LENGTH
    text = text + '--------------------------------------- \r\n'
    text = text + 'CONST_LINEAR_SPEED_FORWARD = %.3f \r\n' % CONST_LINEAR_SPEED_FORWARD
    text = text + 'CONST_ANGULAR_SPEED_FORWARD = %.3f \r\n' % CONST_ANGULAR_SPEED_FORWARD
    text = text + 'CONST_LINEAR_SPEED_TURN = %.3f \r\n' % CONST_LINEAR_SPEED_TURN
    text = text + 'CONST_ANGULAR_SPEED_TURN = %.3f \r\n' % CONST_ANGULAR_SPEED_TURN
    log_sim_params.write(text)

if __name__ == '__main__':
    try:
        global actions, state_space, Q_table
        global T, EPSILON, alpha, gamma
        global ep_steps, ep_reward, episode, steps_per_episode, reward_per_episode, T_per_episode, EPSILON_per_episode
        global ep_reward_arr, reward_max_per_episode, reward_min_per_episode, reward_avg_per_episode
        global crash, t_ep, t_per_episode, t_sim_start, t_step
        global log_sim_info, log_sim_params
        global now_start, now_stop
        global robot_in_pos, first_action_taken

        rospy.init_node('learning_node', anonymous = False)
        rate = rospy.Rate(10)

        setPosPub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size = 10)
        velPub = rospy.Publisher('/cmd_vel', Twist, queue_size = 10)

        initLearning()
        initParams()
        #sleep(5)

        # main loop
        while not rospy.is_shutdown():
            msgScan = rospy.wait_for_message('/scan', LaserScan)


            # Secure the minimum time interval between 2 actions
            step_time = (rospy.Time.now() - t_step).to_sec()
            if step_time > MIN_TIME_BETWEEN_ACTIONS:
                t_step = rospy.Time.now()
                if step_time > 2:
                    text = '\r\nTOO BIG STEP TIME: %.2f s' % step_time
                    print(text)
                    log_sim_info.write(text+'\r\n')

                # End of Learning
                if episode > MAX_EPISODES:
                    # simulation time
                    sim_time = (rospy.Time.now() - t_sim_start).to_sec()
                    sim_time_h = sim_time // 3600
                    sim_time_m = ( sim_time - sim_time_h * 3600 ) // 60
                    sim_time_s = sim_time - sim_time_h * 3600 - sim_time_m * 60

                    # real time
                    now_stop = datetime.now()
                    dt_string_stop = now_stop.strftime("%d/%m/%Y %H:%M:%S")
                    real_time_delta = (now_stop - now_start).total_seconds()
                    real_time_h = real_time_delta // 3600
                    real_time_m = ( real_time_delta - real_time_h * 3600 ) // 60
                    real_time_s = real_time_delta - real_time_h * 3600 - real_time_m * 60

                    # Log learning session info to file
                    text = '--------------------------------------- \r\n\r\n'
                    text = text + 'MAX EPISODES REACHED(%d), LEARNING FINISHED ==> ' % MAX_EPISODES + dt_string_stop + '\r\n'
                    text = text + 'Simulation time: %d:%d:%d  h/m/s \r\n' % (sim_time_h, sim_time_m, sim_time_s)
                    text = text + 'Real time: %d:%d:%d  h/m/s \r\n' % (real_time_h, real_time_m, real_time_s)
                    print(text)
                    log_sim_info.write('\r\n'+text+'\r\n')
                    log_sim_params.write(text+'\r\n')

                    # Log data to file
                    saveQTable(LOG_FILE_DIR+'/Qtable.csv', Q_table)
                    np.savetxt(LOG_FILE_DIR+'/StateSpace.csv', state_space, '%d')
                    np.savetxt(LOG_FILE_DIR+'/steps_per_episode.csv', steps_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/reward_per_episode.csv', reward_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/T_per_episode.csv', T_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/EPSILON_per_episode.csv', EPSILON_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/reward_min_per_episode.csv', reward_min_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/reward_max_per_episode.csv', reward_max_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/reward_avg_per_episode.csv', reward_avg_per_episode, delimiter = ' , ')
                    np.savetxt(LOG_FILE_DIR+'/t_per_episode.csv', t_per_episode, delimiter = ' , ')

                    # Close files and shut down node
                    log_sim_info.close()
                    log_sim_params.close()
                    rospy.signal_shutdown('End of learning')
                else:
                    ep_time = (rospy.Time.now() - t_ep).to_sec()
                    # End of en Episode
                    if crash or ep_steps >= MAX_STEPS_PER_EPISODE:
                        robotStop(velPub)
                        if crash:
                            # get crash position
                            odomMsg = rospy.wait_for_message('/odom', Odometry)
                            ( x_crash , y_crash ) = getPosition(odomMsg)
                            theta_crash = degrees(getRotation(odomMsg))

                        t_ep = rospy.Time.now()
                        reward_min = np.min(ep_reward_arr)
                        reward_max = np.max(ep_reward_arr)
                        reward_avg = np.mean(ep_reward_arr)
                        now = datetime.now()
                        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

                        text = '---------------------------------------\r\n'
                        if crash:
                            text = text + '\r\nEpisode %d ==> CRASH {%.2f,%.2f,%.2f}    ' % (episode, x_crash, y_crash, theta_crash) + dt_string
                        elif ep_steps >= MAX_STEPS_PER_EPISODE:
                            text = text + '\r\nEpisode %d ==> MAX STEPS PER EPISODE REACHED {%d}    ' % (episode, MAX_STEPS_PER_EPISODE) + dt_string
                        else:
                            text = text + '\r\nEpisode %d ==> UNKNOWN TERMINAL CASE    ' % episode + dt_string
                        text = text + '\r\nepisode time: %.2f s (avg step: %.2f s) \r\n' % (ep_time, ep_time / ep_steps)
                        text = text + 'episode steps: %d \r\n' % ep_steps
                        text = text + 'episode reward: %.2f \r\n' % ep_reward
                        text = text + 'episode max | avg | min reward: %.2f | %.2f | %.2f \r\n' % (reward_max, reward_avg, reward_min)
                        if EXPLORATION_FUNCTION == 1:
                            text = text + 'T = %f \r\n' % T
                        else:
                            text = text + 'EPSILON = %f \r\n' % EPSILON
                        print(text)
                        log_sim_info.write('\r\n'+text)

                        steps_per_episode = np.append(steps_per_episode, ep_steps)
                        reward_per_episode = np.append(reward_per_episode, ep_reward)
                        T_per_episode = np.append(T_per_episode, T)
                        EPSILON_per_episode = np.append(EPSILON_per_episode, EPSILON)
                        t_per_episode = np.append(t_per_episode, ep_time)
                        reward_min_per_episode = np.append(reward_min_per_episode, reward_min)
                        reward_max_per_episode = np.append(reward_max_per_episode, reward_max)
                        reward_avg_per_episode = np.append(reward_avg_per_episode, reward_avg)
                        ep_reward_arr = np.array([])
                        ep_steps = 0
                        ep_reward = 0
                        crash = 0
                        robot_in_pos = False
                        first_action_taken = False
                        if T > T_MIN:
                            T = T_GRAD * T
                        if EPSILON > EPSILON_MIN:
                            EPSILON = EPSILON_GRAD * EPSILON
                        episode = episode + 1
                    else:
                        ep_steps = ep_steps + 1
                        # Initial position
                        if not robot_in_pos:
                            robotStop(velPub)
                            ep_steps = ep_steps - 1
                            first_action_taken = False
                            # init pos
                            if RANDOM_INIT_POS:
                                ( x_init , y_init , theta_init ) = robotSetRandomPos(setPosPub)
                            else:
                                ( x_init , y_init , theta_init ) = robotSetPos(setPosPub, X_INIT, Y_INIT, THETA_INIT)

                            odomMsg = rospy.wait_for_message('/odom', Odometry)
                            ( x , y ) = getPosition(odomMsg)
                            theta = degrees(getRotation(odomMsg))
                            # check init pos
                            if abs(x-x_init) < 0.01 and abs(y-y_init) < 0.01 and abs(theta-theta_init) < 1:
                                robot_in_pos = True
                                #sleep(2)
                            else:
                                robot_in_pos = False
                        # First acion
                        elif not first_action_taken:
                            ( lidar, angles ) = lidarScan(msgScan)
                            ( state_ind, x1, x2 ,x3 ,x4 ) = scanDiscretization(state_space, lidar)
                            crash = checkCrash(lidar)

                            if EXPLORATION_FUNCTION == 1 :
                                ( action, status_strat ) = softMaxSelection(Q_table, state_ind, actions, T)
                            else:
                                ( action, status_strat ) = epsiloGreedyExploration(Q_table, state_ind, actions, T)

                            status_rda = robotDoAction(velPub, action)

                            prev_lidar = lidar
                            prev_action = action
                            prev_state_ind = state_ind

                            first_action_taken = True

                            if not (status_strat == 'softMaxSelection => OK' or status_strat == 'epsiloGreedyExploration => OK'):
                                print('\r\n', status_strat, '\r\n')
                                log_sim_info.write('\r\n'+status_strat+'\r\n')

                            if not status_rda == 'robotDoAction => OK':
                                print('\r\n', status_rda, '\r\n')
                                log_sim_info.write('\r\n'+status_rda+'\r\n')

                        # Rest of the algorithm
                        else:
                            ( lidar, angles ) = lidarScan(msgScan)
                            ( state_ind, x1, x2 ,x3 ,x4 ) = scanDiscretization(state_space, lidar)
                            crash = checkCrash(lidar)

                            ( reward, terminal_state ) = getReward(action, prev_action, lidar, prev_lidar, crash)

                            ( Q_table, status_uqt ) = updateQTable(Q_table, prev_state_ind, action, reward, state_ind, alpha, gamma)

                            if EXPLORATION_FUNCTION == 1:
                                ( action, status_strat ) = softMaxSelection(Q_table, state_ind, actions, T)
                            else:
                                ( action, status_strat ) = epsiloGreedyExploration(Q_table, state_ind, actions, T)

                            status_rda = robotDoAction(velPub, action)

                            if not status_uqt == 'updateQTable => OK':
                                print('\r\n', status_uqt, '\r\n')
                                log_sim_info.write('\r\n'+status_uqt+'\r\n')
                            if not (status_strat == 'softMaxSelection => OK' or status_strat == 'epsiloGreedyExploration => OK'):
                                print('\r\n', status_strat, '\r\n')
                                log_sim_info.write('\r\n'+status_strat+'\r\n')
                            if not status_rda == 'robotDoAction => OK':
                                print('\r\n', status_rda, '\r\n')
                                log_sim_info.write('\r\n'+status_rda+'\r\n')

                            ep_reward = ep_reward + reward
                            ep_reward_arr = np.append(ep_reward_arr, reward)
                            prev_lidar = lidar
                            prev_action = action
                            prev_state_ind = state_ind

    except rospy.ROSInterruptException:
        robotStop(velPub)
        print('Simulation terminated!')
        pass
