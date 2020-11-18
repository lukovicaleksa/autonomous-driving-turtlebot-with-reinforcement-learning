#! /usr/bin/env python

import numpy as np
from math import *
import matplotlib.pyplot as plt

LOG_FILE_DIR_1 = 'Log_feedback_1'
LOG_FILE_DIR_2 = 'Log_feedback_2'
LOG_FILE_DIR_3 = 'Log_feedback_3'

def extract_from_file(log_file_dir):
    x_traj = np.genfromtxt(log_file_dir+'/X_traj.csv', delimiter = ' , ')
    y_traj = np.genfromtxt(log_file_dir+'/Y_traj.csv', delimiter = ' , ')
    theta_traj = np.genfromtxt(log_file_dir+'/THETA_traj.csv', delimiter = ' , ')
    x_goal = np.genfromtxt(log_file_dir+'/X_goal.csv', delimiter = ' , ')
    y_goal = np.genfromtxt(log_file_dir+'/Y_goal.csv', delimiter = ' , ')
    theta_goal = np.genfromtxt(log_file_dir+'/THETA_goal.csv', delimiter = ' , ')

    file_params = open(log_file_dir+'/LogSimParams.txt', 'r')
    lines = file_params.readlines()
    file_params.close()

    k_rho = float(lines[1].split()[-1])
    k_alpha = float(lines[2].split()[-1])
    k_beta = float(lines[3].split()[-1])
    v_const = float(lines[4].split()[-1])

    return (x_traj, y_traj, theta_traj, x_goal, y_goal, theta_goal, k_rho, k_alpha, k_beta, v_const)

# Plot feedback control trajectory
def plot_feedback_control(log_file_1, log_file_2, log_file_3):
    # log file 1
    x_traj_1, y_traj_1, theta_traj_1, x_goal_1, y_goal_1, theta_goal_1, k_rho_1, k_alpha_1, k_beta_1, v_const_1 = extract_from_file(LOG_FILE_DIR_1)

    # log file 2
    x_traj_2, y_traj_2, theta_traj_2, x_goal_2, y_goal_2, theta_goal_2, k_rho_2, k_alpha_2, k_beta_2, v_const_2 = extract_from_file(LOG_FILE_DIR_2)

    # log file 2
    x_traj_3, y_traj_3, theta_traj_3, x_goal_3, y_goal_3, theta_goal_3, k_rho_3, k_alpha_3, k_beta_3, v_const_3 = extract_from_file(LOG_FILE_DIR_3)

    plt.style.use('seaborn-ticks')

    plt.figure(1)
    plot_label = r'$k_\rho$ = %.2f  $k_\alpha$ = %.2f  $k_\beta$ = %.2f' % (k_rho_1, k_alpha_1, k_beta_1)
    plt.plot(x_traj_1, y_traj_1, label = plot_label )
    plot_label = r'$k_\rho$ = %.2f  $k_\alpha$ = %.2f  $k_\beta$ = %.2f' % (k_rho_2, k_alpha_2, k_beta_2)
    plt.plot(x_traj_2, y_traj_2, label = plot_label )
    plot_label = r'$k_\rho$ = %.2f  $k_\alpha$ = %.2f  $k_\beta$ = %.2f' % (k_rho_3, k_alpha_3, k_beta_3)
    plt.plot(x_traj_3, y_traj_3, label = plot_label )
    plt.title('X-Y Plane trajectory')
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.xlim(np.min(x_traj_1) - 0.05, np.max(x_traj_1) + 0.05)
    plt.ylim(np.min(y_traj_1) - 0.05, np.max(y_traj_1) + 0.05)
    plt.legend(loc = 4)
    plt.grid()

    plt.figure(2)
    plt.subplot(1,3,1)
    plt.plot(x_goal_1, color = 'red', label = 'reference')
    plt.plot(x_traj_1, color = 'blue', label = 'actual')
    plt.legend(loc = 4)
    plt.title('X coordinate response')
    plt.xlabel('sample time')
    plt.ylabel('x [m]')
    plt.ylim(np.min(x_traj_1) - 0.05, np.max(x_traj_1) + 0.05)
    plt.grid()

    plt.figure(2)
    plt.subplot(1,3,2)
    plt.plot(y_goal_1, color = 'red', label = 'reference')
    plt.plot(y_traj_1, color = 'blue', label = 'actual')
    plt.legend(loc = 4)
    plt.title('Y coordinate response')
    plt.xlabel('sample time')
    plt.ylabel('y [m]')
    plt.ylim(np.min(y_traj_1) - 0.05, np.max(y_traj_1) + 0.05)
    plt.grid()

    plt.figure(2)
    plt.subplot(1,3,3)
    plt.plot(theta_goal_1, color = 'red', label = 'reference')
    plt.plot(theta_traj_1, color = 'blue', label = 'actual')
    plt.legend(loc = 4)
    plt.title('THETA coordinate response')
    plt.xlabel('sample time')
    plt.ylabel('theta [degrees]')
    plt.ylim(np.min(theta_traj_1) - 5, np.max(theta_traj_1) + 5)
    plt.grid()

    plt.show()

# Plot learning parameters
def plot_learning(log_file_dir):
    reward_per_episode = np.genfromtxt(log_file_dir+'/reward_per_episode.csv', delimiter = ' , ')
    steps_per_episode = np.genfromtxt(log_file_dir+'/steps_per_episode.csv', delimiter = ' , ')
    T_per_episode = np.genfromtxt(log_file_dir+'/T_per_episode.csv', delimiter = ' , ')
    reward_min_per_episode = np.genfromtxt(log_file_dir+'/reward_min_per_episode.csv', delimiter = ' , ')
    reward_max_per_episode = np.genfromtxt(log_file_dir+'/reward_max_per_episode.csv', delimiter = ' , ')
    reward_avg_per_episode = np.genfromtxt(log_file_dir+'/reward_avg_per_episode.csv', delimiter = ' , ')
    t_per_episode = np.genfromtxt(log_file_dir+'/t_per_episode.csv', delimiter = ' , ')

    accumulated_reward = np.array([])
    av_steps_per_10_episodes = np.array([])
    episodes_10 = np.arange(10,len(reward_per_episode)+10,10)

    # Accumulated rewards and average steps
    for i in range(len(episodes_10)):
        accumulated_reward = np.append(accumulated_reward, np.sum(reward_per_episode[0:10*(i+1)]))
        av_steps_per_10_episodes = np.append(av_steps_per_10_episodes, np.sum(steps_per_episode[10*i:10*(i+1)]) / 10)

    plt.style.use('seaborn-ticks')

    plt.figure(1)
    plt.subplot(221)
    plt.plot(reward_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Total reward per episode')
    plt.grid()

    plt.figure(1)
    plt.subplot(222)
    plt.plot(steps_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps per episode')
    plt.grid()

    plt.figure(1)
    plt.subplot(223)
    plt.plot(T_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('T')
    plt.title('T per episode')
    plt.grid()

    plt.figure(1)
    plt.subplot(224)
    plt.plot(reward_max_per_episode, label = 'max rewards')
    plt.plot(reward_avg_per_episode, label = 'avg rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward per episode')
    plt.legend(loc = 4)
    plt.grid()

    plt.figure(2)
    plt.plot(episodes_10, accumulated_reward)
    plt.xlabel('Episode')
    plt.ylabel('Accumulated reward')
    plt.title('Accumulated reward per 10 episodes')
    plt.grid()

    plt.figure(3)
    plt.plot(episodes_10, av_steps_per_10_episodes)
    plt.xlabel('Episode')
    plt.ylabel('Average steps')
    plt.title('Average steps per 10 episode')
    plt.grid()

    plt.show()

# Q-table plot
def plot_Q_table(log_file_dir):
    Q_table = np.genfromtxt(log_file_dir+'/Qtable.csv', delimiter = ' , ')

    # Normalized Q-table
    (rows, cols) = Q_table.shape
    Q_table_norm = np.zeros((rows, cols))
    states = np.arange(rows)

    for i in range(rows):
        Q_table_norm[i] = Q_table[i] + np.abs(np.min(Q_table[i]))
        row_sum = np.sum(Q_table_norm[i])
        if row_sum != 0:
            Q_table_norm[i] = Q_table_norm[i] / row_sum * 100.0

    plt.style.use('seaborn-ticks')
    width = 1.0
    fig = plt.figure(4)
    ax = fig.add_subplot(1,1,1)
    ax.bar(states, Q_table_norm[:,0], width, color = 'r')
    ax.bar(states, Q_table_norm[:,1], width, bottom = Q_table_norm[:,0], color = 'b')
    ax.bar(states, Q_table_norm[:,2], width, bottom = Q_table_norm[:,0] + Q_table_norm[:,1], color = 'g')
    ax.set_ylabel('Q-value[%]')
    ax.set_xlabel('State')
    ax.set_title('Q-table')
    ax.set_ylim(-1,101)
    #ax.set_xticks(ind, ('G1', 'G2', 'G3', 'G4', 'G5'))
    #ax.set_yticks(np.arange(0, 100, 10))
    ax.legend(labels = ['action 0', 'action 1', 'action 2'])
    plt.show()

#plot_Q_table(LOG_FILE_DIR)
#plot_learning(LOG_FILE_DIR)
plot_feedback_control(LOG_FILE_DIR_1, LOG_FILE_DIR_2, LOG_FILE_DIR_3)
