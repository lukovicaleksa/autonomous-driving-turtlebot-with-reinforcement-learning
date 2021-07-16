#! /usr/bin/env python

import numpy as np
from math import *
import matplotlib.pyplot as plt

import sys
DATA_PATH = '/home/maestro/catkin_ws/src/master_rad/Data'
sys.path.insert(0, DATA_PATH)

def plot_gamma(log_gamma_1, log_gamma_2, log_gamma_3):
    reward_per_episode_1 = np.genfromtxt(log_gamma_1+'/reward_per_episode.csv', delimiter = ' , ')
    reward_per_episode_2 = np.genfromtxt(log_gamma_2+'/reward_per_episode.csv', delimiter = ' , ')
    reward_per_episode_3 = np.genfromtxt(log_gamma_3+'/reward_per_episode.csv', delimiter = ' , ')

    steps_per_episode_1 = np.genfromtxt(log_gamma_1+'/steps_per_episode.csv', delimiter = ' , ')
    steps_per_episode_2 = np.genfromtxt(log_gamma_2+'/steps_per_episode.csv', delimiter = ' , ')
    steps_per_episode_3 = np.genfromtxt(log_gamma_3+'/steps_per_episode.csv', delimiter = ' , ')

    accumulated_reward_1 = np.array([])
    accumulated_reward_2 = np.array([])
    accumulated_reward_3 = np.array([])

    av_steps_per_10_episodes_1 = np.array([])
    av_steps_per_10_episodes_2 = np.array([])
    av_steps_per_10_episodes_3 = np.array([])

    episodes_10 = np.arange(10,len(reward_per_episode_1)+10,10)

    # Accumulated rewards and average steps
    for i in range(len(episodes_10)):
        accumulated_reward_1 = np.append(accumulated_reward_1, np.sum(reward_per_episode_1[0:10*(i+1)]))
        accumulated_reward_2 = np.append(accumulated_reward_2, np.sum(reward_per_episode_2[0:10*(i+1)]))
        accumulated_reward_3 = np.append(accumulated_reward_3, np.sum(reward_per_episode_3[0:10*(i+1)]))
        av_steps_per_10_episodes_1 = np.append(av_steps_per_10_episodes_1, np.sum(steps_per_episode_1[10*i:10*(i+1)]) / 10)
        av_steps_per_10_episodes_2 = np.append(av_steps_per_10_episodes_2, np.sum(steps_per_episode_2[10*i:10*(i+1)]) / 10)
        av_steps_per_10_episodes_3 = np.append(av_steps_per_10_episodes_3, np.sum(steps_per_episode_3[10*i:10*(i+1)]) / 10)

    plt.style.use('seaborn-ticks')

    plt.figure(1)
    plt.plot(episodes_10, accumulated_reward_1, label = r'$\gamma = 0.9$')
    plt.plot(episodes_10, accumulated_reward_2, label = r'$\gamma = 0.7$')
    plt.plot(episodes_10, accumulated_reward_3, label = r'$\gamma = 0.7$')
    plt.xlabel('Episode')
    plt.ylabel('Accumulated reward')
    plt.title('Accumulated reward per 10 episodes')
    plt.ylim(np.min(accumulated_reward_3) - 500 , np.max(accumulated_reward_3) + 500)
    plt.xlim(np.min(episodes_10), np.max(episodes_10))
    plt.legend()
    plt.grid()

    plt.figure(2)
    plt.plot(episodes_10, av_steps_per_10_episodes_1, label = r'$\gamma = 0.9$')
    plt.plot(episodes_10, av_steps_per_10_episodes_2, label = r'$\gamma = 0.7$')
    plt.plot(episodes_10, av_steps_per_10_episodes_3, label = r'$\gamma = 0.5$')
    plt.xlabel('Episode')
    plt.ylabel('Average steps')
    plt.title('Average steps per 10 episode')
    plt.ylim(np.min(av_steps_per_10_episodes_1) - 10 , np.max(av_steps_per_10_episodes_1) + 10)
    plt.xlim(np.min(episodes_10), np.max(episodes_10))
    plt.legend(loc = 4)
    plt.grid()

    plt.show()


def plot_alpha(log_alpha_1, log_alpha_2, log_alpha_3):
    reward_per_episode_1 = np.genfromtxt(log_alpha_1+'/reward_per_episode.csv', delimiter = ' , ')
    reward_per_episode_2 = np.genfromtxt(log_alpha_2+'/reward_per_episode.csv', delimiter = ' , ')
    reward_per_episode_3 = np.genfromtxt(log_alpha_3+'/reward_per_episode.csv', delimiter = ' , ')

    steps_per_episode_1 = np.genfromtxt(log_alpha_1+'/steps_per_episode.csv', delimiter = ' , ')
    steps_per_episode_2 = np.genfromtxt(log_alpha_2+'/steps_per_episode.csv', delimiter = ' , ')
    steps_per_episode_3 = np.genfromtxt(log_alpha_3+'/steps_per_episode.csv', delimiter = ' , ')

    accumulated_reward_1 = np.array([])
    accumulated_reward_2 = np.array([])
    accumulated_reward_3 = np.array([])

    av_steps_per_10_episodes_1 = np.array([])
    av_steps_per_10_episodes_2 = np.array([])
    av_steps_per_10_episodes_3 = np.array([])

    episodes_10 = np.arange(10,len(reward_per_episode_1)+10,10)

    # Accumulated rewards and average steps
    for i in range(len(episodes_10)):
        accumulated_reward_1 = np.append(accumulated_reward_1, np.sum(reward_per_episode_1[0:10*(i+1)]))
        accumulated_reward_2 = np.append(accumulated_reward_2, np.sum(reward_per_episode_2[0:10*(i+1)]))
        accumulated_reward_3 = np.append(accumulated_reward_3, np.sum(reward_per_episode_3[0:10*(i+1)]))
        av_steps_per_10_episodes_1 = np.append(av_steps_per_10_episodes_1, np.sum(steps_per_episode_1[10*i:10*(i+1)]) / 10)
        av_steps_per_10_episodes_2 = np.append(av_steps_per_10_episodes_2, np.sum(steps_per_episode_2[10*i:10*(i+1)]) / 10)
        av_steps_per_10_episodes_3 = np.append(av_steps_per_10_episodes_3, np.sum(steps_per_episode_3[10*i:10*(i+1)]) / 10)

    plt.style.use('seaborn-ticks')

    plt.figure(1)
    plt.plot(episodes_10, accumulated_reward_1, label = r'$\alpha = 0.3$')
    plt.plot(episodes_10, accumulated_reward_2, label = r'$\alpha = 0.5$')
    plt.plot(episodes_10, accumulated_reward_3, label = r'$\alpha = 0.7$')
    plt.xlabel('Episode')
    plt.ylabel('Accumulated reward')
    plt.title('Accumulated reward per 10 episodes')
    plt.ylim(np.min(accumulated_reward_1) - 500 , np.max(accumulated_reward_1) + 500)
    plt.xlim(np.min(episodes_10), np.max(episodes_10))
    plt.legend()
    plt.grid()

    plt.figure(2)
    plt.plot(episodes_10, av_steps_per_10_episodes_1, label = r'$\alpha = 0.3$')
    plt.plot(episodes_10, av_steps_per_10_episodes_2, label = r'$\alpha = 0.5$')
    plt.plot(episodes_10, av_steps_per_10_episodes_3, label = r'$\alpha = 0.7$')
    plt.xlabel('Episode')
    plt.ylabel('Average steps')
    plt.title('Average steps per 10 episode')
    plt.ylim(np.min(av_steps_per_10_episodes_1) - 10 , np.max(av_steps_per_10_episodes_1) + 10)
    plt.xlim(np.min(episodes_10), np.max(episodes_10))
    plt.legend(loc = 4)
    plt.grid()

    plt.show()


def plot_softmax_epsilon(log_softmax, log_epsilon):
    reward_per_episode_softmax = np.genfromtxt(log_softmax+'/reward_per_episode.csv', delimiter = ' , ')
    steps_per_episode_softmax = np.genfromtxt(log_softmax+'/steps_per_episode.csv', delimiter = ' , ')
    T_per_episode = np.genfromtxt(log_softmax+'/T_per_episode.csv', delimiter = ' , ')
    reward_per_episode_epsilon = np.genfromtxt(log_epsilon+'/reward_per_episode.csv', delimiter = ' , ')
    steps_per_episode_epsilon = np.genfromtxt(log_epsilon+'/steps_per_episode.csv', delimiter = ' , ')
    EPSILON_per_episode = np.genfromtxt(log_epsilon+'/EPSILON_per_episode.csv', delimiter = ' , ')

    accumulated_reward_softmax = np.array([])
    av_steps_per_10_episodes_softmax = np.array([])
    accumulated_reward_epsilon = np.array([])
    av_steps_per_10_episodes_epsilon = np.array([])
    episodes_10 = np.arange(10,len(reward_per_episode_softmax)+10,10)

    # Accumulated rewards and average steps
    for i in range(len(episodes_10)):
        accumulated_reward_softmax = np.append(accumulated_reward_softmax, np.sum(reward_per_episode_softmax[0:10*(i+1)]))
        av_steps_per_10_episodes_softmax = np.append(av_steps_per_10_episodes_softmax, np.sum(steps_per_episode_softmax[10*i:10*(i+1)]) / 10)
        accumulated_reward_epsilon = np.append(accumulated_reward_epsilon, np.sum(reward_per_episode_epsilon[0:10*(i+1)]))
        av_steps_per_10_episodes_epsilon = np.append(av_steps_per_10_episodes_epsilon, np.sum(steps_per_episode_epsilon[10*i:10*(i+1)]) / 10)


    plt.style.use('seaborn-ticks')

    plt.figure(1)
    plt.plot(T_per_episode / np.max(T_per_episode), label = r'$T$')
    plt.plot(EPSILON_per_episode / np.max(EPSILON_per_episode), label = r'$\epsilon$')
    plt.xlabel('Episode')
    plt.ylabel(r'$T$ , $\epsilon$')
    plt.title(r'$T$ , $\epsilon$ per episode')
    plt.legend()
    plt.grid()

    plt.figure(2)
    plt.plot(episodes_10, accumulated_reward_softmax, label = r'$Softmax$')
    plt.plot(episodes_10, accumulated_reward_epsilon, label = r'$\epsilon-Greedy$')
    plt.xlabel('Episode')
    plt.ylabel('Accumulated reward')
    plt.title('Accumulated reward per 10 episodes')
    plt.ylim(np.min(accumulated_reward_epsilon) - 500 , np.max(accumulated_reward_softmax) + 500)
    plt.xlim(np.min(episodes_10), np.max(episodes_10))
    plt.legend()
    plt.grid()

    plt.figure(3)
    plt.plot(episodes_10, av_steps_per_10_episodes_softmax, label = r'$Softmax$')
    plt.plot(episodes_10, av_steps_per_10_episodes_epsilon, label = r'$\epsilon-Greedy$')
    plt.xlabel('Episode')
    plt.ylabel('Average steps')
    plt.title('Average steps per 10 episode')
    plt.ylim(np.min(av_steps_per_10_episodes_softmax) - 10 , np.max(av_steps_per_10_episodes_softmax) + 10)
    plt.xlim(np.min(episodes_10), np.max(episodes_10))
    plt.legend(loc = 2)
    plt.grid()

    plt.tight_layout()

    plt.show()


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
    x_traj_1, y_traj_1, theta_traj_1, x_goal_1, y_goal_1, theta_goal_1, k_rho_1, k_alpha_1, k_beta_1, v_const_1 = extract_from_file(log_file_1)

    # log file 2
    x_traj_2, y_traj_2, theta_traj_2, x_goal_2, y_goal_2, theta_goal_2, k_rho_2, k_alpha_2, k_beta_2, v_const_2 = extract_from_file(log_file_2)

    # log file 2
    x_traj_3, y_traj_3, theta_traj_3, x_goal_3, y_goal_3, theta_goal_3, k_rho_3, k_alpha_3, k_beta_3, v_const_3 = extract_from_file(log_file_3)

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
    plt.subplot(3,1,1)
    plt.plot(x_goal_1, color = 'red', label = 'reference')
    plt.plot(x_traj_1, color = 'blue', label = 'actual')
    plt.legend(loc = 4)
    plt.title('X coordinate response')
    plt.xlabel('sample time')
    plt.ylabel('x [m]')
    plt.xlim(0, len(x_goal_1) + 3)
    plt.ylim(np.min(x_traj_1) - 0.1, np.max(x_traj_1) + 0.1)
    plt.grid()

    plt.figure(2)
    plt.subplot(3,1,2)
    plt.plot(y_goal_1, color = 'red', label = 'reference')
    plt.plot(y_traj_1, color = 'blue', label = 'actual')
    plt.legend(loc = 4)
    plt.title('Y coordinate response')
    plt.xlabel('sample time')
    plt.ylabel('y [m]')
    plt.xlim(0, len(y_goal_1) + 3)
    plt.ylim(np.min(y_traj_1) - 0.075, np.max(y_traj_1) + 0.075)
    plt.grid()

    plt.figure(2)
    plt.subplot(3,1,3)
    plt.plot(theta_goal_1, color = 'red', label = 'reference')
    plt.plot(theta_traj_1, color = 'blue', label = 'actual')
    plt.legend(loc = 1)
    plt.title('$\Theta$ coordinate response')
    plt.xlabel('sample time')
    plt.ylabel('$\Theta$ [degrees]')
    plt.xlim(0, len(theta_goal_1) + 3)
    plt.ylim(np.min(theta_traj_1) - 5, np.max(theta_traj_1) + 5)
    plt.grid()

    plt.tight_layout()

    plt.show()

# Plot learning parameters
def plot_learning(log_file_dir):
    reward_per_episode = np.genfromtxt(log_file_dir+'/reward_per_episode.csv', delimiter = ' , ')
    steps_per_episode = np.genfromtxt(log_file_dir+'/steps_per_episode.csv', delimiter = ' , ')
    T_per_episode = np.genfromtxt(log_file_dir+'/T_per_episode.csv', delimiter = ' , ')
    #EPSILON_per_episode = np.genfromtxt(log_file_dir+'/EPSILON_per_episode.csv', delimiter = ' , ')
    reward_min_per_episode = np.genfromtxt(log_file_dir+'/reward_min_per_episode.csv', delimiter = ' , ')
    reward_max_per_episode = np.genfromtxt(log_file_dir+'/reward_max_per_episode.csv', delimiter = ' , ')
    reward_avg_per_episode = np.genfromtxt(log_file_dir+'/reward_avg_per_episode.csv', delimiter = ' , ')
    #t_per_episode = np.genfromtxt(log_file_dir+'/t_per_episode.csv', delimiter = ' , ')

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
    plt.ylim(np.min(steps_per_episode) - 10, np.max(steps_per_episode) + 10)
    plt.grid()

    plt.figure(1)
    plt.subplot(223)
    plt.plot(T_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('T')
    plt.title('T per episode')
    plt.grid()

    #plt.figure(1)
    #plt.subplot(223)
    #plt.plot(EPSILON_per_episode)
    #plt.xlabel('Episode')
    #plt.ylabel('epsilon')
    #plt.title('epsilon per episode')
    #plt.grid()

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
    plt.ylim(np.min(accumulated_reward) - 500 , np.max(accumulated_reward) + 500)
    plt.xlim(np.min(episodes_10), np.max(episodes_10))
    plt.grid()

    plt.figure(3)
    plt.plot(episodes_10, av_steps_per_10_episodes)
    plt.xlabel('Episode')
    plt.ylabel('Average steps')
    plt.title('Average steps per 10 episode')
    plt.ylim(np.min(av_steps_per_10_episodes) - 10 , np.max(av_steps_per_10_episodes) + 10)
    plt.xlim(np.min(episodes_10), np.max(episodes_10))
    plt.grid()

    plt.tight_layout()

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
    ax.set_ylim(-5,105)
    ax.set_xlim(0, 160)
    ax.legend(labels = ['action 0', 'action 1', 'action 2'])
    plt.tight_layout()

    plt.show()

#plot_softmax_epsilon(DATA_PATH + '/Log_learning_SOFTMAX', DATA_PATH + '/Log_learning_EPSILON')
#plot_Q_table(DATA_PATH + '/Log_learning_RANDOM')
plot_learning(DATA_PATH + '/Log_learning')
#plot_feedback_control(DATA_PATH + '/Log_feedback_1', DATA_PATH + '/Log_feedback_2', DATA_PATH + '/Log_feedback_3')
#plot_alpha(DATA_PATH + '/Log_learning_ALPHA_0dot3', DATA_PATH + '/Log_learning_ALPHA_0dot5', DATA_PATH + '/Log_learning_ALPHA_0dot7')
#plot_gamma(DATA_PATH + '/Log_learning_GAMMA_0dot9', DATA_PATH + '/Log_learning_GAMMA_0dot7', DATA_PATH + '/Log_learning_GAMMA_0dot5')
