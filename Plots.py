#! /usr/bin/env python

import numpy as np
from math import *
import matplotlib.pyplot as plt

LOG_FILE_DIR = 'Log_learning'

# Plot learning parameters
def plot_learning(log_file):
    reward_per_episode = np.genfromtxt(log_file+'/reward_per_episode.csv', delimiter = ' , ')
    steps_per_episode = np.genfromtxt(log_file+'/steps_per_episode.csv', delimiter = ' , ')
    T_per_episode = np.genfromtxt(log_file+'/T_per_episode.csv', delimiter = ' , ')
    reward_min_per_episode = np.genfromtxt(log_file+'/reward_min_per_episode.csv', delimiter = ' , ')
    reward_max_per_episode = np.genfromtxt(log_file+'/reward_max_per_episode.csv', delimiter = ' , ')
    reward_avg_per_episode = np.genfromtxt(log_file+'/reward_avg_per_episode.csv', delimiter = ' , ')
    t_per_episode = np.genfromtxt(log_file+'/t_per_episode.csv', delimiter = ' , ')

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
def plot_Q_table(log_file):
    Q_table = np.genfromtxt(log_file+'/Qtable.csv', delimiter = ' , ')

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

plot_Q_table(LOG_FILE_DIR)
plot_learning(LOG_FILE_DIR)
