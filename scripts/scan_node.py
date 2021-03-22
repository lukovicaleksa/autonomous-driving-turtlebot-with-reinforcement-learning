#! /usr/bin/env python

import rospy
import matplotlib.pyplot as plt
from datetime import datetime

import sys
sys.path.insert(0, '/home/maestro/catkin_ws/src/master_rad/scripts')

from Qlearning import *
from Lidar import *
from Control import *

ANGLE_MAX = 360 - 1
ANGLE_MIN = 1 - 1
HORIZON_WIDTH = 75

MIN_TIME_BETWEEN_SCANS = 0
MAX_SIMULATION_TIME = float('inf')

if __name__ == '__main__':
    try:
        state_space = createStateSpace()

        rospy.init_node('scan_node', anonymous = False)
        rate = rospy.Rate(10)

        now = datetime.now()
        dt_string_start = now.strftime("%d/%m/%Y %H:%M:%S")
        print('SCAN NODE START ==> ', dt_string_start ,'\r\n')

        scan_time = 0
        count = 0

        t_0 = rospy.Time.now()
        t_start = rospy.Time.now()

        # init timer
        while not (t_start > t_0):
            t_start = rospy.Time.now()

        t = t_start

        # Init figure - real time
        plt.style.use('seaborn-ticks')
        fig = plt.figure(1)
        ax = fig.add_subplot(1,1,1)

        # main loop
        while not rospy.is_shutdown():
            msgScan = rospy.wait_for_message('/scan', LaserScan)

            scan_time = (rospy.Time.now() - t).to_sec()
            sim_time = (rospy.Time.now() - t_start).to_sec()
            count = count + 1

            if scan_time > MIN_TIME_BETWEEN_SCANS:
                print('\r\nScan cycle:', count , '\r\nScan time:', scan_time, 's')
                print('Simulation time:', sim_time, 's')
                t = rospy.Time.now()

                # distances in [m], angles in [degrees]
                ( lidar, angles ) = lidarScan(msgScan)
                ( state_ind, x1, x2 ,x3 ,x4 ) = scanDiscretization(state_space, lidar)

                crash = checkCrash(lidar)
                object_nearby = checkObjectNearby(lidar)

                print('state index:', state_ind)
                print('x1 x2 x3 x4')
                print(x1, '', x2, '', x3, '', x4)
                if crash:
                    print('CRASH !')
                if object_nearby:
                    print('OBJECT NEARBY !')

                lidar_horizon = np.concatenate((lidar[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],lidar[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
                #angles_horizon = np.concatenate((angles[(ANGLE_MIN + HORIZON_WIDTH):(ANGLE_MIN):-1],angles[(ANGLE_MAX):(ANGLE_MAX - HORIZON_WIDTH):-1]))
                angles_horizon = np.linspace(90+HORIZON_WIDTH, 90-HORIZON_WIDTH, 150)

                # horizon in x-y plane
                x_horizon = np.array([])
                y_horizon = np.array([])
                for i in range(len(lidar_horizon)):
                    x_horizon = np.append(x_horizon,lidar_horizon[i] * np.cos(radians(angles_horizon[i])))
                    y_horizon = np.append(y_horizon,lidar_horizon[i] * np.sin(radians(angles_horizon[i])))

                ax.clear()
                plt.xlabel('distance[m]')
                plt.ylabel('distance[m]')
                plt.xlim((-1.0,1.0))
                plt.ylim((-0.2,1.2))
                plt.title('Lidar horizon')
                plt.axis('equal')
                ax.plot(x_horizon, y_horizon, 'b.', markersize = 8, label = 'obstacles')
                ax.plot(0, 0, 'r*', markersize = 20, label = 'robot')
                plt.legend(loc = 'lower right', shadow = True)
                plt.draw()
                plt.pause(0.0001)

            if sim_time > MAX_SIMULATION_TIME:
                now = datetime.now()
                dt_string_stop = now.strftime("%d/%m/%Y %H:%M:%S")
                print('\r\nSCAN NODE START ==> ', dt_string_start ,'\r\n')
                print('SCAN NODE STOP ==> ', dt_string_stop ,'\r\n')
                rospy.signal_shutdown('End of simulation')

            rate.sleep()

    except rospy.ROSInterruptException:
        now = datetime.now()
        dt_string_stop = now.strftime("%d/%m/%Y %H:%M:%S")
        print('\r\nSCAN NODE START ==> ', dt_string_start ,'\r\n')
        print('SCAN NODE STOP ==> ', dt_string_stop ,'\r\n')
        rospy.signal_shutdown('End of simulation')

        pass
