#!/usr/bin/env python3

'''
*****************************************************************************************
*
*                   ===============================================
*                            StrataCobot (SC) Theme (eYRC 2026-27)
*                   ===============================================
*
*  This script should be used to implement Task 3A of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ 1449 ]
# Author List:      [ Atharv Mudse, Abhay Potdar, Riddhi Wagh, Yash Waje ]
# Filename:         task3A_navigation.py
# Functions:        [ pathcb, odomcb, wrap_angle, clamp, nearest_index, lookahead_point,
#                     process_navigation, stop_robot, main ]
# Nodes:            ebot_nav_node
#                   Publishing Topics  - [ /cmd_vel ]
#                   Subscribing Topics - [ /ebot_path, /odom ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math

from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path


##################### TASK CONSTANTS #######################

path_topic = '/ebot_path'
odom_topic = '/odom'
cmd_topic = '/cmd_vel'

max_linear_mps = 0.5
max_angular_rps = 1.0

cruise_speed = 0.35             # straight-line speed (m/s)
lookahead_dist = 0.30           # how far ahead on the path to aim (m)
finish_dist = 0.08              # position tolerance at the end of the path (m)
retry_dist = 0.12               # if still further than this after the final turn, creep back to the end point (m)
max_retries = 2                 # how many such corrections per step
finish_yaw = 0.04               # yaw tolerance at the end of the path (rad)
turn_in_place_above = 0.60      # heading error above which the eBot first turns on the spot (rad)
turn_in_place_below = 0.20      # ... and resumes driving once the error is below this (rad)


##################### CLASS DEFINITION #######################

class ebot_nav(Node):
    '''
    ___CLASS___

    Description:    Drives the eBot along whatever path arrives on /ebot_path. Each new
                    path (one per step of the lap) replaces the old one. It only follows
                    the path: it never steers off it, since the planner keeps it clear of rocks.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ebot_nav
        '''

        super().__init__(
            'ebot_nav_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # /ebot_path is held (TRANSIENT_LOCAL) so a path published before this node started still arrives.
        path_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        ############ Topic SUBSCRIPTIONS ############

        self.path_sub = self.create_subscription(Path, path_topic, self.pathcb, path_qos)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)

        ############ Topic PUBLISHERS ############

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        self.timer = self.create_timer(0.05, self.process_navigation)                   # 20 Hz control loop

        self.route = None                                                               # [(x, y), ...] of the current step
        self.final_yaw = 0.0                                                            # yaw to finish with (last pose of the path)
        self.index = 0                                                                  # progress along the route
        self.phase = 'idle'                                                             # idle | turn | track | align | done
        self.retries = 0                                                                # end-point corrections used this step

        self.odom = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.get_logger().info('eBot navigation node started')


    def pathcb(self, data):
        '''
        Description:    Callback for /ebot_path. Every new path replaces the current one.

        Args:
            data (Path):    the path of the step about to be driven

        Returns:
        '''

        try:
            if len(data.poses) < 2:
                self.get_logger().warn('Received a path with fewer than 2 poses, ignored')
                return

            self.route = [(p.pose.position.x, p.pose.position.y) for p in data.poses]

            q = data.poses[-1].pose.orientation                                         # the last pose gives the finishing yaw
            self.final_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))

            self.index = 0
            self.retries = 0
            self.phase = 'turn'

            self.get_logger().info(
                f'New path: {len(self.route)} poses, ends at '
                f'({self.route[-1][0]:.2f}, {self.route[-1][1]:.2f}) facing {self.final_yaw:.2f} rad')

        except Exception as error:
            self.get_logger().error(f'Path callback error: {error}')


    def odomcb(self, data):
        '''
        Description:    Callback for /odom. Stores the position and heading of the base.

        Args:
            data (Odometry):    Pose and velocity of the base

        Returns:
        '''

        try:
            self.odom = data
            self.x = data.pose.pose.position.x
            self.y = data.pose.pose.position.y

            q = data.pose.pose.orientation
            self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                  1.0 - 2.0 * (q.y * q.y + q.z * q.z))

        except Exception as error:
            self.get_logger().error(f'Odometry callback error: {error}')


    def wrap_angle(self, angle):
        '''
        Description:    Wraps an angle to [-pi, pi].
        '''

        return math.atan2(math.sin(angle), math.cos(angle))


    def clamp(self, value, low, high):
        '''
        Description:    Limits 'value' to the range [low, high].
        '''

        return max(low, min(high, value))


    def nearest_index(self):
        '''
        Description:    Index of the route point nearest to the eBot, searching only a short
                        window ahead of the current progress so progress never jumps back.

        Returns:
            int
        '''

        best = self.index
        best_dist = math.inf

        for i in range(self.index, min(len(self.route), self.index + 12)):
            d = math.hypot(self.route[i][0] - self.x, self.route[i][1] - self.y)
            if d < best_dist:
                best_dist = d
                best = i

        return best


    def lookahead_point(self):
        '''
        Description:    The point on the route 'lookahead_dist' of path length ahead of the
                        eBot's progress, or the last point when the end is closer than that.

        Returns:
            tuple:      (x, y)
        '''

        travelled = 0.0

        for i in range(self.index, len(self.route) - 1):
            a, b = self.route[i], self.route[i + 1]
            travelled += math.hypot(b[0] - a[0], b[1] - a[1])
            if travelled >= lookahead_dist:
                return b

        return self.route[-1]


    def process_navigation(self):
        '''
        Description:    Timer function: follows the route, turns to the final yaw, and stops.

        Returns:
        '''

        try:
            if self.route is None or self.odom is None or self.phase == 'idle':
                self.get_logger().info(
                    f'Waiting: path received={self.route is not None}, odom received={self.odom is not None}',
                    throttle_duration_sec=3.0)
                return

            if self.phase == 'done':
                self.stop_robot()
                return

            end_x, end_y = self.route[-1]
            dist_end = math.hypot(end_x - self.x, end_y - self.y)

            ##################### FINAL YAW ALIGNMENT #####################

            if self.phase == 'align':
                error = self.wrap_angle(self.final_yaw - self.yaw)

                if abs(error) < finish_yaw:
                    # The turn on the spot can slide the base. If it ended too far from the end
                    # point, drive the short way back to it and turn again.
                    if dist_end > retry_dist and self.retries < max_retries:
                        self.retries += 1
                        self.route = [(self.x, self.y), (end_x, end_y)]
                        self.index = 0
                        self.phase = 'turn'
                        self.stop_robot()
                        self.get_logger().info(
                            f'Correcting end position (off by {dist_end:.2f} m), try {self.retries}')
                        return

                    self.phase = 'done'
                    self.stop_robot()
                    self.get_logger().info(
                        f'Step finished at ({self.x:.2f}, {self.y:.2f}), yaw {self.yaw:.2f}')
                    return

                msg = Twist()
                turn = self.clamp(2.0 * error, -0.8, 0.8)
                if abs(turn) < 0.12:                                                    # keep a minimum so it never stalls
                    turn = math.copysign(0.12, error)
                msg.angular.z = turn
                self.cmd_pub.publish(msg)
                return

            ##################### PATH TRACKING #####################

            self.index = self.nearest_index()
            target_x, target_y = self.lookahead_point()

            heading_error = self.wrap_angle(
                math.atan2(target_y - self.y, target_x - self.x) - self.yaw)

            # Arrived: close enough to the last point, or just overshot it.
            if dist_end < finish_dist or (dist_end < 0.20 and abs(heading_error) > 1.0):
                self.phase = 'align'
                self.stop_robot()
                return

            msg = Twist()

            if self.phase == 'turn':                                                    # point at the path before driving
                if abs(heading_error) < turn_in_place_below:
                    self.phase = 'track'
                else:
                    turn = self.clamp(1.8 * heading_error, -max_angular_rps, max_angular_rps)
                    if abs(turn) < 0.15:
                        turn = math.copysign(0.15, heading_error)
                    msg.angular.z = turn
                    self.cmd_pub.publish(msg)
                    return

            if abs(heading_error) > turn_in_place_above:                               # lost the heading: turn on the spot
                self.phase = 'turn'
                self.stop_robot()
                return

            msg.angular.z = self.clamp(2.0 * heading_error, -max_angular_rps, max_angular_rps)

            speed = cruise_speed * max(0.25, 1.0 - abs(heading_error) / 0.8)           # slow down in turns
            speed = min(speed, 0.6 * dist_end + 0.04)                                   # slow down towards the end
            msg.linear.x = self.clamp(speed, 0.0, max_linear_mps)

            self.cmd_pub.publish(msg)

        except Exception as error:
            self.get_logger().error(f'Navigation error: {error}')
            self.stop_robot()


    def stop_robot(self):
        '''
        Description:    Publishes a zero velocity command.

        Returns:
        '''

        self.cmd_pub.publish(Twist())


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which initializes the ROS 2 node and spins the
                    ebot_nav class until shutdown.
    '''

    rclpy.init(args=sys.argv)

    ebot_nav_class = ebot_nav()

    try:
        rclpy.spin(ebot_nav_class)

    except KeyboardInterrupt:
        pass

    finally:
        if rclpy.ok():                                                                  # a launch file's Ctrl+C may have shut ROS down already
            ebot_nav_class.stop_robot()
            ebot_nav_class.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()