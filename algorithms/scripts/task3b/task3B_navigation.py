#!/usr/bin/env python3

'''
*****************************************************************************************
*
*                   ===============================================
*                            StrataCobot (SC) Theme (eYRC 2026-27)
*                   ===============================================
*
*  This script should be used to implement Task 3B of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:         task3B_navigation.py
# Functions:        [ pathcb, odomcb, scancb, wrap_angle, clamp, nearest_index,
#                     lookahead_point, avoid_obstacle_steer, process_navigation, stop_robot, main ]
# Nodes:            ebot_nav_node
#                   Publishing Topics  - [ /cmd_vel ]
#                   Subscribing Topics - [ /ebot_path, /odom, /scan ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math

from rclpy.node import Node
from rclpy.qos import (qos_profile_sensor_data, QoSProfile, QoSHistoryPolicy,
                       QoSReliabilityPolicy, QoSDurabilityPolicy)
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan


##################### TASK CONSTANTS #######################

path_topic = '/ebot_path'
odom_topic = '/odom'
scan_topic = '/scan'
cmd_topic = '/cmd_vel'

max_linear_mps = 0.50
max_angular_rps = 0.95

cruise_speed = 0.485            # straight-line speed (m/s)
lookahead_dist = 0.34           # how far ahead on the path to aim (m)
finish_dist = 0.09              # drive right up to the goal (< 9 cm) before final turn
retry_dist = 0.18               # if sliding during turn pushes us > 18 cm off, creep to goal
max_retries = 2                 # how many corrections per step
finish_yaw = 0.065              # yaw tolerance at the end of the path (rad)
turn_in_place_above = 0.75      # heading error above which the eBot first turns on the spot (rad)
turn_in_place_below = 0.25      # ... and resumes driving once the error is below this (rad)

ODOM_EMA_ALPHA = 0.45

# Local reactive avoidance parameters for unmapped rocks discovered mid-route
OBS_FRONT_DIST = 0.52           # start nudging away if a rock is closer than this in the forward cone (m)


##################### CLASS DEFINITION #######################

class ebot_nav(Node):
    '''
    ___CLASS___

    Description:    Drives the eBot along whatever path arrives on /ebot_path, with EMA
                    filtering on /odom and a local /scan skirt to avoid unmapped rocks
                    discovered mid-leg before smoothly rejoining the published path.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ebot_nav
        '''

        super().__init__(
            'ebot_nav_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        path_qos = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        ############ Topic SUBSCRIPTIONS ############

        self.path_sub = self.create_subscription(Path, path_topic, self.pathcb, path_qos)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scancb, qos_profile_sensor_data)

        ############ Topic PUBLISHERS ############

        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        self.timer = self.create_timer(0.04, self.process_navigation)

        self.route = None
        self.final_yaw = 0.0
        self.index = 0
        self.phase = 'idle'
        self.retries = 0

        self.odom = None
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.scan = None

        self.get_logger().info('Task 3B eBot navigation node started')


    def pathcb(self, data):
        '''
        Description:    Callback for /ebot_path. Every new path replaces the current one.
        '''

        try:
            if len(data.poses) < 2:
                self.get_logger().warn('Received a path with fewer than 2 poses, ignored')
                return

            self.route = [(p.pose.position.x, p.pose.position.y) for p in data.poses]

            q = data.poses[-1].pose.orientation
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
        Description:    Callback for /odom with EMA smoothing for Task 3B sensor noise.
        '''

        try:
            raw_x = data.pose.pose.position.x
            raw_y = data.pose.pose.position.y

            q = data.pose.pose.orientation
            raw_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

            if self.odom is None:
                self.x = raw_x
                self.y = raw_y
                self.yaw = raw_yaw
            else:
                a = ODOM_EMA_ALPHA
                self.x = a * raw_x + (1.0 - a) * self.x
                self.y = a * raw_y + (1.0 - a) * self.y
                dyaw = self.wrap_angle(raw_yaw - self.yaw)
                self.yaw = self.wrap_angle(self.yaw + a * dyaw)

            self.odom = data

        except Exception as error:
            self.get_logger().error(f'Odometry callback error: {error}')


    def scancb(self, data):
        '''
        Description:    Stores the latest LaserScan message for reactive obstacle skirting.
        '''

        self.scan = data


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
        Description:    Index of the route point nearest to the eBot.
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
        Description:    The point on the route 'lookahead_dist' ahead of the eBot's progress.
        '''

        travelled = 0.0

        for i in range(self.index, len(self.route) - 1):
            a, b = self.route[i], self.route[i + 1]
            travelled += math.hypot(b[0] - a[0], b[1] - a[1])
            if travelled >= lookahead_dist:
                return b

        return self.route[-1]


    def avoid_obstacle_steer(self, desired_heading_err, dist_end):
        '''
        Description:    Checks the forward LiDAR sectors. If an unmapped rock lies close to the
                        desired heading (and we are not near the goal stop pose), returns a
                        steering nudge and speed scale to skirt cleanly around it.
        '''

        if self.scan is None or dist_end < 0.85:
            return desired_heading_err, 1.0

        ranges = self.scan.ranges
        amin = self.scan.angle_min
        ainc = self.scan.angle_increment
        n = len(ranges)

        min_front = 10.0
        repulse = 0.0

        for i in range(0, n, 2):
            r = ranges[i]
            if not math.isfinite(r) or r < 0.22 or r > OBS_FRONT_DIST:
                continue
            if i + 1 < n and abs(ranges[i + 1] - r) > 0.14:
                continue

            ang = amin + i * ainc
            if abs(ang) > 0.95:
                continue

            if abs(ang - desired_heading_err) < 0.40 and r < min_front:
                min_front = r

            weight = max(0.0, (OBS_FRONT_DIST - r) / OBS_FRONT_DIST) * math.cos(ang * 0.85)
            if ang >= 0.0:
                repulse -= weight * 0.32
            else:
                repulse += weight * 0.32

        if min_front >= OBS_FRONT_DIST:
            return desired_heading_err, 1.0

        adjusted = self.clamp(desired_heading_err + repulse, -1.10, 1.10)
        speed_scale = self.clamp((min_front - 0.24) / (OBS_FRONT_DIST - 0.24), 0.45, 1.0)
        return adjusted, speed_scale


    def process_navigation(self):
        '''
        Description:    Timer function: follows the route, turns to the final yaw, and stops.
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

            ##################### BIDIRECTIONAL CREEP CORRECTION #####################

            if self.phase == 'creep':
                # Drive forward or backward (whichever requires < 90 deg turn) directly to (end_x, end_y)
                dx = end_x - self.x
                dy = end_y - self.y
                if dist_end < finish_dist:
                    self.phase = 'align'
                    self.stop_robot()
                    return

                angle_to_goal = math.atan2(dy, dx)
                fwd_err = self.wrap_angle(angle_to_goal - self.yaw)
                rev_err = self.wrap_angle(angle_to_goal + math.pi - self.yaw)

                msg = Twist()
                if abs(fwd_err) <= abs(rev_err):
                    if abs(fwd_err) > 0.35:
                        msg.angular.z = self.clamp(2.2 * fwd_err, -0.75, 0.75)
                    else:
                        msg.linear.x = self.clamp(1.2 * dist_end + 0.06, 0.08, 0.28)
                        msg.angular.z = self.clamp(2.0 * fwd_err, -0.65, 0.65)
                else:
                    # Goal is behind us: reverse smoothly without spinning 180 deg!
                    if abs(rev_err) > 0.35:
                        msg.angular.z = self.clamp(2.2 * rev_err, -0.75, 0.75)
                    else:
                        msg.linear.x = -self.clamp(1.2 * dist_end + 0.06, 0.08, 0.28)
                        msg.angular.z = self.clamp(2.0 * rev_err, -0.65, 0.65)

                self.cmd_pub.publish(msg)
                return

            ##################### FINAL YAW ALIGNMENT #####################

            if self.phase == 'align':
                error = self.wrap_angle(self.final_yaw - self.yaw)

                if abs(error) < finish_yaw:
                    if dist_end > retry_dist and self.retries < max_retries:
                        self.retries += 1
                        self.phase = 'creep'
                        self.stop_robot()
                        self.get_logger().info(
                            f'Creeping to end position (off by {dist_end:.2f} m), try {self.retries}')
                        return

                    self.phase = 'done'
                    self.stop_robot()
                    self.get_logger().info(
                        f'Step finished at ({self.x:.2f}, {self.y:.2f}), yaw {self.yaw:.2f}')
                    return

                msg = Twist()
                turn = self.clamp(2.5 * error, -0.9, 0.9)
                if abs(turn) < 0.16:
                    turn = math.copysign(0.16, error)
                msg.angular.z = turn
                self.cmd_pub.publish(msg)
                return

            ##################### PATH TRACKING #####################

            self.index = self.nearest_index()
            target_x, target_y = self.lookahead_point()

            raw_heading_error = self.wrap_angle(
                math.atan2(target_y - self.y, target_x - self.x) - self.yaw)

            # Arrived: within finish_dist (0.09 m), or actively tracking and just passed the end point (< 0.13 m)
            if dist_end < finish_dist or (self.phase == 'track' and dist_end < 0.13 and abs(raw_heading_error) > 1.50):
                self.phase = 'align'
                self.stop_robot()
                return

            msg = Twist()

            if self.phase == 'turn':
                if abs(raw_heading_error) < turn_in_place_below:
                    self.phase = 'track'
                else:
                    turn = self.clamp(2.3 * raw_heading_error, -max_angular_rps, max_angular_rps)
                    if abs(turn) < 0.20:
                        turn = math.copysign(0.20, raw_heading_error)
                    msg.angular.z = turn
                    self.cmd_pub.publish(msg)
                    return

            if dist_end > 0.30 and abs(raw_heading_error) > turn_in_place_above:
                self.phase = 'turn'
                self.stop_robot()
                return

            heading_error, obs_speed_scale = self.avoid_obstacle_steer(raw_heading_error, dist_end)

            msg.angular.z = self.clamp(2.2 * heading_error, -max_angular_rps, max_angular_rps)

            speed = cruise_speed * max(0.35, 1.0 - abs(heading_error) / 1.0) * obs_speed_scale
            speed = min(speed, 1.15 * dist_end + 0.08)
            msg.linear.x = self.clamp(speed, 0.0, max_linear_mps)

            self.cmd_pub.publish(msg)

        except Exception as error:
            self.get_logger().error(f'Navigation error: {error}')
            self.stop_robot()


    def stop_robot(self):
        '''
        Description:    Publishes a zero velocity command.
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
        if rclpy.ok():
            ebot_nav_class.stop_robot()
            ebot_nav_class.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()