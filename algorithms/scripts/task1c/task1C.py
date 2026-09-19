#!/usr/bin/env python3

'''
*****************************************************************************************
*
*                   ===============================================
*                            StrataCobot (SC) Theme (eYRC 2026-27)
*                   ===============================================
*
*  This script should be used to implement Task 1C of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:         task1C.py
# Functions:        [ pathcb, odomcb, scancb, wrap_angle, get_sector_distance,
#                     process_navigation, limit_cmd, stop_robot, main ]
# Nodes:            Publishing Topics  - [ /cmd_vel ]
#                   Subscribing Topics - [ /ebot_path, /odom, /scan ]#!/usr/bin/env python3

import rclpy
import sys
import math

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    qos_profile_sensor_data
)

from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan


path_topic = '/ebot_path'
odom_topic = '/odom'
scan_topic = '/scan'
cmd_topic = '/cmd_vel'

max_linear_mps = 0.5
max_angular_rps = 1.0


class ebot_nav(Node):

    def __init__(self):

        super().__init__(
            'ebot_nav_node',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time',
                    rclpy.Parameter.Type.BOOL,
                    True
                )
            ]
        )

        # ============================================================
        # PATH QoS
        # /ebot_path is latched / TRANSIENT_LOCAL
        # ============================================================

        path_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.path_sub = self.create_subscription(
            Path,
            path_topic,
            self.pathcb,
            path_qos
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odomcb,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scancb,
            qos_profile_sensor_data
        )

        self.cmd_pub = self.create_publisher(
            Twist,
            cmd_topic,
            10
        )

        # 20 Hz
        self.timer = self.create_timer(
            0.05,
            self.process_navigation
        )

        # ============================================================
        # STATE
        # ============================================================

        self.route = None
        self.odom = None
        self.scan = None

        self.current_waypoint = 0

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.finished = False

        # Obstacle avoidance
        self.avoid_direction = 0.0
        self.avoiding = False

        self.get_logger().info("eBot navigation node started")

    # ================================================================
    # PATH CALLBACK
    # ================================================================

    def pathcb(self, data):

        try:

            if self.route is not None:
                return

            if len(data.poses) == 0:
                self.get_logger().warn("Received empty path")
                return

            self.route = []

            for pose in data.poses:

                x = pose.pose.position.x
                y = pose.pose.position.y

                self.route.append((x, y))

            self.current_waypoint = 0

            self.get_logger().info(
                f"Received route with {len(self.route)} waypoints "
                f"in frame '{data.header.frame_id}'"
            )

            for i, point in enumerate(self.route):

                self.get_logger().info(
                    f"Waypoint {i + 1}: "
                    f"x={point[0]:.3f}, y={point[1]:.3f}"
                )

        except Exception as e:

            self.get_logger().error(
                f"Path callback error: {e}"
            )

    # ================================================================
    # ODOM CALLBACK
    # ================================================================

    def odomcb(self, data):

        try:

            self.odom = data

            p = data.pose.pose.position
            q = data.pose.pose.orientation

            self.x = p.x
            self.y = p.y

            # Quaternion -> yaw
            siny_cosp = 2.0 * (
                q.w * q.z +
                q.x * q.y
            )

            cosy_cosp = 1.0 - 2.0 * (
                q.y * q.y +
                q.z * q.z
            )

            self.yaw = math.atan2(
                siny_cosp,
                cosy_cosp
            )

        except Exception as e:

            self.get_logger().error(
                f"Odometry callback error: {e}"
            )

    # ================================================================
    # LASER CALLBACK
    # ================================================================

    def scancb(self, data):

        try:
            self.scan = data

        except Exception as e:

            self.get_logger().error(
                f"Scan callback error: {e}"
            )

    # ================================================================
    # ANGLE WRAP
    # ================================================================

    def wrap_angle(self, angle):

        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )

    # ================================================================
    # LASER SECTOR
    # ================================================================

    def get_sector_distance(self, start_angle, end_angle):

        if self.scan is None:
            return float('inf')

        values = []

        angle = self.scan.angle_min

        for r in self.scan.ranges:

            if (
                math.isfinite(r)
                and self.scan.range_min <= r <= self.scan.range_max
            ):

                # Normalize angle
                a = self.wrap_angle(angle)

                # Sector crossing -PI/PI
                if start_angle <= end_angle:

                    if start_angle <= a <= end_angle:
                        values.append(r)

                else:

                    if a >= start_angle or a <= end_angle:
                        values.append(r)

            angle += self.scan.angle_increment

        if not values:
            return float('inf')

        return min(values)

    # ================================================================
    # MAIN NAVIGATION
    # ================================================================

    def process_navigation(self):

        try:

            # --------------------------------------------------------
            # Wait for all required data
            # --------------------------------------------------------

            if self.route is None:
                return

            if self.odom is None:
                return

            if self.scan is None:
                return

            # --------------------------------------------------------
            # Route completed
            # --------------------------------------------------------

            if self.finished:

                self.stop_robot()
                return

            # --------------------------------------------------------
            # Check waypoint index
            # --------------------------------------------------------

            if self.current_waypoint >= len(self.route):

                self.finished = True
                self.stop_robot()

                self.get_logger().info(
                    "All waypoints completed!"
                )

                return

            # --------------------------------------------------------
            # Current target
            # --------------------------------------------------------

            tx, ty = self.route[self.current_waypoint]

            dx = tx - self.x
            dy = ty - self.y

            distance = math.hypot(dx, dy)

            # --------------------------------------------------------
            # Waypoint reached
            # --------------------------------------------------------

            if distance <= 0.30:

                self.get_logger().info(
                    f"Reached waypoint "
                    f"{self.current_waypoint + 1}"
                )

                self.current_waypoint += 1

                if self.current_waypoint >= len(self.route):

                    self.finished = True
                    self.stop_robot()

                    self.get_logger().info(
                        "FINAL WAYPOINT REACHED"
                    )

                return

            # --------------------------------------------------------
            # Heading error
            # --------------------------------------------------------

            target_angle = math.atan2(dy, dx)

            heading_error = self.wrap_angle(
                target_angle - self.yaw
            )

            # ========================================================
            # OBSTACLE DETECTION
            # ========================================================

            # Front: approximately +/- 25 degrees
            front = self.get_sector_distance(
                math.radians(-25),
                math.radians(25)
            )

            # Left: 25 to 70 degrees
            left = self.get_sector_distance(
                math.radians(25),
                math.radians(70)
            )

            # Right: -70 to -25 degrees
            right = self.get_sector_distance(
                math.radians(-70),
                math.radians(-25)
            )

            # ========================================================
            # OBSTACLE AVOIDANCE
            # ========================================================

            obstacle_distance = 0.65

            if front < obstacle_distance:

                self.avoiding = True

                # Choose side having more free space
                if left > right:

                    self.avoid_direction = 1.0

                else:

                    self.avoid_direction = -1.0

            # --------------------------------------------------------
            # Avoid obstacle
            # --------------------------------------------------------

            if self.avoiding:

                # If front is clear again, return to waypoint tracking
                if front > 0.85:

                    self.avoiding = False

                else:

                    msg = Twist()

                    # Turn toward free side
                    msg.angular.z = (
                        self.avoid_direction *
                        0.75
                    )

                    # Move slowly while turning
                    msg.linear.x = 0.08

                    self.limit_cmd(msg)

                    self.cmd_pub.publish(msg)

                    return

            # ========================================================
            # NORMAL WAYPOINT FOLLOWING
            # ========================================================

            # Proportional angular controller
            Kp_angular = 1.8

            angular = Kp_angular * heading_error

            # Limit angular velocity
            angular = max(
                -max_angular_rps,
                min(max_angular_rps, angular)
            )

            # --------------------------------------------------------
            # Slow down while turning
            # --------------------------------------------------------

            heading_abs = abs(heading_error)

            if heading_abs > math.radians(60):

                linear = 0.05

            elif heading_abs > math.radians(35):

                linear = 0.12

            elif heading_abs > math.radians(15):

                linear = 0.22

            else:

                linear = 0.38

            # --------------------------------------------------------
            # Slow down near waypoint
            # --------------------------------------------------------

            if distance < 0.5:

                linear = min(
                    linear,
                    0.15
                )

            # --------------------------------------------------------
            # Extra obstacle precaution
            # --------------------------------------------------------

            if front < 1.0:

                linear = min(
                    linear,
                    0.12
                )

            # --------------------------------------------------------
            # Publish command
            # --------------------------------------------------------

            msg = Twist()

            msg.linear.x = linear
            msg.angular.z = angular

            self.limit_cmd(msg)

            self.cmd_pub.publish(msg)

        except Exception as e:

            # Never allow controller exception to kill node
            self.get_logger().error(
                f"Navigation error: {e}"
            )

            self.stop_robot()

    # ================================================================
    # LIMIT COMMAND
    # ================================================================

    def limit_cmd(self, msg):

        msg.linear.x = max(
            -max_linear_mps,
            min(max_linear_mps, msg.linear.x)
        )

        msg.angular.z = max(
            -max_angular_rps,
            min(max_angular_rps, msg.angular.z)
        )

    # ================================================================
    # STOP
    # ================================================================

    def stop_robot(self):

        msg = Twist()

        msg.linear.x = 0.0
        msg.angular.z = 0.0

        self.cmd_pub.publish(msg)


# ====================================================================
# MAIN
# ====================================================================

def main():

    rclpy.init(args=sys.argv)

    node = ebot_nav()

    try:

        rclpy.spin(node)

    except KeyboardInterrupt:

        pass

    finally:

        node.stop_robot()

        node.destroy_node()

        rclpy.shutdown()


if __name__ == '__main__':

    main()


################### IMPORT MODULES #######################

import rclpy
import sys
import math

from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
    qos_profile_sensor_data
)
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import LaserScan


##################### TASK CONSTANTS #######################

# Route, odometry, lidar and command topics.
path_topic = '/ebot_path'
odom_topic = '/odom'
scan_topic = '/scan'
cmd_topic = '/cmd_vel'

# Maximum velocity limits for the eBot.
max_linear_mps = 0.5
max_angular_rps = 1.0


##################### CLASS DEFINITION #######################

class ebot_nav(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to drive the eBot along the route
                    published on /ebot_path, through its waypoints, in order.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ebot_nav.
        '''

        # use_sim_time is set here so that the node follows the simulation clock.
        super().__init__(
            'ebot_nav_node',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time',
                    rclpy.Parameter.Type.BOOL,
                    True
                )
            ]
        )

        ############ Topic SUBSCRIPTIONS ############

        # /ebot_path is published once and held for late subscribers.
        # TRANSIENT_LOCAL QoS is therefore required to receive the route.
        path_qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.path_sub = self.create_subscription(
            Path,
            path_topic,
            self.pathcb,
            path_qos
        )

        self.odom_sub = self.create_subscription(
            Odometry,
            odom_topic,
            self.odomcb,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            scan_topic,
            self.scancb,
            qos_profile_sensor_data
        )

        ############ Topic PUBLISHERS ############

        self.cmd_pub = self.create_publisher(
            Twist,
            cmd_topic,
            10
        )

        ############ Constructor VARIABLES/OBJECTS ############

        control_rate = 0.05
        self.timer = self.create_timer(
            control_rate,
            self.process_navigation
        )

        self.route = None
        self.odom = None
        self.scan = None

        self.current_waypoint = 0

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0

        self.finished = False

        # Variables used to maintain obstacle avoidance state between cycles.
        self.avoid_direction = 0.0
        self.avoiding = False

        self.get_logger().info('eBot navigation node started')

    def pathcb(self, data):
        '''
        Description:    Callback function for the route topic.
                        Receives the waypoints that the eBot has to follow.

        Args:
            data (Path): The route, as a sequence of poses.

        Returns:
            None
        '''

        try:
            # Take the route only once. Replacing it during navigation would
            # reset the waypoint progress.
            if self.route is not None:
                return

            if len(data.poses) == 0:
                self.get_logger().warn('Received empty path')
                return

            self.route = []

            for pose in data.poses:
                x = pose.pose.position.x
                y = pose.pose.position.y
                self.route.append((x, y))

            self.current_waypoint = 0

            self.get_logger().info(
                f'Received route with {len(self.route)} waypoints '
                f"in frame '{data.header.frame_id}'"
            )

            for index, point in enumerate(self.route):
                self.get_logger().info(
                    f'Waypoint {index + 1}: '
                    f'x={point[0]:.3f}, y={point[1]:.3f}'
                )

        except Exception as error:
            self.get_logger().error(
                f'Path callback error: {error}'
            )

    def odomcb(self, data):
        '''
        Description:    Callback function for the odometry topic.
                        Receives the current position and heading of the eBot.

        Args:
            data (Odometry): Pose and velocity of the base.

        Returns:
            None
        '''

        try:
            self.odom = data

            position = data.pose.pose.position
            orientation = data.pose.pose.orientation

            self.x = position.x
            self.y = position.y

            # Convert quaternion orientation to yaw.
            siny_cosp = 2.0 * (
                orientation.w * orientation.z +
                orientation.x * orientation.y
            )

            cosy_cosp = 1.0 - 2.0 * (
                orientation.y * orientation.y +
                orientation.z * orientation.z
            )

            self.yaw = math.atan2(
                siny_cosp,
                cosy_cosp
            )

        except Exception as error:
            self.get_logger().error(
                f'Odometry callback error: {error}'
            )

    def scancb(self, data):
        '''
        Description:    Callback function for the lidar topic.
                        Stores the latest lidar scan.

        Args:
            data (LaserScan): One lidar sweep.

        Returns:
            None
        '''

        try:
            self.scan = data

        except Exception as error:
            self.get_logger().error(
                f'Scan callback error: {error}'
            )

    def wrap_angle(self, angle):
        '''
        Description:    Wraps an angle to the range [-pi, pi].

        Args:
            angle (float): Angle in radians.

        Returns:
            float: Wrapped angle in radians.
        '''

        return math.atan2(
            math.sin(angle),
            math.cos(angle)
        )

    def get_sector_distance(self, start_angle, end_angle):
        '''
        Description:    Returns the nearest valid lidar measurement within
                        the requested angular sector.

        Args:
            start_angle (float): Starting sector angle in radians.
            end_angle (float): Ending sector angle in radians.

        Returns:
            float: Minimum valid range in the sector, or infinity if none exists.
        '''

        if self.scan is None:
            return float('inf')

        values = []
        angle = self.scan.angle_min

        for range_value in self.scan.ranges:
            if (
                math.isfinite(range_value)
                and self.scan.range_min <= range_value <= self.scan.range_max
            ):
                normalized_angle = self.wrap_angle(angle)

                # Handle both normal sectors and sectors crossing -pi/pi.
                if start_angle <= end_angle:
                    if start_angle <= normalized_angle <= end_angle:
                        values.append(range_value)
                else:
                    if (
                        normalized_angle >= start_angle
                        or normalized_angle <= end_angle
                    ):
                        values.append(range_value)

            angle += self.scan.angle_increment

        if not values:
            return float('inf')

        return min(values)

    def process_navigation(self):
        '''
        Description:    Timer callback used to drive the eBot along the route.
                        It performs waypoint tracking and obstacle avoidance.

        Returns:
            None
        '''

        try:
            # Return until all required data has arrived.
            if self.route is None:
                return

            if self.odom is None:
                return

            if self.scan is None:
                return

            # Stop after the complete route has been traversed.
            if self.finished:
                self.stop_robot()
                return

            # Check that the current waypoint index is valid.
            if self.current_waypoint >= len(self.route):
                self.finished = True
                self.stop_robot()
                self.get_logger().info('All waypoints completed!')
                return

            # Get the current target waypoint.
            target_x, target_y = self.route[self.current_waypoint]

            dx = target_x - self.x
            dy = target_y - self.y

            distance = math.hypot(dx, dy)

            # Move to the next waypoint once the current one is reached.
            if distance <= 0.30:
                self.get_logger().info(
                    f'Reached waypoint {self.current_waypoint + 1}'
                )

                self.current_waypoint += 1

                if self.current_waypoint >= len(self.route):
                    self.finished = True
                    self.stop_robot()
                    self.get_logger().info('FINAL WAYPOINT REACHED')

                return

            # Calculate the heading error towards the current waypoint.
            target_angle = math.atan2(dy, dx)

            heading_error = self.wrap_angle(
                target_angle - self.yaw
            )

            ##################### OBSTACLE DETECTION #####################

            # Front sector: approximately +/- 25 degrees.
            front = self.get_sector_distance(
                math.radians(-25),
                math.radians(25)
            )

            # Left sector: 25 to 70 degrees.
            left = self.get_sector_distance(
                math.radians(25),
                math.radians(70)
            )

            # Right sector: -70 to -25 degrees.
            right = self.get_sector_distance(
                math.radians(-70),
                math.radians(-25)
            )

            ##################### OBSTACLE AVOIDANCE #####################

            obstacle_distance = 0.65

            if front < obstacle_distance:
                self.avoiding = True

                # Choose the side with more free space.
                if left > right:
                    self.avoid_direction = 1.0
                else:
                    self.avoid_direction = -1.0

            if self.avoiding:
                # Resume waypoint tracking after the front is clear.
                if front > 0.85:
                    self.avoiding = False
                else:
                    msg = Twist()

                    # Turn towards the side with more free space.
                    msg.angular.z = self.avoid_direction * 0.75

                    # Move slowly while turning.
                    msg.linear.x = 0.08

                    self.limit_cmd(msg)
                    self.cmd_pub.publish(msg)

                    return

            ##################### NORMAL WAYPOINT FOLLOWING #####################

            # Proportional angular controller.
            kp_angular = 1.8
            angular = kp_angular * heading_error

            # Keep angular velocity within the required limit.
            angular = max(
                -max_angular_rps,
                min(max_angular_rps, angular)
            )

            # Reduce linear velocity as the heading error increases.
            heading_abs = abs(heading_error)

            if heading_abs > math.radians(60):
                linear = 0.05
            elif heading_abs > math.radians(35):
                linear = 0.12
            elif heading_abs > math.radians(15):
                linear = 0.22
            else:
                linear = 0.38

            # Slow down near the current waypoint.
            if distance < 0.5:
                linear = min(linear, 0.15)

            # Add an additional precaution when an obstacle is nearby.
            if front < 1.0:
                linear = min(linear, 0.12)

            # Publish the velocity command.
            msg = Twist()
            msg.linear.x = linear
            msg.angular.z = angular

            self.limit_cmd(msg)
            self.cmd_pub.publish(msg)

        except Exception as error:
            # Prevent an unexpected controller exception from killing the node.
            self.get_logger().error(
                f'Navigation error: {error}'
            )
            self.stop_robot()

    def limit_cmd(self, msg):
        '''
        Description:    Limits the commanded linear and angular velocities.

        Args:
            msg (Twist): Velocity command to be limited.

        Returns:
            None
        '''

        msg.linear.x = max(
            -max_linear_mps,
            min(max_linear_mps, msg.linear.x)
        )

        msg.angular.z = max(
            -max_angular_rps,
            min(max_angular_rps, msg.angular.z)
        )

    def stop_robot(self):
        '''
        Description:    Publishes a zero velocity command to stop the eBot.

        Returns:
            None
        '''

        msg = Twist()
        msg.linear.x = 0.0
        msg.angular.z = 0.0

        self.cmd_pub.publish(msg)


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which initializes the ROS 2 node and spins
                    the ebot_nav class until shutdown.
    '''

    rclpy.init(args=sys.argv)

    ebot_nav_class = ebot_nav()

    try:
        rclpy.spin(ebot_nav_class)

    except KeyboardInterrupt:
        pass

    finally:
        ebot_nav_class.stop_robot()
        ebot_nav_class.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()