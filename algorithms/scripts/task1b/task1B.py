#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 1B of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:\t\t    task1B.py
# Functions:
#                 [ main, arm_waypoints.__init__, tcpposecb, jointstatecb,
#                   armstatuscb, switch_controller, unfold_arm, recover_arm,
#                   calculate_position_velocity, process_waypoints ]
# Nodes:
# Publishing Topics  - [ /delta_twist_cmds, /delta_joint_cmds ]
# Subscribing Topics - [ /tcp_pose_raw, /joint_states, /arm_status ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
from rclpy.node import Node
from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32
from std_srvs.srv import Trigger


##################### TASK CONSTANTS #######################

# Tool positions in base_link, in metres, in the order they must be reached. The tool
# stops at each one and holds it for at least two seconds. Copy the signs as they are:
# base_link is the UR7e's own frame, not the Gazebo world's.
waypoints = [
    (-0.4085, -0.5379, 0.1967),   # 1
    (-0.8000, -0.0005, 0.3967),   # 2
    (-0.7430, 0.5280, 0.1967),   # 3
    (-0.4097, 0.5280, 0.1967),   # 4
    (-0.0763, 0.5280, 0.1967),   # 5
]

path_points = [
    waypoints[0],
    (waypoints[0][0], waypoints[0][1], waypoints[1][2]),
    waypoints[1],
    waypoints[2],
    waypoints[3],
    waypoints[4],
]

path_is_real = [
    True,
    False,
    True,
    True,
    True,
    True,
]

# The two command interfaces. Only ONE is active at a time; messages to the other are
# accepted and ignored.
servo_ns = '/ur_arm_controller'
twist_controller = 'delta_twist_controller'
joint_controller = 'delta_joint_controller'

# The only frame a twist may be stamped with; any other is refused, not converted. Note
# 'base' is base_link turned through 180 degrees, not another name for it.
base_frame = 'base_link'

# JointJog velocities are matched to these names, in this order.
joint_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

# What the servo accepts. A command above one of these is dropped WHOLE, not clamped.
cap_linear_mps = 0.15     # magnitude of a twist's linear part
cap_angular_rps = 0.35    # magnitude of its angular part
cap_joint_rps = 0.35      # per joint

unfold_target = {
    'shoulder_pan_joint': 0.0,
    'shoulder_lift_joint': -1.20,
    'elbow_joint': -1.60,
    'wrist_1_joint': -1.90,
    'wrist_2_joint': 1.57,
    'wrist_3_joint': 1.57,
}

recovery_target = {
    'shoulder_pan_joint': 0.0,
    'shoulder_lift_joint': -0.523599,
    'elbow_joint': -2.443461,
    'wrist_1_joint': -0.174533,
    'wrist_2_joint': 1.57,
    'wrist_3_joint': 1.57,
}

unfold_kp = 0.6

# Dead-man switch: the arm stops this long after the last message it received.
command_timeout_s = 0.15

position_tolerance = 0.005   # 5 mm

##################### CLASS DEFINITION #######################


class arm_waypoints(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to drive the UR7e's tool through the
                    given waypoints using the arm's velocity command interfaces.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class arm_waypoints
        '''

        # use_sim_time is set here, not on the command line, so this node runs on the
        # simulation clock however it is started.
        super().__init__(                                                               # registering node
            'arm_waypoints_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        ############ Topic PUBLISHERS ############

        self.twist_pub = self.create_publisher(
            TwistStamped,
            '/delta_twist_cmds',
            10)    # end-effector velocity, in base_link
        self.joint_pub = self.create_publisher(
            JointJog, '/delta_joint_cmds', 10)        # per-joint velocity

        ############ Topic SUBSCRIPTIONS ############

        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.jointstatecb, 50)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)

        ############ Constructor VARIABLES/OBJECTS ############

        # rate of time to run one control cycle (seconds)
        control_rate = 0.05
        self.switch_cli = self.create_client(                                           # client used to pick which command topic is live
            SwitchController, f'{servo_ns}/switch_controller')
        # creating a timer based function which gets called on every 0.05 seconds
        # (as defined by 'control_rate' variable)
        self.timer = self.create_timer(control_rate, self.process_waypoints)

        # tool pose variable (from tcpposecb())
        self.tcp_pose = None
        # joint feedback variable (from jointstatecb())
        self.joint_angles = None
        # arm state code variable (from armstatuscb())
        self.arm_status = None
        self.recovery_required = False
        self.in_recovery = False

        # index of the current waypoint to reach
        self.path_idx = 0
        # time spent holding at the current waypoint
        self.holding = False
        # whether the arm is currently holding at a waypoint
        self.hold_start_time = None
        self.unfolded = False

        self.switch_future = None
        self.switch_target = None

        self.motion_state = 'INIT'
        self.recovery_state = 'NONE'

        self.unlock_client = self.create_client(
            Trigger,
            '/ur_arm_controller/unlock_protective_stop'
        )

        self.unlock_future = None

        ############ ADD YOUR CODE HERE ############

        # INSTRUCTIONS & HELP :

        # ->  Add any variable your motion needs to keep between cycles.
        #       ->  HINT: Which waypoint you are on, and what the arm is doing about it.

        ############################################

    def tcpposecb(self, data):
        '''
        Description:    Callback function for the tool pose topic.
                        Use this function to receive where the tool currently is.

        Args:
            data (PoseStamped):    Pose of the tool, reported in base_link

        Returns:
        '''

        self.tcp_pose = data.pose

    def jointstatecb(self, data):
        '''
        Description:    Callback function for the joint states topic.
                        Use this function to receive the current angle of each joint.

        Args:
            data (JointState):    Joint feedback published by the arm

        Returns:
        '''

        self.joint_angles = [
            data.position[data.name.index(joint)]
            for joint in joint_names
        ]

    def armstatuscb(self, data):
        '''
        Description:
            Callback function for the arm status topic.

        Args:
            data (Int32): One state code describing the arm state.

        Returns:
            None
        '''

        self.arm_status = data.data

        if 20 <= self.arm_status <= 29:
            if not self.in_recovery:
                self.recovery_required = True

    def switch_controller(self, controller):
        '''
        Description:    Function to make one of the arm's two command interfaces the active
                        one, so that commands published to it are acted on.

        Args:
            controller  (str):      Name of the controller to activate, either
                                    'twist_controller' or 'joint_controller'

        Returns:
            success     (bool):     Whether the controller was activated
        '''

        req = SwitchController.Request()
        req.activate_controllers = [controller]

        if controller == twist_controller:
            other_controller = joint_controller
        else:
            other_controller = twist_controller

        req.deactivate_controllers = [other_controller]
        req.strictness = SwitchController.Request.STRICT

        if not self.switch_cli.wait_for_service(timeout_sec=20.0):
            self.get_logger().error('Controller switch service not available')
            return False

        self.switch_future = self.switch_cli.call_async(req)
        self.switch_target = controller

        return None

    def unfold_arm(self):
        '''
        Description:
            Move all arm joints toward a safe unfolded configuration
            before beginning Cartesian waypoint motion.

        Returns:
            bool: True when the arm has been sufficiently unfolded.
        '''

        velocities = []
        max_error = 0.0

        for joint in joint_names:

            joint_index = joint_names.index(joint)

            current_angle = self.joint_angles[joint_index]
            target_angle = unfold_target[joint]

            error = target_angle - current_angle

            max_error = max(max_error, abs(error))

            velocity = unfold_kp * error

            velocity = max(
                min(velocity, cap_joint_rps * 0.9),
                -cap_joint_rps * 0.9
            )

            velocities.append(velocity)

        if max_error < 0.03:

            msg = JointJog()

            msg.header.stamp = self.get_clock().now().to_msg()
            msg.joint_names = joint_names
            msg.velocities = [0.0] * len(joint_names)

            self.joint_pub.publish(msg)

            self.unfolded = True

            self.get_logger().info(
                'Arm unfolded. Switching to twist controller.'
            )

            return True

        msg = JointJog()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names
        msg.velocities = velocities

        self.joint_pub.publish(msg)

        return False

    def recover_arm(self):
        '''
        Description:
            Move the shoulder lift joint upward to raise the arm
            away from the platform.

        Returns:
            bool: True when the shoulder has moved sufficiently.
        '''

        joint_index = joint_names.index('shoulder_lift_joint')

        current_angle = self.joint_angles[joint_index]

        # Positive shoulder-lift velocity raises the arm from
        # the observed recovery configuration.
        velocity = 0.05

        msg = JointJog()

        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names

        velocities = [0.0] * len(joint_names)
        velocities[joint_index] = velocity

        msg.velocities = velocities

        self.joint_pub.publish(msg)

        # Move only a small amount for this test.
        return current_angle > -2.80

    def calculate_position_velocity(self, target):
        '''
        Description:
            Calculate a proportional Cartesian velocity command
            to move the TCP toward the target position.

        Args:
            target (tuple): Target x, y, z position in base_link.

        Returns:
            tuple: Linear velocity vx, vy, vz and position error magnitude.
        '''

        # Current TCP position.
        current_x = self.tcp_pose.position.x
        current_y = self.tcp_pose.position.y
        current_z = self.tcp_pose.position.z

        # Position error.
        error_x = target[0] - current_x
        error_y = target[1] - current_y
        error_z = target[2] - current_z

        # Proportional gain.
        kp = 0.2

        # Proportional velocity.
        vx = kp * error_x
        vy = kp * error_y
        vz = kp * error_z

        # Magnitude of the velocity vector.
        velocity_magnitude = math.sqrt(
            vx * vx +
            vy * vy +
            vz * vz
        )

        # Scale the whole vector if it exceeds the allowed limit.
        if velocity_magnitude > cap_linear_mps:

            scale = (cap_linear_mps * 0.9) / velocity_magnitude

            vx *= scale
            vy *= scale
            vz *= scale

        # Position error magnitude.
        distance = math.sqrt(
            error_x * error_x +
            error_y * error_y +
            error_z * error_z
        )

        return vx, vy, vz, distance

    def process_waypoints(self):

        if self.recovery_required:

            self.twist_pub.publish(TwistStamped())

            if self.unlock_future is None:

                if self.unlock_client.service_is_ready():
                    request = Trigger.Request()

                    self.unlock_future = (
                        self.unlock_client.call_async(request)
                    )

                    self.get_logger().warn(
                        'Protective stop detected. Unlock request sent.'
                    )

                return

            if not self.unlock_future.done():
                return

            response = self.unlock_future.result()
            self.unlock_future = None

            if response is not None and response.success:
                self.get_logger().warn(
                    'Protective stop unlocked. Switching to joint recovery.'
                )

                self.recovery_required = False
                self.in_recovery = True
                self.recovery_state = 'SWITCH_TO_JOINT'
                self.switch_controller(joint_controller)

            else:
                self.get_logger().error(
                    'Protective stop unlock failed.'
                )

            return

        '''
        Description:
            Main control loop for the arm motion sequence.
        '''

        # Wait until all required feedback has been received.
        if self.tcp_pose is None:
            return

        if self.joint_angles is None:
            return

        if self.arm_status is None:
            return

        # ---------------------------------------------------------
        # Check whether a controller switch is still in progress.
        # ---------------------------------------------------------
        if self.switch_future is not None:

            if not self.switch_future.done():
                return

            response = self.switch_future.result()

            self.switch_future = None

            if response is None or not response.ok:
                message = 'No response received' if response is None else response.message

                self.get_logger().error(
                    f'Failed to switch to {self.switch_target}: {message}'
                )

                self.motion_state = 'ERROR'
                return

            self.get_logger().info(
                f'Switched to {self.switch_target}'
            )

            if self.switch_target == joint_controller:

                if self.recovery_state == 'SWITCH_TO_JOINT':
                    self.motion_state = 'RECOVERY'
                    self.recovery_state = 'RECOVERY'
                else:
                    self.motion_state = 'UNFOLDING'

            else:
                self.motion_state = 'WAYPOINTS'

            self.switch_target = None

        # ---------------------------------------------------------
        # Initial controller selection.
        # ---------------------------------------------------------
        if self.motion_state == 'INIT':

            self.get_logger().info(
                'Starting arm sequence: switching to joint controller'
            )

            self.motion_state = 'SWITCH_TO_JOINT'
            self.switch_controller(joint_controller)
            return

        # ---------------------------------------------------------
        # Unfold the arm.
        # ---------------------------------------------------------
        if self.motion_state == 'UNFOLDING':

            if self.unfold_arm():
                self.get_logger().info(
                    'Unfolding complete. Switching to twist controller.'
                )

                self.motion_state = 'SWITCH_TO_TWIST'
                self.switch_controller(twist_controller)

            return

        # ---------------------------------------------------------
        # Recover from a protective stop.
        # ---------------------------------------------------------
        if self.motion_state == 'RECOVERY':

            recovery_complete = self.recover_arm()

            if self.arm_status == 0 and recovery_complete:

                self.in_recovery = False
                self.recovery_state = 'NONE'
                self.motion_state = 'SWITCH_TO_TWIST'

                self.switch_controller(twist_controller)

                self.get_logger().warn(
                    'Recovery complete. Switching back to twist controller.'
                )

                return

        # ---------------------------------------------------------
        # Waypoint motion will be implemented later.
        # ---------------------------------------------------------

        if self.motion_state == 'WAYPOINTS':

            if self.path_idx >= len(path_points):
                msg = TwistStamped()
                msg.header.stamp = self.get_clock().now().to_msg()
                msg.header.frame_id = base_frame
                self.twist_pub.publish(msg)
                return

            target = path_points[self.path_idx]

            vx, vy, vz, distance = self.calculate_position_velocity(target)

            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = base_frame

            msg.twist.linear.x = vx
            msg.twist.linear.y = vy
            msg.twist.linear.z = vz

            self.twist_pub.publish(msg)

            if distance < position_tolerance:

                if not self.holding:
                    self.holding = True
                    self.hold_start_time = self.get_clock().now()

                elapsed = (
                    self.get_clock().now() - self.hold_start_time
                ).nanoseconds / 1e9

                if elapsed >= 2.0:

                    self.get_logger().warn(
                        f'Waypoint {self.path_idx + 1} reached.'
                    )

                    self.path_idx += 1
                    self.holding = False
                    self.hold_start_time = None

            else:
                self.holding = False
                self.hold_start_time = None

            return

        # ---------------------------------------------------------
        # Error state.
        # ---------------------------------------------------------
        if self.motion_state == 'ERROR':
            return

##################### FUNCTION DEFINITION #######################


def main(args=None):
    rclpy.init(args=args)

    node = rclpy.create_node('arm_waypoints_process')
    node.get_logger().info('Node created: Arm waypoints process')

    arm_waypoints_class = arm_waypoints()

    arm_waypoints_class.get_logger().set_level(
        rclpy.logging.LoggingSeverity.WARN
    )

    rclpy.spin(arm_waypoints_class)

    arm_waypoints_class.destroy_node()
    node.destroy_node()
    rclpy.shutdown()                                              # shutdown process


if __name__ == '__main__':

    main()