#!/usr/bin/env python3

'''
# Team ID:          [ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:		    task1b_boilerplate.py
# Functions:
# [ Comma separated list of functions in this file ]
# Nodes:		    Add your publishing and subscribing node
#                   Example:
# Publishing Topics  - [ /delta_twist_cmds, /delta_joint_cmds ]
#                   Subscribing Topics - [ /tcp_pose_raw, /joint_states, /etc... ]
'''

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
from scipy.spatial.transform import Rotation
from std_srvs.srv import Trigger

waypoints = [
    (-0.4085, -0.5379, 0.1967),   # 1
    (-0.8000, -0.0005, 0.3967),   # 2
    (-0.7430, 0.5280, 0.1967),   # 3
    (-0.4097, 0.5280, 0.1967),   # 4
    (-0.0763, 0.5280, 0.1967),   # 5
]

path_points = [
    waypoints[0],                                              # WP1 - real
    (waypoints[0][0], waypoints[0][1], waypoints[1][2]),       # via: rise in Z first
    waypoints[1],                                              # WP2 - real
    waypoints[2],                                              # WP3 - real, direct from WP2
    waypoints[3],                                              # WP4 - real
    waypoints[4],                                              # WP5 - real
]
path_is_real = [True, False, True, True, True, True]

servo_ns = '/ur_arm_controller'
twist_controller = 'delta_twist_controller'
joint_controller = 'delta_joint_controller'

base_frame = 'base_link'

joint_names = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]

cap_linear_mps = 0.15
cap_angular_rps = 0.35
cap_joint_rps = 0.35

command_timeout_s = 0.15


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

        super().__init__(
            'arm_waypoints_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        self.twist_pub = self.create_publisher(
            TwistStamped,
            '/delta_twist_cmds',
            10)
        self.joint_pub = self.create_publisher(
            JointJog, '/delta_joint_cmds', 10)

        self.tcp_sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20)
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.jointstatecb, 50)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.armstatuscb, 10)

        self.unlock_cli = self.create_client(Trigger, f'{servo_ns}/unlock_protective_stop')
        self.unlock_future = None
        self.last_safe_position = waypoints[0]

        control_rate = 0.05
        self.switch_cli = self.create_client(
            SwitchController, f'{servo_ns}/switch_controller')

        self.timer = self.create_timer(control_rate, self.process_waypoints)

        self.tcp_pose = None

        self.joint_angles = None

        self.arm_status = None

        self.recovery_controller_set = False

        self.path_idx = 0
        self.hold_start_time = None
        self.target_orientation = None
        self.unfold_target = {
            'shoulder_pan_joint': 0.0, 'shoulder_lift_joint': -1.20, 'elbow_joint': -1.60,
            'wrist_1_joint': -1.90, 'wrist_2_joint': 1.57, 'wrist_3_joint': 1.57,
        }
        self.unfolded = False
        self.unfold_controller_set = False
        self.switch_future = None

        self.controller_ready = self.switch_controller('twist_controller')
        if not self.controller_ready:
            self.get_logger().error('Failed to activate twist controller')

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

        try:
            self.joint_angles = [data.position[data.name.index(j)] for j in joint_names]
        except ValueError as e:
            self.get_logger().warn(f'Joint name not found: {e}')

    def armstatuscb(self, data):
        '''
        Description:    Callback function for the arm status topic.
                        Use this function to receive the arm's current state code.

        Args:
            data (Int32):    One state code describing what the arm is doing

        Returns:
        '''

        if self.arm_status != data.data:
            self.get_logger().info(f'Arm status changed: {data.data}')
        self.arm_status = data.data

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

        target = twist_controller if controller == 'twist_controller' else joint_controller
        other = joint_controller if controller == 'twist_controller' else twist_controller

        req = SwitchController.Request()
        req.activate_controllers = [target]
        req.deactivate_controllers = [other]
        req.strictness = SwitchController.Request.STRICT

        if not self.switch_cli.wait_for_service(timeout_sec=20.0):
            self.get_logger().error('switch_controller service not available')
            return False

        future = self.switch_cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        if future.result() is None:
            self.get_logger().warn('First switch_controller attempt failed, retrying')
            future = self.switch_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)

        return future.result() is not None and future.result().ok

    def request_switch(self, controller):
        '''Non-blocking version, safe to call every timer tick. Returns True once
        the switch has completed successfully; False while pending or on failure.'''
        if self.switch_future is not None:
            if self.switch_future.done():
                result = self.switch_future.result()
                self.switch_future = None
                if result is not None and result.ok:
                    return True
                self.get_logger().warn(f'switch_controller to {controller} failed, retrying')
                return False
            return False  # still waiting on the service

        target = twist_controller if controller == 'twist_controller' else joint_controller
        other = joint_controller if controller == 'twist_controller' else twist_controller
        req = SwitchController.Request()
        req.activate_controllers = [target]
        req.deactivate_controllers = [other]
        req.strictness = SwitchController.Request.STRICT

        if not self.switch_cli.service_is_ready():
            return False

        self.switch_future = self.switch_cli.call_async(req)
        return False

    def unfold_step(self):
        if not self.unfold_controller_set:
            if self.request_switch('joint_controller'):
                self.unfold_controller_set = True
            return

        if self.joint_angles is None:
            return

        Kp_joint = 0.6
        velocities = []
        max_err = 0.0
        for name in joint_names:
            idx = joint_names.index(name)
            cur = self.joint_angles[idx]
            tgt = self.unfold_target[name]
            err = tgt - cur
            max_err = max(max_err, abs(err))
            v = Kp_joint * err
            v = max(min(v, cap_joint_rps * 0.9), -cap_joint_rps * 0.9)
            velocities.append(v)

        if max_err < 0.03:
            if self.request_switch('twist_controller'):
                self.unfolded = True
                self.target_orientation = self.tcp_pose.orientation
                self.get_logger().info('Unfold complete, switching to twist control')
            return

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names
        msg.velocities = velocities
        self.joint_pub.publish(msg)

    def process_waypoints(self):
        '''
        Description:    Timer function used to drive the tool through the waypoints.

        Args:
        Returns:
        '''
        if self.arm_status is not None and 20 <= self.arm_status < 30:
            self.handle_recovery()
            return

        if self.tcp_pose is None or self.joint_angles is None or self.arm_status is None:
            return

        if not self.unfolded:
            self.unfold_step()
            return

        if self.path_idx >= len(path_points):
            self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            return

        target = path_points[self.path_idx]
        is_real = path_is_real[self.path_idx]
        pos_tolerance = 0.005 if is_real else 0.02  # via points: looser, just pass through

        cx, cy, cz = self.tcp_pose.position.x, self.tcp_pose.position.y, self.tcp_pose.position.z
        ex, ey, ez = target[0] - cx, target[1] - cy, target[2] - cz
        pos_error = math.sqrt(ex * ex + ey * ey + ez * ez)

        Kp_lin = 0.2
        vx, vy, vz = Kp_lin * ex, Kp_lin * ey, Kp_lin * ez
        lin_mag = math.sqrt(vx * vx + vy * vy + vz * vz)
        if lin_mag > cap_linear_mps:
            scale = (cap_linear_mps * 0.9) / lin_mag
            vx, vy, vz = vx * scale, vy * scale, vz * scale

        wx = wy = wz = 0.0
        if self.target_orientation is not None:
            cur_q = [self.tcp_pose.orientation.x, self.tcp_pose.orientation.y,
                     self.tcp_pose.orientation.z, self.tcp_pose.orientation.w]
            tgt_q = [self.target_orientation.x, self.target_orientation.y,
                     self.target_orientation.z, self.target_orientation.w]
            r_err = Rotation.from_quat(tgt_q) * Rotation.from_quat(cur_q).inv()
            rotvec = r_err.as_rotvec()

            Kp_ang = 2.0
            wx, wy, wz = Kp_ang * rotvec[0], Kp_ang * rotvec[1], Kp_ang * rotvec[2]
            ang_mag = math.sqrt(wx * wx + wy * wy + wz * wz)
            if ang_mag > cap_angular_rps:
                scale = (cap_angular_rps * 0.9) / ang_mag
                wx, wy, wz = wx * scale, wy * scale, wz * scale

        if pos_error < pos_tolerance:
            if is_real:
                if self.hold_start_time is None:
                    self.hold_start_time = self.get_clock().now()
                    self.get_logger().info(
                        f'Reached real waypoint at path idx {
                            self.path_idx}, holding')
                elapsed = (self.get_clock().now() - self.hold_start_time).nanoseconds / 1e9
                if elapsed >= 2.0:
                    self.get_logger().info(f'Hold complete at path idx {self.path_idx}')
                    self.last_safe_position = target
                    self.path_idx += 1
                    self.hold_start_time = None
                self.publish_twist(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
                return
            else:
                self.path_idx += 1
        else:
            self.hold_start_time = None

        self.publish_twist(vx, vy, vz, wx, wy, wz)

    def publish_twist(self, vx, vy, vz, wx, wy, wz):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        msg.twist.angular.x = wx
        msg.twist.angular.y = wy
        msg.twist.angular.z = wz
        self.twist_pub.publish(msg)

    def handle_recovery(self):
        if self.unlock_future is None:
            if not self.unlock_cli.service_is_ready():
                return
            self.unlock_future = self.unlock_cli.call_async(Trigger.Request())
            return
        if not self.unlock_future.done():
            return
        result = self.unlock_future.result()
        self.unlock_future = None
        if result is None or not result.success:
            self.get_logger().error(f'Unlock failed: {result.message if result else "no result"}')
            return
        self.get_logger().warn(f'Unlocked: {result.message}')

        if not self.recovery_controller_set:
            if self.request_switch('joint_controller'):
                self.recovery_controller_set = True
            return

        if self.joint_angles is None:
            return

        Kp_joint = 0.4
        velocities = []
        max_err = 0.0
        for name in joint_names:
            idx = joint_names.index(name)
            cur = self.joint_angles[idx]
            tgt = self.unfold_target[name]
            err = tgt - cur
            max_err = max(max_err, abs(err))
            v = Kp_joint * err
            v = max(min(v, cap_joint_rps * 0.8), -cap_joint_rps * 0.8)
            velocities.append(v)

        if max_err < 0.1:
            if self.request_switch('twist_controller'):
                self.recovery_controller_set = False
                self.get_logger().info('Recovered clear of singularity, resuming twist control')
            return

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names
        msg.velocities = velocities
        self.joint_pub.publish(msg)


def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    arm_waypoints class to perform its task
    '''

    rclpy.init(args=sys.argv)                                       # initialisation

    node = rclpy.create_node('arm_waypoints_process')               # creating ROS node

    node.get_logger().info('Node created: Arm waypoints process')   # logging information

    # creating a new object for class 'arm_waypoints'
    arm_waypoints_class = arm_waypoints()

    # spining on the object to make it alive in ROS 2 DDS
    rclpy.spin(arm_waypoints_class)

    arm_waypoints_class.destroy_node()                              # destroy node after spin ends

    rclpy.shutdown()                                                # shutdown process


if __name__ == '__main__':

    main()