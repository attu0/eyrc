#!/usr/bin/env python3

'''
StrataCobot (SC) Theme (eYRC 2026-27)
Task 2A - Ore manipulation

This node combines the Task 1B UR7e velocity-control approach with:
  - TF lookup of ore positions published by Task 2A perception
  - magnet service control
  - force-based grasp verification
  - pick -> lift -> transport -> drop sequencing

No motion-planning package is used.
'''

import rclpy
import sys
import math
import tf2_ros

from rclpy.node import Node
from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import PoseStamped, TwistStamped
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32, Float32
from std_srvs.srv import Trigger, SetBool


# ------------------------- TASK CONSTANTS -------------------------

servo_ns = '/ur_arm_controller'
twist_controller = 'delta_twist_controller'
joint_controller = 'delta_joint_controller'

base_frame = 'base_link'

joint_names = [
    'shoulder_pan_joint',
    'shoulder_lift_joint',
    'elbow_joint',
    'wrist_1_joint',
    'wrist_2_joint',
    'wrist_3_joint',
]

# Task 1B container/tool positions, in base_link.
DROP_POINTS = {
    'azurite_ore': (-0.0700, 0.5500, 0.1500),
    'malachite_ore': (-0.4000, 0.5500, 0.1500),
    'vanadinite_ore': (-0.6500, 0.5500, 0.1500),
}

# Four ores expected in Task 2A.
ORE_ORDER = [
    'azurite_ore_1',
    'azurite_ore_2',
    'malachite_ore_1',
    'vanadinite_ore_1',
]

# The tool first approaches above the ore, then descends for capture.
APPROACH_HEIGHT = 0.08
# The detected TF is slightly above/below the visual centre depending on the
# depth patch. Move 20 mm lower than the previous +15 mm pick target.
PICK_OFFSET_Z = -0.005
LIFT_HEIGHT = 0.16

cap_linear_mps = 0.15
cap_angular_rps = 0.35
cap_joint_rps = 0.35

unfold_target = {
    'shoulder_pan_joint': 0.0,
    'shoulder_lift_joint': -1.20,
    'elbow_joint': -1.60,
    'wrist_1_joint': -1.90,
    'wrist_2_joint': 1.57,
    'wrist_3_joint': 1.57,
}

position_tolerance = 0.005
unfold_kp = 0.6
control_rate = 0.05

# Grasp verification. Keep this as a constant so it is easy to tune
# against the force values reported by the Task 2A simulation.
FORCE_THRESHOLD_N = 0.20
FORCE_WAIT_S = 1.5

# Time to allow the magnet service to complete and the ore to settle.
MAGNET_SETTLE_S = 0.5
DROP_SETTLE_S = 0.8

# Safe transport path while the ore is held.
# The arm must pass through waypoint #2 before approaching the container.
TRANSPORT_WAYPOINT = (-0.8000, -0.0005, 0.3967)
TRANSPORT_KP = 0.10
TRANSPORT_MAX_SPEED = 0.08
DROP_KP = 0.10
DROP_MAX_SPEED = 0.08
DROP_APPROACH_HEIGHT = 0.10


class arm_task2a(Node):

    def __init__(self):
        super().__init__(
            'task2a_manipulation',
            parameter_overrides=[
                rclpy.parameter.Parameter(
                    'use_sim_time',
                    rclpy.Parameter.Type.BOOL,
                    True
                )
            ]
        )

        # ------------------------- PUBLISHERS -------------------------

        self.twist_pub = self.create_publisher(
            TwistStamped, '/delta_twist_cmds', 10
        )
        self.joint_pub = self.create_publisher(
            JointJog, '/delta_joint_cmds', 10
        )

        # ------------------------- SUBSCRIBERS -------------------------

        self.tcp_sub = self.create_subscription(
            PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.jointstatecb, 50
        )
        self.status_sub = self.create_subscription(
            Int32, '/arm_status', self.armstatuscb, 10
        )
        self.force_sub = self.create_subscription(
            Float32,
            '/ur7e/end_effector/force_magnitude',
            self.forcecb,
            20
        )

        # ------------------------- SERVICES -------------------------

        self.switch_cli = self.create_client(
            SwitchController,
            f'{servo_ns}/switch_controller'
        )
        self.unlock_client = self.create_client(
            Trigger,
            '/ur_arm_controller/unlock_protective_stop'
        )
        self.magnet_client = self.create_client(
            SetBool,
            '/magnet'
        )

        # ------------------------- TF -------------------------

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        # ------------------------- FEEDBACK -------------------------

        self.tcp_pose = None
        self.joint_angles = None
        self.arm_status = None
        self.force_magnitude = None

        # ------------------------- STATE -------------------------

        self.motion_state = 'INIT'

        self.switch_future = None
        self.switch_target = None

        self.unlock_future = None
        self.recovery_required = False
        self.in_recovery = False
        self.recovery_state = 'NONE'

        self.unfolded = False

        self.ore_index = 0
        self.current_ore = None
        self.current_ore_position = None
        self.current_drop_position = None
        self.transport_waypoint = TRANSPORT_WAYPOINT
        self.transport_phase = None

        self.hold_start_time = None
        self.magnet_future = None
        self.magnet_target = None
        self.magnet_on_time = None

        self.force_baseline = None
        self.force_wait_start = None

        self.phase = None
        self.phase_start_time = None

        self.timer = self.create_timer(
            control_rate,
            self.process
        )

        self.get_logger().info(
            'Task 2A manipulation node started.'
        )

    # ------------------------- CALLBACKS -------------------------

    def tcpposecb(self, data):
        self.tcp_pose = data.pose

    def jointstatecb(self, data):
        try:
            self.joint_angles = [
                data.position[data.name.index(joint)]
                for joint in joint_names
            ]
        except (ValueError, IndexError):
            self.joint_angles = None

    def armstatuscb(self, data):
        self.arm_status = data.data

        if 20 <= self.arm_status <= 29:
            if not self.in_recovery:
                self.recovery_required = True

    def forcecb(self, data):
        self.force_magnitude = float(data.data)

    # ------------------------- BASIC HELPERS -------------------------

    def now_seconds(self):
        return self.get_clock().now().nanoseconds / 1e9

    def stop_twist(self):
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        self.twist_pub.publish(msg)

    def set_phase(self, phase):
        self.phase = phase
        self.phase_start_time = self.now_seconds()
        self.get_logger().info(
            f'Ore={self.current_ore}, phase={phase}'
        )

    def phase_elapsed(self):
        if self.phase_start_time is None:
            return 0.0
        return self.now_seconds() - self.phase_start_time

    def calculate_position_velocity(self, target):
        current_x = self.tcp_pose.position.x
        current_y = self.tcp_pose.position.y
        current_z = self.tcp_pose.position.z

        error_x = target[0] - current_x
        error_y = target[1] - current_y
        error_z = target[2] - current_z

        kp = 0.20

        vx = kp * error_x
        vy = kp * error_y
        vz = kp * error_z

        magnitude = math.sqrt(
            vx * vx + vy * vy + vz * vz
        )

        if magnitude > cap_linear_mps:
            scale = (cap_linear_mps * 0.9) / magnitude
            vx *= scale
            vy *= scale
            vz *= scale

        distance = math.sqrt(
            error_x * error_x +
            error_y * error_y +
            error_z * error_z
        )

        return vx, vy, vz, distance

    def publish_twist_to(self, target):
        vx, vy, vz, distance = self.calculate_position_velocity(target)

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz

        self.twist_pub.publish(msg)
        return distance

    def publish_slow_twist_to(self, target, kp=0.10, max_speed=0.08):
        current_x = self.tcp_pose.position.x
        current_y = self.tcp_pose.position.y
        current_z = self.tcp_pose.position.z

        error_x = target[0] - current_x
        error_y = target[1] - current_y
        error_z = target[2] - current_z

        vx = kp * error_x
        vy = kp * error_y
        vz = kp * error_z

        magnitude = math.sqrt(vx * vx + vy * vy + vz * vz)
        if magnitude > max_speed:
            scale = max_speed / magnitude
            vx *= scale
            vy *= scale
            vz *= scale

        distance = math.sqrt(
            error_x * error_x + error_y * error_y + error_z * error_z
        )

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz
        self.twist_pub.publish(msg)
        return distance

    # ------------------------- CONTROLLER SWITCHING -------------------------

    def switch_controller(self, controller):
        req = SwitchController.Request()
        req.activate_controllers = [controller]

        other_controller = (
            joint_controller
            if controller == twist_controller
            else twist_controller
        )

        req.deactivate_controllers = [other_controller]
        req.strictness = SwitchController.Request.STRICT

        if not self.switch_cli.wait_for_service(timeout_sec=20.0):
            self.get_logger().error(
                'Controller switch service not available.'
            )
            return False

        self.switch_future = self.switch_cli.call_async(req)
        self.switch_target = controller
        return True

    # ------------------------- UNFOLD / RECOVERY -------------------------

    def unfold_arm(self):
        velocities = []
        max_error = 0.0

        for joint in joint_names:
            idx = joint_names.index(joint)
            current = self.joint_angles[idx]
            target = unfold_target[joint]

            error = target - current
            max_error = max(max_error, abs(error))

            velocity = unfold_kp * error
            velocity = max(
                min(velocity, cap_joint_rps * 0.9),
                -cap_joint_rps * 0.9
            )
            velocities.append(velocity)

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names
        msg.velocities = velocities
        self.joint_pub.publish(msg)

        if max_error < 0.03:
            msg.velocities = [0.0] * len(joint_names)
            self.joint_pub.publish(msg)
            return True

        return False

    def recover_arm(self):
        idx = joint_names.index('shoulder_lift_joint')
        current = self.joint_angles[idx]

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names

        velocities = [0.0] * len(joint_names)
        velocities[idx] = 0.05
        msg.velocities = velocities

        self.joint_pub.publish(msg)

        return current > -2.80

    # ------------------------- TF / ORE -------------------------

    def lookup_ore(self, ore_id):
        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame,
                ore_id,
                rclpy.time.Time()
            )

            t = transform.transform.translation
            return (float(t.x), float(t.y), float(t.z))

        except Exception:
            return None

    def select_next_ore(self):
        while self.ore_index < len(ORE_ORDER):
            ore_id = ORE_ORDER[self.ore_index]
            position = self.lookup_ore(ore_id)

            if position is None:
                self.get_logger().info(
                    f'Waiting for TF: {ore_id}'
                )
                return False

            if ore_id.startswith('azurite_ore'):
                ore_type = 'azurite_ore'
            elif ore_id.startswith('malachite_ore'):
                ore_type = 'malachite_ore'
            elif ore_id.startswith('vanadinite_ore'):
                ore_type = 'vanadinite_ore'
            else:
                ore_type = None

            if ore_type not in DROP_POINTS:
                self.get_logger().error(
                    f'No drop point for {ore_id}'
                )
                self.motion_state = 'ERROR'
                return False

            self.current_ore = ore_id
            self.current_ore_position = position
            self.current_drop_position = DROP_POINTS[ore_type]
            self.transport_waypoint = TRANSPORT_WAYPOINT
            self.transport_phase = None

            self.get_logger().info(
                f'Picked target {ore_id}: '
                f'x={position[0]:.3f}, '
                f'y={position[1]:.3f}, '
                f'z={position[2]:.3f}'
            )

            self.set_phase('APPROACH')
            self.motion_state = 'PICK'
            return True

        self.stop_twist()
        self.get_logger().info(
            'Task 2A complete: all ores processed.'
        )
        self.motion_state = 'DONE'
        return False

    def ore_approach_target(self):
        x, y, z = self.current_ore_position
        return (x, y, z + APPROACH_HEIGHT)

    def ore_pick_target(self):
        x, y, z = self.current_ore_position
        return (x, y, z + PICK_OFFSET_Z)

    def ore_lift_target(self):
        x, y, z = self.current_ore_position
        return (x, y, z + LIFT_HEIGHT)

    # ------------------------- MAGNET -------------------------

    def request_magnet(self, enabled):
        if self.magnet_future is not None:
            return

        if not self.magnet_client.wait_for_service(timeout_sec=2.0):
            self.get_logger().error(
                'Magnet service /magnet is not available.'
            )
            self.motion_state = 'ERROR'
            return

        request = SetBool.Request()
        request.data = enabled

        self.magnet_target = enabled
        self.magnet_future = self.magnet_client.call_async(
            request
        )

    def process_magnet_future(self):
        if self.magnet_future is None:
            return False

        if not self.magnet_future.done():
            return True

        try:
            response = self.magnet_future.result()
        except Exception as error:
            self.get_logger().error(
                f'Magnet service failed: {error}'
            )
            self.motion_state = 'ERROR'
            self.magnet_future = None
            return True

        self.magnet_future = None

        if response is None or not response.success:
            message = (
                'no response'
                if response is None
                else response.message
            )
            self.get_logger().error(
                f'Magnet command failed: {message}'
            )
            self.motion_state = 'ERROR'
            return True

        self.get_logger().info(
            f'Magnet {"ON" if self.magnet_target else "OFF"}: '
            f'{response.message}'
        )

        if self.magnet_target:
            self.magnet_on_time = self.now_seconds()
            self.force_wait_start = self.now_seconds()
            self.set_phase('VERIFY_PICK')
        else:
            self.set_phase('DROP_SETTLE')

        return True

    # ------------------------- MAIN TASK STATE MACHINE -------------------------

    def process(self):
        # Never continue motion blindly during a protective stop.
        if self.recovery_required:
            self.stop_twist()

            if self.unlock_future is None:
                if self.unlock_client.service_is_ready():
                    self.unlock_future = (
                        self.unlock_client.call_async(
                            Trigger.Request()
                        )
                    )
                    self.get_logger().warn(
                        'Protective stop detected; unlock requested.'
                    )
                return

            if not self.unlock_future.done():
                return

            try:
                response = self.unlock_future.result()
            except Exception:
                response = None

            self.unlock_future = None

            if response is not None and response.success:
                self.recovery_required = False
                self.in_recovery = True
                self.recovery_state = 'SWITCH_TO_JOINT'
                self.switch_controller(joint_controller)
                self.motion_state = 'WAIT_SWITCH'
                self.get_logger().warn(
                    'Protective stop unlocked; starting recovery.'
                )
            else:
                self.get_logger().error(
                    'Protective stop unlock failed.'
                )
            return

        # A controller switch is asynchronous.
        if self.switch_future is not None:
            if not self.switch_future.done():
                return

            response = self.switch_future.result()
            target = self.switch_target

            self.switch_future = None
            self.switch_target = None

            if response is None or not response.ok:
                message = (
                    'no response'
                    if response is None
                    else response.message
                )
                self.get_logger().error(
                    f'Controller switch failed: {message}'
                )
                self.motion_state = 'ERROR'
                return

            self.get_logger().info(
                f'Switched to {target}'
            )

            if target == joint_controller:
                if self.recovery_state == 'SWITCH_TO_JOINT':
                    self.motion_state = 'RECOVERY'
                else:
                    self.motion_state = 'UNFOLDING'
            else:
                if self.in_recovery:
                    self.in_recovery = False
                    self.recovery_state = 'NONE'
                    self.motion_state = 'READY'
                elif self.motion_state == 'UNFOLDING':
                    self.motion_state = 'READY'
                else:
                    self.motion_state = 'READY'

            return

        # Magnet service result is also asynchronous.
        if self.magnet_future is not None:
            self.process_magnet_future()
            return

        # Wait for all mandatory feedback.
        if self.tcp_pose is None:
            return

        if self.joint_angles is None:
            return

        if self.arm_status is None:
            return

        # ------------------------- INIT -------------------------

        if self.motion_state == 'INIT':
            self.get_logger().info(
                'Starting Task 2A: switching to joint controller.'
            )
            if self.switch_controller(joint_controller):
                self.motion_state = 'WAIT_SWITCH'
            return

        # ------------------------- UNFOLD -------------------------

        if self.motion_state == 'UNFOLDING':
            if self.unfold_arm():
                self.get_logger().info(
                    'Arm unfolded; switching to twist controller.'
                )
                self.switch_controller(twist_controller)
            return

        # ------------------------- RECOVERY -------------------------

        if self.motion_state == 'RECOVERY':
            if self.recover_arm():
                if self.arm_status == 0:
                    self.in_recovery = False
                    self.recovery_state = 'NONE'
                    self.switch_controller(twist_controller)
            return

        # ------------------------- READY -------------------------

        if self.motion_state == 'READY':
            if self.ore_index >= len(ORE_ORDER):
                self.motion_state = 'DONE'
                return

            self.motion_state = 'FIND_ORE'
            return

        # ------------------------- FIND ORE -------------------------

        if self.motion_state == 'FIND_ORE':
            self.select_next_ore()
            return

        # ------------------------- PICK -------------------------

        if self.motion_state == 'PICK':
            if self.phase == 'APPROACH':
                distance = self.publish_twist_to(
                    self.ore_approach_target()
                )

                if distance < position_tolerance:
                    self.stop_twist()
                    self.set_phase('DESCEND')

                return

            if self.phase == 'DESCEND':
                distance = self.publish_twist_to(
                    self.ore_pick_target()
                )

                if distance < position_tolerance:
                    self.stop_twist()

                    self.force_baseline = self.force_magnitude
                    self.request_magnet(True)

                return

            if self.phase == 'VERIFY_PICK':
                self.stop_twist()

                force = self.force_magnitude
                baseline = (
                    0.0
                    if self.force_baseline is None
                    else self.force_baseline
                )

                if force is not None and (
                    force >= FORCE_THRESHOLD_N or
                    force >= baseline + FORCE_THRESHOLD_N
                ):
                    self.get_logger().info(
                        f'Pick confirmed for {self.current_ore}; '
                        f'force={force:.3f} N'
                    )
                    self.set_phase('LIFT')
                    return

                if self.phase_elapsed() > FORCE_WAIT_S:
                    self.get_logger().error(
                        f'Pick verification failed for '
                        f'{self.current_ore}; '
                        f'force={force}'
                    )
                    self.request_magnet(False)
                    self.motion_state = 'ERROR'

                return

            if self.phase == 'LIFT':
                distance = self.publish_twist_to(
                    self.ore_lift_target()
                )

                if distance < position_tolerance:
                    self.stop_twist()
                    self.transport_phase = 'WAYPOINT_2'
                    self.set_phase('TRANSPORT_WAYPOINT')

                return

        # ------------------------- SAFE TRANSPORT PATH -------------------------

        if self.motion_state == 'PICK' and self.phase == 'TRANSPORT_WAYPOINT':
            # Keep the ore attached and route through the requested waypoint #2.
            distance = self.publish_slow_twist_to(
                self.transport_waypoint,
                kp=TRANSPORT_KP,
                max_speed=TRANSPORT_MAX_SPEED
            )

            if distance < position_tolerance:
                self.stop_twist()
                self.get_logger().info(
                    f'Reached transport waypoint #2: {self.transport_waypoint}'
                )
                self.transport_phase = 'ABOVE_DROP'
                self.set_phase('DROP_APPROACH')

            return

        # Move above the container first, then descend vertically. This avoids
        # taking a long diagonal directly into the container area.
        if self.motion_state == 'PICK' and self.phase == 'DROP_APPROACH':
            dx, dy, dz = self.current_drop_position
            target = (dx, dy, dz + DROP_APPROACH_HEIGHT)

            distance = self.publish_slow_twist_to(
                target,
                kp=DROP_KP,
                max_speed=DROP_MAX_SPEED
            )

            if distance < position_tolerance:
                self.stop_twist()
                self.set_phase('DROP_DESCEND')

            return

        if self.motion_state == 'PICK' and self.phase == 'DROP_DESCEND':
            distance = self.publish_slow_twist_to(
                self.current_drop_position,
                kp=DROP_KP,
                max_speed=DROP_MAX_SPEED
            )

            if distance < position_tolerance:
                self.stop_twist()
                self.request_magnet(False)

            return

        # ------------------------- DROP SETTLE -------------------------

        if self.motion_state == 'PICK' and self.phase == 'DROP_SETTLE':
            self.stop_twist()

            if self.phase_elapsed() >= DROP_SETTLE_S:
                self.get_logger().info(
                    f'Dropped {self.current_ore} into '
                    f'{self.current_ore.rsplit("_", 2)[0]} container.'
                )

                self.ore_index += 1
                self.current_ore = None
                self.current_ore_position = None
                self.current_drop_position = None
                self.transport_waypoint = TRANSPORT_WAYPOINT
                self.transport_phase = None

                self.motion_state = 'FIND_ORE'

            return

        # ------------------------- DONE -------------------------

        if self.motion_state == 'DONE':
            self.stop_twist()
            return

        # ------------------------- ERROR -------------------------

        if self.motion_state == 'ERROR':
            self.stop_twist()
            return


def main(args=None):
    rclpy.init(args=args)

    node = arm_task2a()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
