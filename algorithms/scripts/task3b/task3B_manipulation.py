#!/usr/bin/env python3

'''
*****************************************************************************************
*            ===============================================
*                       StrataCobot (SC) Theme (eYRC 2026-27)
*            ===============================================
*
*  This script implements Task 3B ore manipulation for StrataCobot (SC).
*  It sorts the ores into their containers and moves the ore package into the crusher,
*  with closed-loop force compliance during VERIFY for 1st-attempt picks.
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*****************************************************************************************
'''

# Team ID:          [ 1449 ]
# Author List:      [ Atharv Mudse, Abhay Potdar, Riddhi Wagh, Yash Waje ]
# Filename:         task3B_manipulation.py
# Functions:        quat_to_matrix, __init__, callbacks, motion helpers,
#                   controller/recovery helpers, ore selection, run_phase, main
# Global variables: controller names, joint names, drop points and task constants
# Nodes:             Publishing Topics  - [ /delta_twist_cmds, /delta_joint_cmds, /ore_rescan,
#                                           /handover/package_taken ]
#                    Subscribing Topics - [ /tcp_pose_raw, /joint_states, /arm_status,
#                                           /ore_rescan_done, /ur7e/end_effector/force_magnitude,
#                                           /handover/ebot_ready ]

import math
import re
import sys

import rclpy
import tf2_ros

from control_msgs.msg import JointJog
from controller_manager_msgs.srv import SwitchController
from geometry_msgs.msg import Pose, PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32, Int32, String
from std_srvs.srv import SetBool, Trigger


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

# Container drop positions in base_link.
DROP_POINTS = {
    'azurite_ore': (-0.0700, 0.5750, 0.1750),
    'malachite_ore': (-0.4000, 0.5750, 0.1750),
    'vanadinite_ore': (-0.6500, 0.5750, 0.1750),
}

# Ore package (Task 3B).
PACKAGE_FRAME = 'ore_package'
PACKAGE_FORCE_MIN_N = 1.80
PACKAGE_FORCE_MAX_N = 4.45
PACKAGE_STABLE_RANGE_N = 1.30
PACKAGE_CLEAR_LIFT_M = 0.13

CRUSHER_DROP = (-0.7250, 0.0050, 0.3100)

ORE_RE = re.compile(r'^(azurite|malachite|vanadinite)_ore_\d+$')

EXPECTED_ORES = 4
WAIT_ORES_TIMEOUT_S = 20.0
IDLE_WAIT_S = 15.0

# Heights above the ore TF.
# Restored PICK_OFFSET_Z = -0.001 so tool stops at z ~ -0.061 to -0.063 m without crushing the ore.
HOVER_HEIGHT = 0.09
APPROACH_HEIGHT = 0.038
PICK_OFFSET_Z = -0.001
PKG_PICK_OFFSET_Z = -0.002
PICK_Z_TRIES = (0.0, -0.002, 0.002)
MAX_PICK_ATTEMPTS = 10
MAX_RETRY_ROUNDS = 2
SEARCH_OFFSETS = (
    (0.0, 0.0), (0.005, 0.0), (-0.005, 0.0), (0.0, 0.005), (0.0, -0.005),
    (0.010, 0.0), (-0.010, 0.0), (0.0, 0.010), (0.0, -0.010),
    (0.005, 0.005), (-0.005, -0.005),
)
# Must be larger than BACKOFF tolerance (0.0025 m) so backoff actually lifts the tool!
CONTACT_BACKOFF_M = 0.007
RESCAN_TIMEOUT_S = 4.5
REDETECT_TIMEOUT_S = 1.8
LIFT_HEIGHT = 0.15

PROTECTIVE_STOP_CODES = (20, 21, 22, 23)
MAX_STOPS_PER_ORE = 2

cap_linear_mps = 0.15
cap_angular_rps = 0.35
cap_joint_rps = 0.35

unfold_target = {
    'shoulder_pan_joint': 0.0,
    'shoulder_lift_joint': -1.20,
    'elbow_joint': -1.60,
    'wrist_1_joint': -1.9124,
    'wrist_2_joint': 1.5708,
    'wrist_3_joint': 1.5708,
}

unfold_kp = 4.2
control_rate = 0.03

position_tolerance = 0.0065
TOL_FINE = 0.015
TOL_COARSE = 0.030

TRAVEL_KP, TRAVEL_SPEED = 4.5, 0.148
APPROACH_KP, APPROACH_SPEED = 4.0, 0.146
DESCEND_KP, DESCEND_SPEED = 2.8, 0.105
LIFT_KP, LIFT_SPEED = 4.5, 0.148
TRANSPORT_KP, TRANSPORT_MAX_SPEED = 4.5, 0.148
DROP_KP, DROP_MAX_SPEED = 4.5, 0.148
DROP_DESCEND_SPEED = 0.115

PKG_LIFT_SPEED = 0.125
PKG_TRANSPORT_SPEED = 0.138
PKG_DESCEND_SPEED = 0.075
PKG_CONTACT_FORCE_N = 5.5

MOVE_TIMEOUT_S = 40.0

FORCE_HELD_MIN_N = 0.75
FORCE_HELD_MAX_N = 2.65
FORCE_STABLE_RANGE_N = 0.95
FORCE_STABLE_WINDOW_S = 0.20
CONTACT_FORCE_N = 3.8
FORCE_WAIT_S = 2.2
MAGNET_SETTLE_S = 0.14
DROP_SETTLE_S = 0.04

# Elevated travel height (0.26 m) clears all container dividers and crusher walls.
TRAVEL_Z = 0.26

R_CLEAR = 0.30
R_ARC = 0.55
ARC_STEP_DEG = 35.0
VIA_RADIUS = 0.42
TOL_ARC = 0.055

PAN_SOFT_LIMIT = 5.6
LIMIT_MARGIN = 0.20
JOINT_LIMITS = {
    'shoulder_pan_joint': 6.2832,
    'shoulder_lift_joint': 6.2832,
    'elbow_joint': 3.1416,
    'wrist_1_joint': 6.2832,
    'wrist_2_joint': 6.2832,
    'wrist_3_joint': 6.2832,
}
DROP_APPROACH_HEIGHT = 0.025

DROP_ORES_FROM_HEIGHT = True
DROP_PACKAGE_FROM_HEIGHT = True

ANG_KP = 2.5
ANG_MAX_RPS = 0.33
LEVEL_TOL_RAD = 0.035
LEVEL_TIMEOUT_S = 8.0

TCP_EMA_ALPHA = 0.70


def quat_to_matrix(q):
    """Converts a quaternion into a 3x3 rotation matrix."""
    x, y, z, w = q.x, q.y, q.z, q.w
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    s = 2.0 / n
    return [
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ]


class arm_task2a(Node):

    """Controls the arm to pick ores and place them in type-specific containers."""

    def __init__(self):
        """Initialises ROS 2 publishers, subscribers, services, state and timer."""
        super().__init__(
            'task3b_manipulation',
            parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, True)
            ]
        )

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

        self.tcp_sub = self.create_subscription(
            PoseStamped, '/tcp_pose_raw', self.tcpposecb, 20
        )
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self.jointstatecb, 50
        )
        self.status_sub = self.create_subscription(
            Int32, '/arm_status', self.armstatuscb, 10
        )
        self.rescan_pub = self.create_publisher(String, '/ore_rescan', 10)
        self.rescan_done_sub = self.create_subscription(
            String, '/ore_rescan_done', self.rescandonecb, 10
        )
        self.force_sub = self.create_subscription(
            Float32, '/ur7e/end_effector/force_magnitude', self.forcecb, 20
        )

        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.taken_pub = self.create_publisher(Bool, '/handover/package_taken', latched)
        self.ready_sub = self.create_subscription(
            Bool, '/handover/ebot_ready', self.ebotreadycb, latched
        )

        self.switch_cli = self.create_client(
            SwitchController, f'{servo_ns}/switch_controller'
        )
        self.unlock_client = self.create_client(
            Trigger, '/ur_arm_controller/unlock_protective_stop'
        )
        self.magnet_client = self.create_client(SetBool, '/magnet')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.tcp_pose = None
        self.joint_angles = None
        self.arm_status = None
        self.force_magnitude = None
        self.contact_hits = 0

        self.motion_state = 'INIT'
        self.phase = None
        self.phase_start_time = None
        self.start_time = None

        self.active_controller = None
        self.switch_future = None
        self.switch_target = None
        self.switch_retries = 0

        self.unlock_future = None
        self.recovery_required = False
        self.in_recovery = False
        self.unfolded = False
        self.recover_target = None

        self.magnet_future = None
        self.magnet_request = None
        self.magnet_target = None
        self.magnet_reason = None

        self.done_ores = set()
        self.package_ready = False
        self.package_warned = False
        self.package_taken_sent = False
        self.redetect_names = []
        self.redetect_acks = set()
        self.redetect_resent = False
        self.skipped_ores = set()
        self.stop_counts = {}
        self.idle_start = None

        self.current_ore = None
        self.current_ore_position = None
        self.current_drop_position = None
        self.pick_attempt = 0
        self.force_baseline = None
        self.force_hist = []
        self.pick_attempts = {}
        self.tool_axis = None
        self.ang_sign = 1.0
        self.level_tilt0 = None
        self.level_checked = False
        self.retry_rounds = 0
        self.retry_target = None
        self.backoff_target = None
        self.rescan_done = False
        self.rescan_last = -100.0
        self.path = None
        self.path_idx = 0

        self.timer = self.create_timer(control_rate, self.process)

        self.get_logger().info('Task 3B manipulation node started.')

    def tcpposecb(self, data):
        """Stores the EMA-smoothed tool-centre-point pose received from the arm."""
        if self.tcp_pose is None:
            self.tcp_pose = data.pose
            return

        a = TCP_EMA_ALPHA
        filtered = Pose()
        filtered.position.x = a * data.pose.position.x + (1.0 - a) * self.tcp_pose.position.x
        filtered.position.y = a * data.pose.position.y + (1.0 - a) * self.tcp_pose.position.y
        filtered.position.z = a * data.pose.position.z + (1.0 - a) * self.tcp_pose.position.z
        filtered.orientation = data.pose.orientation
        self.tcp_pose = filtered

    def jointstatecb(self, data):
        """Stores joint positions in the configured joint order."""
        try:
            self.joint_angles = [
                data.position[data.name.index(joint)] for joint in joint_names
            ]
        except (ValueError, IndexError):
            self.joint_angles = None

    def armstatuscb(self, data):
        """Stores arm status and flags protective-stop recovery when needed."""
        self.arm_status = data.data

        if self.arm_status in PROTECTIVE_STOP_CODES and not self.in_recovery:
            self.recovery_required = True

    def ebotreadycb(self, data):
        '''The eBot reports it stands beside the arm with the package on its deck.'''
        if data.data:
            self.package_ready = True

    def rescandonecb(self, data):
        """Records completion of the perception node re-scan request."""
        if data.data == self.current_ore:
            self.rescan_done = True
        self.redetect_acks.add(data.data)

    def forcecb(self, data):
        """Stores force feedback and maintains a short history for verification."""
        self.force_magnitude = float(data.data)
        now = self.now_seconds()
        self.force_hist.append((now, self.force_magnitude))
        self.force_hist = [(t, f) for t, f in self.force_hist if now - t <= 3.0]

        limit_force = PKG_CONTACT_FORCE_N if self.current_ore == PACKAGE_FRAME else CONTACT_FORCE_N
        if self.force_magnitude > limit_force:
            self.contact_hits += 1
        else:
            self.contact_hits = 0

    def now_seconds(self):
        """Returns the current ROS clock time in seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    def stop_twist(self):
        """Publishes a zero Cartesian velocity command to stop arm motion."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        self.twist_pub.publish(msg)

    def publish_vertical_nudge(self, vz):
        """Publishes a gentle vertical velocity while keeping the tool levelled."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        msg.twist.linear.z = vz
        wx, wy, wz = self.orientation_rate()
        msg.twist.angular.x = wx
        msg.twist.angular.y = wy
        msg.twist.angular.z = wz
        self.twist_pub.publish(msg)

    def set_phase(self, phase):
        """Changes the current task phase and resets its elapsed-time reference."""
        now = self.now_seconds()
        previous = getattr(self, 'phase', None)
        started = getattr(self, 'phase_start_time', None)
        took = (now - started) if started is not None else 0.0

        moved = 0.0
        start_xyz = getattr(self, 'phase_start_xyz', None)
        if start_xyz is not None and self.tcp_pose is not None:
            p = self.tcp_pose.position
            moved = math.dist(start_xyz, (p.x, p.y, p.z))

        self.phase = phase
        self.phase_start_time = now
        self.contact_hits = 0
        self.phase_start_xyz = (
            (self.tcp_pose.position.x, self.tcp_pose.position.y, self.tcp_pose.position.z)
            if self.tcp_pose is not None else None
        )

        tool_txt = 'n/a'
        if self.tcp_pose is not None:
            tp = self.tcp_pose.position
            tool_txt = f'({tp.x:.4f}, {tp.y:.4f}, {tp.z:.4f})'

        self.get_logger().info(
            f'Ore={self.current_ore}, phase={phase} '
            f'(previous {previous}: {took:.1f} s, tool moved {moved:.2f} m) '
            f'tool_xyz(base_link)={tool_txt}'
        )

        if phase == 'MAGNET_ON':
            self.get_logger().info(
                f'[EVENT] PICK  {self.current_ore} tool_xyz={tool_txt} t={now:.1f}s')
        elif phase == 'DROP_RELEASE':
            self.get_logger().info(
                f'[EVENT] DROP  {self.current_ore} tool_xyz={tool_txt} t={now:.1f}s')
        elif phase == 'LIFT':
            self.get_logger().info(
                f'[EVENT] LIFT  {self.current_ore} tool_xyz={tool_txt} t={now:.1f}s')

    def phase_elapsed(self):
        """Returns the elapsed time since the current phase began."""
        if self.phase_start_time is None:
            return 0.0
        return self.now_seconds() - self.phase_start_time

    def publish_slow_twist_to(self, target, kp, max_speed, hold_speed=False):
        """Publishes a bounded Cartesian velocity toward a target pose."""
        error_x = target[0] - self.tcp_pose.position.x
        error_y = target[1] - self.tcp_pose.position.y
        error_z = target[2] - self.tcp_pose.position.z

        vx = kp * error_x
        vy = kp * error_y
        vz = kp * error_z

        max_speed = min(max_speed, cap_linear_mps * 0.98)
        magnitude = math.sqrt(vx * vx + vy * vy + vz * vz)
        dist = math.sqrt(error_x ** 2 + error_y ** 2 + error_z ** 2)

        min_speed = 0.050 if dist > 0.008 else 0.020

        if magnitude > max_speed or (hold_speed and magnitude > 1e-9):
            scale = max_speed / magnitude
            vx *= scale
            vy *= scale
            vz *= scale
        elif 1e-9 < magnitude < min_speed:
            scale = min_speed / magnitude
            vx *= scale
            vy *= scale
            vz *= scale

        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
        msg.twist.linear.x = vx
        msg.twist.linear.y = vy
        msg.twist.linear.z = vz

        wx, wy, wz = self.orientation_rate()
        msg.twist.angular.x = wx
        msg.twist.angular.y = wy
        msg.twist.angular.z = wz

        self.twist_pub.publish(msg)

        return dist

    def choose_tool_axis(self):
        """Selects the tool axis that points most nearly downward."""
        R = quat_to_matrix(self.tcp_pose.orientation)
        best = None
        for idx in range(3):
            for sign in (1.0, -1.0):
                down = -sign * R[2][idx]
                if best is None or down > best[0]:
                    best = (down, idx, sign)
        self.tool_axis = (best[1], best[2])

    def tool_vector(self):
        """Returns the selected tool axis expressed in the base frame."""
        R = quat_to_matrix(self.tcp_pose.orientation)
        idx, sign = self.tool_axis
        return (sign * R[0][idx], sign * R[1][idx], sign * R[2][idx])

    def tool_tilt(self):
        """Angle (rad) between the tool axis and straight down."""
        v = self.tool_vector()
        return math.acos(max(-1.0, min(1.0, -v[2])))

    def orientation_rate(self):
        """Angular velocity (base frame) that turns the tool axis toward straight down."""
        if self.tool_axis is None:
            return (0.0, 0.0, 0.0)

        vx, vy, _ = self.tool_vector()
        wx = self.ang_sign * ANG_KP * (-vy)
        wy = self.ang_sign * ANG_KP * vx

        mag = math.hypot(wx, wy)
        if mag > ANG_MAX_RPS:
            wx *= ANG_MAX_RPS / mag
            wy *= ANG_MAX_RPS / mag

        return (wx, wy, 0.0)

    def go(self, target, kp, max_speed, tol):
        '''Drive the tool toward target. True when reached (or timed out).'''

        distance = self.publish_slow_twist_to(target, kp, max_speed)

        if distance < tol:
            self.stop_twist()
            return True

        if self.phase_elapsed() > MOVE_TIMEOUT_S:
            self.get_logger().warn(
                f'Move timeout in phase {self.phase}; continuing '
                f'(remaining {distance:.3f} m).'
            )
            self.stop_twist()
            return True

        return False

    def segment_clearance(self, a, b):
        '''Distance from the base axis to the straight segment a-b.'''

        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        if length2 < 1e-9:
            return math.hypot(a[0], a[1])

        t = max(0.0, min(1.0, -(a[0] * dx + a[1] * dy) / length2))
        return math.hypot(a[0] + t * dx, a[1] + t * dy)

    def plan_path_xy(self, start, end):
        '''XY waypoints (end included) for a level move that keeps clear of the base axis
        and the ore crusher hopper at (-X, y ~ 0).'''

        sx, sy = start
        ex, ey = end

        if math.hypot(sx, sy) < R_CLEAR or math.hypot(ex, ey) < R_CLEAR:
            return [end]

        dx, dy = ex - sx, ey - sy
        length2 = dx * dx + dy * dy
        if length2 < 1e-9:
            return [end]

        if (sy < -0.20 and ey > 0.20) or (sy > 0.20 and ey < -0.20):
            t_zero = -sy / (ey - sy)
            x_cross = sx + t_zero * dx
            if not (-0.43 <= x_cross <= -0.32):
                return [(-0.38, 0.0), end]

        t = max(0.0, min(1.0, -(sx * dx + sy * dy) / length2))
        if math.hypot(sx + t * dx, sy + t * dy) >= R_CLEAR:
            return [end]

        theta_s = math.atan2(sy, sx)
        theta_e = math.atan2(ey, ex)
        inc = (theta_e - theta_s) % (2.0 * math.pi)
        dec = inc - 2.0 * math.pi

        delta = inc if math.cos(theta_s + inc / 2.0) < 0.0 else dec

        if self.joint_angles is not None:
            pan = self.joint_angles[0]
            other = dec if delta == inc else inc
            if abs(pan + delta) > PAN_SOFT_LIMIT and abs(pan + other) < abs(pan + delta):
                self.get_logger().warn(
                    f'Pan at {pan:.2f} rad: routing the other way, ending at {pan + other:.2f}.')
                delta = other

        if VIA_RADIUS is not None:
            mid = theta_s + delta / 2.0
            via = (VIA_RADIUS * math.cos(mid), VIA_RADIUS * math.sin(mid))
            if (self.segment_clearance(start, via) >= R_CLEAR
                    and self.segment_clearance(via, end) >= R_CLEAR):
                return [via, end]

        n = max(1, int(math.ceil(abs(delta) / math.radians(ARC_STEP_DEG))))
        points = []
        for k in range(1, n):
            angle = theta_s + delta * k / n
            points.append((R_ARC * math.cos(angle), R_ARC * math.sin(angle)))
        points.append(end)

        return points

    def start_path(self, end3, approach_end=None):
        """Plans level waypoints for a safe move to a target, optionally ending at approach_end."""
        points = self.plan_path_xy(
            (self.tcp_pose.position.x, self.tcp_pose.position.y),
            (end3[0], end3[1])
        )
        self.path = [(x, y, end3[2]) for x, y in points]
        if approach_end is not None:
            self.path.append(approach_end)
        self.path_idx = 0

        if len(points) > 1:
            self.get_logger().info(
                f'Routing around the base with {len(points) - 1} arc point(s).'
            )

    def follow_path(self, kp, speed, tol_final):
        '''Follow self.path. True when the last point is reached (or timed out).'''

        if self.path is None:
            return True

        target = self.path[self.path_idx]
        last = self.path_idx == len(self.path) - 1

        distance = self.publish_slow_twist_to(
            target, kp, speed, hold_speed=not last
        )

        if distance < (tol_final if last else TOL_ARC):
            if last:
                self.stop_twist()
                self.path = None
                return True
            self.path_idx += 1
            return False

        if self.phase_elapsed() > 2.0 * MOVE_TIMEOUT_S:
            self.get_logger().warn(
                f'Path timeout in phase {self.phase}; continuing.'
            )
            self.stop_twist()
            self.path = None
            return True

        return False

    def force_steady_in_band(self):
        '''True when the median of recent force readings sits in the held-ore band.'''

        now = self.now_seconds()
        samples = sorted(f for t, f in self.force_hist if now - t <= FORCE_STABLE_WINDOW_S)

        if len(samples) < 3:
            return False

        if self.current_ore == PACKAGE_FRAME:
            low, high = PACKAGE_FORCE_MIN_N, PACKAGE_FORCE_MAX_N
            max_range = PACKAGE_STABLE_RANGE_N
        else:
            low, high = FORCE_HELD_MIN_N, FORCE_HELD_MAX_N
            max_range = FORCE_STABLE_RANGE_N

        mid = samples[len(samples) // 2]
        core = samples[1:-1] if len(samples) >= 5 else samples

        return (
            low <= mid <= high
            and min(core) >= low * 0.85
            and max(core) <= high * 1.10
            and (max(core) - min(core)) <= max_range
        )

    def switch_controller(self, controller, strict=True):
        """Requests a switch between the arm joint and twist controllers."""
        if self.active_controller == controller:
            self.on_switch_done(controller)
            return True

        req = SwitchController.Request()
        req.activate_controllers = [controller]
        req.deactivate_controllers = [
            joint_controller if controller == twist_controller else twist_controller
        ]
        req.strictness = (
            SwitchController.Request.STRICT
            if strict else SwitchController.Request.BEST_EFFORT
        )

        if not self.switch_cli.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Controller switch service not available.')
            return False

        self.switch_future = self.switch_cli.call_async(req)
        self.switch_target = controller
        self.motion_state = 'WAIT_SWITCH'
        return True

    def on_switch_done(self, target):
        """Updates the controller state and chooses the next motion state."""
        self.active_controller = target
        self.switch_retries = 0
        self.get_logger().info(f'Switched to {target}')

        if target == joint_controller:
            self.motion_state = 'RECOVERY' if self.in_recovery else 'UNFOLDING'
            return

        if self.in_recovery:
            self.in_recovery = False

            if not self.unfolded:
                self.switch_controller(joint_controller)
                return

            self.motion_state = 'RUN'
            self.set_phase('LEVEL')
            return

        if not self.unfolded:
            self.unfolded = True
            self.motion_state = 'RUN'
            self.set_phase('LEVEL')
            return

        self.motion_state = 'RUN'

    def unfold_arm(self):
        """Moves the arm joints toward the configured unfolded posture."""
        velocities = []
        max_error = 0.0

        for idx, joint in enumerate(joint_names):
            error = unfold_target[joint] - self.joint_angles[idx]
            max_error = max(max_error, abs(error))

            velocity = max(
                min(unfold_kp * error, cap_joint_rps * 0.98),
                -cap_joint_rps * 0.98
            )
            velocities.append(velocity)

        msg = JointJog()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.joint_names = joint_names
        msg.velocities = velocities
        self.joint_pub.publish(msg)

        if max_error < 0.035:
            msg.velocities = [0.0] * len(joint_names)
            self.joint_pub.publish(msg)
            return True

        return False

    def recover_arm(self):
        """Returns the arm joints safely toward unfold_target to escape singularities and joint limits."""
        return self.unfold_arm()

    def handle_recovery(self):
        """Handles protective-stop unlocking and safe arm recovery."""
        self.stop_twist()

        if self.unlock_future is None:
            if self.unlock_client.service_is_ready():
                self.unlock_future = self.unlock_client.call_async(Trigger.Request())
                self.get_logger().warn('Protective stop detected; unlock requested.')
            return

        if not self.unlock_future.done():
            return

        try:
            response = self.unlock_future.result()
        except Exception:
            response = None

        self.unlock_future = None

        if response is None or not response.success:
            self.get_logger().error('Protective stop unlock failed; retrying.')
            return

        self.recovery_required = False
        self.in_recovery = True
        self.path = None
        self.get_logger().warn('Protective stop unlocked; starting recovery.')

        self.request_magnet(False, 'recovery')

        ore = self.current_ore
        if ore is not None and ore not in self.done_ores:
            if ore == PACKAGE_FRAME:
                msg = String()
                msg.data = PACKAGE_FRAME
                self.rescan_pub.publish(msg)
            self.stop_counts[ore] = self.stop_counts.get(ore, 0) + 1
            if self.stop_counts[ore] >= MAX_STOPS_PER_ORE:
                self.get_logger().warn(f'Skipping {ore} after repeated stops.')
                self.skipped_ores.add(ore)

        self.switch_controller(joint_controller)

    def lookup_ore(self, ore_id):
        """Looks up an ore transform and returns its position in base_link."""
        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame, ore_id, rclpy.time.Time()
            )
            t = transform.transform.translation
            pos = (float(t.x), float(t.y), float(t.z))
            if pos[2] > 0.40:
                return None
            return pos
        except Exception:
            return None

    def discover_ores(self):
        """Finds ore TF frame names published by the perception node."""
        try:
            text = self.tf_buffer.all_frames_as_string()
        except Exception:
            return []

        names = set(re.findall(r'Frame (\S+) exists with parent', text))
        return sorted(n for n in names if ORE_RE.match(n))

    def select_next_ore(self):
        """Selects the nearest remaining ore or handles task completion."""
        found = self.discover_ores()
        remaining = [
            o for o in found
            if o not in self.done_ores and o not in self.skipped_ores
        ]

        if self.package_ready and PACKAGE_FRAME not in self.done_ores \
                and PACKAGE_FRAME not in self.skipped_ores:
            if CRUSHER_DROP is None:
                if not self.package_warned:
                    self.package_warned = True
                    self.get_logger().error('CRUSHER_DROP is not set, so the package is not picked.')
            elif self.lookup_ore(PACKAGE_FRAME) is not None:
                remaining = [PACKAGE_FRAME] + remaining

        if not remaining:
            if self.skipped_ores and self.retry_rounds < MAX_RETRY_ROUNDS:
                self.retry_rounds += 1
                self.get_logger().warn(
                    f'Trying the skipped ores again: {sorted(self.skipped_ores)}'
                )
                for ore in self.skipped_ores:
                    self.pick_attempts[ore] = 0
                    self.stop_counts[ore] = 0
                self.skipped_ores.clear()
                return

            handled = len([o for o in self.done_ores if o != PACKAGE_FRAME]) \
                + len([o for o in self.skipped_ores if o != PACKAGE_FRAME])

            if self.idle_start is None:
                self.idle_start = self.now_seconds()

            idle = self.now_seconds() - self.idle_start

            package_pending = (
                CRUSHER_DROP is not None
                and PACKAGE_FRAME not in self.done_ores
                and PACKAGE_FRAME not in self.skipped_ores
            )
            if package_pending:
                self.stop_twist()
                self.get_logger().info(
                    'Waiting for the package to arrive...', throttle_duration_sec=3.0
                )
                return

            if handled >= EXPECTED_ORES or idle > IDLE_WAIT_S:
                self.stop_twist()
                self.get_logger().info(
                    f'Task 3B complete: dropped={sorted(self.done_ores)}, '
                    f'skipped={sorted(self.skipped_ores)}'
                )
                self.motion_state = 'DONE'
            else:
                self.get_logger().info(
                    'Waiting for remaining ore TFs...', throttle_duration_sec=2.0
                )
            return

        self.idle_start = None

        best = None
        best_dist = float('inf')

        for ore_id in remaining:
            position = self.lookup_ore(ore_id)
            if position is None:
                continue

            dist = math.hypot(
                position[0] - self.tcp_pose.position.x,
                position[1] - self.tcp_pose.position.y,
            )
            if ore_id == PACKAGE_FRAME:
                dist = -1.0
            if dist < best_dist:
                best = (ore_id, position)
                best_dist = dist

        if best is None:
            return

        ore_id, position = best
        ore_type = ore_id.rsplit('_', 1)[0]

        self.current_ore = ore_id
        self.current_ore_position = position
        self.current_drop_position = (
            CRUSHER_DROP if ore_id == PACKAGE_FRAME else DROP_POINTS[ore_type]
        )
        self.pick_attempt = self.pick_attempts.get(ore_id, 0)

        self.get_logger().info(
            f'Target {ore_id}: x={position[0]:.3f}, '
            f'y={position[1]:.3f}, z={position[2]:.3f}'
        )
        self.start_path(self.ore_hover_target(), approach_end=self.ore_approach_target())
        self.set_phase('HOVER')

    def package_waiting(self):
        '''True when the eBot is beside the arm and the package can be picked now.'''

        return (
            self.package_ready
            and CRUSHER_DROP is not None
            and PACKAGE_FRAME not in self.done_ores
            and PACKAGE_FRAME not in self.skipped_ores
            and self.lookup_ore(PACKAGE_FRAME) is not None
        )

    def start_redetect(self):
        '''Asks perception to detect every remaining ore again.'''

        self.redetect_names = [
            o for o in self.discover_ores()
            if o not in self.done_ores and o not in self.skipped_ores
        ]
        self.redetect_acks = set()
        self.redetect_resent = False

        if not self.redetect_names:
            self.set_phase('SELECT')
            return

        for name in self.redetect_names:
            msg = String()
            msg.data = name
            self.rescan_pub.publish(msg)

        self.get_logger().info(f'Detecting again after the drop: {self.redetect_names}')
        self.set_phase('REDETECT')

    def ore_hover_target(self):
        """Returns a safe hover target above the current ore."""
        x, y, z = self.current_ore_position
        return (x, y, max(TRAVEL_Z, z + HOVER_HEIGHT))

    def search_offset(self):
        """Returns the XY search offset for the current pick attempt."""
        return SEARCH_OFFSETS[self.pick_attempt % len(SEARCH_OFFSETS)]

    def ore_approach_target(self):
        """Returns the approach target above the current ore."""
        x, y, z = self.current_ore_position
        ox, oy = self.search_offset()
        return (x + ox, y + oy, z + APPROACH_HEIGHT)

    def ore_pick_target(self):
        """Returns the magnet contact target for the current pick attempt."""
        x, y, z = self.current_ore_position
        ox, oy = self.search_offset()
        extra = PICK_Z_TRIES[self.pick_attempt % len(PICK_Z_TRIES)]
        offset_z = PKG_PICK_OFFSET_Z if self.current_ore == PACKAGE_FRAME else PICK_OFFSET_Z
        return (x + ox, y + oy, z + offset_z + extra)

    def pick_failed(self):
        """Records a failed pick and schedules a retry or skip sequence."""
        ore = self.current_ore
        self.pick_attempts[ore] = self.pick_attempts.get(ore, 0) + 1
        self.pick_attempt = self.pick_attempts[ore]

        self.set_phase('RELEASE_WAIT')

        if self.pick_attempt >= MAX_PICK_ATTEMPTS:
            self.get_logger().warn(
                f'{ore}: {self.pick_attempt} failed picks; leaving it for a later round.'
            )
            self.skipped_ores.add(ore)
            self.request_magnet(False, 'skip')
        else:
            self.get_logger().warn(
                f'Pick not confirmed for {ore}; re-checking its position and '
                f'trying again (attempt {self.pick_attempt + 1}).'
            )
            self.request_magnet(False, 'retry')

    def ore_lift_target(self):
        """Returns the vertical lift target after a confirmed pick."""
        x, y, z = self.current_ore_position
        return (x, y, max(TRAVEL_Z, z + LIFT_HEIGHT))

    def drop_travel_target(self):
        """Returns the safe travel target above the destination container."""
        dx, dy, dz = self.current_drop_position
        return (dx, dy, max(TRAVEL_Z, dz + DROP_APPROACH_HEIGHT))

    def drop_approach_target(self):
        """Returns the approach target above the destination container."""
        dx, dy, dz = self.current_drop_position
        return (dx, dy, dz + DROP_APPROACH_HEIGHT)

    def publish_package_taken(self):
        '''Tells the eBot the package is off its deck, so it may drive home (sent once).'''
        if self.package_taken_sent:
            return

        self.package_taken_sent = True
        msg = Bool()
        msg.data = True
        self.taken_pub.publish(msg)
        self.get_logger().info('Package lifted clear of the deck; eBot may leave.')

    def request_magnet(self, enabled, reason):
        '''Queue a magnet command; it is sent as soon as the service is ready.'''
        self.magnet_request = (enabled, reason)

    def pump_magnet(self):
        """Sends a queued magnet request when the service is ready."""
        if self.magnet_future is not None or self.magnet_request is None:
            return

        if not self.magnet_client.service_is_ready():
            self.get_logger().warn(
                'Waiting for /magnet service...', throttle_duration_sec=2.0
            )
            return

        enabled, reason = self.magnet_request
        self.magnet_request = None

        request = SetBool.Request()
        request.data = enabled

        self.magnet_target = enabled
        self.magnet_reason = reason
        self.magnet_future = self.magnet_client.call_async(request)

    def check_magnet_future(self):
        """Processes the result of an asynchronous magnet service call."""
        if not self.magnet_future.done():
            return

        try:
            response = self.magnet_future.result()
            message = '' if response is None else response.message
            self.get_logger().info(
                f'Magnet {"ON" if self.magnet_target else "OFF"}: {message}'
            )
        except Exception as error:
            self.get_logger().error(f'Magnet service call failed: {error}')

        self.magnet_future = None
        reason = self.magnet_reason

        if reason == 'pick':
            self.set_phase('VERIFY')
        elif reason in ('retry', 'skip'):
            self.retry_target = (
                self.tcp_pose.position.x,
                self.tcp_pose.position.y,
                max(TRAVEL_Z, self.tcp_pose.position.z),
            )
            self.rescan_done = False
            self.rescan_last = -100.0
            self.set_phase('RETRY_LIFT' if reason == 'retry' else 'SKIP_LIFT')
        elif reason == 'drop':
            self.set_phase('DROP_SETTLE')

    def process(self):
        """Runs one control-loop iteration for the current manipulation state."""
        self.pump_magnet()

        if self.recovery_required:
            self.handle_recovery()
            return

        if self.switch_future is not None:
            if not self.switch_future.done():
                return

            response = self.switch_future.result()
            target = self.switch_target
            self.switch_future = None
            self.switch_target = None

            if response is None or not response.ok:
                self.switch_retries += 1
                self.get_logger().error(
                    f'Controller switch to {target} failed '
                    f'(attempt {self.switch_retries}).'
                )
                if self.switch_retries > 5:
                    self.motion_state = 'ERROR'
                else:
                    self.switch_controller(target, strict=False)
                return

            self.on_switch_done(target)
            return

        if self.magnet_future is not None:
            self.check_magnet_future()
            return

        if self.magnet_request is not None:
            self.stop_twist()
            return

        if self.tcp_pose is None or self.joint_angles is None or self.arm_status is None:
            return

        state = self.motion_state

        if state == 'INIT':
            self.start_time = self.now_seconds()
            self.get_logger().info('Unfolding arm immediately while perception detects ores.')
            if not self.switch_controller(joint_controller):
                self.motion_state = 'WAIT_ORES'
            return

        if state == 'WAIT_ORES':
            found = self.discover_ores()
            elapsed = self.now_seconds() - self.start_time

            if len(found) >= EXPECTED_ORES or (found and elapsed > WAIT_ORES_TIMEOUT_S):
                self.get_logger().info(
                    f'Ore transforms available: {found}. Unfolding arm.'
                )
                if not self.switch_controller(joint_controller):
                    self.motion_state = 'WAIT_ORES'
            else:
                self.get_logger().info(
                    f'Found {len(found)} ore frame(s)...', throttle_duration_sec=2.0
                )
            return

        if state == 'UNFOLDING':
            if self.unfold_arm():
                self.get_logger().info('Arm unfolded; switching to twist controller.')
                self.switch_controller(twist_controller)
            return

        if state == 'RECOVERY':
            if self.recover_arm() and self.arm_status == 0:
                self.switch_controller(twist_controller)
            return

        if state == 'RUN':
            self.run_phase()
            return

        if state in ('DONE', 'ERROR'):
            self.stop_twist()
            return

    def run_phase(self):
        """Executes the actions associated with the active per-ore phase."""
        p = self.phase

        if p == 'LEVEL':
            if self.tool_axis is None:
                self.choose_tool_axis()
                self.level_tilt0 = self.tool_tilt()
                self.level_checked = False
                self.get_logger().info(
                    f'Tool axis {self.tool_axis}; tilt from vertical = '
                    f'{math.degrees(self.level_tilt0):.1f} deg.'
                )

            tilt = self.tool_tilt()
            elapsed = self.phase_elapsed()

            if not self.level_checked and elapsed > 1.0:
                self.level_checked = True
                if tilt > self.level_tilt0 + 0.02:
                    self.ang_sign = -self.ang_sign
                    self.get_logger().warn('Tilt grew; reversing the rotation sign.')

            if tilt < LEVEL_TOL_RAD or elapsed > LEVEL_TIMEOUT_S:
                self.stop_twist()
                self.get_logger().info(
                    f'Tool levelled: tilt {math.degrees(tilt):.2f} deg.'
                )
                self.set_phase('SELECT')
                return

            msg = TwistStamped()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = base_frame
            wx, wy, wz = self.orientation_rate()
            msg.twist.angular.x = wx
            msg.twist.angular.y = wy
            msg.twist.angular.z = wz
            self.twist_pub.publish(msg)
            return

        if p == 'SELECT':
            self.select_next_ore()
            return

        if p == 'HOVER':
            if self.current_ore != PACKAGE_FRAME and self.package_waiting():
                self.path = None
                self.stop_twist()
                self.get_logger().info('eBot is beside the arm: taking the package first.')
                self.set_phase('SELECT')
                return

            if self.follow_path(TRAVEL_KP, TRAVEL_SPEED, TOL_FINE):
                self.set_phase('DESCEND')
            return

        if p == 'APPROACH':
            if self.go(self.ore_approach_target(), APPROACH_KP, APPROACH_SPEED, TOL_FINE):
                self.set_phase('DESCEND')
            return

        if p == 'DESCEND':
            force = self.force_magnitude
            limit_force = PKG_CONTACT_FORCE_N if self.current_ore == PACKAGE_FRAME else CONTACT_FORCE_N
            contact = (
                self.contact_hits >= 2
                and force is not None
                and force > limit_force
                and self.phase_elapsed() > 0.25
            )

            if contact:
                self.stop_twist()
                self.backoff_target = (
                    self.tcp_pose.position.x,
                    self.tcp_pose.position.y,
                    self.tcp_pose.position.z + CONTACT_BACKOFF_M,
                )
                self.get_logger().info(
                    f'Contact at {force:.1f} N; backing off before the magnet.'
                )
                self.set_phase('BACKOFF')
                return

            descend_spd = PKG_DESCEND_SPEED if self.current_ore == PACKAGE_FRAME else DESCEND_SPEED
            if self.go(
                self.ore_pick_target(), DESCEND_KP, descend_spd,
                position_tolerance
            ):
                self.set_phase('MAGNET_ON')
                self.request_magnet(True, 'pick')
            return

        if p == 'BACKOFF':
            if self.go(self.backoff_target, 2.2, 0.05, 0.0025):
                self.set_phase('MAGNET_ON')
                self.request_magnet(True, 'pick')
            return

        if p in ('MAGNET_ON', 'RELEASE_WAIT'):
            self.stop_twist()
            return

        if p == 'VERIFY':
            force = self.force_magnitude
            elapsed = self.phase_elapsed()

            self.get_logger().info(
                f'Verifying pick: force={force}',
                throttle_duration_sec=0.5
            )

            if elapsed >= MAGNET_SETTLE_S and self.force_steady_in_band():
                self.stop_twist()
                self.get_logger().info(
                    f'Pick confirmed for {self.current_ore}; force={force:.3f} N'
                )
                self.set_phase('LIFT')
                return

            # Closed-loop compliance while magnet is ON:
            # 1) If force > high_band, the tool is pressing down on the ore/conveyor:
            #    gently lift (+0.025 m/s) with the magnet ON so the ore hangs freely in-band!
            # 2) If force < low_band after 0.22 s, the tool stopped a few mm above the ore:
            #    gently creep down (-0.018 m/s) with the magnet ON until it snaps on!
            high_band = PACKAGE_FORCE_MAX_N if self.current_ore == PACKAGE_FRAME else FORCE_HELD_MAX_N
            low_band = PACKAGE_FORCE_MIN_N if self.current_ore == PACKAGE_FRAME else FORCE_HELD_MIN_N

            if force is not None and elapsed >= MAGNET_SETTLE_S:
                if force > high_band:
                    self.publish_vertical_nudge(0.028)
                elif force < low_band and elapsed < 0.85:
                    self.publish_vertical_nudge(-0.018)
                else:
                    self.stop_twist()
            else:
                self.stop_twist()

            if elapsed > FORCE_WAIT_S:
                self.stop_twist()
                self.pick_failed()
            return

        if p == 'RETRY_LIFT':
            if self.go(self.retry_target, APPROACH_KP, APPROACH_SPEED, TOL_FINE):
                self.set_phase('RESCAN')
            return

        if p == 'RESCAN':
            self.stop_twist()
            elapsed = self.phase_elapsed()

            if elapsed - self.rescan_last > 1.5 and not self.rescan_done:
                self.rescan_last = elapsed
                msg = String()
                msg.data = self.current_ore
                self.rescan_pub.publish(msg)

            if self.rescan_done or elapsed > RESCAN_TIMEOUT_S:
                position = self.lookup_ore(self.current_ore)
                if position is not None:
                    old = self.current_ore_position
                    self.current_ore_position = position
                    self.get_logger().info(
                        f'{self.current_ore} position now '
                        f'({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f}), '
                        f'shift {1000.0 * math.dist(old, position):.0f} mm.'
                    )
                self.start_path(self.ore_hover_target(), approach_end=self.ore_approach_target())
                self.set_phase('HOVER')
            return

        if p == 'SKIP_LIFT':
            if self.go(self.retry_target, APPROACH_KP, APPROACH_SPEED, TOL_FINE):
                self.set_phase('SELECT')
            return

        if p == 'LIFT':
            lift_spd = PKG_LIFT_SPEED if self.current_ore == PACKAGE_FRAME else LIFT_SPEED
            if (self.current_ore == PACKAGE_FRAME and not self.package_taken_sent
                    and self.tcp_pose.position.z >= self.current_ore_position[2] + PACKAGE_CLEAR_LIFT_M):
                self.publish_package_taken()

            if self.go(self.ore_lift_target(), LIFT_KP, lift_spd, TOL_COARSE):
                if self.current_ore == PACKAGE_FRAME:
                    self.publish_package_taken()
                self.start_path(self.drop_travel_target())
                self.set_phase('TRANSIT')
            return

        if p == 'TRANSIT':
            trans_spd = PKG_TRANSPORT_SPEED if self.current_ore == PACKAGE_FRAME else TRANSPORT_MAX_SPEED
            if self.follow_path(TRANSPORT_KP, trans_spd, TOL_COARSE):
                from_height = (
                    DROP_PACKAGE_FROM_HEIGHT if self.current_ore == PACKAGE_FRAME
                    else DROP_ORES_FROM_HEIGHT
                )
                if from_height:
                    self.set_phase('DROP_RELEASE')
                    self.request_magnet(False, 'drop')
                else:
                    self.set_phase('DROP_APPROACH')
            return

        if p == 'DROP_APPROACH':
            if self.go(self.drop_approach_target(), DROP_KP, DROP_MAX_SPEED, TOL_COARSE):
                from_height = (
                    DROP_PACKAGE_FROM_HEIGHT if self.current_ore == PACKAGE_FRAME
                    else DROP_ORES_FROM_HEIGHT
                )
                if from_height:
                    self.set_phase('DROP_RELEASE')
                    self.request_magnet(False, 'drop')
                else:
                    self.set_phase('DROP_DESCEND')
            return

        if p == 'DROP_DESCEND':
            if self.go(self.current_drop_position, DROP_KP, DROP_DESCEND_SPEED, position_tolerance):
                self.set_phase('DROP_RELEASE')
                self.request_magnet(False, 'drop')
            return

        if p == 'DROP_RELEASE':
            self.stop_twist()
            return

        if p == 'DROP_SETTLE':
            self.stop_twist()

            if self.phase_elapsed() >= DROP_SETTLE_S:
                self.done_ores.add(self.current_ore)
                self.get_logger().info(
                    f'Dropped {self.current_ore} into '
                    f'{self.current_ore.rsplit("_", 1)[0]} container.'
                )
                self.set_phase('DROP_RETREAT')
            return

        if p == 'DROP_RETREAT':
            from_height = (
                DROP_PACKAGE_FROM_HEIGHT if self.current_ore == PACKAGE_FRAME
                else DROP_ORES_FROM_HEIGHT
            )
            if from_height or self.go(self.drop_travel_target(), DROP_KP,
                                      DROP_MAX_SPEED, TOL_COARSE):
                self.set_phase('SELECT')
            return

        if p == 'REDETECT':
            self.stop_twist()
            elapsed = self.phase_elapsed()
            pending = [n for n in self.redetect_names if n not in self.redetect_acks]

            if pending and elapsed > 1.2 and not self.redetect_resent:
                self.redetect_resent = True
                for name in pending:
                    msg = String()
                    msg.data = name
                    self.rescan_pub.publish(msg)

            if not pending or elapsed > REDETECT_TIMEOUT_S:
                self.set_phase('SELECT')
            return

        if p == 'RECOVER_UP':
            if self.recover_target is None or self.go(
                self.recover_target, APPROACH_KP, 0.05, TOL_FINE
            ):
                self.set_phase('LEVEL')
            return


def main(args=None):
    """Initialises ROS 2, spins the manipulation node and shuts down cleanly."""
    rclpy.init(args=args)

    node = arm_task2a()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':
    main()