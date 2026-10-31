#!/usr/bin/env python3

'''
*****************************************************************************************
*            ===============================================
*                       StrataCobot (SC) Theme (eYRC 2026-27)
*            ===============================================
*
*  This script implements Task 3A ore manipulation for StrataCobot (SC).
*  It sorts the ores into their containers and moves the ore package into the crusher.
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*****************************************************************************************
'''

# Team ID:          [ 1449 ]
# Author List:      [ Atharv Mudse, Abhay Potdar, Riddhi Wagh, Yash Waje ]
# Filename:         task3A_manipulation.py
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
from geometry_msgs.msg import PoseStamped, TwistStamped
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
    'azurite_ore': (-0.0700, 0.5750, 0.2000),
    'malachite_ore': (-0.4000, 0.5750, 0.2000),
    'vanadinite_ore': (-0.6500, 0.5750, 0.2000),
}

# Ore package (Task 3A). The package is heavier than an ore: the magnet reads about 2.94 N.
PACKAGE_FRAME = 'ore_package'
PACKAGE_FORCE_MIN_N = 2.2
PACKAGE_FORCE_MAX_N = 3.8
# The eBot is told it may leave once the package is this far above where it was picked,
# i.e. clear of the deck, without waiting for the arm to reach travel height.
PACKAGE_CLEAR_LIFT_M = 0.12

# Where the tool goes to release the package, (x, y, z) in base_link, like DROP_POINTS (measured).
# The package hangs below the tool face, so z must leave its underside inside the hopper.
# Left as None the package is never picked, and the node says so.
CRUSHER_DROP = (-0.7250, 0.0050, 0.4000)

ORE_RE = re.compile(r'^(azurite|malachite|vanadinite)_ore_\d+$')

EXPECTED_ORES = 4
WAIT_ORES_TIMEOUT_S = 20.0
IDLE_WAIT_S = 15.0

# Heights above the ore TF.
HOVER_HEIGHT = 0.18
APPROACH_HEIGHT = 0.08
PICK_OFFSET_Z = -0.002
# Extra z offsets tried on successive pick attempts.
PICK_Z_TRIES = (0.0, -0.002, 0.002)
# A pick is never skipped for a bad force reading: the arm keeps trying, nudging
# the tool around the detected position each time, and re-detects the ore after
# every failure. Only after MAX_PICK_ATTEMPTS is the ore left for a later round.
MAX_PICK_ATTEMPTS = 10
MAX_RETRY_ROUNDS = 2
SEARCH_OFFSETS = (
    (0.0, 0.0), (0.005, 0.0), (-0.005, 0.0), (0.0, 0.005), (0.0, -0.005),
    (0.010, 0.0), (-0.010, 0.0), (0.0, 0.010), (0.0, -0.010),
    (0.005, 0.005), (-0.005, -0.005),
)
# After touching something, back off this far before switching the magnet on.
CONTACT_BACKOFF_M = 0.004
RESCAN_TIMEOUT_S = 8.0
# After each drop the remaining ores are detected again; perception answers within 5 s.
REDETECT_TIMEOUT_S = 3.5
LIFT_HEIGHT = 0.16

# Protective stops are codes 20 to 23.
PROTECTIVE_STOP_CODES = (20, 21, 22, 23)
MAX_STOPS_PER_ORE = 2

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

unfold_kp = 2.5
control_rate = 0.05

# Position tolerances (m).
position_tolerance = 0.005
TOL_FINE = 0.008
TOL_COARSE = 0.015

# Speed profiles (kp, max m/s). All maxima stay below cap_linear_mps.
TRAVEL_KP, TRAVEL_SPEED = 3.0, 0.135
APPROACH_KP, APPROACH_SPEED = 2.5, 0.13
DESCEND_KP, DESCEND_SPEED = 1.5, 0.07
LIFT_KP, LIFT_SPEED = 3.0, 0.13
TRANSPORT_KP, TRANSPORT_MAX_SPEED = 3.0, 0.135
DROP_KP, DROP_MAX_SPEED = 3.0, 0.13
DROP_DESCEND_SPEED = 0.08

MOVE_TIMEOUT_S = 40.0

# Grasp verification.
# A held ore reads about 1.47 N (measured). Accept a steady reading in this band.
FORCE_HELD_MIN_N = 0.8
FORCE_HELD_MAX_N = 2.5
FORCE_STABLE_RANGE_N = 0.15
FORCE_STABLE_WINDOW_S = 0.3
# Stop descending if the tool presses on the ore/conveyor this hard.
CONTACT_FORCE_N = 4.0
FORCE_WAIT_S = 2.0
MAGNET_SETTLE_S = 0.4
DROP_SETTLE_S = 0.5

# Safe height (base_link z) for all horizontal moves. The arm only ever moves
# vertically near the ore and the container, and level at this height in between.
TRAVEL_Z = 0.30

# Path planning around the arm's own base. A level move whose straight line passes
# closer than R_CLEAR to the base axis is replaced by an arc of radius R_ARC that
# goes round the back (x < 0) of the base.
R_CLEAR = 0.25
R_ARC = 0.60
ARC_STEP_DEG = 30.0
# A move that would pass the base is first tried as two straight legs through one point behind
# the base at this radius (about half the length of the arc). It is used only when both legs
# keep at least R_CLEAR from the base axis. Set to None to always use the arc.
VIA_RADIUS = 0.40
TOL_ARC = 0.04
DROP_APPROACH_HEIGHT = 0.10

# Release the magnet at the hover height above the container instead of descending to the drop
# point first. This skips the descent and the climb back, but the ore falls the extra height
# (about DROP_APPROACH_HEIGHT), so test that every ore stays in its container before relying on it.
# The package is heavier and the hopper is the one drop that must not fail, so it still descends.
DROP_ORES_FROM_HEIGHT = True
DROP_PACKAGE_FROM_HEIGHT = False

# Tool orientation: keep the tool's pointing axis straight down so the magnet face
# lies flat on the ore. The axis is found automatically (the tcp axis closest to
# straight down once the arm is unfolded).
ANG_KP = 1.5
ANG_MAX_RPS = 0.25
LEVEL_TOL_RAD = 0.01
LEVEL_TIMEOUT_S = 20.0


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
            'task3a_manipulation',
            parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, True)
            ]
        )

        # ------------------------- PUBLISHERS -------------------------

        self.twist_pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.joint_pub = self.create_publisher(JointJog, '/delta_joint_cmds', 10)

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
        self.rescan_pub = self.create_publisher(String, '/ore_rescan', 10)
        self.rescan_done_sub = self.create_subscription(
            String, '/ore_rescan_done', self.rescandonecb, 10
        )
        self.force_sub = self.create_subscription(
            Float32, '/ur7e/end_effector/force_magnitude', self.forcecb, 20
        )

        # ------------------------- HANDOVER -------------------------

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

        # ------------------------- SERVICES -------------------------

        self.switch_cli = self.create_client(
            SwitchController, f'{servo_ns}/switch_controller'
        )
        self.unlock_client = self.create_client(
            Trigger, '/ur_arm_controller/unlock_protective_stop'
        )
        self.magnet_client = self.create_client(SetBool, '/magnet')

        # ------------------------- TF -------------------------

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ------------------------- FEEDBACK -------------------------

        self.tcp_pose = None
        self.joint_angles = None
        self.arm_status = None
        self.force_magnitude = None

        # ------------------------- STATE -------------------------

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

        self.get_logger().info('Task 3A manipulation node started.')

    # ------------------------- CALLBACKS -------------------------

    def tcpposecb(self, data):
        """Stores the latest tool-centre-point pose received from the arm."""
        self.tcp_pose = data.pose

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

    # ------------------------- BASIC HELPERS -------------------------

    def now_seconds(self):
        """Returns the current ROS clock time in seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    def stop_twist(self):
        """Publishes a zero Cartesian velocity command to stop arm motion."""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = base_frame
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
        self.phase_start_xyz = (
            (self.tcp_pose.position.x, self.tcp_pose.position.y, self.tcp_pose.position.z)
            if self.tcp_pose is not None else None
        )

        self.get_logger().info(
            f'Ore={self.current_ore}, phase={phase} '
            f'(previous {previous}: {took:.1f} s, tool moved {moved:.2f} m)'
        )

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

        max_speed = min(max_speed, cap_linear_mps * 0.9)
        magnitude = math.sqrt(vx * vx + vy * vy + vz * vz)

        if magnitude > max_speed or (hold_speed and magnitude > 1e-9):
            scale = max_speed / magnitude
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

        return math.sqrt(error_x ** 2 + error_y ** 2 + error_z ** 2)

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

    # ------------------------- PATHS AROUND THE BASE -------------------------

    def segment_clearance(self, a, b):
        '''Distance from the base axis to the straight segment a-b.'''

        dx, dy = b[0] - a[0], b[1] - a[1]
        length2 = dx * dx + dy * dy
        if length2 < 1e-9:
            return math.hypot(a[0], a[1])

        t = max(0.0, min(1.0, -(a[0] * dx + a[1] * dy) / length2))
        return math.hypot(a[0] + t * dx, a[1] + t * dy)

    def plan_path_xy(self, start, end):
        '''XY waypoints (end included) for a level move that keeps clear of the
        base axis. Direct when the straight line is far enough from the base,
        otherwise an arc of radius R_ARC passing behind the base (x < 0).'''

        sx, sy = start
        ex, ey = end

        if math.hypot(sx, sy) < R_CLEAR or math.hypot(ex, ey) < R_CLEAR:
            return [end]

        dx, dy = ex - sx, ey - sy
        length2 = dx * dx + dy * dy
        if length2 < 1e-9:
            return [end]

        t = max(0.0, min(1.0, -(sx * dx + sy * dy) / length2))
        if math.hypot(sx + t * dx, sy + t * dy) >= R_CLEAR:
            return [end]

        theta_s = math.atan2(sy, sx)
        theta_e = math.atan2(ey, ex)
        inc = (theta_e - theta_s) % (2.0 * math.pi)
        dec = inc - 2.0 * math.pi

        delta = inc if math.cos(theta_s + inc / 2.0) < 0.0 else dec

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

    def start_path(self, end3):
        """Plans and stores level waypoints for a safe move to a target."""
        points = self.plan_path_xy(
            (self.tcp_pose.position.x, self.tcp_pose.position.y),
            (end3[0], end3[1])
        )
        self.path = [(x, y, end3[2]) for x, y in points]
        self.path_idx = 0

        if len(self.path) > 1:
            self.get_logger().info(
                f'Routing around the base with {len(self.path) - 1} arc point(s).'
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
        '''True when recent force readings sit steadily in the held-ore band.'''

        now = self.now_seconds()
        samples = [f for t, f in self.force_hist if now - t <= FORCE_STABLE_WINDOW_S]

        if len(samples) < 2:
            return False

        if self.current_ore == PACKAGE_FRAME:
            low, high = PACKAGE_FORCE_MIN_N, PACKAGE_FORCE_MAX_N
        else:
            low, high = FORCE_HELD_MIN_N, FORCE_HELD_MAX_N

        return (
            min(samples) >= low
            and max(samples) <= high
            and max(samples) - min(samples) <= FORCE_STABLE_RANGE_N
        )

    # ------------------------- CONTROLLER SWITCHING -------------------------

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

        # Twist controller is active.
        if self.in_recovery:
            self.in_recovery = False

            if not self.unfolded:
                self.switch_controller(joint_controller)
                return

            self.motion_state = 'RUN'
            if self.tcp_pose is not None:
                self.recover_target = (
                    self.tcp_pose.position.x,
                    self.tcp_pose.position.y,
                    max(self.tcp_pose.position.z + 0.08, TRAVEL_Z),
                )
            self.set_phase('RECOVER_UP')
            return

        if not self.unfolded:
            self.unfolded = True
            self.motion_state = 'RUN'
            self.set_phase('LEVEL')
            return

        self.motion_state = 'RUN'

    # ------------------------- UNFOLD / RECOVERY -------------------------

    def unfold_arm(self):
        """Moves the arm joints toward the configured unfolded posture."""
        velocities = []
        max_error = 0.0

        for idx, joint in enumerate(joint_names):
            error = unfold_target[joint] - self.joint_angles[idx]
            max_error = max(max_error, abs(error))

            velocity = max(
                min(unfold_kp * error, cap_joint_rps * 0.9),
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
        """Raises the shoulder joint during protective-stop recovery."""
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

        # Release anything held, and avoid repeating a pick that keeps stopping.
        self.request_magnet(False, 'recovery')

        ore = self.current_ore
        if ore is not None and ore not in self.done_ores:
            self.stop_counts[ore] = self.stop_counts.get(ore, 0) + 1
            if self.stop_counts[ore] >= MAX_STOPS_PER_ORE:
                self.get_logger().warn(f'Skipping {ore} after repeated stops.')
                self.skipped_ores.add(ore)

        self.switch_controller(joint_controller)

    # ------------------------- TF / ORE -------------------------

    def lookup_ore(self, ore_id):
        """Looks up an ore transform and returns its position in base_link."""
        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame, ore_id, rclpy.time.Time()
            )
            t = transform.transform.translation
            return (float(t.x), float(t.y), float(t.z))
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

        # The package, once the eBot is beside the arm, goes before any ore: the eBot
        # then drives home while the arm carries on sorting.
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

            # The run is not over until the package is in the crusher (or given up).
            package_pending = (
                CRUSHER_DROP is not None
                and PACKAGE_FRAME not in self.done_ores
                and PACKAGE_FRAME not in self.skipped_ores
            )
            if package_pending:
                self.get_logger().info(
                    'Waiting for the package to arrive...', throttle_duration_sec=3.0
                )
                return

            if handled >= EXPECTED_ORES or idle > IDLE_WAIT_S:
                self.stop_twist()
                self.get_logger().info(
                    f'Task 3A complete: dropped={sorted(self.done_ores)}, '
                    f'skipped={sorted(self.skipped_ores)}'
                )
                self.motion_state = 'DONE'
            else:
                self.get_logger().info(
                    'Waiting for remaining ore TFs...', throttle_duration_sec=2.0
                )
            return

        self.idle_start = None

        # Nearest remaining ore (in the horizontal plane) first.
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
                dist = -1.0                                                             # the package always wins
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
        self.start_path(self.ore_hover_target())
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
        '''Asks perception to detect every remaining ore again, leaving out the dropped ones.'''

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
        return (x + ox, y + oy, z + PICK_OFFSET_Z + extra)

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

    # ------------------------- MAGNET -------------------------

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

    # ------------------------- MAIN LOOP -------------------------

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
            self.get_logger().info('Waiting for ore transforms from perception.')
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

    # ------------------------- PER-ORE PHASES -------------------------

    def run_phase(self):
        """Executes the actions associated with the active per-ore phase."""
        p = self.phase

        if p == 'LEVEL':
            # Turn the tool until it points straight down.
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

            # If the tilt grows, the rotation sign is the wrong way round.
            if not self.level_checked and elapsed > 1.5:
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

        # ---- Pick ----

        if p == 'HOVER':
            # Travelling empty towards an ore at travel height: the package goes first.
            if self.current_ore != PACKAGE_FRAME and self.package_waiting():
                self.path = None
                self.stop_twist()
                self.get_logger().info('eBot is beside the arm: taking the package first.')
                self.set_phase('SELECT')
                return

            if self.follow_path(TRAVEL_KP, TRAVEL_SPEED, TOL_COARSE):
                self.set_phase('APPROACH')
            return

        if p == 'APPROACH':
            if self.go(self.ore_approach_target(), APPROACH_KP, APPROACH_SPEED, TOL_FINE):
                self.set_phase('DESCEND')
            return

        if p == 'DESCEND':
            force = self.force_magnitude
            contact = (
                force is not None
                and force > CONTACT_FORCE_N
                and self.phase_elapsed() > 0.5
            )

            if contact:
                # Touched something before reaching the target height. Back off
                # a few mm, then switch the magnet on from there.
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

            if self.go(
                self.ore_pick_target(), DESCEND_KP, DESCEND_SPEED,
                position_tolerance
            ):
                self.set_phase('MAGNET_ON')
                self.request_magnet(True, 'pick')
            return

        if p == 'BACKOFF':
            if self.go(self.backoff_target, 1.5, 0.03, 0.003):
                self.set_phase('MAGNET_ON')
                self.request_magnet(True, 'pick')
            return

        if p in ('MAGNET_ON', 'RELEASE_WAIT'):
            self.stop_twist()
            return

        if p == 'VERIFY':
            self.stop_twist()

            force = self.force_magnitude
            elapsed = self.phase_elapsed()

            self.get_logger().info(
                f'Verifying pick: force={force}',
                throttle_duration_sec=0.5
            )

            if elapsed >= MAGNET_SETTLE_S and self.force_steady_in_band():
                self.get_logger().info(
                    f'Pick confirmed for {self.current_ore}; force={force:.3f} N'
                )
                self.set_phase('LIFT')
                return

            # Pressing on something hard, or nothing held: the ore is not on the
            # magnet. Never carry on without a confirmed pick.
            blocked = (
                force is not None
                and force > 2.0 * CONTACT_FORCE_N
                and elapsed > 0.6
            )
            if blocked or elapsed > FORCE_WAIT_S:
                self.pick_failed()
            return

        if p == 'RETRY_LIFT':
            if self.go(self.retry_target, APPROACH_KP, APPROACH_SPEED, TOL_FINE):
                self.set_phase('RESCAN')
            return

        if p == 'RESCAN':
            # Ask perception for a fresh detection of this ore (it may have been
            # nudged), then use the new transform for the next attempt.
            self.stop_twist()
            elapsed = self.phase_elapsed()

            if elapsed - self.rescan_last > 2.0 and not self.rescan_done:
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
                self.start_path(self.ore_hover_target())
                self.set_phase('HOVER')
            return

        if p == 'SKIP_LIFT':
            if self.go(self.retry_target, APPROACH_KP, APPROACH_SPEED, TOL_FINE):
                self.set_phase('SELECT')
            return

        if p == 'LIFT':
            if (self.current_ore == PACKAGE_FRAME and not self.package_taken_sent
                    and self.tcp_pose.position.z >= self.current_ore_position[2] + PACKAGE_CLEAR_LIFT_M):
                self.publish_package_taken()                                            # off the deck: the eBot may go

            if self.go(self.ore_lift_target(), LIFT_KP, LIFT_SPEED, TOL_COARSE):
                if self.current_ore == PACKAGE_FRAME:
                    self.publish_package_taken()
                self.start_path(self.drop_travel_target())
                self.set_phase('TRANSIT')
            return

        # ---- Transport and drop ----

        if p == 'TRANSIT':
            if self.follow_path(TRANSPORT_KP, TRANSPORT_MAX_SPEED, TOL_COARSE):
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

        # ---- Leave the container area ----

        if p == 'DROP_RETREAT':
            if self.go(self.drop_travel_target(), DROP_KP, DROP_MAX_SPEED, TOL_COARSE):
                self.start_redetect()
            return

        if p == 'REDETECT':
            self.stop_twist()
            elapsed = self.phase_elapsed()
            pending = [n for n in self.redetect_names if n not in self.redetect_acks]

            if pending and elapsed > 2.0 and not self.redetect_resent:
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