#!/usr/bin/env python3

'''
*****************************************************************************************
*
*            ===============================================
*                       StrataCobot (SC) Theme (eYRC 2026-27)
*            ===============================================
*
*  This script implements Task 3A perception of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:         task3A_perception.py
# Functions:        detect_ores, quat_to_matrix, __init__, depthimagecb, colorimagecb,
#                   caminfocb, sample_depth, pixel_to_base, match_track,
#                   publish_locked, process_image, main
# Global variables: ore_types, color_topic, depth_topic, camera_info_topic, base_frame
# Nodes:             Publishing Topics  - [ /tf ]
#                    Subscribing Topics - [ /camera/camera/color/image_raw,
#                                           /camera/camera/aligned_depth_to_color/image_raw,
#                                           /camera/camera/color/camera_info,
#                                           /handover/ebot_ready, /ore_rescan ]


################### IMPORT MODULES #######################

import sys

import cv2
import numpy as np
import rclpy
import tf2_ros

from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import (qos_profile_sensor_data, QoSProfile, QoSHistoryPolicy,
                       QoSReliabilityPolicy, QoSDurabilityPolicy)
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String


##################### TASK CONSTANTS #######################

ore_types = [
    'azurite_ore',
    'malachite_ore',
    'vanadinite_ore'
]

color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'

base_frame = 'base_link'
cam_frame_id = 'camera_color_optical_frame'

# Total ores on the conveyor in Task 2A.
EXPECTED_ORES = 4

# Number of consistent detections needed before an ore position is locked.
N_SAMPLES = 8

# Detections of one ore must stay inside this pixel radius to count as the same ore.
TRACK_RADIUS_PX = 20

# Contour area limits (pixels) for something to be an ore.
MIN_AREA = 150
MAX_AREA = 8000

# Container positions (x, y) in base_link. Anything detected near them is a
# container outline or an ore that was already dropped, never an ore to pick.
CONTAINER_POINTS = [(-0.07, 0.55), (-0.40, 0.55), (-0.65, 0.55)]
CONTAINER_EXCLUDE_RADIUS = 0.15

# Real ores lie on the conveyor, at about this height in base_link and within the arm's reach.
# Red or orange rocks elsewhere in the world match the vanadinite colour and are rejected here.
ORE_Z = -0.065
ORE_Z_TOL = 0.06
ORE_MAX_RADIUS = 1.2

# ---- Ore package (Task 3A) ----
# The package is detected only after the eBot reports it is beside the arm.
PACKAGE_FRAME = 'ore_package'

# HSV range of the package colour, ((h_lo, s_lo, v_lo), (h_hi, s_hi, v_hi)).
# The package is #f700d7 (hue 308 deg = 154 in OpenCV, full saturation). The ranges are wide
# enough to cover its shaded faces.
PACKAGE_HSV = ((140, 120, 60), (165, 255, 255))
PACKAGE_MIN_AREA = 250                                                           # the package is small and far in the image (about 900 px)
PACKAGE_MAX_AREA = 60000

# Seconds the node spends re-detecting one ore when the arm asks for it.
RESCAN_TIMEOUT_S = 5.0


##################### FUNCTION DEFINITIONS #######################

def detect_ores(image):
    '''
    Description:
        Detects ores in a colour image and returns the centre pixel and
        type of each detected ore.

    Args:
        image (numpy.ndarray): BGR colour image.

    Returns:
        center_ore_list (list): (cX, cY) of every detected ore.
        ore_type_list (list): type of each detected ore.
    '''

    center_ore_list = []
    ore_type_list = []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_bounds = {
        'azurite_ore': ((95, 150, 100), (115, 255, 255)),
        'malachite_ore': ((60, 150, 100), (90, 255, 255)),
        'vanadinite_ore': ((0, 200, 180), (20, 255, 255))
    }

    for ore_type, (lower, upper) in color_bounds.items():

        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            area = cv2.contourArea(contour)
            if area < MIN_AREA or area > MAX_AREA:
                continue

            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue

            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])

            center_ore_list.append((center_x, center_y))
            ore_type_list.append(ore_type)

    return center_ore_list, ore_type_list


def quat_to_matrix(q):
    '''
    Description:
        Converts a geometry_msgs Quaternion into a 3x3 rotation matrix.
    '''

    x, y, z, w = q.x, q.y, q.z, q.w
    n = x * x + y * y + z * z + w * w
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n

    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)]
    ])


##################### CLASS DEFINITION #######################

class ore_tf(Node):
    '''
    ___CLASS___

    Description:
        Detects ores, converts them to base_link and broadcasts one TF
        frame per ore. An ore position is the median of N_SAMPLES
        detections and is then locked, so it stays valid when the arm
        hides the ore or carries it away.
    '''

    def __init__(self):
        """Initialises ROS 2 publishers, subscribers, services, state and timer."""
        super().__init__(
            'ore_tf_publisher',
            parameter_overrides=[
                Parameter('use_sim_time', Parameter.Type.BOOL, True)
            ]
        )

        self.color_cam_sub = self.create_subscription(
            Image, color_topic, self.colorimagecb, qos_profile_sensor_data
        )
        self.depth_cam_sub = self.create_subscription(
            Image, depth_topic, self.depthimagecb, qos_profile_sensor_data
        )
        self.cam_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.caminfocb, qos_profile_sensor_data
        )

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)

        # The arm node asks for a fresh detection of one ore on /ore_rescan
        # (data = ore frame name) and gets the answer on /ore_rescan_done.
        self.rescanning = {}
        self.rescan_done_pub = self.create_publisher(String, '/ore_rescan_done', 10)
        self.rescan_sub = self.create_subscription(
            String, '/ore_rescan', self.rescancb, 10
        )

        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.ready_sub = self.create_subscription(
            Bool, '/handover/ebot_ready', self.ebotreadycb, latched
        )

        self.timer = self.create_timer(0.1, self.process_image)

        self.cv_image = None
        self.depth_image = None
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Candidate ores being tracked: dicts with type, pixel, samples, name.
        self.tracks = []

        # Locked ore positions in base_link: name -> (x, y, z).
        self.locked = {}

        self._saved_detection_image = False

        # The ore package: searched for once the eBot is beside the arm.
        self.package_search = False
        self.package_pos = None
        self.package_samples = []
        self.package_rescanning = False
        self.package_scan_start = 0.0
        self.package_debug_saved = False

        self.get_logger().info('Task 3A perception started.')

    def depthimagecb(self, data):
        '''
        Description:
            Stores the aligned depth image in metres.
        '''

        try:
            depth = self.bridge.imgmsg_to_cv2(
                data, desired_encoding='passthrough'
            )

            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0

            self.depth_image = depth

        except CvBridgeError as error:
            self.get_logger().error(f'Depth image conversion failed: {error}')

    def colorimagecb(self, data):
        '''
        Description:
            Stores the colour image as BGR.
        '''

        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(
                data, desired_encoding='bgr8'
            )
        except CvBridgeError as error:
            self.get_logger().error(f'Color image conversion failed: {error}')

    def caminfocb(self, data):
        '''
        Description:
            Stores the camera intrinsics.
        '''

        self.fx = data.k[0]
        self.fy = data.k[4]
        self.cx = data.k[2]
        self.cy = data.k[5]

    def now_seconds(self):
        """Returns the current ROS clock time in seconds."""
        return self.get_clock().now().nanoseconds / 1e9

    def publish_done(self, name):
        """Publishes the completion acknowledgement for an ore re-scan."""
        msg = String()
        msg.data = name
        self.rescan_done_pub.publish(msg)

    def ebotreadycb(self, msg):
        '''
        Description:
            The eBot stands beside the arm with the package on its deck:
            start detecting the package.
        '''

        if msg.data and not self.package_search:
            self.package_search = True
            self.package_scan_start = self.now_seconds()
            self.get_logger().info('eBot is beside the arm; detecting the package.')

    def rescancb(self, msg):
        '''
        Description:
            Forgets the tracks of one ore and re-detects it from scratch.
            The old TF stays published until the new position is locked.
        '''

        name = msg.data.strip()

        if name == PACKAGE_FRAME:                                                       # forget the package position, detect again
            self.package_samples = []
            self.package_rescanning = True
            self.package_scan_start = self.now_seconds()
            return

        if name not in self.locked:
            self.publish_done(name)
            return

        if name in self.rescanning:
            return

        self.tracks = [t for t in self.tracks if t['name'] != name]
        self.rescanning[name] = {
            'old': self.locked[name],
            'start': self.now_seconds()
        }
        self.get_logger().info(f'Re-detecting {name}.')

    def sample_depth(self, center_x, center_y):
        '''
        Description:
            Median valid depth in a 9x9 window around a pixel, or None.
        '''

        patch = self.depth_image[
            max(center_y - 4, 0):center_y + 5,
            max(center_x - 4, 0):center_x + 5
        ]

        valid = patch[np.isfinite(patch) & (patch > 0.0)]

        if valid.size == 0:
            return None

        return float(np.median(valid))

    def pixel_to_base(self, center_x, center_y, depth, transform):
        '''
        Description:
            Deprojects a pixel with its depth and transforms the point
            from the camera optical frame to base_link.

        Returns:
            numpy array [x, y, z] in base_link.
        '''

        x = (center_x - self.cx) * depth / self.fx
        y = (center_y - self.cy) * depth / self.fy
        z = depth

        rotation = quat_to_matrix(transform.transform.rotation)
        t = transform.transform.translation

        point = rotation @ np.array([x, y, z]) + np.array([t.x, t.y, t.z])

        return point

    def match_track(self, ore_type, center_x, center_y):
        '''
        Description:
            Returns the existing track of this type nearest to the pixel
            (inside TRACK_RADIUS_PX), or None.
        '''

        best = None
        best_dist = TRACK_RADIUS_PX

        for track in self.tracks:
            if track['type'] != ore_type:
                continue

            dist = np.hypot(
                track['px'][0] - center_x, track['px'][1] - center_y
            )

            if dist <= best_dist:
                best = track
                best_dist = dist

        return best

    def publish_locked(self, stamp):
        '''
        Description:
            Broadcasts every locked ore as a TF frame in base_link.
        '''

        for name, position in self.locked.items():

            msg = TransformStamped()
            msg.header.stamp = stamp
            msg.header.frame_id = base_frame
            msg.child_frame_id = name

            msg.transform.translation.x = position[0]
            msg.transform.translation.y = position[1]
            msg.transform.translation.z = position[2]

            msg.transform.rotation.x = 0.0
            msg.transform.rotation.y = 0.0
            msg.transform.rotation.z = 0.0
            msg.transform.rotation.w = 1.0

            self.br.sendTransform(msg)

    def publish_package(self, stamp):
        '''
        Description:
            Broadcasts the locked package position as the ore_package frame in base_link.
        '''

        if self.package_pos is None:
            return

        msg = TransformStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = base_frame
        msg.child_frame_id = PACKAGE_FRAME

        msg.transform.translation.x = self.package_pos[0]
        msg.transform.translation.y = self.package_pos[1]
        msg.transform.translation.z = self.package_pos[2]
        msg.transform.rotation.w = 1.0

        self.br.sendTransform(msg)

    def process_package(self):
        '''
        Description:
            Finds the package by colour, takes the median of N_SAMPLES detections
            of its top face and locks it. A re-scan asked for by the arm repeats this.
        '''

        if not self.package_search:
            return

        now = self.now_seconds()

        if self.package_rescanning and now - self.package_scan_start > RESCAN_TIMEOUT_S:
            self.package_rescanning = False
            self.get_logger().warn('Re-detection of the package timed out; keeping the old position.')
            self.publish_done(PACKAGE_FRAME)
            return

        if self.package_pos is not None and not self.package_rescanning:
            return

        if PACKAGE_HSV is None:
            self.get_logger().error(
                'PACKAGE_HSV is not set, so the package cannot be detected.',
                throttle_duration_sec=5.0
            )
            return

        if self.cv_image is None or self.depth_image is None or self.fx is None:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame, cam_frame_id, rclpy.time.Time()
            )
        except Exception as error:
            self.get_logger().warn(f'TF lookup failed: {error}', throttle_duration_sec=2.0)
            return

        hsv = cv2.cvtColor(self.cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(PACKAGE_HSV[0]), np.array(PACKAGE_HSV[1]))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        best_area = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if PACKAGE_MIN_AREA <= area <= PACKAGE_MAX_AREA and area > best_area:
                best = contour
                best_area = area

        if best is None:
            largest = max((cv2.contourArea(c) for c in contours), default=0.0)
            self.get_logger().warn(
                f'Package not seen: {cv2.countNonZero(mask)} magenta pixels, largest blob '
                f'{largest:.0f} px (wanted {PACKAGE_MIN_AREA}-{PACKAGE_MAX_AREA}). '
                'Saved package_debug.png. Check PACKAGE_HSV and the eBot stop pose.',
                throttle_duration_sec=3.0
            )

            if not self.package_debug_saved:                                            # camera frame beside the colour mask
                cv2.imwrite('package_debug.png',
                            np.hstack([self.cv_image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)]))
                self.package_debug_saved = True

            return

        moments = cv2.moments(best)
        if moments['m00'] == 0:
            return

        center_x = int(moments['m10'] / moments['m00'])
        center_y = int(moments['m01'] / moments['m00'])

        depth = self.sample_depth(center_x, center_y)
        if depth is None:
            return

        self.package_samples.append(self.pixel_to_base(center_x, center_y, depth, transform))

        if len(self.package_samples) >= N_SAMPLES:
            median = np.median(np.array(self.package_samples), axis=0)
            self.package_pos = (float(median[0]), float(median[1]), float(median[2]))
            self.package_samples = []

            was_rescanning = self.package_rescanning
            self.package_rescanning = False

            self.get_logger().info(
                f'Locked {PACKAGE_FRAME}: x={median[0]:.3f}, y={median[1]:.3f}, z={median[2]:.3f}'
            )

            if was_rescanning:
                self.publish_done(PACKAGE_FRAME)

    def process_image(self):
        '''
        Description:
            Timer callback: detects ores, locks their positions and
            broadcasts the TF frames.
        '''

        self.publish_locked(self.get_clock().now().to_msg())
        self.publish_package(self.get_clock().now().to_msg())
        self.process_package()

        now = self.now_seconds()
        for name, info in list(self.rescanning.items()):
            if now - info['start'] > RESCAN_TIMEOUT_S:
                del self.rescanning[name]
                self.get_logger().warn(
                    f'Re-detection of {name} timed out; keeping the old position.'
                )
                self.publish_done(name)

        # All ores found: nothing more to detect. This also keeps the
        # carried ore and dropped ores from creating extra frames.
        if len(self.locked) >= EXPECTED_ORES and not self.rescanning:
            return

        if (
            self.cv_image is None
            or self.depth_image is None
            or self.fx is None
        ):
            self.get_logger().info(
                'Waiting for camera data...', throttle_duration_sec=2.0
            )
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                base_frame, cam_frame_id, rclpy.time.Time()
            )
        except Exception as error:
            self.get_logger().warn(
                f'TF lookup failed: {error}', throttle_duration_sec=2.0
            )
            return

        display = self.cv_image.copy()
        center_ore_list, ore_type_list = detect_ores(self.cv_image)

        for (center_x, center_y), ore_type in zip(center_ore_list, ore_type_list):

            depth = self.sample_depth(center_x, center_y)
            if depth is None:
                continue

            position = self.pixel_to_base(center_x, center_y, depth, transform)

            # Ignore containers and ores already dropped in them.
            near_container = any(
                np.hypot(position[0] - cxm, position[1] - cym)
                < CONTAINER_EXCLUDE_RADIUS
                for cxm, cym in CONTAINER_POINTS
            )
            if near_container:
                continue

            # Not on the conveyor (a rock on the ground, or out of reach): not an ore.
            if (abs(position[2] - ORE_Z) > ORE_Z_TOL
                    or np.hypot(position[0], position[1]) > ORE_MAX_RADIUS):
                continue

            track = self.match_track(ore_type, center_x, center_y)

            if track is None:
                track = {
                    'type': ore_type,
                    'px': (center_x, center_y),
                    'samples': [],
                    'name': None
                }
                self.tracks.append(track)

            if track['name'] is None:
                track['samples'].append(position)

                if len(track['samples']) >= N_SAMPLES:
                    median = np.median(np.array(track['samples']), axis=0)
                    candidates = [
                        n for n in self.rescanning
                        if n.startswith(ore_type + '_')
                    ]

                    if candidates:
                        name = min(
                            candidates,
                            key=lambda n: float(np.linalg.norm(
                                np.array(self.rescanning[n]['old']) - median
                            ))
                        )
                        old = self.rescanning.pop(name)['old']
                        moved = float(np.linalg.norm(np.array(old) - median))
                        self.get_logger().info(
                            f'Re-detected {name}: moved {moved * 1000:.0f} mm.'
                        )
                    else:
                        index = 1
                        while f'{ore_type}_{index}' in self.locked:
                            index += 1
                        name = f'{ore_type}_{index}'

                    track['name'] = name
                    self.locked[name] = (
                        float(median[0]), float(median[1]), float(median[2])
                    )
                    self.get_logger().info(
                        f'Locked {name}: x={median[0]:.3f}, '
                        f'y={median[1]:.3f}, z={median[2]:.3f}'
                    )

                    if candidates:
                        self.publish_done(name)

            label = track['name'] if track['name'] else ore_type
            cv2.circle(display, (center_x, center_y), 8, (255, 255, 255), 2)
            cv2.putText(
                display, label, (center_x - 40, center_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA
            )

        if len(self.locked) >= EXPECTED_ORES and not self._saved_detection_image:
            cv2.imwrite('SC#1449_task3A_detection.png', display)
            self._saved_detection_image = True
            self.get_logger().info('All ores locked; detection image saved.')


##################### MAIN #######################

def main():
    '''
    Description:
        Creates the ore_tf node and spins it.
    '''

    rclpy.init(args=sys.argv)

    node = ore_tf()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()

    if rclpy.ok():
        rclpy.shutdown()


if __name__ == '__main__':

    main()