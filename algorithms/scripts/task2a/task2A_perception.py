#!/usr/bin/env python3

'''
*****************************************************************************************
*
*            ===============================================
*                       StrataCobot (SC) Theme (eYRC 2026-27)
*            ===============================================
*
*  This script implements Task 2A perception of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:         task2A_perception.py
# Functions:        detect_ores, quat_to_matrix, __init__, depthimagecb, colorimagecb,
#                   caminfocb, sample_depth, pixel_to_base, match_track,
#                   publish_locked, process_image, main
# Global variables: ore_types, color_topic, depth_topic, camera_info_topic, base_frame
# Nodes:             Publishing Topics  - [ /tf ]
#                    Subscribing Topics - [ /camera/camera/color/image_raw,
#                                           /camera/camera/aligned_depth_to_color/image_raw,
#                                           /camera/camera/color/camera_info ]


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
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String


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

        self.get_logger().info('Task 2A perception started.')

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

    def rescancb(self, msg):
        '''
        Description:
            Forgets the tracks of one ore and re-detects it from scratch.
            The old TF stays published until the new position is locked.
        '''

        name = msg.data.strip()

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

    def process_image(self):
        '''
        Description:
            Timer callback: detects ores, locks their positions and
            broadcasts the TF frames.
        '''

        self.publish_locked(self.get_clock().now().to_msg())

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
            cv2.imwrite('SC#1449_task2A_detection.png', display)
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
