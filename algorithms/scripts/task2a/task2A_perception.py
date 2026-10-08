#!/usr/bin/env python3


'''
*****************************************************************************************
*
*            ===============================================
*                       StrataCobot (SC) Theme (eYRC 2026-27)
*            ===============================================
*
*  This script should be used to implement Task 2A perception of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Functions:        detect_ores, __init__, depthimagecb, colorimagecb,
#                   caminfocb, process_image, main
# Global variables: ore_types, color_topic, depth_topic,
#                   camera_info_topic, base_frame
# Nodes:             Publishing Topics  - [ /tf ]
#                    Subscribing Topics - [ /camera/camera/color/image_raw,
#                                           /camera/camera/aligned_depth_to_color/image_raw,
#                                           /camera/camera/color/camera_info ]


################### IMPORT MODULES #######################

import rclpy
import sys
import cv2
import tf2_ros
import numpy as np

from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, TransformStamped
from sensor_msgs.msg import CameraInfo, Image
from tf2_geometry_msgs import do_transform_point


##################### TASK CONSTANTS #######################

# Task 2A uses four ores. Each type may use IDs 1 and 2.
# Ore transforms are published as <type>_ore_1 / <type>_ore_2.
ore_types = [
    'azurite_ore',
    'malachite_ore',
    'vanadinite_ore'
]

# RealSense colour image topic.
color_topic = '/camera/camera/color/image_raw'

# RealSense depth image aligned with the colour image.
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'

# RealSense camera information topic.
camera_info_topic = '/camera/camera/color/camera_info'

# Parent frame for every ore transform.
base_frame = 'base_link'


##################### FUNCTION DEFINITIONS #######################

def detect_ores(image):
    '''
    Description:
        Detects ores present in a colour image and returns the center
        pixel location and type of each detected ore.

    Args:
        image (numpy.ndarray):
            Input colour image frame received from the camera topic.

    Returns:
        center_ore_list (list):
            Center pixel coordinates (cX, cY) of every detected ore.

        ore_type_list (list):
            Type of each detected ore.
    '''

    ############ Function VARIABLES ############

    center_ore_list = []
    ore_type_list = []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # HSV colour ranges for the three ore types.
    color_bounds = {
        'azurite_ore': ((95, 150, 100), (115, 255, 255)),
        'malachite_ore': ((60, 150, 100), (90, 255, 255)),
        'vanadinite_ore': ((0, 200, 180), (20, 255, 255))
    }

    for ore_type, (lower, upper) in color_bounds.items():

        mask = cv2.inRange(
            hsv,
            np.array(lower),
            np.array(upper)
        )

        # Remove small noise from the binary mask.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((5, 5), np.uint8)
        )

        # Fill small gaps inside detected regions.
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            np.ones((5, 5), np.uint8)
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            # Ignore regions that are too small to be ores.
            if cv2.contourArea(contour) < 150:
                continue

            moments = cv2.moments(contour)

            # Protect against division by zero.
            if moments['m00'] == 0:
                continue

            center_x = int(moments['m10'] / moments['m00'])
            center_y = int(moments['m01'] / moments['m00'])

            center_ore_list.append((center_x, center_y))
            ore_type_list.append(ore_type)

    return center_ore_list, ore_type_list


##################### CLASS DEFINITION #######################

class ore_tf(Node):
    '''
    ___CLASS___

    Description:
        Class which serves the purpose of detecting ores in the cell
        and broadcasting their positions as TF frames with respect
        to the robot base frame.
    '''

    def __init__(self):
        '''
        Description:
            Initialization of class ore_tf.
        '''

        super().__init__('ore_tf_publisher')

        ############ Topic SUBSCRIPTIONS ############

        self.color_cam_sub = self.create_subscription(
            Image,
            color_topic,
            self.colorimagecb,
            10
        )

        self.depth_cam_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depthimagecb,
            10
        )

        self.cam_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.caminfocb,
            10
        )

        ############ Constructor VARIABLES/OBJECTS ############

        # Time interval between image processing cycles.
        image_processing_rate = 0.2

        # CvBridge object used for ROS image conversion.
        self.bridge = CvBridge()

        # TF buffer used for looking up camera-to-base transforms.
        self.tf_buffer = tf2_ros.buffer.Buffer()

        # TF listener used to receive available transforms.
        self.listener = tf2_ros.TransformListener(
            self.tf_buffer,
            self
        )

        # TF broadcaster used to publish ore transforms.
        self.br = tf2_ros.TransformBroadcaster(self)

        # Timer used to process the images periodically.
        self.timer = self.create_timer(
            image_processing_rate,
            self.process_image
        )

        # Latest colour image.
        self.cv_image = None

        # Latest depth image.
        self.depth_image = None

        # Latest camera intrinsic information.
        self.cam_info = None

        # Stores the last known pixel position of each ore ID.
        self.ore_positions = {}
        self.last_base_positions = {}

        # Stores whether the detection image has already been saved.
        self._saved_detection_image = False

        # Stores whether the depth data type has already been logged.
        self._logged_depth_dtype = False

    def depthimagecb(self, data):
        '''
        Description:
            Callback function for the aligned depth camera topic.
            Converts the received ROS depth image into a NumPy array.

        Args:
            data (Image):
                Input depth image received from the aligned depth
                camera topic.

        Returns:
            None
        '''

        try:

            depth = self.bridge.imgmsg_to_cv2(
                data,
                desired_encoding='passthrough'
            )

            if not self._logged_depth_dtype:
                self.get_logger().info(
                    f'Depth dtype: {depth.dtype}'
                )
                self._logged_depth_dtype = True

            # Convert millimetres to metres when using 16-bit depth.
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0

            self.depth_image = depth

        except CvBridgeError as error:
            self.get_logger().error(
                f'Depth image conversion failed: {error}'
            )

    def colorimagecb(self, data):
        '''
        Description:
            Callback function for the colour camera topic.
            Converts the received ROS image into an OpenCV BGR image.

        Args:
            data (Image):
                Input colour image received from the camera topic.

        Returns:
            None
        '''

        try:

            self.cv_image = self.bridge.imgmsg_to_cv2(
                data,
                desired_encoding='bgr8'
            )

        except CvBridgeError as error:
            self.get_logger().error(
                f'Color image conversion failed: {error}'
            )

    def caminfocb(self, data):
        '''
        Description:
            Callback function for the camera information topic.
            Stores the camera intrinsic parameters required for
            3D point calculation.

        Args:
            data (CameraInfo):
                Camera calibration information.

        Returns:
            None
        '''

        self.cam_info = data.k

        # Camera focal lengths.
        self.fx = data.k[0]
        self.fy = data.k[4]

        # Camera principal point.
        self.cx = data.k[2]
        self.cy = data.k[5]

    def process_image(self):
        '''
        Description:
            Timer function used to detect ores, calculate their
            3D positions and publish their TF frames.

        Args:
            None

        Returns:
            None
        '''

        ############ Check Required Data ############

        # Return until a colour image is available.
        if self.cv_image is None:
            self.get_logger().info(
                'Waiting for color image...'
            )
            return

        display = self.cv_image.copy()

        # Detect all ores in the current colour frame.
        center_ore_list, ore_type_list = detect_ores(
            self.cv_image
        )

        ############ Group Detections by Type ############

        by_type = {}

        for center, ore_type in zip(
            center_ore_list,
            ore_type_list
        ):
            by_type.setdefault(
                ore_type,
                []
            ).append(center)

        ############ Assign Stable Ore IDs ############

        all_assignments = []

        for ore_type, detections in by_type.items():

            candidate_ids = [
                f'{ore_type}_1',
                f'{ore_type}_2'
            ]

            if len(detections) == 1:

                center_x, center_y = detections[0]

                best_id = None
                best_distance = float('inf')

                for ore_id in candidate_ids:

                    if ore_id in self.ore_positions:

                        previous_x, previous_y = (
                            self.ore_positions[ore_id]
                        )

                        distance = (
                            (previous_x - center_x) ** 2 +
                            (previous_y - center_y) ** 2
                        )

                        if distance < best_distance:
                            best_distance = distance
                            best_id = ore_id

                if best_id is None:
                    best_id = candidate_ids[0]

                assignment = [
                    (best_id, (center_x, center_y))
                ]

            elif len(detections) >= 2:

                detection_1 = detections[0]
                detection_2 = detections[1]

                known_ids = [
                    ore_id
                    for ore_id in candidate_ids
                    if ore_id in self.ore_positions
                ]

                if len(known_ids) == 2:

                    id_1, id_2 = candidate_ids

                    position_1 = self.ore_positions[id_1]
                    position_2 = self.ore_positions[id_2]

                    cost_a = (
                        (position_1[0] - detection_1[0]) ** 2 +
                        (position_1[1] - detection_1[1]) ** 2 +
                        (position_2[0] - detection_2[0]) ** 2 +
                        (position_2[1] - detection_2[1]) ** 2
                    )

                    cost_b = (
                        (position_2[0] - detection_1[0]) ** 2 +
                        (position_2[1] - detection_1[1]) ** 2 +
                        (position_1[0] - detection_2[0]) ** 2 +
                        (position_1[1] - detection_2[1]) ** 2
                    )

                    if cost_a <= cost_b:
                        assignment = [
                            (id_1, detection_1),
                            (id_2, detection_2)
                        ]
                    else:
                        assignment = [
                            (id_2, detection_1),
                            (id_1, detection_2)
                        ]

                elif len(known_ids) == 1:

                    known_id = known_ids[0]

                    if known_id == candidate_ids[0]:
                        other_id = candidate_ids[1]
                    else:
                        other_id = candidate_ids[0]

                    previous_position = self.ore_positions[known_id]

                    distance_1 = (
                        (previous_position[0] - detection_1[0]) ** 2 +
                        (previous_position[1] - detection_1[1]) ** 2
                    )

                    distance_2 = (
                        (previous_position[0] - detection_2[0]) ** 2 +
                        (previous_position[1] - detection_2[1]) ** 2
                    )

                    if distance_1 <= distance_2:
                        assignment = [
                            (known_id, detection_1),
                            (other_id, detection_2)
                        ]
                    else:
                        assignment = [
                            (known_id, detection_2),
                            (other_id, detection_1)
                        ]

                else:

                    assignment = [
                        (candidate_ids[0], detection_1),
                        (candidate_ids[1], detection_2)
                    ]

            else:

                assignment = []

            for ore_id, (center_x, center_y) in assignment:

                self.ore_positions[ore_id] = (
                    center_x,
                    center_y
                )

                all_assignments.append(
                    (ore_id, center_x, center_y)
                )

        ############ Draw Detected Ores ############

        for ore_id, center_x, center_y in all_assignments:

            cv2.circle(
                display,
                (center_x, center_y),
                8,
                (255, 255, 255),
                2
            )

            cv2.putText(
                display,
                ore_id,
                (center_x - 40, center_y - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

        cv2.imshow(
            'color_feed',
            display
        )

        cv2.waitKey(1)

        ############ Save Detection Image ############

        if (
            not self._saved_detection_image
            and len(all_assignments) >= 4
        ):

            save_result = cv2.imwrite(
                'SC#1449_task2A_detection.png',
                display
            )

            self._saved_detection_image = True

            self.get_logger().info(
                f'Detection image save result: {save_result}'
            )

        ############ Broadcast Ore TF ############

        if (
            self.depth_image is not None
            and hasattr(self, 'fx')
        ):

            cam_frame_id = 'camera_color_optical_frame'

            current_time = self.get_clock().now().to_msg()

            for ore_id, center_x, center_y in all_assignments:

                # Extract a 9x9 depth window around the ore center.
                patch = self.depth_image[
                    max(center_y - 4, 0):center_y + 5,
                    max(center_x - 4, 0):center_x + 5
                ]

                # Keep only valid depth measurements.
                valid_depth = patch[
                    np.isfinite(patch) &
                    (patch > 0.0)
                ]

                if valid_depth.size == 0:
                    continue

                # Median depth provides a more stable measurement.
                depth = float(
                    np.median(valid_depth)
                )

                # Deproject the pixel into the camera optical frame.
                x = (
                    (center_x - self.cx) *
                    depth /
                    self.fx
                )

                y = (
                    (center_y - self.cy) *
                    depth /
                    self.fy
                )

                z = depth

                ############ Transform Point to Base Frame ############

                point_in_camera = PointStamped()

                point_in_camera.header.frame_id = cam_frame_id
                point_in_camera.header.stamp = (
                    rclpy.time.Time().to_msg()
                )

                point_in_camera.point.x = x
                point_in_camera.point.y = y
                point_in_camera.point.z = z

                try:

                    transform = self.tf_buffer.lookup_transform(
                        base_frame,
                        cam_frame_id,
                        rclpy.time.Time()
                    )

                    point_in_base = do_transform_point(
                        point_in_camera,
                        transform
                    )

                except Exception as error:

                    self.get_logger().warn(
                        f'TF lookup failed for {ore_id}: {error}'
                    )

                    continue

                base_x = point_in_base.point.x
                base_y = point_in_base.point.y
                base_z = point_in_base.point.z

                # Keep the last valid 3-D position. The arm can hide an ore
                # from the camera while it is approaching, so the last valid
                # TF must remain available to the manipulation node.
                self.last_base_positions[ore_id] = (
                    float(base_x),
                    float(base_y),
                    float(base_z)
                )

                ############ Publish Transform ############

                transform_msg = TransformStamped()

                transform_msg.header.stamp = current_time
                transform_msg.header.frame_id = base_frame
                transform_msg.child_frame_id = ore_id

                transform_msg.transform.translation.x = float(base_x)
                transform_msg.transform.translation.y = float(base_y)
                transform_msg.transform.translation.z = float(base_z)

                transform_msg.transform.rotation.w = 1.0

                self.br.sendTransform(
                    transform_msg
                )

        # Re-publish cached transforms for ores that are temporarily hidden
        # by the arm/tool. This keeps their last valid base_link position
        # available even after the camera view is occluded.
        visible_ids = {
            item[0] for item in all_assignments
        }

        for ore_id, position in self.last_base_positions.items():

            if ore_id in visible_ids:
                continue

            transform_msg = TransformStamped()

            transform_msg.header.stamp = current_time
            transform_msg.header.frame_id = base_frame
            transform_msg.child_frame_id = ore_id

            transform_msg.transform.translation.x = position[0]
            transform_msg.transform.translation.y = position[1]
            transform_msg.transform.translation.z = position[2]

            transform_msg.transform.rotation.x = 0.0
            transform_msg.transform.rotation.y = 0.0
            transform_msg.transform.rotation.z = 0.0
            transform_msg.transform.rotation.w = 1.0

            self.br.sendTransform(
                transform_msg
            )


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:
        Main function which creates a ROS node and spins around for
        the ore_tf class to perform its task.

    Args:
        None

    Returns:
        None
    '''

    rclpy.init(args=sys.argv)

    node = rclpy.create_node(
        'ore_tf_process'
    )

    node.get_logger().info(
        'Node created: Ore tf process'
    )

    ore_tf_class = ore_tf()

    rclpy.spin(
        ore_tf_class
    )

    ore_tf_class.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':

    main()