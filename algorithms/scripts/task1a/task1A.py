#!/usr/bin/env python3

'''# Team ID:          [ 1449 ]
# Author List:		[ Atharv Mudse, Abhay Potdar, Riddhi Wagh, Yash Waje ]
# Filename:		    task1A.py
# Functions:        detect_ores,  __init__, depthimagecb, colorimagecb, caminfocb, process_image, main
# Global variables: ore_types,
#                   color_topic,
#                   depth_topic,
#                   camera_info_topic,
#                   base_frame
# Nodes:		    Publishing Topics  - [ /tf ]
#                   Subscribing Topics - [ /camera/camera/color/image_raw,
#                                          /camera/camera/aligned_depth_to_color/image_raw,
#                                          /camera/camera/color/camera_info ]
'''

import rclpy
import sys
import cv2
import math
import tf2_ros
import numpy as np
from rclpy.node import Node
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped, PointStamped
from tf2_geometry_msgs import do_transform_point


ore_types = ['azurite_ore', 'malachite_ore', 'vanadinite_ore']
color_topic = '/camera/camera/color/image_raw'
depth_topic = '/camera/camera/aligned_depth_to_color/image_raw'
camera_info_topic = '/camera/camera/color/camera_info'
base_frame = 'base_link'


def detect_ores(image):
    '''
    Purpose:
    ---
    Detects the ores present in a colour image and returns the pixel
    location and type of each detected ore.

    Input Arguments:
    ---
    `image` :  [ numpy.ndarray ]
        Input colour image frame received from the camera topic.

    Returns:
    ---
    `center_ore_list` :  [ list ]
        Centre pixel coordinates (cX, cY) of every detected ore.

    `ore_type_list` :  [ list ]
        Type of each detected ore.

    Example call:
    ---
    `detect_ores(image)`
    '''

    center_ore_list = []
    ore_type_list = []

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    color_bounds = {
        'azurite_ore': ((95, 150, 100), (115, 255, 255)),
        'malachite_ore': ((60, 150, 100), (90, 255, 255)),
        'vanadinite_ore': ((0, 200, 180), (20, 255, 255)),
    }

    for ore_type, (lower, upper) in color_bounds.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            if cv2.contourArea(c) < 150:
                continue
            M = cv2.moments(c)
            if M['m00'] == 0:
                continue
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
            center_ore_list.append((cX, cY))
            ore_type_list.append(ore_type)

    return center_ore_list, ore_type_list


class ore_tf(Node):
    '''
    Purpose:
    ---
    Detects ores using colour and depth camera data and broadcasts
    their position as TF frames with respect to the robot base frame.

    Subscribes to:
    ---
    `/camera/camera/color/image_raw` : [ sensor_msgs/msg/Image ]

    `/camera/camera/aligned_depth_to_color/image_raw` :
    [ sensor_msgs/msg/Image ]

    `/camera/camera/color/camera_info` : [ sensor_msgs/msg/CameraInfo ]

    Publishes to:
    ---
    `/tf` : [ tf2_msgs/msg/TFMessage ]
    '''

    def __init__(self):
        '''
        Purpose:
        ---
        Initializes the ROS node, camera subscriptions, TF listener,
        TF broadcaster and image processing timer.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        Called automatically when the node is constructed.
        '''

        super().__init__('ore_tf_publisher')

        self.color_cam_sub = self.create_subscription(Image, color_topic, self.colorimagecb, 10)
        self.depth_cam_sub = self.create_subscription(Image, depth_topic, self.depthimagecb, 10)
        self.cam_info_sub = self.create_subscription(
            CameraInfo, camera_info_topic, self.caminfocb, 10)

        image_processing_rate = 0.2
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.buffer.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.br = tf2_ros.TransformBroadcaster(self)
        self.timer = self.create_timer(image_processing_rate, self.process_image)

        self.cv_image = None
        self.depth_image = None
        self.cam_info = None
        self.ore_positions = {}

    def depthimagecb(self, data):
        '''
        Purpose:
        ---
        Receives the depth image from the camera and converts it into
        a NumPy array for further processing.

        Input Arguments:
        ---
        `data` :  [ sensor_msgs/msg/Image ]
            Depth image received from the camera topic.

        Returns:
        ---
        None

        Example call:
        ---
        Called by the ROS executor whenever a depth image is received.
        '''

        try:
            depth = self.bridge.imgmsg_to_cv2(data, desired_encoding='passthrough')

            if not hasattr(self, '_logged_depth_dtype'):
                self.get_logger().info(f'Depth dtype: {depth.dtype}')
                self._logged_depth_dtype = True

            if depth.dtype == np.uint16:
                depth = depth.astype(np.float32) / 1000.0  # mm -> m

            self.depth_image = depth
        except CvBridgeError as e:
            self.get_logger().error(f'Depth image conversion failed: {e}')

    def colorimagecb(self, data):
        '''
        Purpose:
        ---
        Receives the colour image from the camera and converts it into
        an OpenCV BGR image.

        Input Arguments:
        ---
        `data` :  [ sensor_msgs/msg/Image ]
            Colour image received from the camera topic.

        Returns:
        ---
        None

        Example call:
        ---
        Called by the ROS executor whenever a colour image is received.
        '''
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Color image conversion failed: {e}')

    def caminfocb(self, data):
        '''
        Purpose:
        ---
        Receives camera calibration information and stores the camera
        intrinsic parameters required for 3D point calculation.

        Input Arguments:
        ---
        `data` :  [ sensor_msgs/msg/CameraInfo ]
            Camera calibration information.

        Returns:
        ---
        None

        Example call:
        ---
        Called by the ROS executor whenever camera information is received.
        '''

        self.cam_info = data.k
        self.fx = data.k[0]
        self.fy = data.k[4]
        self.cx = data.k[2]
        self.cy = data.k[5]

    def process_image(self):
        '''
        Purpose:
        ---
        Processes the latest colour and depth images, detects ores,
        calculates their 3D positions and broadcasts their TF frames
        with respect to the robot base frame.

        Input Arguments:
        ---
        None

        Returns:
        ---
        None

        Example call:
        ---
        Called automatically by the ROS timer at the configured
        image processing rate.
        '''

        if self.cv_image is None:
            self.get_logger().info('Waiting for color image...')
            return

        display = self.cv_image.copy()
        center_ore_list, ore_type_list = detect_ores(self.cv_image)

        # Group detections by type
        by_type = {}
        for (cX, cY), ore_type in zip(center_ore_list, ore_type_list):
            by_type.setdefault(ore_type, []).append((cX, cY))

        # Assign stable IDs for every detection, BEFORE drawing or broadcasting
        all_assignments = []  # list of (best_id, cX, cY)
        for ore_type, dets in by_type.items():
            candidate_ids = [f'{ore_type}_1', f'{ore_type}_2']

            if len(dets) == 1:
                (cX, cY) = dets[0]
                best_id, best_dist = None, float('inf')
                for oid in candidate_ids:
                    if oid in self.ore_positions:
                        px, py = self.ore_positions[oid]
                        dist = (px - cX) ** 2 + (py - cY) ** 2
                        if dist < best_dist:
                            best_dist, best_id = dist, oid
                if best_id is None:
                    best_id = candidate_ids[0]
                assignment = [(best_id, (cX, cY))]

            elif len(dets) >= 2:
                d1, d2 = dets[0], dets[1]
                known = [oid for oid in candidate_ids if oid in self.ore_positions]

                if len(known) == 2:
                    id1, id2 = candidate_ids
                    p1, p2 = self.ore_positions[id1], self.ore_positions[id2]
                    cost_a = (p1[0] - d1[0])**2 + (p1[1] - d1[1])**2 + \
                        (p2[0] - d2[0])**2 + (p2[1] - d2[1])**2
                    cost_b = (p2[0] - d1[0])**2 + (p2[1] - d1[1])**2 + \
                        (p1[0] - d2[0])**2 + (p1[1] - d2[1])**2
                    assignment = [
                        (id1, d1), (id2, d2)] if cost_a <= cost_b else [
                        (id2, d1), (id1, d2)]
                elif len(known) == 1:
                    known_id = known[0]
                    other_id = candidate_ids[1] if known_id == candidate_ids[0] else candidate_ids[0]
                    p = self.ore_positions[known_id]
                    dist1 = (p[0] - d1[0])**2 + (p[1] - d1[1])**2
                    dist2 = (p[0] - d2[0])**2 + (p[1] - d2[1])**2
                    assignment = [
                        (known_id, d1), (other_id, d2)] if dist1 <= dist2 else [
                        (known_id, d2), (other_id, d1)]
                else:
                    assignment = [(candidate_ids[0], d1), (candidate_ids[1], d2)]
            else:
                assignment = []

            for best_id, (cX, cY) in assignment:
                self.ore_positions[best_id] = (cX, cY)
                all_assignments.append((best_id, cX, cY))

        # Draw using the ASSIGNED ids, matching what gets broadcast
        for best_id, cX, cY in all_assignments:
            cv2.circle(display, (cX, cY), 8, (255, 255, 255), 2)
            cv2.putText(display, best_id, (cX - 40, cY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('color_feed', display)
        cv2.waitKey(1)

        if not hasattr(self, '_saved_detection_image') and len(all_assignments) == 6:
            ok = cv2.imwrite('SC#1449_task1A_detection.png', display)
            self._saved_detection_image = True
            self.get_logger().info(f'Detection image save result: {ok}')

        # Broadcast TF using the same all_assignments
        if self.depth_image is not None and hasattr(self, 'fx'):
            cam_frame_id = 'camera_color_optical_frame'
            now = self.get_clock().now().to_msg()

            for best_id, cX, cY in all_assignments:
                patch = self.depth_image[max(cY - 4, 0):cY + 5, max(cX - 4, 0):cX + 5]
                valid = patch[np.isfinite(patch) & (patch > 0.0)]
                if valid.size == 0:
                    continue
                z = float(np.median(valid))
                x = (cX - self.cx) * z / self.fx
                y = (cY - self.cy) * z / self.fy

                point_in_camera = PointStamped()
                point_in_camera.header.frame_id = cam_frame_id
                point_in_camera.header.stamp = rclpy.time.Time().to_msg()
                point_in_camera.point.x = x
                point_in_camera.point.y = y
                point_in_camera.point.z = z

                try:
                    tf = self.tf_buffer.lookup_transform(
                        base_frame, cam_frame_id, rclpy.time.Time())
                    point_in_base = do_transform_point(point_in_camera, tf)
                except Exception as e:
                    self.get_logger().warn(f'TF lookup failed for {best_id}: {e}')
                    continue

                bx, by, bz = point_in_base.point.x, point_in_base.point.y, point_in_base.point.z

                t = TransformStamped()
                t.header.stamp = now
                t.header.frame_id = base_frame
                t.child_frame_id = best_id
                t.transform.translation.x = float(bx)
                t.transform.translation.y = float(by)
                t.transform.translation.z = float(bz)
                t.transform.rotation.w = 1.0
                self.br.sendTransform(t)


def main():
    '''
    Purpose:
    ---
    Initializes ROS, creates the ore detection node and starts
    the ROS event loop.

    Input Arguments:
    ---
    None

    Returns:
    ---
    None

    Example call:
    ---
    Called automatically by the Operating System.
    '''

    rclpy.init(args=sys.argv)
    node = rclpy.create_node('ore_tf_process')
    node.get_logger().info('Node created: Ore tf process')
    ore_tf_class = ore_tf()
    rclpy.spin(ore_tf_class)
    ore_tf_class.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':

    main()
