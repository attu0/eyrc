#!/usr/bin/env python3

'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 3B of StrataCobot (SC) Theme (eYRC 2026-27).
*
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:          [ 1449 ]
# Author List:		[ Atharv Mudse, Abhay Potdar, Riddhi Wagh, Yash Waje ]
# Filename:		    task3B_planning.py
# Functions:
# [ mapcb, odomcb, scancb, world_to_cell, cell_to_world,
#                     compute_clearance, cell_free, point_free, segment_free, astar, shortcut,
#                     densify, plan_path, publish_path, request_ore_package, taken_cb,
#                     end_reached, process_planning, main ]
# Nodes:		    ebot_planner_node
# Publishing Topics  - [ /ebot_path, /handover/ebot_ready ]
#                   Subscribing Topics - [ /map, /odom, /scan, /handover/package_taken ]
#                   Service Clients    - [ /spawn_ore_package ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
import heapq
import time
from rclpy.node import Node
from rclpy.qos import (qos_profile_sensor_data, QoSProfile, QoSHistoryPolicy,
                       QoSReliabilityPolicy, QoSDurabilityPolicy)
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


##################### TASK CONSTANTS #######################

map_topic = '/map'
path_topic = '/ebot_path'
odom_topic = '/odom'
scan_topic = '/scan'
spawn_service = '/spawn_ore_package'

ready_topic = '/handover/ebot_ready'
taken_topic = '/handover/package_taken'

map_frame = 'map'

home_pose = (0.0, 0.0, 0.0)
ore_drop_pose = (6.0, 1.9, 3.14)
arm_pose = (0.0, 1.9, 3.14)

steps = [
    (home_pose, ore_drop_pose),     # step 1: then request the ore package
    (ore_drop_pose, arm_pose),      # step 2
    (arm_pose, home_pose),          # step 3
]

step_yaws = [
    (0.0, 3.14),
    (3.14, 3.14),
    (3.14, 0.0),
]

end_yaw_options = [
    (0.0, 3.14),        # ore drop pose: either
    (0.0, 3.14),        # arm pose: either
    (0.0,),             # home: 0.0 only
]


##################### PLANNER TUNING #######################

clearance_levels = (0.32, 0.29, 0.25, 0.22)

soft_margin = 0.35
soft_weight = 0.40

endpoint_exempt = 0.38
path_spacing = 0.25
planning_res = 0.020
occupied_threshold = 50

# Matches the task's 0.30 m / 0.15 rad arrival rule while waiting 0.25 s of stillness
# so ebot_nav finishes its final yaw alignment first.
end_pos_tol = 0.25
start_pos_tol = 0.35
end_yaw_tol = 0.12
still_lin = 0.075
still_ang = 0.105
still_hold = 0.40               # seconds the eBot must stay still
ore_settle_time = 0.35          # seconds to let the package settle on the deck
ore_retry_period = 0.25         # seconds between /spawn_ore_package attempts

ODOM_EMA_ALPHA = 0.40

ARM_STRUCT_X = 0.0
ARM_STRUCT_Y = 2.45
ARM_STRUCT_R = 0.45


##################### CLASS DEFINITION #######################

class ebot_planner(Node):
    '''
    ___CLASS___

    Description:    Reads the map, fuses live /scan obstacle observations to handle map-world
                    mismatches, plans a path for each step, publishes it on /ebot_path, and
                    requests the ore package at the end of step 1.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ebot_planner
        '''

        super().__init__(
            'ebot_planner_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        ############ Topic SUBSCRIPTIONS ############

        self.map_sub = self.create_subscription(OccupancyGrid, map_topic, self.mapcb, latched)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scancb, qos_profile_sensor_data)

        ############ Topic PUBLISHERS ############

        self.path_pub = self.create_publisher(Path, path_topic, latched)

        ############ Service CLIENTS ############

        self.spawn_client = self.create_client(Trigger, spawn_service)

        self.ready_pub = self.create_publisher(Bool, ready_topic, latched)
        self.taken_sub = self.create_subscription(Bool, taken_topic, self.taken_cb, latched)

        ############ Constructor VARIABLES/OBJECTS ############

        plan_rate = 0.04
        self.timer = self.create_timer(plan_rate, self.process_planning)

        self.grid = None
        self.odom = None
        self.step = 0

        self.width = 0
        self.height = 0
        self.res = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.occ = []
        self.scan_votes = []
        self.clear = []
        self.map_dirty = False

        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.still_since = None
        self.lin_speed = 0.0
        self.ang_speed = 0.0
        self.pose_hist = []

        self.stage = 'plan'
        self.ore_pending = False
        self.ore_done = False
        self.ore_next_try = 0.0
        self.settle_until = 0.0
        self.package_taken = False

    def now_sec(self):
        '''
        Description:    Current (simulation) time in seconds.
        '''

        return self.get_clock().now().nanoseconds * 1e-9

    def is_still(self):
        '''
        Description:    True when the eBot has been standing still for still_hold seconds.
        '''

        now = self.now_sec()
        if self.still_since is not None and (now - self.still_since) >= still_hold:
            return True

        recent = [(t, px, py, pyaw) for t, px, py, pyaw in self.pose_hist if now - t <= 0.30]
        if len(recent) >= 6 and (now - recent[0][0]) >= 0.22:
            xs = [p[1] for p in recent]
            ys = [p[2] for p in recent]
            yaws = [p[3] for p in recent]
            dyaw = max(abs(math.atan2(math.sin(y - yaws[0]), math.cos(y - yaws[0]))) for y in yaws)
            if math.hypot(max(xs) - min(xs), max(ys) - min(ys)) < 0.020 and dyaw < 0.030:
                return True

        return False

    def mapcb(self, data):
        '''
        Description:    Callback function for the map topic.
        '''

        try:
            fine_w = data.info.width
            fine_h = data.info.height
            fine_res = data.info.resolution
            factor = max(1, int(round(planning_res / fine_res)))

            self.width = -(-fine_w // factor)
            self.height = -(-fine_h // factor)
            self.res = fine_res * factor
            self.origin_x = data.info.origin.position.x
            self.origin_y = data.info.origin.position.y

            started = time.time()

            self.occ = [False] * (self.width * self.height)
            self.scan_votes = [0] * (self.width * self.height)
            rocks = 0
            for i, v in enumerate(data.data):
                if v >= occupied_threshold:
                    row, col = divmod(i, fine_w)
                    idx = (row // factor) * self.width + (col // factor)
                    self.occ[idx] = True
                    self.scan_votes[idx] = 2
                    rocks += 1

            self.clear = self.compute_clearance()
            self.grid = data

            self.get_logger().info(
                f'Map received: {fine_w} x {fine_h} cells at {fine_res:.3f} m, origin '
                f'({self.origin_x:.2f}, {self.origin_y:.2f}), {rocks} rock cells. Planning grid '
                f'{self.width} x {self.height} at {self.res:.3f} m, built in {time.time() - started:.2f} s')

        except Exception as error:
            self.get_logger().error(f'Map callback error: {error}')

    def odomcb(self, data):
        '''
        Description:    Callback function for the odometry topic with EMA noise filtering.
        '''

        try:
            p = data.pose.pose.position
            q = data.pose.pose.orientation
            raw_x = p.x
            raw_y = p.y
            raw_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

            raw_lin = math.hypot(data.twist.twist.linear.x, data.twist.twist.linear.y)
            raw_ang = abs(data.twist.twist.angular.z)

            if self.odom is None:
                self.x = raw_x
                self.y = raw_y
                self.yaw = raw_yaw
                self.lin_speed = raw_lin
                self.ang_speed = raw_ang
            else:
                a = ODOM_EMA_ALPHA
                self.x = a * raw_x + (1.0 - a) * self.x
                self.y = a * raw_y + (1.0 - a) * self.y
                dyaw = math.atan2(math.sin(raw_yaw - self.yaw), math.cos(raw_yaw - self.yaw))
                self.yaw = math.atan2(math.sin(self.yaw + a * dyaw), math.cos(self.yaw + a * dyaw))
                self.lin_speed = a * raw_lin + (1.0 - a) * self.lin_speed
                self.ang_speed = a * raw_ang + (1.0 - a) * self.ang_speed

            now = self.now_sec()
            self.pose_hist.append((now, self.x, self.y, self.yaw))
            self.pose_hist = [(t, px, py, pyaw)
                              for t, px, py, pyaw in self.pose_hist if now - t <= 0.45]

            if self.lin_speed < still_lin and self.ang_speed < still_ang:
                if self.still_since is None:
                    self.still_since = now
            else:
                self.still_since = None

            self.odom = data

        except Exception as error:
            self.get_logger().error(f'Odometry callback error: {error}')

    def scancb(self, msg):
        '''
        Description:    Integrates LaserScan readings into the occupancy grid while the eBot is
                        moving slowly or stopped so unmapped rocks are added and phantom mapped
                        rocks are cleared before the next leg is planned.
        '''

        if self.grid is None or self.odom is None:
            return

        if self.ang_speed > 0.25:
            return

        try:
            rx, ry, ryaw = self.x, self.y, self.yaw
            angle = msg.angle_min
            inc = msg.angle_increment
            rmin = max(0.25, msg.range_min)
            rmax = min(6.5, msg.range_max)
            w = self.width
            changed = False

            for i in range(0, len(msg.ranges), 2):
                r = msg.ranges[i]
                ray_angle = ryaw + angle + i * inc

                if not math.isfinite(r) or r < rmin:
                    continue

                hit = r < (rmax - 0.10)
                clear_dist = min(r - 0.10, 3.2)

                step_d = self.res * 1.5
                d = 0.30
                c_cos = math.cos(ray_angle)
                c_sin = math.sin(ray_angle)
                while d < clear_dist:
                    cell = self.world_to_cell(rx + d * c_cos, ry + d * c_sin)
                    if cell is not None:
                        idx = cell[1] * w + cell[0]
                        if self.scan_votes[idx] > -3:
                            self.scan_votes[idx] -= 1
                            if self.occ[idx] and self.scan_votes[idx] <= -2:
                                self.occ[idx] = False
                                changed = True
                    d += step_d

                if hit and r < 6.0:
                    if i + 1 < len(msg.ranges) and abs(msg.ranges[i + 1] - r) > 0.16:
                        continue

                    hx = rx + r * c_cos
                    hy = ry + r * c_sin

                    if math.hypot(hx - ARM_STRUCT_X, hy - ARM_STRUCT_Y) < ARM_STRUCT_R:
                        continue
                    if any(math.hypot(hx - gx, hy - gy) < 0.35 for gx,
                           gy, _ in (home_pose, ore_drop_pose, arm_pose)):
                        continue

                    cell = self.world_to_cell(hx, hy)
                    if cell is not None:
                        idx = cell[1] * w + cell[0]
                        if self.scan_votes[idx] < 6:
                            self.scan_votes[idx] += 2
                        if not self.occ[idx] and self.scan_votes[idx] >= 2:
                            self.occ[idx] = True
                            changed = True

            if changed:
                self.map_dirty = True

        except Exception as error:
            self.get_logger().error(f'Scan callback error: {error}')

    ##################### GRID HELPERS #######################

    def world_to_cell(self, x, y):
        '''
        Description:    Converts map coordinates to a (col, row) cell.
        '''

        col = int(math.floor((x - self.origin_x) / self.res))
        row = int(math.floor((y - self.origin_y) / self.res))

        if 0 <= col < self.width and 0 <= row < self.height:
            return (col, row)

        return None

    def cell_to_world(self, col, row):
        '''
        Description:    Centre of a cell in map coordinates.
        '''

        return (self.origin_x + (col + 0.5) * self.res,
                self.origin_y + (row + 0.5) * self.res)

    def compute_clearance(self):
        '''
        Description:    Exact Euclidean distance transform (Felzenszwalb-Huttenlocher).
        '''

        big = 1e12
        w, h = self.width, self.height

        def edt_1d(f):
            n = len(f)
            d = [0.0] * n
            v = [0] * n
            z = [0.0] * (n + 1)
            k = 0
            z[0] = -math.inf
            z[1] = math.inf
            for q in range(1, n):
                s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
                while s <= z[k]:
                    k -= 1
                    s = ((f[q] + q * q) - (f[v[k]] + v[k] * v[k])) / (2.0 * q - 2.0 * v[k])
                k += 1
                v[k] = q
                z[k] = s
                z[k + 1] = math.inf
            k = 0
            for q in range(n):
                while z[k + 1] < q:
                    k += 1
                d[q] = (q - v[k]) ** 2 + f[v[k]]
            return d

        grid = [0.0 if self.occ[i] else big for i in range(w * h)]

        for col in range(w):
            column = edt_1d([grid[row * w + col] for row in range(h)])
            for row in range(h):
                grid[row * w + col] = column[row]

        for row in range(h):
            line = edt_1d(grid[row * w:(row + 1) * w])
            grid[row * w:(row + 1) * w] = line

        half = 0.5 * self.res
        return [max(0.0, math.sqrt(v) * self.res - half) if v < big else 1e6 for v in grid]

    def cell_free(self, col, row, hard_r, ends):
        '''
        Description:    Whether the eBot's centre may stand in this cell.
        '''

        if not (0 <= col < self.width and 0 <= row < self.height):
            return False

        i = row * self.width + col

        if self.occ[i]:
            return False

        if self.clear[i] >= hard_r:
            return True

        cx, cy = self.cell_to_world(col, row)
        return any(math.hypot(cx - ex, cy - ey) < endpoint_exempt for ex, ey in ends)

    def point_free(self, x, y, hard_r, ends):
        '''
        Description:    cell_free() for a position in metres.
        '''

        cell = self.world_to_cell(x, y)
        return cell is not None and self.cell_free(cell[0], cell[1], hard_r, ends)

    def segment_free(self, p, q, hard_r, ends):
        '''
        Description:    Whether the straight line p -> q keeps the required clearance.
        '''

        dist = math.hypot(q[0] - p[0], q[1] - p[1])
        n = max(1, int(math.ceil(dist / (0.5 * self.res))))

        for i in range(n + 1):
            t = i / n
            if not self.point_free(p[0] + t * (q[0] - p[0]), p[1] +
                                   t * (q[1] - p[1]), hard_r, ends):
                return False

        return True

    ##################### SEARCH #######################

    def astar(self, start, end, hard_r):
        '''
        Description:    A* over the 8-connected grid.
        '''

        s = self.world_to_cell(start[0], start[1])
        g = self.world_to_cell(end[0], end[1])

        if s is None or g is None:
            return None

        ends = ((start[0], start[1]), (end[0], end[1]))
        w = self.width

        self.occ[s[1] * w + s[0]] = False
        self.occ[g[1] * w + g[0]] = False

        soft_r = hard_r + soft_margin
        neighbours = [(1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                      (1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)),
                      (-1, 1, math.sqrt(2)), (-1, -1, math.sqrt(2))]

        best = {s: 0.0}
        parent = {}
        closed = set()
        heap = [(0.0, 0.0, s)]

        while heap:
            _, cost, cell = heapq.heappop(heap)

            if cell in closed:
                continue
            closed.add(cell)

            if cell == g:
                path = [cell]
                while cell in parent:
                    cell = parent[cell]
                    path.append(cell)
                path.reverse()
                return path

            for dc, dr, length in neighbours:
                nc, nr = cell[0] + dc, cell[1] + dr

                if not self.cell_free(nc, nr, hard_r, ends):
                    continue

                if dc != 0 and dr != 0:
                    if not (self.cell_free(cell[0] + dc, cell[1], hard_r, ends)
                            and self.cell_free(cell[0], cell[1] + dr, hard_r, ends)):
                        continue

                clearance = self.clear[nr * w + nc]
                penalty = max(0.0, (soft_r - clearance) / soft_r)
                new_cost = cost + length * self.res * (1.0 + soft_weight * penalty)

                if new_cost < best.get((nc, nr), math.inf):
                    best[(nc, nr)] = new_cost
                    parent[(nc, nr)] = cell
                    heuristic = math.hypot(g[0] - nc, g[1] - nr) * self.res
                    heapq.heappush(heap, (new_cost + heuristic, new_cost, (nc, nr)))

        return None

    def shortcut(self, pts, hard_r, ends):
        '''
        Description:    Thins out a cell path via line-of-sight shortcutting.
        '''

        def pull(seq):
            out = [seq[0]]
            i = 0
            while i < len(seq) - 1:
                j = len(seq) - 1
                while j > i + 1 and not self.segment_free(seq[i], seq[j], hard_r, ends):
                    j -= 1
                out.append(seq[j])
                i = j
            return out

        def length(seq):
            return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(seq[:-1], seq[1:]))

        forward = pull(pts)
        backward = pull(pts[::-1])[::-1]

        return forward if length(forward) <= length(backward) else backward

    def densify(self, pts):
        '''
        Description:    Adds points along each straight segment so poses are at most
                        'path_spacing' apart.
        '''

        out = [pts[0]]

        for a, b in zip(pts[:-1], pts[1:]):
            dist = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, int(math.ceil(dist / path_spacing)))
            for k in range(1, n + 1):
                t = k / n
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))

        out[-1] = pts[-1]

        return out

    def plan_path(self, start, end):
        '''
        Description:    Plans a path from 'start' to 'end' that stays clear of every rock.
        '''

        points = None

        try:
            if self.map_dirty:
                self.clear = self.compute_clearance()
                self.map_dirty = False

            ends = ((start[0], start[1]), (end[0], end[1]))

            for hard_r in clearance_levels:
                cells = self.astar(start, end, hard_r)

                if cells is None:
                    self.get_logger().warn(f'No path with {hard_r:.2f} m clearance, trying less')
                    continue

                pts = [self.cell_to_world(c, r) for c, r in cells]
                pts[0] = (start[0], start[1])
                pts[-1] = (end[0], end[1])

                pts = self.shortcut(pts, hard_r, ends)
                points = self.densify(pts)

                length = sum(math.hypot(b[0] - a[0], b[1] - a[1])
                             for a, b in zip(points[:-1], points[1:]))
                self.get_logger().info(
                    f'Planned step {self.step + 1}: clearance {hard_r:.2f} m, '
                    f'{len(points)} poses, length {length:.2f} m')
                break

        except Exception as error:
            self.get_logger().error(f'Planning error: {error}')
            points = None

        return points

    def publish_path(self, points, start, end):
        '''
        Description:    Publishes 'points' on /ebot_path as a nav_msgs/Path.
        '''

        path = Path()
        path.header.frame_id = map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        start_yaw, end_yaw = step_yaws[self.step]
        last = len(points) - 1

        for i, (x, y) in enumerate(points):
            if i == 0:
                yaw = start_yaw
            elif i == last:
                yaw = end_yaw
            else:
                nx, ny = points[i + 1]
                yaw = math.atan2(ny - y, nx - x)

            pose = PoseStamped()
            pose.header.frame_id = map_frame
            pose.header.stamp = path.header.stamp
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            path.poses.append(pose)

        self.path_pub.publish(path)
        self.get_logger().info(
            f'[EVENT] EBOT PATH step {
                self.step +
                1}: start=({
                points[0][0]:.2f}, {
                points[0][1]:.2f}) '
            f'end=({points[-1][0]:.2f}, {points[-1][1]:.2f}) t={self.now_sec():.1f}s')
        self.get_logger().info(f'Published path for step {self.step + 1} ({len(points)} poses)')

    def request_ore_package(self):
        '''
        Description:    Calls /spawn_ore_package once the eBot stands still on the ore drop pose.
        '''

        if not self.spawn_client.service_is_ready():
            self.get_logger().warn('/spawn_ore_package not available yet', throttle_duration_sec=5.0)
            self.ore_next_try = self.now_sec() + ore_retry_period
            return

        self.ore_pending = True
        future = self.spawn_client.call_async(Trigger.Request())
        future.add_done_callback(self.ore_reply)

    def ore_reply(self, future):
        '''
        Description:    Handles the reply of /spawn_ore_package.
        '''

        try:
            result = future.result()
            self.get_logger().info(
                f'Ore package reply: success={
                    result.success}, message="{
                    result.message}"')

            if result.success:
                self.ore_done = True
            else:
                self.ore_next_try = self.now_sec() + ore_retry_period

        except Exception as error:
            self.get_logger().error(f'Ore package call failed: {error}')
            self.ore_next_try = self.now_sec() + ore_retry_period

        finally:
            self.ore_pending = False

    def taken_cb(self, msg):
        '''
        Description:    Callback for /handover/package_taken: the arm has lifted the package off the deck.
        '''

        if msg.data:
            self.package_taken = True

    def end_reached(self, end, options):
        '''
        Description:    Whether the eBot stands still at the end pose of the current step.
        '''

        if math.hypot(self.x - end[0], self.y - end[1]) > end_pos_tol:
            return False

        if not any(abs(math.atan2(math.sin(self.yaw - o), math.cos(self.yaw - o))) <= end_yaw_tol
                   for o in options):
            return False

        return self.is_still()

    def process_planning(self):
        '''
        Description:    Timer function used to plan, publish and request, step by step.
        '''

        try:
            if self.grid is None or self.odom is None:
                self.get_logger().info(
                    f'Waiting for map ({
                        self.grid is not None}) and odometry ({
                        self.odom is not None})',
                    throttle_duration_sec=3.0)
                return

            if self.stage in ('done', 'failed'):
                return

            if self.step >= len(steps):
                self.stage = 'done'
                self.get_logger().info('Lap complete: all three steps driven')
                return

            start, end = steps[self.step]

            if self.stage == 'plan':
                if not self.is_still():
                    self.get_logger().info(
                        f'Waiting for the eBot to stand still before planning step {self.step + 1} '
                        f'(odom velocity: lin {self.lin_speed:.3f} m/s, ang {self.ang_speed:.3f} rad/s)',
                        throttle_duration_sec=3.0)
                    return

                if math.hypot(self.x - start[0], self.y - start[1]) > start_pos_tol:
                    self.get_logger().warn(
                        f'Not at the start of step {self.step + 1} yet', throttle_duration_sec=5.0)
                    return

                points = self.plan_path(start, end)

                if points is None:
                    self.get_logger().error(f'No path found for step {self.step + 1}')
                    self.stage = 'failed'
                    return

                self.publish_path(points, start, end)
                self.stage = 'drive'

            elif self.stage == 'drive':
                if self.end_reached(end, end_yaw_options[self.step]):
                    self.get_logger().info(f'Step {self.step + 1} finished')
                    err = math.hypot(self.x - end[0], self.y - end[1])
                    self.get_logger().info(
                        f'[EVENT] EBOT REACHED step {
                            self.step +
                            1}: pos=({
                            self.x:.3f}, {
                            self.y:.3f}) '
                        f'yaw={self.yaw:.3f} | goal=({end[0]:.2f}, {end[1]:.2f}) | '
                        f'pos_err={err:.3f} m | t={self.now_sec():.1f}s')

                    if self.step == 0:
                        self.request_ore_package()
                        self.stage = 'ore'
                    elif self.step == 1:
                        ready = Bool()
                        ready.data = True
                        self.ready_pub.publish(ready)
                        self.stage = 'handover'
                        self.get_logger().info('Beside the arm: waiting for it to take the package')
                    else:
                        self.step += 1
                        self.stage = 'plan'

            elif self.stage == 'ore':
                if self.ore_done:
                    self.settle_until = self.now_sec() + ore_settle_time
                    self.stage = 'settle'
                elif not self.ore_pending and self.now_sec() >= self.ore_next_try:
                    self.request_ore_package()

            elif self.stage == 'settle':
                if self.now_sec() >= self.settle_until:
                    self.step += 1
                    self.stage = 'plan'

            elif self.stage == 'handover':
                if self.package_taken:
                    self.step += 1
                    self.stage = 'plan'

        except Exception as error:
            self.get_logger().error(f'Planning cycle error: {error}')


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_planner class to perform its task
    '''

    rclpy.init(args=sys.argv)

    ebot_planner_class = ebot_planner()

    try:
        rclpy.spin(ebot_planner_class)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            ebot_planner_class.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':

    main()
