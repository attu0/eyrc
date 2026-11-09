#!/usr/bin/env python3


'''
*****************************************************************************************
*
*        		===============================================
*           		        StrataCobot (SC) Theme (eYRC 2026-27)
*        		===============================================
*
*  This script should be used to implement Task 3A of StrataCobot (SC) Theme (eYRC 2026-27).
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
# Filename:		    task3A_planning.py
# Functions:
#			        [ mapcb, odomcb, world_to_cell, cell_to_world, compute_clearance,
#                     cell_free, point_free, segment_free, astar, shortcut, densify,
#                     plan_path, publish_path, request_ore_package, taken_cb, end_reached,
#                     process_planning, main ]
# Nodes:		    ebot_planner_node
#			        Publishing Topics  - [ /ebot_path, /handover/ebot_ready ]
#                   Subscribing Topics - [ /map, /odom, /handover/package_taken ]
#                   Service Clients    - [ /spawn_ore_package ]


################### IMPORT MODULES #######################

import rclpy
import sys
import math
import heapq
import time
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy, QoSDurabilityPolicy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


##################### TASK CONSTANTS #######################

# The map is published ONCE, as the world loads, and held for late subscribers.
map_topic = '/map'

# Your follower reads the path from here, with the same held-message QoS.
path_topic = '/ebot_path'

odom_topic = '/odom'
spawn_service = '/spawn_ore_package'

# Handover with the arm (latched, so a late subscriber still gets the last value).
ready_topic = '/handover/ebot_ready'            # eBot -> arm: stopped beside the arm, package on deck
taken_topic = '/handover/package_taken'         # arm -> eBot: package lifted off the deck, you may leave

# Every pose in the path, and the path itself, is in this frame.
map_frame = 'map'

# The three points of the lap: (x, y) in metres and yaw in radians, in the map frame.
# At the ore drop pose and the arm pose either yaw, 0.0 or 3.14, is valid.
home_pose = (0.0, 0.0, 0.0)
ore_drop_pose = (6.0, 1.9, 0.0)
# CALIBRATE: this is the stop pose beside the arm. It must put the package inside the arm's reach
# and the camera's view, and outside the space the arm sweeps while sorting ores.
arm_pose = (0.0, 1.9, 0.0)

# The lap, as (start, end) of each step, in the order they are driven.
steps = [
    (home_pose, ore_drop_pose),     # step 1: then request the ore package
    (ore_drop_pose, arm_pose),      # step 2
    (arm_pose, home_pose),          # step 3
]

# Yaw written into the first and last pose of each step: (start_yaw, end_yaw).
# Step 2 runs back across the arena, so it ends facing 3.14 (the nearer valid yaw).
step_yaws = [
    (0.0, 0.0),
    (0.0, 3.14),
    (3.14, 0.0),
]

# Yaws that count as "arrived" at the end of each step (radians).
end_yaw_options = [
    (0.0, 3.14),        # ore drop pose: either
    (0.0, 3.14),        # arm pose: either
    (0.0,),             # home: 0.0 only
]


##################### PLANNER TUNING #######################

# 0.32 m keeps the eBot well clear of rocks while taking shorter, more direct gaps on the first try.
clearance_levels = (0.32, 0.28, 0.24)

# Beyond the hard clearance the search is still nudged away from rocks over this extra
# distance (m), so that it picks the roomier side of a gap.
soft_margin = 0.35
soft_weight = 0.0               # 0 = pure shortest path (best EP).

# Around the start and end points the clearance rule is relaxed (m) so a rock near
# the pose can never make the step unplannable. Occupied cells stay forbidden.
endpoint_exempt = 0.35

# Distance between neighbouring poses of the published path (m).
path_spacing = 0.25

# Coarser planning grid (0.02 m) builds EDT and runs A* ~3x faster at startup without losing accuracy.
planning_res = 0.020

# A cell with at least this value is a rock.
occupied_threshold = 50

# Step-complete test, within the task's 0.3 m / 0.15 rad.
end_pos_tol = 0.26
end_yaw_tol = 0.13
still_lin = 0.025
still_ang = 0.025
still_hold = 0.12               # seconds the eBot must stay still
ore_settle_time = 0.45          # seconds to let the package settle on the deck
ore_retry_period = 0.30         # seconds between /spawn_ore_package attempts


##################### CLASS DEFINITION #######################

class ebot_planner(Node):
    '''
    ___CLASS___

    Description:    Class which serves the purpose to read the map, plan a path for each
                    step of the lap, publish it on /ebot_path, and request the ore package
                    at the end of step 1.
    '''

    def __init__(self):
        '''
        Description:    Initialization of class ebot_planner
        '''

        super().__init__(                                                               # registering node
            'ebot_planner_node',
            parameter_overrides=[rclpy.parameter.Parameter(
                'use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # A held message: kept by the publisher and handed to anyone who subscribes later.
        latched = QoSProfile(
            depth=1,
            history=QoSHistoryPolicy.KEEP_LAST,
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)

        ############ Topic SUBSCRIPTIONS ############

        self.map_sub = self.create_subscription(OccupancyGrid, map_topic, self.mapcb, latched)
        self.odom_sub = self.create_subscription(Odometry, odom_topic, self.odomcb, 10)

        ############ Topic PUBLISHERS ############

        self.path_pub = self.create_publisher(Path, path_topic, latched)                # your follower drives what you publish here

        ############ Service CLIENTS ############

        self.spawn_client = self.create_client(Trigger, spawn_service)                  # drops the ore package onto the eBot

        self.ready_pub = self.create_publisher(Bool, ready_topic, latched)
        self.taken_sub = self.create_subscription(Bool, taken_topic, self.taken_cb, latched)

        ############ Constructor VARIABLES/OBJECTS ############

        plan_rate = 0.04                                                                # 25 Hz planning state machine
        self.timer = self.create_timer(plan_rate, self.process_planning)

        self.grid = None                                                                # the map (from mapcb())
        self.odom = None                                                                # where the base is (from odomcb())
        self.step = 0                                                                   # which step of the lap is next (index into 'steps')

        ############ ADD YOUR CODE HERE ############

        # Grid description (filled in by mapcb)
        self.width = 0
        self.height = 0
        self.res = 0.05
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.occ = []                                                                   # flat list of bool, True = rock
        self.clear = []                                                                 # flat list of float, metres to the nearest rock

        # Base state (filled in by odomcb)
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.still_since = None                                                         # time (s) from which the base has been still
        self.lin_speed = 0.0
        self.ang_speed = 0.0

        # Step state machine: 'plan' -> 'drive' -> ('ore' -> 'settle') -> next step
        self.stage = 'plan'
        self.ore_pending = False                                                        # a /spawn_ore_package call is in flight
        self.ore_done = False                                                           # the package has been delivered
        self.ore_next_try = 0.0
        self.settle_until = 0.0
        self.package_taken = False                                                      # the arm has lifted the package off the deck

        ############################################


    def now_sec(self):
        '''
        Description:    Current (simulation) time in seconds.

        Args:
        Returns:
            float:      time in seconds
        '''

        return self.get_clock().now().nanoseconds * 1e-9


    def mapcb(self, data):
        '''
        Description:    Callback function for the map topic.
                        Use this function to receive the occupancy grid of the arena.

        Args:
            data (OccupancyGrid):    The arena, one value per cell

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:
            fine_w = data.info.width
            fine_h = data.info.height
            fine_res = data.info.resolution
            factor = max(1, int(round(planning_res / fine_res)))                        # fine cells per coarse cell

            self.width = -(-fine_w // factor)                                           # ceil division
            self.height = -(-fine_h // factor)
            self.res = fine_res * factor
            self.origin_x = data.info.origin.position.x
            self.origin_y = data.info.origin.position.y

            started = time.time()

            self.occ = [False] * (self.width * self.height)                             # every cell is free (0) or occupied (100)
            rocks = 0
            for i, v in enumerate(data.data):
                if v >= occupied_threshold:
                    row, col = divmod(i, fine_w)
                    self.occ[(row // factor) * self.width + (col // factor)] = True
                    rocks += 1

            self.clear = self.compute_clearance()
            self.grid = data

            self.get_logger().info(
                f'Map received: {fine_w} x {fine_h} cells at {fine_res:.3f} m, origin '
                f'({self.origin_x:.2f}, {self.origin_y:.2f}), {rocks} rock cells. Planning grid '
                f'{self.width} x {self.height} at {self.res:.3f} m, built in {time.time() - started:.2f} s')

        except Exception as error:
            self.get_logger().error(f'Map callback error: {error}')

        ############################################


    def odomcb(self, data):
        '''
        Description:    Callback function for the odometry topic.
                        Use this function to receive where the base currently is.

        Args:
            data (Odometry):    Pose and velocity of the base

        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:
            p = data.pose.pose.position
            q = data.pose.pose.orientation
            self.x = p.x
            self.y = p.y
            self.yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                  1.0 - 2.0 * (q.y * q.y + q.z * q.z))

            lin = math.hypot(data.twist.twist.linear.x, data.twist.twist.linear.y)
            ang = abs(data.twist.twist.angular.z)
            self.lin_speed = lin
            self.ang_speed = ang

            if lin < still_lin and ang < still_ang:                                     # standing still: remember since when
                if self.still_since is None:
                    self.still_since = self.now_sec()
            else:
                self.still_since = None

            self.odom = data

        except Exception as error:
            self.get_logger().error(f'Odometry callback error: {error}')

        ############################################


    ##################### GRID HELPERS #######################

    def world_to_cell(self, x, y):
        '''
        Description:    Converts map coordinates to a (col, row) cell.

        Args:
            x (float), y (float):   position in the map frame (m)

        Returns:
            tuple:      (col, row), or None if outside the grid
        '''

        col = int(math.floor((x - self.origin_x) / self.res))
        row = int(math.floor((y - self.origin_y) / self.res))

        if 0 <= col < self.width and 0 <= row < self.height:
            return (col, row)

        return None


    def cell_to_world(self, col, row):
        '''
        Description:    Centre of a cell in map coordinates.

        Args:
            col (int), row (int):   cell indices

        Returns:
            tuple:      (x, y) in metres
        '''

        return (self.origin_x + (col + 0.5) * self.res,
                self.origin_y + (row + 0.5) * self.res)


    def compute_clearance(self):
        '''
        Description:    Exact Euclidean distance transform (Felzenszwalb-Huttenlocher):
                        for every cell, the distance in metres from its centre to the
                        nearest rock, less half a cell because a rock cell is a square.

        Args:
        Returns:
            list:       flat list (row * width + col) of clearances in metres
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

        for col in range(w):                                                            # pass 1: down each column
            column = edt_1d([grid[row * w + col] for row in range(h)])
            for row in range(h):
                grid[row * w + col] = column[row]

        for row in range(h):                                                            # pass 2: along each row
            line = edt_1d(grid[row * w:(row + 1) * w])
            grid[row * w:(row + 1) * w] = line

        half = 0.5 * self.res
        return [max(0.0, math.sqrt(v) * self.res - half) if v < big else 1e6 for v in grid]


    def cell_free(self, col, row, hard_r, ends):
        '''
        Description:    Whether the eBot's centre may stand in this cell.

        Args:
            col (int), row (int):   cell indices
            hard_r (float):         required clearance from rocks (m)
            ends (tuple):           the (x, y) start and end points of the step

        Returns:
            bool
        '''

        if not (0 <= col < self.width and 0 <= row < self.height):
            return False

        i = row * self.width + col

        if self.occ[i]:
            return False

        if self.clear[i] >= hard_r:
            return True

        cx, cy = self.cell_to_world(col, row)                                           # relaxed rule next to the step's end points
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

        Args:
            p (tuple), q (tuple):   (x, y) end points
            hard_r (float):         required clearance (m)
            ends (tuple):           the step's start and end points

        Returns:
            bool
        '''

        dist = math.hypot(q[0] - p[0], q[1] - p[1])
        n = max(1, int(math.ceil(dist / (0.5 * self.res))))

        for i in range(n + 1):
            t = i / n
            if not self.point_free(p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1]), hard_r, ends):
                return False

        return True


    ##################### SEARCH #######################

    def astar(self, start, end, hard_r):
        '''
        Description:    A* over the 8-connected grid. Cells closer than 'hard_r' to a rock
                        are forbidden; cells a bit further are made more expensive.

        Args:
            start (tuple), end (tuple):   (x, y, yaw) of the step
            hard_r (float):               required clearance (m)

        Returns:
            list:       [(col, row), ...] from start to end, or None
        '''

        s = self.world_to_cell(start[0], start[1])
        g = self.world_to_cell(end[0], end[1])

        if s is None or g is None:
            return None

        ends = ((start[0], start[1]), (end[0], end[1]))
        w = self.width

        if self.occ[s[1] * w + s[0]] or self.occ[g[1] * w + g[0]]:
            return None

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

                if dc != 0 and dr != 0:                                                 # no squeezing diagonally between two rocks
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
        Description:    Thins out a cell path: from each point, jumps to the furthest later
                        point that can be reached in a straight line with the clearance kept.

        Args:
            pts (list):      [(x, y), ...]
            hard_r (float):  required clearance (m)
            ends (tuple):    the step's start and end points

        Returns:
            list:       [(x, y), ...] with far fewer points
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
        backward = pull(pts[::-1])[::-1]                                                # pulled from the other end

        return forward if length(forward) <= length(backward) else backward


    def densify(self, pts):
        '''
        Description:    Adds points along each straight segment so poses are at most
                        'path_spacing' apart. Keeps every original point.

        Args:
            pts (list):     [(x, y), ...]

        Returns:
            list:           [(x, y), ...]
        '''

        out = [pts[0]]

        for a, b in zip(pts[:-1], pts[1:]):
            dist = math.hypot(b[0] - a[0], b[1] - a[1])
            n = max(1, int(math.ceil(dist / path_spacing)))
            for k in range(1, n + 1):
                t = k / n
                out.append((a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])))

        out[-1] = pts[-1]                                                               # exact end position, no float drift

        return out


    def plan_path(self, start, end):
        '''
        Description:    Plans a path from 'start' to 'end' that stays clear of every rock.

        Args:
            start (tuple):    (x, y, yaw) where the step begins, in the map frame
            end (tuple):      (x, y, yaw) where the step ends, in the map frame

        Returns:
            list:             [(x, y), ...] from 'start' to 'end', or None if no path was found
        '''

        points = None

        ############ ADD YOUR CODE HERE ############

        try:
            ends = ((start[0], start[1]), (end[0], end[1]))

            for hard_r in clearance_levels:                                             # roomiest margin first, then squeeze
                cells = self.astar(start, end, hard_r)

                if cells is None:
                    self.get_logger().warn(f'No path with {hard_r:.2f} m clearance, trying less')
                    continue

                pts = [self.cell_to_world(c, r) for c, r in cells]
                pts[0] = (start[0], start[1])                                           # exact start and end positions
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

        ############################################

        return points


    def publish_path(self, points, start, end):
        '''
        Description:    Publishes 'points' on /ebot_path as a nav_msgs/Path.

        Args:
            points (list):    [(x, y), ...] as returned by plan_path()
            start (tuple):    (x, y, yaw) where the step begins
            end (tuple):      (x, y, yaw) where the step ends

        Returns:
        '''

        path = Path()
        path.header.frame_id = map_frame
        path.header.stamp = self.get_clock().now().to_msg()

        ############ ADD YOUR CODE HERE ############

        start_yaw, end_yaw = step_yaws[self.step]
        last = len(points) - 1

        for i, (x, y) in enumerate(points):
            if i == 0:
                yaw = start_yaw
            elif i == last:
                yaw = end_yaw
            else:                                                                       # in between: face along the path
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
            f'[EVENT] EBOT PATH step {self.step + 1}: start=({points[0][0]:.2f}, {points[0][1]:.2f}) '
            f'end=({points[-1][0]:.2f}, {points[-1][1]:.2f}) t={self.now_sec():.1f}s')
        self.get_logger().info(f'Published path for step {self.step + 1} ({len(points)} poses)')

        ############################################


    def request_ore_package(self):
        '''
        Description:    Calls /spawn_ore_package once the eBot stands still on the ore drop pose.

        Args:
        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        if not self.spawn_client.service_is_ready():
            self.get_logger().warn('/spawn_ore_package not available yet', throttle_duration_sec=5.0)
            self.ore_next_try = self.now_sec() + ore_retry_period
            return

        self.ore_pending = True
        future = self.spawn_client.call_async(Trigger.Request())
        future.add_done_callback(self.ore_reply)                                        # never wait for the result here

        ############################################


    def ore_reply(self, future):
        '''
        Description:    Handles the reply of /spawn_ore_package.

        Args:
            future:     the finished service call

        Returns:
        '''

        try:
            result = future.result()
            self.get_logger().info(f'Ore package reply: success={result.success}, message="{result.message}"')

            if result.success:
                self.ore_done = True
            else:
                self.ore_next_try = self.now_sec() + ore_retry_period                   # try again shortly

        except Exception as error:
            self.get_logger().error(f'Ore package call failed: {error}')
            self.ore_next_try = self.now_sec() + ore_retry_period

        finally:
            self.ore_pending = False


    def taken_cb(self, msg):
        '''
        Description:    Callback for /handover/package_taken: the arm has lifted the package off the deck.

        Args:
            msg (Bool):     True once the package is clear of the deck

        Returns:
        '''

        if msg.data:
            self.package_taken = True

    def end_reached(self, end, options):
        '''
        Description:    Whether the eBot stands still at the end pose of the current step.

        Args:
            end (tuple):        (x, y, yaw) of the step's end
            options (tuple):    valid end yaws

        Returns:
            bool
        '''

        if math.hypot(self.x - end[0], self.y - end[1]) > end_pos_tol:
            return False

        if not any(abs(math.atan2(math.sin(self.yaw - o), math.cos(self.yaw - o))) <= end_yaw_tol
                   for o in options):
            return False

        return self.still_since is not None and self.now_sec() - self.still_since >= still_hold


    def process_planning(self):
        '''
        Description:    Timer function used to plan, publish and request, step by step.

        Args:
        Returns:
        '''

        ############ ADD YOUR CODE HERE ############

        try:
            if self.grid is None or self.odom is None:                                  # wait for map and odometry
                self.get_logger().info(
                    f'Waiting for map ({self.grid is not None}) and odometry ({self.odom is not None})',
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
                still = self.still_since is not None and self.now_sec() - self.still_since >= still_hold
                if not still:
                    self.get_logger().info(
                        f'Waiting for the eBot to stand still before planning step {self.step + 1} '
                        f'(odom velocity: lin {self.lin_speed:.3f} m/s, ang {self.ang_speed:.3f} rad/s)',
                        throttle_duration_sec=3.0)
                    return                                                              # publish only while standing still

                if math.hypot(self.x - start[0], self.y - start[1]) > end_pos_tol:
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
                        f'[EVENT] EBOT REACHED step {self.step + 1}: pos=({self.x:.3f}, {self.y:.3f}) '
                        f'yaw={self.yaw:.3f} | goal=({end[0]:.2f}, {end[1]:.2f}) | '
                        f'pos_err={err:.3f} m | t={self.now_sec():.1f}s')

                    if self.step == 0:
                        self.stage = 'ore'
                    elif self.step == 1:                                                # beside the arm: tell it the package is here
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
                    self.settle_until = self.now_sec() + ore_settle_time                # let the package come to rest
                    self.stage = 'settle'
                elif not self.ore_pending and self.now_sec() >= self.ore_next_try:
                    self.request_ore_package()

            elif self.stage == 'settle':
                if self.now_sec() >= self.settle_until:
                    self.step += 1
                    self.stage = 'plan'

            elif self.stage == 'handover':
                if self.package_taken:                                                  # package is off the deck: drive home
                    self.step += 1
                    self.stage = 'plan'

        except Exception as error:                                                      # nothing here may kill the node
            self.get_logger().error(f'Planning cycle error: {error}')

        ############################################


##################### FUNCTION DEFINITION #######################

def main():
    '''
    Description:    Main function which creates a ROS node and spins around for the
                    ebot_planner class to perform its task
    '''

    rclpy.init(args=sys.argv)                                           # initialisation

    node = rclpy.create_node('ebot_planner_process')                    # creating ROS node

    node.get_logger().info('Node created: eBot planner process')        # logging information

    ebot_planner_class = ebot_planner()                                 # creating a new object for class 'ebot_planner'

    rclpy.spin(ebot_planner_class)                                      # spining on the object to make it alive in ROS 2 DDS

    ebot_planner_class.destroy_node()                                   # destroy node after spin ends

    rclpy.shutdown()                                                    # shutdown process


if __name__ == '__main__':

    main()