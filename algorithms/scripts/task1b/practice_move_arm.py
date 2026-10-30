#!/usr/bin/env python3
'''
PRACTICE ONLY - not part of the submission. Never run this during an evaluation.

Moves the arm's tool by keyboard (Cartesian velocity in base_link) and prints the tool position,
so you can read off the hopper position for CRUSHER_DROP.

Keys:  w / s = +x / -x      a / d = +y / -y      r / f = up / down
       + / - = faster / slower      space = stop      u = unlock a protective stop      q = quit
Hold a key to keep moving; the arm stops 0.3 s after the last key press.
'''

import sys
import select
import termios
import tty
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, TwistStamped
from std_msgs.msg import Int32
from std_srvs.srv import Trigger

KEYS = {
    'w': (1, 0, 0), 's': (-1, 0, 0),
    'a': (0, 1, 0), 'd': (0, -1, 0),
    'r': (0, 0, 1), 'f': (0, 0, -1),
}
MAX_SPEED = 0.50                 # m/s, below the arm's 0.15 m/s cap


class probe(Node):
    def __init__(self):
        super().__init__('practice_move_arm')
        self.pub = self.create_publisher(TwistStamped, '/delta_twist_cmds', 10)
        self.sub = self.create_subscription(PoseStamped, '/tcp_pose_raw', self.posecb, 20)
        self.status_sub = self.create_subscription(Int32, '/arm_status', self.statuscb, 10)
        self.unlock_client = self.create_client(Trigger, '/ur_arm_controller/unlock_protective_stop')
        self.status = None
        self.pos = None
        self.speed = 0.04
        self.direction = (0, 0, 0)
        self.last_key = 0.0

    def posecb(self, msg):
        self.pos = msg.pose.position

    def statuscb(self, msg):
        self.status = msg.data

    def unlock(self):
        '''The documented recovery from a protective stop (codes 20 to 23); it does not remove the stop.'''
        if self.unlock_client.service_is_ready():
            self.unlock_client.call_async(Trigger.Request())

    def send(self):
        if time.time() - self.last_key > 0.3:
            self.direction = (0, 0, 0)
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.twist.linear.x = self.direction[0] * self.speed
        msg.twist.linear.y = self.direction[1] * self.speed
        msg.twist.linear.z = self.direction[2] * self.speed
        self.pub.publish(msg)

    def show(self):
        if self.pos is not None:
            sys.stdout.write(
                f'\rx={self.pos.x:+.3f}  y={self.pos.y:+.3f}  z={self.pos.z:+.3f}  speed={self.speed:.2f} m/s  arm_status={self.status}   ')
            sys.stdout.flush()


def main():
    rclpy.init()
    node = probe()
    settings = termios.tcgetattr(sys.stdin)
    print(__doc__)

    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)

            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == 'q':
                    break
                if key in KEYS:
                    node.direction = KEYS[key]
                    node.last_key = time.time()
                elif key == ' ':
                    node.direction = (0, 0, 0)
                elif key == 'u':
                    node.unlock()
                elif key == '+':
                    node.speed = min(MAX_SPEED, node.speed + 0.01)
                elif key == '-':
                    node.speed = max(0.01, node.speed - 0.01)

            node.send()
            node.show()

    except KeyboardInterrupt:
        pass

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        node.direction = (0, 0, 0)
        node.send()
        node.destroy_node()
        rclpy.shutdown()
        print()


if __name__ == '__main__':
    main()