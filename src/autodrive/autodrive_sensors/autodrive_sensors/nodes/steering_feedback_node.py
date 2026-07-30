"""DEPRECATED -- steering feedback is now published by the Arduino serial
port OWNER, not by a separate node.

The steering loop closed inside the firmware (mega_steer_closed_loop.ino,
POT on A6) which streams FB telemetry. Since only ONE node may hold the
Arduino serial port at a time, the port owner republishes the measured
steering angle on /vehicle/steering_feedback:
  - manual driving : keyboard_teleop_node
  - autonomous     : arduino_bridge_node

A standalone node opening the same port would conflict with the driver, so
this node no longer opens serial or publishes -- it just logs the deprecation
and idles. It has been removed from the launch files; kept only so any stale
reference/entry-point still resolves.
"""
from typing import Optional

import rclpy
from rclpy.node import Node


class SteeringFeedbackNode(Node):
    def __init__(self) -> None:
        super().__init__('steering_feedback_node')
        self.get_logger().warn(
            'steering_feedback_node is DEPRECATED -- /vehicle/steering_feedback is '
            'now published by the Arduino port owner (keyboard_teleop_node or '
            'arduino_bridge_node). This node does nothing.')


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = SteeringFeedbackNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
