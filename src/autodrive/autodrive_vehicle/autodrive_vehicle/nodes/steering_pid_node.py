"""DEPRECATED -- the steering control loop moved INTO the firmware.

A potentiometer on A6 now measures the real steering angle, so
mega_steer_closed_loop.ino closes the loop on the Arduino: the host sends a
target angle (`SA`) and the firmware servos to it. There is no longer a
host-side steering PWM to compute, so this node -- which produced the
open-loop angle->pulse PWM on /vehicle/steering_pwm -- is obsolete.

arduino_bridge_node now maps the desired steering angle from /safety/command
straight to a target ADC and sends it. This node has been removed from the
launch files; kept only so any stale reference/entry-point still resolves.
It does nothing (no output topic, no serial).

See autodrive_vehicle.core.steering_pot for the angle<->ADC mapping and
autodrive_vehicle.core.steering_pid / steering_open_loop (also now unused for
control) for the retired host-side attempts.
"""
from typing import Optional

import rclpy
from rclpy.node import Node


class SteeringPidNode(Node):
    def __init__(self) -> None:
        super().__init__('steering_pid_node')
        self.get_logger().warn(
            'steering_pid_node is DEPRECATED -- steering is closed-loop in the '
            'firmware now; arduino_bridge_node sends the target angle directly. '
            'This node does nothing.')


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = SteeringPidNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
