"""ROS2 node arbitrating the final vehicle command.

Subscribes to the controller's /control/command and republishes the
arbitrated result on /safety/command -- the ONLY topic autodrive_vehicle
is allowed to consume for driving commands.
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray

from autodrive_safety.core.command_policy import CommandPolicy, VehicleCommand


class CommandArbiterNode(Node):
    """Subscribes to /control/command, publishes /safety/command and /diagnostics."""

    def __init__(self) -> None:
        super().__init__('command_arbiter_node')

        self._policy = CommandPolicy()
        # TODO: wire these flags to real sources (e-stop button/topic,
        # watchdog /diagnostics) instead of hardcoded False.
        self._emergency_stop = False
        self._failure_stop = False

        self._control_command_sub = self.create_subscription(
            AckermannDriveStamped, '/control/command', self._on_control_command, 10)
        self._safety_command_pub = self.create_publisher(
            AckermannDriveStamped, '/safety/command', 10)
        self._diagnostics_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)

        self.get_logger().info('command_arbiter_node started')

    def _on_control_command(self, msg: AckermannDriveStamped) -> None:
        """Arbitrate the incoming controller command and republish it.

        TODO: also publish a DiagnosticArray entry describing which
        arbitration branch was taken.
        """
        controller_command = VehicleCommand(
            steering_angle=msg.drive.steering_angle, speed=msg.drive.speed)

        arbitrated = self._policy.arbitrate(
            controller_command, self._emergency_stop, self._failure_stop)

        out = AckermannDriveStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.drive.steering_angle = arbitrated.steering_angle
        out.drive.speed = arbitrated.speed
        self._safety_command_pub.publish(out)

        # TODO: populate a meaningful DiagnosticArray/DiagnosticStatus.
        diagnostics = DiagnosticArray()
        diagnostics.header.stamp = out.header.stamp
        self._diagnostics_pub.publish(diagnostics)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = CommandArbiterNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
