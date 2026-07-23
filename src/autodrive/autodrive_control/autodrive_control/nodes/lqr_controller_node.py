"""ROS2 node wrapping LQRController.

Handles message conversion and pub/sub only; the actual control law lives
in autodrive_control.core.lqr. delta is the LQR OUTPUT, not its input --
the input state is (e_y, e_psi) read from /control/tracking_error.
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Vector3Stamped

from autodrive_control.core.lqr import LQRController


class LqrControllerNode(Node):
    """Subscribes to /control/tracking_error, publishes /control/command."""

    def __init__(self) -> None:
        super().__init__('lqr_controller_node')

        # TODO: these are placeholder LQR weights; tune once the linear
        # vehicle model and gain computation are implemented.
        self.declare_parameter('q_e_y', 1.0)
        self.declare_parameter('q_e_psi', 1.0)
        self.declare_parameter('r_delta', 1.0)

        q_e_y = self.get_parameter('q_e_y').get_parameter_value().double_value
        q_e_psi = self.get_parameter('q_e_psi').get_parameter_value().double_value
        r_delta = self.get_parameter('r_delta').get_parameter_value().double_value

        self._controller = LQRController(q=(q_e_y, q_e_psi), r=(r_delta,))

        self._error_sub = self.create_subscription(
            Vector3Stamped, '/control/tracking_error', self._on_tracking_error, 10)
        self._command_pub = self.create_publisher(
            AckermannDriveStamped, '/control/command', 10)

        self.get_logger().info('lqr_controller_node started')

    def _on_tracking_error(self, msg: Vector3Stamped) -> None:
        """Compute the desired steering angle and publish a vehicle command.

        TODO: also decide the commanded speed (currently left at 0.0
        placeholder) once a speed-planning strategy exists.
        """
        e_y = msg.vector.x
        e_psi = msg.vector.y
        delta: float = self._controller.compute_control((e_y, e_psi))

        command = AckermannDriveStamped()
        command.header.stamp = self.get_clock().now().to_msg()
        command.drive.steering_angle = delta
        # TODO: set command.drive.speed once a speed command source exists.
        self._command_pub.publish(command)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = LqrControllerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
