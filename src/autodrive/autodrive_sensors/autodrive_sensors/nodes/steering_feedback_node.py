"""ROS2 node publishing measured steering angle feedback.

Note on message choice: /vehicle/steering_feedback is not in the standard
message table, so this skeleton uses std_msgs/Float32 (steering angle in
radians) as the simplest standard-message placeholder. Revisit if a richer
message is needed later.
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32

from autodrive_sensors.drivers.serial_driver import SerialDriver


class SteeringFeedbackNode(Node):
    """Publishes /vehicle/steering_feedback (std_msgs/Float32, radians)."""

    def __init__(self) -> None:
        super().__init__('steering_feedback_node')

        self.declare_parameter('serial_port', '')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('publish_rate_hz', 50.0)

        serial_port: str = self.get_parameter('serial_port').get_parameter_value().string_value
        baudrate: int = self.get_parameter('baudrate').get_parameter_value().integer_value
        publish_rate_hz: float = self.get_parameter('publish_rate_hz').get_parameter_value().double_value

        self._driver = SerialDriver(port=serial_port or None, baudrate=baudrate)

        self._feedback_pub = self.create_publisher(
            Float32, '/vehicle/steering_feedback', 10)

        period_sec = 1.0 / publish_rate_hz if publish_rate_hz > 0.0 else 1.0
        self._timer = self.create_timer(period_sec, self._on_timer)

        self.get_logger().info('steering_feedback_node started')

    def _on_timer(self) -> None:
        """Read the measured steering angle and publish it.

        TODO: parse the actual Arduino feedback payload via self._driver
        and convert it into a steering angle in radians.
        """
        raw: Optional[bytes] = self._driver.read()
        if raw is None:
            return
        # TODO: decode raw feedback and publish Float32.


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
