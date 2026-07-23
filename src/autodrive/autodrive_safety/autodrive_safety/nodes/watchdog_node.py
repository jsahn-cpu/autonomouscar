"""ROS2 node watching for stale sensor/localization/control data.

Tracks the last-received time of: camera, localization pose, controller
command, and steering feedback. Publishes a summary on /diagnostics.
Timeout thresholds are read from parameters (see
autodrive_bringup/config/safety.yaml) -- placeholder values below.
"""
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from ackermann_msgs.msg import AckermannDriveStamped
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32


class WatchdogNode(Node):
    """Publishes /diagnostics based on last-received timestamps of key topics."""

    _WATCHED_KEYS = ('camera', 'localization_pose', 'controller_command', 'steering_feedback')

    def __init__(self) -> None:
        super().__init__('watchdog_node')

        # TODO: tune these placeholder timeouts once real timing is characterized.
        self.declare_parameter('camera_timeout_sec', 1.0)
        self.declare_parameter('localization_pose_timeout_sec', 1.0)
        self.declare_parameter('controller_command_timeout_sec', 0.5)
        self.declare_parameter('steering_feedback_timeout_sec', 0.5)
        self.declare_parameter('check_rate_hz', 10.0)

        self._timeouts: Dict[str, float] = {
            'camera': self.get_parameter('camera_timeout_sec').get_parameter_value().double_value,
            'localization_pose': self.get_parameter(
                'localization_pose_timeout_sec').get_parameter_value().double_value,
            'controller_command': self.get_parameter(
                'controller_command_timeout_sec').get_parameter_value().double_value,
            'steering_feedback': self.get_parameter(
                'steering_feedback_timeout_sec').get_parameter_value().double_value,
        }
        self._last_received: Dict[str, Optional[Time]] = {key: None for key in self._WATCHED_KEYS}

        self.create_subscription(
            CompressedImage, '/camera/front/image/compressed',
            lambda msg: self._mark_received('camera'), 10)
        self.create_subscription(
            PoseWithCovarianceStamped, '/localization/pose',
            lambda msg: self._mark_received('localization_pose'), 10)
        self.create_subscription(
            AckermannDriveStamped, '/control/command',
            lambda msg: self._mark_received('controller_command'), 10)
        self.create_subscription(
            Float32, '/vehicle/steering_feedback',
            lambda msg: self._mark_received('steering_feedback'), 10)

        self._diagnostics_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 10)

        check_rate_hz: float = self.get_parameter('check_rate_hz').get_parameter_value().double_value
        period_sec = 1.0 / check_rate_hz if check_rate_hz > 0.0 else 0.1
        self._timer = self.create_timer(period_sec, self._on_check)

        self.get_logger().info('watchdog_node started')

    def _mark_received(self, key: str) -> None:
        self._last_received[key] = self.get_clock().now()

    def _on_check(self) -> None:
        """Check each watched topic's staleness and publish a DiagnosticArray.

        TODO: implement actual staleness -> DiagnosticStatus.level mapping
        (OK / WARN / ERROR / STALE) using self._timeouts.
        """
        diagnostics = DiagnosticArray()
        diagnostics.header.stamp = self.get_clock().now().to_msg()
        for key in self._WATCHED_KEYS:
            status = DiagnosticStatus()
            status.name = key
            # TODO: compute elapsed time since self._last_received[key] and
            # compare against self._timeouts[key] to set status.level/message.
            diagnostics.status.append(status)
        self._diagnostics_pub.publish(diagnostics)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = WatchdogNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
