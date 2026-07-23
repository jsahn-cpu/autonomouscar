"""Development tool: logs last-received time for key topics to the console.

Not a safety component -- see autodrive_safety.watchdog_node for the
production watchdog. This is purely for interactive debugging.
"""
from typing import Dict, Optional

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32


class SensorMonitorNode(Node):
    """Logs staleness of camera / localization pose / control command / steering feedback."""

    _WATCHED_KEYS = ('camera', 'localization_pose', 'controller_command', 'steering_feedback')

    def __init__(self) -> None:
        super().__init__('sensor_monitor_node')

        self.declare_parameter('report_rate_hz', 1.0)
        report_rate_hz: float = self.get_parameter('report_rate_hz').get_parameter_value().double_value

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

        period_sec = 1.0 / report_rate_hz if report_rate_hz > 0.0 else 1.0
        self._timer = self.create_timer(period_sec, self._on_report)

        self.get_logger().info('sensor_monitor_node started')

    def _mark_received(self, key: str) -> None:
        self._last_received[key] = self.get_clock().now()

    def _on_report(self) -> None:
        """Log each watched topic's last-received status.

        TODO: report actual elapsed-time-since-last-message once needed
        beyond a simple received/not-received indication.
        """
        for key in self._WATCHED_KEYS:
            received = self._last_received[key] is not None
            self.get_logger().info(f'{key}: {"received" if received else "no data yet"}')


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = SensorMonitorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
