"""Development tool: publishes a static dummy pose on /localization/pose.

Useful for exercising downstream nodes (control, safety) without a real
localization stack running.
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped


class DummyPosePublisher(Node):
    """Publishes /localization/pose (geometry_msgs/PoseWithCovarianceStamped)."""

    def __init__(self) -> None:
        super().__init__('dummy_pose_publisher')

        self.declare_parameter('publish_rate_hz', 10.0)
        self.declare_parameter('frame_id', 'map')

        publish_rate_hz: float = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self._frame_id: str = self.get_parameter('frame_id').get_parameter_value().string_value

        self._pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/localization/pose', 10)

        period_sec = 1.0 / publish_rate_hz if publish_rate_hz > 0.0 else 0.1
        self._timer = self.create_timer(period_sec, self._on_timer)

        self.get_logger().info('dummy_pose_publisher started')

    def _on_timer(self) -> None:
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        # Dummy pose at the origin. TODO: add configurable dummy trajectories
        # if more elaborate downstream testing is needed.
        msg.pose.pose.orientation.w = 1.0
        self._pose_pub.publish(msg)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = DummyPosePublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
