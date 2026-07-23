"""Development tool: publishes a static straight-line dummy path.

Useful for exercising downstream nodes (tracking error, LQR) without a
real reference-path source running.
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path


class DummyPathPublisher(Node):
    """Publishes /planning/reference_path (nav_msgs/Path)."""

    def __init__(self) -> None:
        super().__init__('dummy_path_publisher')

        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('frame_id', 'map')
        self.declare_parameter('num_waypoints', 20)
        self.declare_parameter('waypoint_spacing_m', 0.5)

        publish_rate_hz: float = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self._frame_id: str = self.get_parameter('frame_id').get_parameter_value().string_value
        self._num_waypoints: int = self.get_parameter('num_waypoints').get_parameter_value().integer_value
        self._spacing: float = self.get_parameter('waypoint_spacing_m').get_parameter_value().double_value

        self._path_pub = self.create_publisher(Path, '/planning/reference_path', 10)

        period_sec = 1.0 / publish_rate_hz if publish_rate_hz > 0.0 else 1.0
        self._timer = self.create_timer(period_sec, self._on_timer)

        self.get_logger().info('dummy_path_publisher started')

    def _on_timer(self) -> None:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id

        # Dummy straight line along +x. TODO: add configurable dummy
        # curved/curvy paths if more elaborate downstream testing is needed.
        for i in range(self._num_waypoints):
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = i * self._spacing
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self._path_pub.publish(msg)


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = DummyPathPublisher()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
