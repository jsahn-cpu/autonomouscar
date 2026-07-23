"""ROS2 node wrapping GlobalMapMatcher.

Handles message conversion and pub/sub only; the actual matching logic
lives in autodrive_localization.core.map_matcher.
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid

from autodrive_localization.core.map_matcher import GlobalMapMatcher
from autodrive_localization.core.motion_model import VehicleState


class GlobalMapMatcherNode(Node):
    """Subscribes to the local map and publishes a camera pose measurement."""

    def __init__(self) -> None:
        super().__init__('global_map_matcher_node')

        self.declare_parameter('global_map_path', '')
        global_map_path: str = self.get_parameter('global_map_path').get_parameter_value().string_value

        self._matcher = GlobalMapMatcher(global_map_path=global_map_path or None)

        self._local_map_sub = self.create_subscription(
            OccupancyGrid, '/perception/local_map', self._on_local_map, 10)
        self._camera_pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/localization/camera_pose', 10)

        self.get_logger().info('global_map_matcher_node started')

    def _on_local_map(self, msg: OccupancyGrid) -> None:
        """Match the incoming local map against the global map.

        TODO: convert the resulting VehicleState into
        PoseWithCovarianceStamped and publish on /localization/camera_pose.
        """
        matched_state: Optional[VehicleState] = self._matcher.match(msg)
        if matched_state is None:
            return
        # TODO: build PoseWithCovarianceStamped from matched_state and publish.


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = GlobalMapMatcherNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
