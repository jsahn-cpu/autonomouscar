"""ROS2 node wrapping TrackingErrorCalculator.

Handles message conversion and pub/sub only; the actual error geometry
lives in autodrive_control.core.tracking_error.

Message convention for /control/tracking_error
(geometry_msgs/Vector3Stamped):
  x: lateral error e_y [m]
  y: heading error e_psi [rad]
  z: reserved (unused)
"""
from typing import List, Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped, Vector3Stamped
from nav_msgs.msg import Path

from autodrive_control.core.tracking_error import Pose2D, TrackingErrorCalculator


class TrackingErrorNode(Node):
    """Subscribes to pose + reference path, publishes /control/tracking_error."""

    def __init__(self) -> None:
        super().__init__('tracking_error_node')

        self._calculator = TrackingErrorCalculator()
        self._reference_path: List[Pose2D] = []
        self._latest_pose: Optional[Pose2D] = None

        self._pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/localization/pose', self._on_pose, 10)
        self._path_sub = self.create_subscription(
            Path, '/planning/reference_path', self._on_path, 10)
        self._error_pub = self.create_publisher(
            Vector3Stamped, '/control/tracking_error', 10)

        self.get_logger().info('tracking_error_node started')

    def _on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """Store the latest estimated pose and (eventually) publish tracking error.

        TODO: convert msg into Pose2D, compute (e_y, e_psi) via
        self._calculator.compute(), and publish Vector3Stamped with
        x=e_y, y=e_psi, z=0.0 (reserved).
        """
        # TODO: convert msg -> Pose2D and store in self._latest_pose.
        if self._latest_pose is None or not self._reference_path:
            return
        # TODO: compute error and publish.

    def _on_path(self, msg: Path) -> None:
        """Store the latest reference path.

        TODO: convert msg.poses into a List[Pose2D].
        """
        # TODO: convert msg -> List[Pose2D] and store in self._reference_path.
        pass


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = TrackingErrorNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
