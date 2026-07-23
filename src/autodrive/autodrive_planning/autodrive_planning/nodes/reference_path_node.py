"""ROS2 node publishing a pre-stored reference path.

Handles message conversion and publishing only; the actual file loading
logic lives in autodrive_planning.core.path_loader. Path-planning
algorithms (A*, Hybrid A*, RRT) and mission planning are not implemented
here.
"""
from pathlib import Path
from typing import List, Optional

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path as PathMsg

from autodrive_planning.core.path_loader import PathLoader, Waypoint


class ReferencePathNode(Node):
    """Publishes /planning/reference_path (nav_msgs/Path)."""

    def __init__(self) -> None:
        super().__init__('reference_path_node')

        self.declare_parameter('path_file', '')
        self.declare_parameter('publish_rate_hz', 1.0)
        self.declare_parameter('frame_id', 'map')

        path_file: str = self.get_parameter('path_file').get_parameter_value().string_value
        publish_rate_hz: float = self.get_parameter('publish_rate_hz').get_parameter_value().double_value
        self._frame_id: str = self.get_parameter('frame_id').get_parameter_value().string_value

        self._path_loader = PathLoader()
        self._waypoints: List[Waypoint] = []
        if path_file:
            # TODO: pick CSV vs YAML loader based on file extension.
            self._waypoints = self._path_loader.load_from_csv(Path(path_file))

        self._path_pub = self.create_publisher(PathMsg, '/planning/reference_path', 10)

        period_sec = 1.0 / publish_rate_hz if publish_rate_hz > 0.0 else 1.0
        self._timer = self.create_timer(period_sec, self._on_timer)

        self.get_logger().info('reference_path_node started')

    def _on_timer(self) -> None:
        """Publish the loaded reference path.

        TODO: convert self._waypoints into a nav_msgs/Path and publish.
        """
        if not self._waypoints:
            return
        # TODO: build PathMsg from self._waypoints and publish.


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = ReferencePathNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
