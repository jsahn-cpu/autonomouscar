"""ROS2 node wrapping LocalMapBuilder.

Handles message conversion and pub/sub only; the actual map-building
logic lives in autodrive_perception.core.local_map_builder.
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image

from autodrive_perception.core.local_map_builder import LocalMapBuilder


class LocalMapNode(Node):
    """Subscribes to the BEV image and publishes a local occupancy map."""

    def __init__(self) -> None:
        super().__init__('local_map_node')

        self.declare_parameter('resolution', 0.05)
        self.declare_parameter('width_cells', 200)
        self.declare_parameter('height_cells', 200)
        self.declare_parameter('frame_id', 'base_link')

        resolution: float = self.get_parameter('resolution').get_parameter_value().double_value
        width_cells: int = self.get_parameter('width_cells').get_parameter_value().integer_value
        height_cells: int = self.get_parameter('height_cells').get_parameter_value().integer_value
        self._frame_id: str = self.get_parameter('frame_id').get_parameter_value().string_value

        self._local_map_builder = LocalMapBuilder(
            resolution=resolution, width_cells=width_cells, height_cells=height_cells)

        self._bev_sub = self.create_subscription(
            Image, '/perception/bev/image', self._on_bev_image, 10)
        self._local_map_pub = self.create_publisher(
            OccupancyGrid, '/perception/local_map', 10)

        self.get_logger().info('local_map_node started')

    def _on_bev_image(self, msg: Image) -> None:
        """Build a local occupancy map from an incoming BEV image.

        TODO: decode msg, run self._local_map_builder.build(), and publish
        the result as a nav_msgs/OccupancyGrid on /perception/local_map.
        """
        grid_data: Optional[object] = self._local_map_builder.build(msg)
        if grid_data is None:
            return
        # TODO: populate OccupancyGrid fields and publish.


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = LocalMapNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
