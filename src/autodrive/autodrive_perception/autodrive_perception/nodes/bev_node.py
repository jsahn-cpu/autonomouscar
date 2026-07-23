"""ROS2 node wrapping BEVTransform.

Handles message conversion and pub/sub only; the actual transform math
lives in autodrive_perception.core.bev_transform.
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image

from autodrive_perception.core.bev_transform import BEVTransform


class BevNode(Node):
    """Subscribes to the front camera and publishes a BEV debug image."""

    def __init__(self) -> None:
        super().__init__('bev_node')

        self.declare_parameter('output_frame_id', 'base_link')
        self._output_frame_id: str = self.get_parameter('output_frame_id').get_parameter_value().string_value

        self._bev_transform = BEVTransform()

        self._image_sub = self.create_subscription(
            CompressedImage, '/camera/front/image/compressed', self._on_image, 10)
        self._bev_pub = self.create_publisher(Image, '/perception/bev/image', 10)

        self.get_logger().info('bev_node started')

    def _on_image(self, msg: CompressedImage) -> None:
        """Run the BEV transform on an incoming compressed image.

        TODO: decode msg.data, run self._bev_transform.compute(), and
        encode/publish the result as a sensor_msgs/Image on
        /perception/bev/image.
        """
        bev_image: Optional[object] = self._bev_transform.compute(msg)
        if bev_image is None:
            return
        # TODO: convert bev_image to sensor_msgs/Image and publish.


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = BevNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
