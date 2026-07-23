"""Development tool skeleton for front camera intrinsic/extrinsic calibration.

TODO: implement actual calibration (e.g. checkerboard-based intrinsics,
and extrinsic calibration feeding back into
autodrive_description/urdf/autodrive.urdf.xacro).
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class CameraCalibrationNode(Node):
    """Subscribes to the front camera stream to support calibration capture."""

    def __init__(self) -> None:
        super().__init__('camera_calibration')

        self._image_sub = self.create_subscription(
            CompressedImage, '/camera/front/image/compressed', self._on_image, 10)

        self.get_logger().info('camera_calibration started')

    def _on_image(self, msg: CompressedImage) -> None:
        """Capture calibration frames.

        TODO: implement checkerboard detection / intrinsic solving.
        """
        pass


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = CameraCalibrationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
