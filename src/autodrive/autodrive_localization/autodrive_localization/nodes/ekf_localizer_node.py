"""ROS2 node wrapping the EKF pose estimator.

Handles message conversion and pub/sub only; the actual filter math
lives in autodrive_localization.core.ekf.
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseWithCovarianceStamped

from autodrive_localization.core.ekf import EKF
from autodrive_localization.core.motion_model import BicycleMotionModel


class EkfLocalizerNode(Node):
    """Fuses camera pose measurements into /localization/pose.

    TODO: wire an actual motion-model input source (e.g. steering
    feedback / velocity) instead of only reacting to camera pose updates.
    """

    def __init__(self) -> None:
        super().__init__('ekf_localizer_node')

        self.declare_parameter('wheelbase', 0.545)
        wheelbase: float = self.get_parameter('wheelbase').get_parameter_value().double_value

        self._ekf = EKF(motion_model=BicycleMotionModel(wheelbase=wheelbase))

        self._camera_pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/localization/camera_pose', self._on_camera_pose, 10)
        self._pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, '/localization/pose', 10)

        self.get_logger().info('ekf_localizer_node started')

    def _on_camera_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """Update the EKF with a camera pose measurement and publish the estimate.

        TODO: convert msg into the EKF measurement type, call
        self._ekf.update(), and publish the resulting state as
        PoseWithCovarianceStamped on /localization/pose.
        """
        # TODO: convert msg -> VehicleState measurement.
        estimated_state = self._ekf.state
        # TODO: build PoseWithCovarianceStamped from estimated_state and publish.


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = EkfLocalizerNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
