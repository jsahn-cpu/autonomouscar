"""Development tool skeleton for steering angle calibration.

TODO: implement actual calibration (e.g. sweeping commanded steering
angle via /safety/command and recording /vehicle/steering_feedback to
derive a PWM-to-angle mapping).
"""
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32


class SteeringCalibrationNode(Node):
    """Subscribes to steering feedback to support calibration sweeps."""

    def __init__(self) -> None:
        super().__init__('steering_calibration')

        self._feedback_sub = self.create_subscription(
            Float32, '/vehicle/steering_feedback', self._on_feedback, 10)

        self.get_logger().info('steering_calibration started')

    def _on_feedback(self, msg: Float32) -> None:
        """Record a calibration sample.

        TODO: implement the sweep/recording/fitting logic.
        """
        pass


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = SteeringCalibrationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
