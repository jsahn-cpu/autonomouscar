"""Extended Kalman Filter core logic for pose estimation.

State vector: [x, y, yaw]. No rclpy dependency: this module only deals
with plain data so it can be unit tested independently of ROS.
"""
from typing import Optional

from autodrive_localization.core.motion_model import BicycleMotionModel, VehicleState


class EKF:
    """Fuses motion-model prediction with camera pose measurements.

    TODO: implement the actual EKF math (state covariance propagation,
    Jacobians, Kalman gain, measurement update).
    """

    def __init__(self, motion_model: Optional[BicycleMotionModel] = None) -> None:
        self._motion_model = motion_model or BicycleMotionModel()
        self._state = VehicleState()
        # TODO: initialize and maintain the state covariance matrix.

    def predict(self, velocity: float, steering_angle: float, dt: float) -> VehicleState:
        """Propagate the state using the motion model.

        TODO: also propagate the covariance matrix.
        """
        self._state = self._motion_model.predict(self._state, velocity, steering_angle, dt)
        return self._state

    def update(self, measurement: VehicleState) -> VehicleState:
        """Correct the predicted state using a camera pose measurement.

        TODO: implement the actual Kalman gain / measurement update.
        Currently returns the un-corrected predicted state.
        """
        return self._state

    @property
    def state(self) -> VehicleState:
        return self._state
