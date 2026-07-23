"""Kinematic bicycle motion model used for EKF prediction.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.
"""
from dataclasses import dataclass


@dataclass
class VehicleState:
    """Minimal vehicle state used across localization core modules."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


class BicycleMotionModel:
    """Predicts the next vehicle state using a kinematic bicycle model.

    TODO: implement the actual kinematic bicycle equations, e.g.
        x'   = x + v * cos(yaw) * dt
        y'   = y + v * sin(yaw) * dt
        yaw' = yaw + (v / wheelbase) * tan(delta) * dt
    """

    def __init__(self, wheelbase: float = 0.545) -> None:
        self._wheelbase = wheelbase

    def predict(self, state: VehicleState, velocity: float, steering_angle: float, dt: float) -> VehicleState:
        """Return the predicted state after dt seconds.

        TODO: implement using self._wheelbase, velocity and steering_angle.
        Currently returns the input state unchanged.
        """
        return state
