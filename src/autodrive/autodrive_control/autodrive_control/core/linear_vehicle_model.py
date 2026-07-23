"""Linearized kinematic bicycle model used to build the LQR state-space matrices.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.
"""
from typing import Any, Tuple


class LinearVehicleModel:
    """Provides the linearized A, B matrices for the LQR state x = [e_y, e_psi].

    TODO: implement the actual linearization of the kinematic bicycle
    model around a reference velocity/curvature.
    """

    def __init__(self, wheelbase: float = 0.545) -> None:
        self._wheelbase = wheelbase

    def get_state_space(self, velocity: float) -> Tuple[Any, Any]:
        """Return (A, B) matrices linearized at the given velocity.

        TODO: implement using self._wheelbase and velocity. Returns
        (None, None) until implemented.
        """
        return None, None
