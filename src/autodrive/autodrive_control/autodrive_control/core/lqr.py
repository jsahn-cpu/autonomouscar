"""LQR steering controller core logic.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.

Controller input state: x = [e_y, e_psi]^T
Controller output: u = delta (desired steering angle, radians)

PathTrackingController is an abstract base so Pure Pursuit and RL
controllers can be added later and swapped in without changing the node
that calls them.
"""
from abc import ABC, abstractmethod
from typing import Any, Optional, Tuple


class PathTrackingController(ABC):
    """Common interface for path-tracking controllers (LQR, Pure Pursuit, RL, ...)."""

    @abstractmethod
    def compute_control(self, state: Tuple[float, float]) -> float:
        """Return the desired steering angle delta [rad] for state (e_y, e_psi)."""
        raise NotImplementedError


class LQRController(PathTrackingController):
    """LQR controller: x = [e_y, e_psi] -> u = delta.

    TODO: implement compute_gain() (discrete algebraic Riccati equation)
    and use the resulting gain matrix in compute_control().
    """

    def __init__(self, q: Optional[Any] = None, r: Optional[Any] = None) -> None:
        self._q = q
        self._r = r
        self._gain: Optional[Any] = None

    def compute_gain(self, a: Any, b: Any) -> Optional[Any]:
        """Solve the LQR gain K for the given (A, B) state-space matrices.

        TODO: implement the discrete algebraic Riccati equation solve.
        """
        self._gain = None
        return self._gain

    def compute_control(self, state: Tuple[float, float]) -> float:
        """Return delta given state = (e_y, e_psi).

        TODO: implement u = -K @ state using self._gain. Returns 0.0 until
        implemented.
        """
        return 0.0
