"""Path-tracking error computation core logic.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.

Convention: state is (e_y, e_psi) where e_y is the lateral error [m] and
e_psi is the heading error [rad].
"""
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Pose2D:
    """Minimal 2D pose used by the control package's core modules."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


class TrackingErrorCalculator:
    """Computes lateral and heading error against a reference path.

    TODO: implement nearest-point search on the reference path and the
    actual e_y / e_psi geometry.
    """

    def compute(self, pose: Pose2D, reference_path: List[Pose2D]) -> Tuple[float, float]:
        """Return (e_y, e_psi).

        TODO: implement using pose and reference_path. Currently returns
        zeros until implemented.
        """
        return 0.0, 0.0
