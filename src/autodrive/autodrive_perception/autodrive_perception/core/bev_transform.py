"""Bird's-eye-view transform core logic.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.
"""
from typing import Any, Optional


class BEVTransform:
    """Converts a front-camera image into a bird's-eye-view image.

    TODO: implement the actual perspective transform (e.g. homography
    derived from camera extrinsics/intrinsics). Obstacle/traffic-light/
    parking perception must NOT be added here.
    """

    def __init__(self, homography: Optional[Any] = None) -> None:
        # TODO: store/compute the homography matrix once calibration exists.
        self._homography = homography

    def compute(self, image: Any) -> Optional[Any]:
        """Return the BEV-transformed image.

        TODO: implement using self._homography. Returns None until implemented.
        """
        return None
