"""Global map matching core logic.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.
"""
from typing import Any, Optional

from autodrive_localization.core.motion_model import VehicleState


class GlobalMapMatcher:
    """Matches a local occupancy map against a pre-built global map.

    TODO: implement the actual matching algorithm (e.g. scan matching,
    correlative matching) and load the global map from storage.
    """

    def __init__(self, global_map_path: Optional[str] = None) -> None:
        self._global_map_path = global_map_path
        # TODO: load the global map from self._global_map_path.

    def match(self, local_map: Any) -> Optional[VehicleState]:
        """Return the camera pose measurement matched against the global map.

        TODO: implement using local_map. Returns None until implemented.
        """
        return None
