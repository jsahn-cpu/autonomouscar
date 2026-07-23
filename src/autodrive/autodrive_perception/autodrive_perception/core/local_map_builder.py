"""Local occupancy map construction core logic.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.
"""
from typing import Any, List, Optional


class LocalMapBuilder:
    """Builds a local occupancy grid from a BEV image.

    TODO: implement the actual occupancy grid construction. Obstacle
    detection must NOT be added here -- this is a generic drivable-area
    style local map only.
    """

    def __init__(self, resolution: float = 0.05, width_cells: int = 200, height_cells: int = 200) -> None:
        self._resolution = resolution
        self._width_cells = width_cells
        self._height_cells = height_cells

    def build(self, bev_image: Any) -> Optional[List[int]]:
        """Return a flat row-major occupancy grid (values 0-100 or -1).

        TODO: implement using bev_image. Returns None until implemented.
        """
        return None
