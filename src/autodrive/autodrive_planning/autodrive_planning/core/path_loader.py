"""Reference path loading core logic.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Waypoint:
    """A single reference-path waypoint in the map/global frame."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


class PathLoader:
    """Loads a pre-stored reference path from CSV or YAML.

    TODO: finalize the on-disk file format and implement actual parsing.
    """

    def load_from_csv(self, path: Path) -> List[Waypoint]:
        """Load waypoints from a CSV file.

        TODO: implement actual CSV parsing (columns: x, y, yaw).
        """
        return []

    def load_from_yaml(self, path: Path) -> List[Waypoint]:
        """Load waypoints from a YAML file.

        TODO: implement actual YAML parsing.
        """
        return []
