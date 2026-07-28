"""2D LaserScan clustering for obstacle/vehicle detection.

No rclpy dependency -- pure geometry so it can be unit tested and reused
outside a node (see scan_cluster_node.py for the ROS wrapper). Turns one
LaserScan into a list of Cluster objects (centroid + oriented bounding box
+ point count), which the parking mission then gates by size and pairs into
"the two parked cars flanking the slot".

Why adjacent-point segmentation (not DBSCAN): a LaserScan's points are
already ORDERED by beam angle, so a single O(n) left-to-right pass that
starts a new cluster whenever the gap to the next point is too big
recovers the same connected objects a full DBSCAN would, far cheaper. The
gap threshold is RANGE-ADAPTIVE (grows with distance) because the spacing
between adjacent beams' hit points grows with range -- a fixed threshold
would over-segment far objects and merge near ones.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class Cluster:
    """One detected object in the lidar frame (x forward, y left, meters)."""
    centroid: Tuple[float, float]          # mean of the points
    box_center: Tuple[float, float]        # min-area-rect center
    length: float                          # longer bbox side (m)
    width: float                           # shorter bbox side (m)
    yaw: float                             # bbox orientation (rad, -pi/2..pi/2)
    n_points: int
    points: np.ndarray                     # (N, 2) member points, for debug/viz

    @property
    def range(self) -> float:
        return float(np.hypot(*self.centroid))


def _polar_to_xy(angle_min, angle_increment, ranges, range_min, range_max):
    """Valid (x, y) points + their original beam indices. Invalid beams
    (nan/inf/out-of-range) are dropped, but their indices are remembered so
    a run of them breaks a cluster (an occlusion gap between two objects)."""
    ranges = np.asarray(ranges, dtype=np.float64)
    idx = np.arange(ranges.size)
    valid = np.isfinite(ranges) & (ranges >= range_min) & (ranges <= range_max)
    idx = idx[valid]
    r = ranges[valid]
    ang = angle_min + idx * angle_increment
    xy = np.stack([r * np.cos(ang), r * np.sin(ang)], axis=1)
    return xy, idx, r


def _in_roi(xy: np.ndarray, roi: Optional[Tuple[float, float, float, float]]) -> np.ndarray:
    if roi is None:
        return np.ones(len(xy), dtype=bool)
    x_min, x_max, y_min, y_max = roi
    return (xy[:, 0] >= x_min) & (xy[:, 0] <= x_max) & (xy[:, 1] >= y_min) & (xy[:, 1] <= y_max)


def _oriented_box(points: np.ndarray) -> Tuple[Tuple[float, float], float, float, float]:
    """(center, length, width, yaw) of the min-area rectangle around points.
    length >= width; yaw in radians of the length axis."""
    pts = points.astype(np.float32)
    (cx, cy), (w, h), angle_deg = cv2.minAreaRect(pts)
    length, width = (w, h) if w >= h else (h, w)
    # cv2 angle is of the 'w' side; make yaw refer to the LENGTH axis
    yaw_deg = angle_deg if w >= h else angle_deg + 90.0
    yaw = np.deg2rad(((yaw_deg + 90.0) % 180.0) - 90.0)  # wrap to (-90, 90]
    return (float(cx), float(cy)), float(length), float(width), float(yaw)


def cluster_scan(
    angle_min: float,
    angle_increment: float,
    ranges,
    range_min: float,
    range_max: float,
    roi: Optional[Tuple[float, float, float, float]] = None,
    seg_dist_base: float = 0.05,      # C0: gap threshold at range 0 (m)
    seg_dist_range_coeff: float = 0.05,  # C1: extra gap allowed per meter of range
    min_points: int = 4,
) -> List[Cluster]:
    """Cluster one scan. seg threshold between consecutive points is
    seg_dist_base + seg_dist_range_coeff * range (adaptive). A run of
    invalid/out-of-ROI beams also breaks a cluster."""
    xy, idx, r = _polar_to_xy(angle_min, angle_increment, ranges, range_min, range_max)
    if len(xy) == 0:
        return []
    keep = _in_roi(xy, roi)
    xy, idx, r = xy[keep], idx[keep], r[keep]
    if len(xy) < min_points:
        return []

    clusters: List[Cluster] = []
    start = 0
    for i in range(1, len(xy) + 1):
        cut = i == len(xy)
        if not cut:
            beam_gap = idx[i] - idx[i - 1] > 1  # dropped beam(s) in between
            step = float(np.hypot(*(xy[i] - xy[i - 1])))
            thresh = seg_dist_base + seg_dist_range_coeff * r[i - 1]
            cut = beam_gap or step > thresh
        if cut:
            seg = xy[start:i]
            if len(seg) >= min_points:
                center, length, width, yaw = _oriented_box(seg)
                clusters.append(Cluster(
                    centroid=(float(seg[:, 0].mean()), float(seg[:, 1].mean())),
                    box_center=center, length=length, width=width, yaw=yaw,
                    n_points=len(seg), points=seg,
                ))
            start = i
    return clusters


def passes_vehicle_gate(
    c: Cluster, min_length: float, max_length: float,
    min_width: float, max_width: float, min_points: int,
) -> bool:
    """Is this cluster the size/shape of a parked car (vs noise or a wall)?
    A wall is too long; noise is too small/few-point. Width uses only a
    lower bound by default (a car seen end-on can look thin)."""
    return (
        c.n_points >= min_points
        and min_length <= c.length <= max_length
        and min_width <= c.width <= max_width
    )


def find_flanking_pair(
    vehicles: List[Cluster], gap_min: float, gap_max: float,
) -> Optional[Tuple[Cluster, Cluster]]:
    """Among vehicle clusters, the pair whose centroids are gap_min..gap_max
    apart -- the two parked cars bounding an empty slot. Returns the closest
    (smallest-range) such pair, or None. The empty slot is the space between
    them; the mission builds the parking target from their inner edges."""
    best = None
    best_key = None
    for a in range(len(vehicles)):
        for b in range(a + 1, len(vehicles)):
            va, vb = vehicles[a], vehicles[b]
            d = float(np.hypot(va.centroid[0] - vb.centroid[0],
                               va.centroid[1] - vb.centroid[1]))
            if gap_min <= d <= gap_max:
                key = min(va.range, vb.range)  # prefer the nearest pair
                if best_key is None or key < best_key:
                    best_key, best = key, (va, vb)
    return best
