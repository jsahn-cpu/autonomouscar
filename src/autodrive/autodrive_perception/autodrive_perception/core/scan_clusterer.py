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
    max_beam_gap: int = 6,
) -> List[Cluster]:
    """Cluster one scan. seg threshold between consecutive points is
    seg_dist_base + seg_dist_range_coeff * range (adaptive). A gap of MORE
    than max_beam_gap dropped beams also breaks a cluster -- allowing a few
    dropouts keeps a single object (whose surface reflects nothing for a
    beam or two, e.g. a dark/glossy patch) from splitting in the middle,
    while a real occlusion gap between two objects still breaks it (either
    via many dropped beams OR the spatial-distance threshold on the
    surviving points)."""
    n_beams = len(ranges)
    xy, idx, r = _polar_to_xy(angle_min, angle_increment, ranges, range_min, range_max)
    if len(xy) == 0:
        return []
    keep = _in_roi(xy, roi)
    xy, idx, r = xy[keep], idx[keep], r[keep]
    if len(xy) < min_points:
        return []

    # First pass: break into raw segments (point arrays) on a spatial jump or
    # too many dropped beams. Segments are NOT yet size-filtered so the
    # wraparound merge below can still join a split object.
    segments = []  # each: dict(pts, first_idx, last_idx)
    start = 0
    for i in range(1, len(xy) + 1):
        cut = i == len(xy)
        if not cut:
            beam_gap = idx[i] - idx[i - 1] > max_beam_gap  # too many dropped beams
            step = float(np.hypot(*(xy[i] - xy[i - 1])))
            thresh = seg_dist_base + seg_dist_range_coeff * r[i - 1]
            cut = beam_gap or step > thresh
        if cut:
            segments.append({'pts': xy[start:i], 'first_idx': int(idx[start]), 'last_idx': int(idx[i - 1])})
            start = i

    # Wraparound merge: a full-circle scan is a ring, so the last beam
    # (angle ~+pi) is adjacent to the first (angle ~-pi). An object sitting
    # on that seam (here: straight behind the lidar) lands split across the
    # two ends of the array. Stitch the last and first segments if they're
    # spatially close and the beam gap ACROSS the seam is small.
    if len(segments) >= 2 and n_beams > 0:
        first, last = segments[0], segments[-1]
        seam_beam_gap = (n_beams - 1 - last['last_idx']) + first['first_idx']
        seam_step = float(np.hypot(*(first['pts'][0] - last['pts'][-1])))
        seam_thresh = seg_dist_base + seg_dist_range_coeff * float(np.hypot(*last['pts'][-1]))
        if seam_beam_gap <= max_beam_gap and seam_step <= seam_thresh:
            last['pts'] = np.vstack([last['pts'], first['pts']])
            segments = segments[1:]           # drop the now-merged first
            segments[-1] = last               # keep the combined one

    clusters: List[Cluster] = []
    for seg in segments:
        pts = seg['pts']
        if len(pts) < min_points:
            continue
        center, length, width, yaw = _oriented_box(pts)
        clusters.append(Cluster(
            centroid=(float(pts[:, 0].mean()), float(pts[:, 1].mean())),
            box_center=center, length=length, width=width, yaw=yaw,
            n_points=len(pts), points=pts,
        ))
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
