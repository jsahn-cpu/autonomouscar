"""Fit continuous lane curves from a segmentation class map.

The segmentation model fires only where there is actual white paint, so a
dashed line comes out as separate dashes and a worn/occluded solid line comes
out fragmented. Both are just "points scattered along a line": per-row centroid
+ a low-degree polynomial fit `x = f(y)` bridges the gaps and yields ONE
continuous curve per class. This is the "connect the line" step -- done here in
fitting, not in the model.

No rclpy dependency -- pure geometry, unit-testable. The node
(lane_curve_node.py) wraps this, adds temporal smoothing, and publishes the
2nd-lane centre path for the (future) pulse controller.

Class ids (label/lane_dataset): 1 left_solid, 2 center_dashed, 3 right_solid.
For the 2nd lane (right of the dashed) the boundaries are class 2 (left) and
class 3 (right); the target path is their midpoint.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class LaneCurve:
    """x = f(y) fitted through a class's per-row centroids. Evaluation clamps y
    to the observed [y_min, y_max] so the curve holds flat instead of wildly
    extrapolating past where it actually had pixels."""
    poly: np.ndarray          # np.polyfit coeffs (highest degree first)
    y_min: int
    y_max: int
    n_rows: int

    def x_at(self, ys) -> np.ndarray:
        ys = np.clip(np.asarray(ys, dtype=np.float64), self.y_min, self.y_max)
        return np.polyval(self.poly, ys)


def per_row_centroids(mask: np.ndarray, min_px_per_row: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """(ys, xs): for each row with >= min_px_per_row set pixels, the mean x.
    Vectorised so it's cheap on a full-res class map."""
    m = mask.astype(np.float64)
    cols = np.arange(m.shape[1], dtype=np.float64)
    counts = m.sum(axis=1)
    sums = (m * cols).sum(axis=1)
    valid = counts >= min_px_per_row
    ys = np.nonzero(valid)[0].astype(np.float64)
    xs = sums[valid] / counts[valid]
    return ys, xs


def fit_curve(
    mask: np.ndarray, degree: int = 2, min_rows: int = 6, min_px_per_row: int = 2,
    min_y_spread: int = 40,
) -> Optional[LaneCurve]:
    """Fit x = f(y) through the class's per-row centroids. Uses degree 1 unless
    the points span enough rows (min_y_spread) AND there are enough of them for
    a stable higher-degree fit -- a curve fit to a short/sparse dash cluster
    just overfits noise. Returns None if too few rows to trust."""
    ys, xs = per_row_centroids(mask, min_px_per_row)
    if len(ys) < min_rows:
        return None
    spread = ys.max() - ys.min()
    deg = degree if (spread >= min_y_spread and len(ys) >= degree + 4) else 1
    poly = np.polyfit(ys, xs, deg)
    return LaneCurve(poly=poly, y_min=int(ys.min()), y_max=int(ys.max()), n_rows=int(len(ys)))


def lane_center_x(
    left: Optional[LaneCurve], right: Optional[LaneCurve], ys,
    half_lane_px: Optional[float] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """Centre x of the 2nd lane at rows `ys`.

    - both boundaries (dashed=left, right_solid=right): midpoint (best).
    - only one boundary + half_lane_px: offset from it by half a lane width.
    - neither: (None, 'none').
    Returns (xs or None, mode)."""
    ys = np.asarray(ys, dtype=np.float64)
    if left is not None and right is not None:
        return 0.5 * (left.x_at(ys) + right.x_at(ys)), 'both'
    if left is not None and half_lane_px is not None:
        return left.x_at(ys) + half_lane_px, 'left_only'   # dashed + half-lane to the right
    if right is not None and half_lane_px is not None:
        return right.x_at(ys) - half_lane_px, 'right_only'
    return None, 'none'
