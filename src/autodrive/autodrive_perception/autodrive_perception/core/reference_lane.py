"""Builds a reference lane (path to follow) from the tracked lane lines.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.

The vehicle drives a single 2-lane road, keeping to the right, so the
target path is the right boundary line offset left by a fixed pixel amount
-- not a full "which of these lines forms which lane" model, which would
need correctly identifying every line's role (see lane_tracker.py's
_select_evenly_spaced docstring for why that's still an open problem).

"Right boundary" here means whichever confirmed line currently has the
largest x_near (rightmost track), an assumption rather than something
verified against a real label -- solid/dashed classification was
intentionally dropped (see lane_detector.py's class docstring). If a
non-boundary marking is ever the rightmost confirmed track, this will
follow that instead; the same caveat that already applies to
/perception/lane_lines applies here.

The output is in image pixel coordinates, NOT a world/map frame -- this is
a direct camera-space target curve (closer to visual servoing than a real
planned path), not the same thing as /planning/reference_path, which is a
world-frame nav_msgs/Path produced later in the full stack from
localization + a stored reference route.
"""
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


class ReferenceLaneBuilder:
    """Fits a curve to the rightmost tracked line and offsets it inward."""

    def __init__(
        self,
        lane_offset_px: float = 0.0,
        poly_degree: int = 2,
        num_samples: int = 20,
    ) -> None:
        self._lane_offset_px = lane_offset_px
        self._poly_degree = poly_degree
        self._num_samples = num_samples

    def build(self, confirmed: Sequence[Dict[str, Any]]) -> Optional[np.ndarray]:
        """Return sampled (x, y) points of the reference path (near to far),
        or None if there's no line to build one from.

        Fits x = f(y) to the chosen line's actual detected points rather
        than using LaneTracker's own near/far straight chord -- that chord
        is a fine proxy for "is this near the same line as before" (its
        actual job) but a poor stand-in for a curved line's real shape (see
        lane_tracker.py's module docstring), which matters much more here
        since this curve is what a steering target would actually follow.
        """
        if not confirmed:
            return None

        # Picking by 'x_near' (the Kalman-projected intercept) sounds right
        # but isn't: on a sharp curve, a single physical boundary line often
        # fragments into several separate tracks (the collinearity check
        # that clusters Hough segments rejects merging across too much
        # curvature -- see lane_detector.py's _is_consistent_line). A short
        # fragment's own points may only cover a small stretch far from the
        # near/far reference rows, so its projected x_near is a long,
        # noise-amplified extrapolation -- exactly the failure mode that
        # produced runaway values and made the reference path visibly cut
        # across the actual curve instead of following it. The x of each
        # candidate's own closest (largest-y) real point is never
        # extrapolated, so it stays anchored to what was actually seen.
        def nearest_point_x(det: Dict[str, Any]) -> float:
            pts = det['points']
            return float(pts[np.argmax(pts[:, 1]), 0])

        right_line = max(confirmed, key=nearest_point_x)
        points = right_line['points']
        if len(points) < 3:
            return None  # not enough points for a stable curve fit

        ys = points[:, 1].astype(np.float64)
        xs = points[:, 0].astype(np.float64)
        degree = min(self._poly_degree, len(points) - 1)
        coeffs = np.polyfit(ys, xs, degree)

        y_samples = np.linspace(ys.min(), ys.max(), self._num_samples)
        x_samples = np.polyval(coeffs, y_samples) - self._lane_offset_px

        return np.stack([x_samples, y_samples], axis=1)
