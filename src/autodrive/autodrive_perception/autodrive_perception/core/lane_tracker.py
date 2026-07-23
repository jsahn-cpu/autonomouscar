"""Kalman-filter-based frame-to-frame tracking for LaneDetector output.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.

LaneDetector.detect() analyzes each frame completely independently, so a
real, physically continuous lane line can flicker on/off between frames, and
one-off unrelated markings (e.g. a parking-space divider) can look exactly
as line-like as a real lane boundary in a single frame. LaneTracker tracks
each line's position across frames with a Kalman filter and only accepts a
new frame's detection into an existing track if it falls close enough to
that track's *predicted* position -- an established track can't be knocked
off course by an unrelated line that happens to appear nearby, and a track
is only reported once it has matched for several consecutive-ish frames.

This does NOT distinguish "real lane line" from "other marking" by meaning
-- it only protects a line already being tracked from being hijacked by a
one-off outlier. A marking that is itself consistently visible for many
frames (e.g. the camera pointed at a parking area with no real lane in
view) will still be tracked and reported like any other line, since nothing
here knows what a lane boundary means semantically.

The Kalman state (a line's x-intercepts at two fixed reference rows) exists
only to decide which detection belongs to which track -- a straight chord
between those rows is a reasonable stand-in for "is this near the same line
as before" but a poor one for the line's actual shape on a sharp curve. What
gets returned is each track's last actually-matched raw detection, not a
line reconstructed from the Kalman state.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from autodrive_perception.core.kalman_filter import KalmanFilter


class LaneTracker:
    """Tracks each frame's LaneDetector detections with a Kalman filter per
    line and only returns tracks confirmed over multiple frames (with a
    small grace period for frames where a real line is briefly missed)."""

    def __init__(
        self,
        roi_top_ratio: float = 0.45,
        min_confirm_frames: int = 3,
        max_missed_frames: int = 3,
        process_noise: float = 4.0,
        measurement_noise: float = 25.0,
        initial_velocity_variance: float = 400.0,
        gating_mahalanobis: float = 3.0,
        parallel_angle_tol_deg: float = 10.0,
        spacing_tol_ratio: float = 0.35,
        min_consistent_group_size: int = 2,
    ) -> None:
        # Must match LaneDetector's roi_top_ratio -- this is where the "far"
        # reference row for each line's x-intercept is measured, and the two
        # detectors need to agree on what a line's position even means.
        self._roi_top_ratio = roi_top_ratio
        self._min_confirm_frames = min_confirm_frames
        self._max_missed_frames = max_missed_frames
        self._gating_mahalanobis = gating_mahalanobis
        # The track layout is a fixed number of parallel, evenly-spaced
        # boundary/divider lines -- something that persists long enough to
        # become its own confirmed track (e.g. a parking-space divider) but
        # doesn't fit that pattern with the others is not one of those lines,
        # even though nothing about its own shape says so.
        self._parallel_angle_tol_deg = parallel_angle_tol_deg
        self._spacing_tol_ratio = spacing_tol_ratio
        self._min_consistent_group_size = min_consistent_group_size

        # State = [x_near, x_far, vx_near, vx_far]: a line is represented by
        # where it crosses two fixed reference rows (near/bottom and
        # far/top of the ROI) rather than an angle, since angle wraps
        # around (mod pi) and doesn't fit a linear state space cleanly.
        # Constant-velocity model per intercept (dt=1 frame).
        self._F = np.array([
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self._H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        # Discrete white-noise-acceleration model: position and velocity
        # noise are coupled (both derive from the same random per-step
        # acceleration), not independent diagonal terms.
        q = process_noise
        self._Q = q * np.array([
            [0.25, 0.0, 0.5, 0.0],
            [0.0, 0.25, 0.0, 0.5],
            [0.5, 0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0, 1.0],
        ])
        self._R = measurement_noise * np.eye(2)
        self._measurement_noise = measurement_noise
        self._initial_velocity_variance = initial_velocity_variance
        self._tracks: List[Dict[str, Any]] = []

    @staticmethod
    def _fit_line(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vx, vy, x0, y0 = cv2.fitLine(
            points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return np.array([vx, vy], dtype=np.float64), np.array([x0, y0], dtype=np.float64)

    def _project(
        self, det: Dict[str, Any], near_row: float, far_row: float, max_abs_x: float,
    ) -> Optional[Tuple[float, float]]:
        """Where this detection's fitted line crosses the near/far reference
        rows, or None if that crossing isn't numerically meaningful.

        A line running close to horizontal barely changes row per unit
        length, so its crossing point races off toward +-infinity for a tiny
        change in angle -- both hard to track with a stable Kalman gain and,
        incidentally, exactly the shape of a marking that runs perpendicular
        to the direction of travel (e.g. a parking-space divider) rather
        than along it like a real lane line does.
        """
        direction, origin = self._fit_line(det['points'])
        vy = direction[1]
        if abs(vy) < 1e-2:
            return None
        x_near = origin[0] + (near_row - origin[1]) / vy * direction[0]
        x_far = origin[0] + (far_row - origin[1]) / vy * direction[0]
        if not (np.isfinite(x_near) and np.isfinite(x_far)):
            return None
        if abs(x_near) > max_abs_x or abs(x_far) > max_abs_x:
            return None
        return float(x_near), float(x_far)

    def _select_evenly_spaced(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Keep only the largest run of tracks that are mutually parallel
        AND roughly evenly spaced, dropping everything else."""
        if len(tracks) < self._min_consistent_group_size:
            return tracks

        n = len(tracks)
        angles = []
        for t in tracks:
            direction, _ = self._fit_line(t['last_detection']['points'])
            angles.append(float(np.arctan2(direction[1], direction[0]) % np.pi))

        # Union-find clustering by pairwise angle agreement (circular
        # distance, since angle wraps at pi).
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        tol = np.deg2rad(self._parallel_angle_tol_deg)
        for i in range(n):
            for j in range(i + 1, n):
                diff = abs(angles[i] - angles[j])
                diff = min(diff, np.pi - diff)
                if diff <= tol:
                    ri, rj = find(i), find(j)
                    if ri != rj:
                        parent[ri] = rj

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        best_group = max(groups.values(), key=len)
        if len(best_group) < self._min_consistent_group_size:
            return []

        ordered = sorted(best_group, key=lambda i: tracks[i]['kf'].x[0])
        if len(ordered) == 1:
            return [tracks[i] for i in ordered]
        xs = [tracks[i]['kf'].x[0] for i in ordered]
        gaps = [xs[k + 1] - xs[k] for k in range(len(xs) - 1)]

        # Longest run of consecutive gaps that agree with their own run's
        # running mean -- adaptive rather than a single fixed spacing, since
        # different tracks may show a different (but each internally
        # consistent) real lane width.
        best_start, best_len = 0, 1
        run_start, run_len, run_sum = 0, 1, gaps[0]
        for k in range(1, len(gaps)):
            run_mean = run_sum / run_len
            if abs(gaps[k] - run_mean) <= self._spacing_tol_ratio * run_mean:
                run_len += 1
                run_sum += gaps[k]
            else:
                if run_len > best_len:
                    best_len, best_start = run_len, run_start
                run_start, run_len, run_sum = k, 1, gaps[k]
        if run_len > best_len:
            best_len, best_start = run_len, run_start

        selected = ordered[best_start:best_start + best_len + 1]
        return [tracks[i] for i in selected]

    def _new_track(self, x_near: float, x_far: float, det: Dict[str, Any]) -> Dict[str, Any]:
        x0 = np.array([x_near, x_far, 0.0, 0.0])
        P0 = np.diag([
            self._measurement_noise, self._measurement_noise,
            self._initial_velocity_variance, self._initial_velocity_variance,
        ])
        kf = KalmanFilter(x0, P0, self._F, self._H, self._Q, self._R)
        return {'kf': kf, 'seen_count': 1, 'missed_count': 0, 'last_detection': det}

    def update(
        self, detections: Sequence[Dict[str, Any]], image_height: int, image_width: int,
    ) -> List[Dict[str, Any]]:
        """Feed one frame's detections in, get the confirmed subset out."""
        near_row = float(image_height - 1)
        far_row = float(image_height) * self._roi_top_ratio
        max_abs_x = image_width * 3.0

        # The Kalman state (x_near, x_far) is only used to decide WHICH
        # detection belongs to which track (gating) -- a straight chord
        # between two fixed rows is a fine proxy for "is this near the same
        # line as before", but a bad stand-in for the line's actual shape on
        # a sharp curve. What gets returned/drawn is each track's last
        # actually-matched detection (raw Hough segments), not this.
        measurements, measurement_dets = [], []
        for det in detections:
            projected = self._project(det, near_row, far_row, max_abs_x)
            if projected is not None:
                measurements.append(np.array(projected))
                measurement_dets.append(det)

        for track in self._tracks:
            track['kf'].predict()

        # Greedy nearest-first assignment over every (track, measurement)
        # pair within the gate: taking the single closest pair first (rather
        # than looping tracks/measurements in order) means a track isn't
        # stolen by a farther match just because of iteration order:
        pairs = []
        for ti, track in enumerate(self._tracks):
            for mi, z in enumerate(measurements):
                dist = track['kf'].mahalanobis(z)
                if dist <= self._gating_mahalanobis:
                    pairs.append((dist, ti, mi))
        pairs.sort(key=lambda p: p[0])

        matched_tracks, matched_meas = set(), set()
        for dist, ti, mi in pairs:
            if ti in matched_tracks or mi in matched_meas:
                continue
            self._tracks[ti]['kf'].update(measurements[mi])
            self._tracks[ti]['seen_count'] += 1
            self._tracks[ti]['missed_count'] = 0
            self._tracks[ti]['last_detection'] = measurement_dets[mi]
            matched_tracks.add(ti)
            matched_meas.add(mi)

        for mi, z in enumerate(measurements):
            if mi not in matched_meas:
                self._tracks.append(self._new_track(float(z[0]), float(z[1]), measurement_dets[mi]))
                # Just created FROM this frame's measurement, so it was seen
                # this frame -- not a miss.
                matched_tracks.add(len(self._tracks) - 1)

        surviving = []
        for ti, track in enumerate(self._tracks):
            if ti not in matched_tracks:
                track['missed_count'] += 1
                if track['missed_count'] > self._max_missed_frames:
                    continue
            surviving.append(track)
        self._tracks = surviving

        confirmed = [t for t in self._tracks if t['seen_count'] >= self._min_confirm_frames]
        selected = self._select_evenly_spaced(confirmed)
        # x_near/x_far come from the Kalman state (smoothed across frames),
        # not the last raw detection -- this is what a consumer wanting a
        # stable line position (e.g. for a future lateral-offset topic)
        # should read, rather than re-deriving it from noisy raw points.
        return [
            {**t['last_detection'], 'x_near': float(t['kf'].x[0]), 'x_far': float(t['kf'].x[1])}
            for t in selected
        ]
