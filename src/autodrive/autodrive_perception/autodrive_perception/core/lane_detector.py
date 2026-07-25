"""White lane-marking masking and Hough-line clustering.

No rclpy dependency: this module only deals with plain data so it can be
unit tested independently of ROS.

Works on a raw (perspective) camera image, not a BEV image: a straight lane
line in a raw/perspective view is still a straight line in pixel space (it
just converges toward a vanishing point rather than staying at a constant
column like it would in a BEV image), so line segments are grouped by
collinearity (similar line angle + small perpendicular offset) instead of by
column position.

Obstacle/traffic-light/parking perception must NOT be added here.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class _Segment:
    points: np.ndarray  # 2x2 float32: [[x1, y1], [x2, y2]] segment endpoints
    centroid: np.ndarray  # (x, y) segment midpoint
    direction: np.ndarray  # unit vector (vx, vy); sign is arbitrary


class LaneDetector:
    """Detects white lane markings (solid or dashed) in a raw camera image
    via the Hough transform.

    Lines are not classified solid vs dashed -- the vehicle must not cross
    either kind, so lane-keeping only needs each line's position, not its
    type.
    """

    def __init__(
        self,
        adaptive_block_size: int = 151,
        adaptive_c: float = -15.0,
        blur_kernel_size: int = 5,
        hough_threshold: int = 30,
        hough_min_line_length: int = 20,
        hough_max_line_gap: int = 8,
        hough_scale: float = 0.75,
        angle_tol_deg: float = 8.0,
        collinear_dist_tol_px: float = 25.0,
        max_segments: int = 500,
        roi_top_ratio: float = 0.45,
        roi_top_width_ratio: float = 0.5,
        roi_bottom_width_ratio: float = 1.0,
        roi_left_ratio: float = 0.0,
        filter_outside_track_boundary: bool = True,
        min_boundary_span_ratio: float = 0.3,
        min_line_aspect_ratio: float = 3.0,
        min_line_length_px: float = 60.0,
        corner_tol_px: float = 30.0,
    ) -> None:
        # A pixel counts as line paint only if it's brighter than its own
        # local neighborhood average by at least -adaptive_c gray levels
        # (block_size must be odd, and wide enough to span more than a lane
        # line's own width -- otherwise a line dominates its own neighborhood
        # average and stops registering as "brighter than its surroundings").
        # A single fixed global cutoff was tried first, but overhead lighting
        # reflects unevenly off the floor: a broad, gently-lit patch (glare,
        # a person standing nearby) can cross a fixed cutoff and get masked
        # in as if it were paint, even though it's nowhere near as bright as
        # its surroundings as real paint is.
        self._adaptive_block_size = adaptive_block_size
        self._adaptive_c = adaptive_c
        # adaptiveThreshold runs on the same downscaled copy as Hough (see
        # hough_scale below and mask_white/_find_segments) -- the block size
        # is a spatial extent (must stay wider than a lane line at whatever
        # resolution it's actually applied to), so it's rescaled along with
        # the image, not left at its original-resolution value. Must stay
        # odd and >=3 for cv2.adaptiveThreshold.
        scaled_block = max(3, int(round(adaptive_block_size * hough_scale)))
        self._scaled_adaptive_block_size = scaled_block if scaled_block % 2 == 1 else scaled_block + 1
        # Smooths small-scale noise (floor scratch marks, texture) before
        # thresholding, so fewer spurious tiny Hough segments get generated
        # from it in the first place. 0/1 disables blurring; must be odd
        # otherwise (GaussianBlur's kernel size requirement).
        self._blur_kernel_size = blur_kernel_size
        # HoughLinesP: threshold is the accumulator vote count needed to
        # accept a line; min_line_length drops short spurious segments;
        # max_line_gap is kept small so a dashed line's gaps are NOT bridged
        # into what would look like one continuous (solid) segment.
        self._hough_threshold = hough_threshold
        self._hough_min_line_length = hough_min_line_length
        self._hough_max_line_gap = hough_max_line_gap
        # HoughLinesP dominates detect()'s runtime on the full-resolution
        # mask; running it on a downscaled copy instead (coordinates are
        # scaled back up afterward) cuts that cost roughly quadratically for
        # a small loss of precision -- see _find_segments.
        self._hough_scale = hough_scale
        self._angle_tol_rad = np.deg2rad(angle_tol_deg)
        self._collinear_dist_tol_px = collinear_dist_tol_px
        # Clustering compares every segment pair; a noisy/overexposed frame
        # (e.g. bright ceiling/reflections) can otherwise produce hundreds of
        # spurious segments and make that O(n^2) pass pathologically slow.
        # Keep only the longest segments beyond this cap.
        self._max_segments = max_segments
        # Fraction of image height (from the top) to exclude from masking --
        # the camera looks forward, not just at the floor, so the ceiling/
        # walls/whiteboard above this row are never track floor. The
        # remaining region is a trapezoid (narrower at the top/far edge,
        # roi_top_width_ratio of the image width, widening to
        # roi_bottom_width_ratio at the bottom/near edge) rather than a full
        # rectangle, since a forward-facing camera's floor view narrows
        # toward the vanishing point -- the rectangle's upper corners are
        # usually wall/background, not floor.
        self._roi_top_ratio = roi_top_ratio
        self._roi_top_width_ratio = roi_top_width_ratio
        self._roi_bottom_width_ratio = roi_bottom_width_ratio
        # Fraction of the image width (from the left) to exclude entirely --
        # e.g. 0.3 keeps only the rightmost 70% of each row. Meant to drop
        # the dashed center line and anything left of it (other parking-lot
        # markings, the left ROI half in general) so only the right solid
        # boundary is ever a detection candidate, rather than relying on
        # tracking/selection logic to pick the right one out of several.
        self._roi_left_ratio = roi_left_ratio
        self._filter_outside_track_boundary = filter_outside_track_boundary
        # A detection must span at least this fraction of the ROI's height to
        # be treated as a candidate track-boundary line -- filters out short
        # segments (car body, reflections) that would otherwise get picked as
        # a boundary just for being leftmost/rightmost.
        self._min_boundary_span_ratio = min_boundary_span_ratio
        # A detected cluster whose overall shape (via minAreaRect) isn't at
        # least this elongated is probably not a lane line at all -- a wall
        # edge, doorframe, or other white object that happened to produce a
        # few locally-collinear segments. Dropped outright rather than
        # labeled, since it isn't a lane marking either way.
        self._min_line_aspect_ratio = min_line_aspect_ratio
        self._min_line_length_px = min_line_length_px
        # How close two detections' endpoints must be to count as meeting at
        # a shared corner -- see _remove_closed_shapes.
        self._corner_tol_px = corner_tol_px
        self._last_mask: Optional[Any] = None
        self._last_segments: List[_Segment] = []
        self._last_raw_clusters: List[List[_Segment]] = []

    @property
    def last_mask(self) -> Optional[Any]:
        """The binary mask from the most recent detect() call, or None
        before the first call."""
        return self._last_mask

    def _roi_polygon(self, width: int, height: int) -> np.ndarray:
        """Trapezoid ROI corners: narrow at the top/far edge, wide at the
        bottom/near edge (see __init__ docstring for why), then clipped on
        the left at roi_left_ratio if set."""
        top_row = height * self._roi_top_ratio
        bottom_row = height - 1
        center_x = width / 2.0
        top_half_w = width * self._roi_top_width_ratio / 2.0
        bottom_half_w = width * self._roi_bottom_width_ratio / 2.0
        left_bound = width * self._roi_left_ratio
        top_left_x = max(center_x - top_half_w, left_bound)
        bottom_left_x = max(center_x - bottom_half_w, left_bound)
        return np.array([
            [top_left_x, top_row],
            [center_x + top_half_w, top_row],
            [center_x + bottom_half_w, bottom_row],
            [bottom_left_x, bottom_row],
        ], dtype=np.int32)

    def crop_to_roi(self, image: Any) -> Any:
        """Warp the ROI trapezoid to fill a rectangle the same size as the
        input, instead of a plain bounding-box crop -- the trapezoid's top
        edge is narrower than its bottom (see _roi_polygon), so a bounding-
        box crop still leaves two unused triangular corners at the top; this
        maps all four trapezoid corners onto the four rectangle corners so
        nothing outside the ROI is visible at all.

        This is NOT the bird's-eye-view/ground-plane homography that was
        dropped for this project (see class docstring): that one assumed a
        fixed camera height/pitch to project onto real-world ground
        coordinates, and vehicle vibration drifts the real pose enough to
        make that unreliable. This warp carries no such assumption -- the
        ROI trapezoid is just an already-tuned image-space shape, not
        derived from any camera height/pitch/extrinsic, so there's no
        real-world mapping here to get wrong. The tradeoff is purely visual:
        a straight line in the original perspective view generally looks
        curved after this warp, since the stretch varies by row. Display
        only -- call it on an already-rendered debug image, never on
        anything fed into detect()."""
        h, w = image.shape[:2]
        roi_polygon = self._roi_polygon(w, h).astype(np.float32)
        dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
        transform = cv2.getPerspectiveTransform(roi_polygon, dst)
        return cv2.warpPerspective(image, transform, (w, h))

    def mask_white(self, image: Any) -> Any:
        """Return a binary (0/255) mask of near-white pixels, restricted to
        the trapezoid floor ROI (see `_roi_polygon`), downscaled by
        hough_scale.

        Accepts either a BGR frame or an already-grayscale one (2D array) --
        this is the first thing detect() does with the input either way, so
        a caller that only needs detection (not color) can skip decoding
        color at all and hand in grayscale directly.

        The resize happens here (before blur/threshold), not just before
        Hough -- adaptiveThreshold's cost scales with pixel count same as
        Hough's does, and running it on the full-resolution image only to
        immediately downscale the result for Hough was doing that single
        most expensive step at the wrong resolution. Detected line
        coordinates are scaled back up to the original resolution before
        leaving _find_segments, so nothing downstream (clustering, the
        tracker, published topics) sees anything but original-resolution
        pixel coordinates -- only the working copy used for masking/Hough is
        smaller, not the image the caller gets back or any output."""
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self._hough_scale != 1.0:
            gray = cv2.resize(gray, None, fx=self._hough_scale, fy=self._hough_scale,
                               interpolation=cv2.INTER_AREA)
        if self._blur_kernel_size > 1:
            gray = cv2.GaussianBlur(gray, (self._blur_kernel_size, self._blur_kernel_size), 0)
        mask = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            self._scaled_adaptive_block_size, self._adaptive_c)
        roi_mask = np.zeros_like(mask)
        cv2.fillPoly(roi_mask, [self._roi_polygon(mask.shape[1], mask.shape[0])], 255)
        return cv2.bitwise_and(mask, roi_mask)

    def _find_segments(self, mask: Any) -> List[_Segment]:
        # mask is already downscaled by hough_scale (see mask_white) --
        # HoughLinesP's cost scales with pixel count/density, and on the
        # full 1920x1080 mask it used to dominate detect()'s total runtime
        # (~110ms/frame measured, vs a few ms for everything else combined).
        # Line markings are thick enough that finding them on a downscaled
        # mask and scaling the resulting segment coordinates back up loses
        # little useful precision while cutting that cost roughly with the
        # square of hough_scale.
        # min_line_length/max_line_gap are user-facing config in original-
        # image pixels, so scale them into the downscaled mask's pixel units.
        lines = cv2.HoughLinesP(
            mask, rho=1, theta=np.pi / 180,
            threshold=self._hough_threshold,
            minLineLength=max(1, int(self._hough_min_line_length * self._hough_scale)),
            maxLineGap=max(1, int(self._hough_max_line_gap * self._hough_scale)))
        if lines is None:
            return []

        inv_scale = 1.0 / self._hough_scale
        segments: List[_Segment] = []
        for line in lines:
            x1, y1, x2, y2 = (line[0].astype(np.float64) * inv_scale)
            direction = np.array([x2 - x1, y2 - y1])
            length = np.linalg.norm(direction)
            if length < 1e-6:
                continue
            segments.append(_Segment(
                points=np.array([[x1, y1], [x2, y2]], dtype=np.float32),
                centroid=np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0]),
                direction=direction / length,
            ))

        if len(segments) > self._max_segments:
            segments.sort(
                key=lambda s: np.linalg.norm(s.points[1] - s.points[0]), reverse=True)
            segments = segments[:self._max_segments]
        return segments

    def _cluster_collinear(self, segments: Sequence[_Segment]) -> List[List[int]]:
        """Validated union-find grouping of segments whose lines are collinear.

        Pairwise angle/distance is computed vectorized (numpy, O(n^2) memory
        but no per-pair Python-level overhead) since a Python double loop
        doing per-pair numpy calls is orders of magnitude slower and can
        stall the node on a noisy frame with hundreds of segments. Only the
        (much sparser) list of actually-adjacent pairs is then walked in
        Python to decide merges.

        A plain (unvalidated) union-find would let a chain of pairwise-close
        segments transitively bridge two runs that are nowhere near collinear
        overall -- e.g. a near and a far part of the same nominal boundary,
        joined only through a segment that happens to sit between them. That
        merged group then fails the straightness check applied to the whole
        group later and gets discarded wholesale, taking a perfectly good
        sub-run down with it. So each candidate merge here is checked first,
        rejecting only the bridging link instead of the whole chain.

        Each candidate merge is checked with the same straightness test used
        on the final groups (_is_consistent_line) before committing, so only
        the bridging link is rejected instead of the whole chain. Merging two
        still-lone segments is skipped from that check -- already fully
        covered by the pairwise angle/distance test above -- since Hough
        emits many near-duplicate overlapping segments per stroke (see
        _find_segments), and re-verifying every one of those trivial
        merges was the dominant cost in early profiling.
        """
        n = len(segments)
        if n == 0:
            return []
        directions = np.stack([s.direction for s in segments])  # (n, 2)
        centroids = np.stack([s.centroid for s in segments])  # (n, 2)

        # Angle between every pair of directions, modulo pi (segment
        # direction sign is arbitrary).
        cos_matrix = np.clip(np.abs(directions @ directions.T), -1.0, 1.0)
        angle_matrix = np.arccos(cos_matrix)

        # dist_to_line[i, j] = perpendicular distance from centroid[j] to the
        # line through centroid[i] with direction[i]. Not symmetric, so take
        # the max of both directions as the pairwise "collinear distance".
        diff = centroids[None, :, :] - centroids[:, None, :]  # diff[i, j] = c_j - c_i
        proj_len = np.einsum('ijk,ik->ij', diff, directions)
        perp = diff - proj_len[:, :, None] * directions[:, None, :]
        dist_to_line = np.linalg.norm(perp, axis=2)  # dist_to_line[i, j]
        symmetric_dist = np.maximum(dist_to_line, dist_to_line.T)

        adjacency = (angle_matrix <= self._angle_tol_rad) & (symmetric_dist <= self._collinear_dist_tol_px)

        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        pair_i, pair_j = np.where(np.triu(adjacency, k=1))
        # Merge the most confidently-collinear pairs first: a real straight
        # run gets fully joined before any shakier "bridge" pair is tried, so
        # rejecting a bad bridge doesn't also split up a run that was fine.
        order = np.argsort(symmetric_dist[pair_i, pair_j])

        group_points: Dict[int, np.ndarray] = {i: segments[i].points for i in range(n)}
        for k in order.tolist():
            i, j = int(pair_i[k]), int(pair_j[k])
            ri, rj = find(i), find(j)
            if ri == rj:
                continue
            merged_points = np.vstack([group_points[ri], group_points[rj]])
            if len(merged_points) > 4 and not self._is_consistent_line(merged_points):
                continue
            parent[ri] = rj
            group_points[rj] = merged_points
            del group_points[ri]

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)
        return list(groups.values())

    @staticmethod
    def _boundary_x_at(boundary_points: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Interpolate a boundary line's x position at given y values.

        `boundary_points` are sorted by y so np.interp can walk a (possibly
        curved) line; y values outside the boundary's own range clamp to its
        nearest end (np.interp's default behavior) rather than extrapolating.
        """
        order = np.argsort(boundary_points[:, 1])
        ys_sorted = boundary_points[order, 1]
        xs_sorted = boundary_points[order, 0]
        return np.interp(y, ys_sorted, xs_sorted)

    def _filter_outside_boundary(
        self, detections: List[Dict[str, Any]], image_height: int
    ) -> List[Dict[str, Any]]:
        """Drop detections that fall outside the track's left/right boundary
        lines (the leftmost/rightmost detections that are long enough to
        plausibly be a boundary line, not a short segment like a reflection
        or the car body). Needs at least 2 boundary candidates; otherwise
        there's nothing reliable to bound with, so all detections are kept
        as-is (single frame only -- this doesn't track boundaries across
        frames)."""
        roi_height = image_height * (1.0 - self._roi_top_ratio)
        min_span = self._min_boundary_span_ratio * roi_height

        def y_span(det: Dict[str, Any]) -> float:
            ys = det['points'][:, 1]
            return float(ys.max() - ys.min())

        candidates = [d for d in detections if y_span(d) >= min_span]
        if len(candidates) < 2:
            return detections

        candidates_by_x = sorted(candidates, key=lambda d: d['points'][:, 0].mean())
        left_boundary, right_boundary = candidates_by_x[0], candidates_by_x[-1]
        if left_boundary is right_boundary:
            return detections

        kept = []
        for det in detections:
            if det is left_boundary or det is right_boundary:
                kept.append(det)
                continue
            ys = det['points'][:, 1]
            xs = det['points'][:, 0]
            left_x = self._boundary_x_at(left_boundary['points'], ys)
            right_x = self._boundary_x_at(right_boundary['points'], ys)
            inside = (xs >= left_x) & (xs <= right_x)
            if inside.mean() >= 0.5:  # majority of the cluster is inside the track
                kept.append(det)
        return kept

    def _is_consistent_line(self, points: np.ndarray) -> bool:
        """True if every point lies close to ONE overall fitted line.

        `_cluster_collinear` unions segments pairwise, so a chain of locally-
        collinear segments (e.g. scratch marks/texture noise on the floor)
        can transitively merge into one cluster even though the two ends are
        nowhere near collinear with each other -- pairwise closeness doesn't
        bound the whole group's shape. Refit one line through all the
        cluster's points and reject if any point strays further than the
        same tolerance used for pairwise clustering.
        """
        vx, vy, x0, y0 = cv2.fitLine(
            points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        direction = np.array([vx, vy], dtype=np.float64)
        origin = np.array([x0, y0], dtype=np.float64)
        offset = points.astype(np.float64) - origin
        perp = offset - np.outer(offset @ direction, direction)
        max_residual = np.linalg.norm(perp, axis=1).max()
        return max_residual <= self._collinear_dist_tol_px

    def _is_line_like(self, points: np.ndarray) -> bool:
        """True if `points` form an elongated (line-like) shape rather than a
        blob -- a wall corner, doorframe, or other curved/round white object
        can produce a few segments that pass the collinearity check yet still
        look nothing like a lane line overall. Aspect ratio alone isn't
        enough: e.g. a chord of a circle is thin and "line-like" by that
        measure alone despite being short, so a minimum length is required
        too (real lane lines span a meaningful part of the ROI; incidental
        edge fragments from other objects tend to be short)."""
        if len(points) < 2:
            return False
        if not self._is_consistent_line(points):
            return False
        (_, _), (w, h), _ = cv2.minAreaRect(points.astype(np.float32))
        long_side, short_side = max(w, h), max(min(w, h), 1e-6)
        if long_side < self._min_line_length_px:
            return False
        return (long_side / short_side) >= self._min_line_aspect_ratio

    def _remove_closed_shapes(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Drop detections that are sides of a small painted pictogram (e.g.
        a parking-space box) meeting at corners, rather than a real lane
        line.

        A real lane line's own two ends don't connect to anything -- it's an
        isolated open stroke, whether solid or one dash of a dashed line
        (dash gaps are intentionally left un-bridged by hough_max_line_gap
        and are far wider than corner_tol_px). A painted box's sides, though,
        meet end-to-end at each corner. Adjacent sides of a box also fail the
        collinearity check by a wide margin (~90 degrees apart, vs. the ~8
        degree tolerance used for clustering), so they always show up as
        separate detections here rather than being merged into one -- which
        is what makes this endpoint-proximity check meaningful in the first
        place.

        Hough rarely detects a painted box as a perfectly closed 4-sided
        loop (a corner is often missed, or picked up as an extra spurious
        fragment), so this doesn't require the group to fully close -- 3 or
        more mutually corner-connected detections is already implausible for
        real, independent lane lines and is dropped as a unit.
        """
        n = len(detections)
        if n < 3:
            return detections

        endpoints = []
        for det in detections:
            direction, origin = self._fit_line(det['points'])
            proj = (det['points'] - origin) @ direction
            endpoints.append((
                origin + proj.min() * direction,
                origin + proj.max() * direction,
            ))

        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for i in range(n):
            for ei in range(2):
                for j in range(i + 1, n):
                    for ej in range(2):
                        if np.linalg.norm(endpoints[i][ei] - endpoints[j][ej]) <= self._corner_tol_px:
                            ri, rj = find(i), find(j)
                            if ri != rj:
                                parent[ri] = rj

        groups: Dict[int, List[int]] = {}
        for i in range(n):
            groups.setdefault(find(i), []).append(i)

        closed_members = set()
        for members in groups.values():
            if len(members) >= 3:
                closed_members.update(members)

        if not closed_members:
            return detections
        return [d for i, d in enumerate(detections) if i not in closed_members]

    @staticmethod
    def _fit_line(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        vx, vy, x0, y0 = cv2.fitLine(
            points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        return np.array([vx, vy], dtype=np.float64), np.array([x0, y0], dtype=np.float64)

    def detect(self, image: Any) -> List[Dict[str, Any]]:
        """Return one entry per detected lane line: {points, segments,
        blob_count}, where `segments` is the list of (x1, y1, x2, y2) Hough
        line segments making up that lane (for drawing the actual detected
        line rather than a bounding box).

        `image` may be BGR or already-grayscale -- see mask_white()."""
        mask = self.mask_white(image)
        # Cached rather than recomputed by callers that also want to inspect
        # intermediate pipeline stages (e.g. for debug topics) --
        # mask_white()/_find_segments()/_cluster_collinear() are already the
        # expensive steps here, so this is just stashing references to what
        # detect() computes anyway, not doing any extra work in the normal
        # path. See draw_pipeline_debug() for what these are used for.
        self._last_mask = mask
        segs = self._find_segments(mask)
        self._last_segments = segs
        if not segs:
            self._last_raw_clusters = []
            return []

        groups = self._cluster_collinear(segs)
        self._last_raw_clusters = [[segs[i] for i in group_idx] for group_idx in groups]

        detections = []
        for group_segs in self._last_raw_clusters:
            points = np.vstack([s.points for s in group_segs])
            if not self._is_line_like(points):
                continue
            detections.append({
                'points': points,
                'segments': [tuple(s.points.flatten().tolist()) for s in group_segs],
                'blob_count': len(group_segs),
            })

        detections = self._remove_closed_shapes(detections)
        if self._filter_outside_track_boundary:
            detections = self._filter_outside_boundary(detections, image.shape[0])
        return detections

    def draw_debug(self, bgr: Any, detections: Sequence[Dict[str, Any]]) -> Any:
        """Draw the actual detected line segments on a copy of the input
        image, plus the floor ROI cutoff line (blue) for visual
        sanity-checking."""
        out = bgr.copy()
        roi_polygon = self._roi_polygon(out.shape[1], out.shape[0])
        cv2.polylines(out, [roi_polygon], isClosed=True, color=(255, 128, 0), thickness=1)
        for det in detections:
            for x1, y1, x2, y2 in det['segments']:
                # LaneTracker reports a line as its x-intercepts at the ROI's
                # near/far rows, straight-line-connected -- on a curve this
                # can shoot far outside the ROI's left/right edges well
                # before reaching those rows, so draw only the portion
                # actually inside the ROI instead of the raw (often
                # off-frame) endpoints.
                clipped = self._clip_segment_to_polygon(x1, y1, x2, y2, roi_polygon)
                if clipped is None:
                    continue
                cx1, cy1, cx2, cy2 = clipped
                cv2.line(out, (int(cx1), int(cy1)), (int(cx2), int(cy2)), (0, 200, 0), 3)
        return out

    def draw_pipeline_debug(
        self, image: Any, detections: Sequence[Dict[str, Any]], crop_to_roi: bool = False,
    ) -> Any:
        """Compose a 2x2 tiled view of the detection pipeline's stages: mask,
        raw Hough segments (pre-clustering), clusters (color-coded,
        pre-filter), and the final result (same as draw_debug()).

        None of those intermediate stages are visible from the final debug
        image alone, which is fine for normal operation but not for
        re-tuning thresholds/ROI/etc. from scratch -- e.g. after the
        camera's mounting position (height, pitch) changes and everything
        tuned against the old framing needs re-verifying.

        Must be called after detect() on the same frame -- reads
        self._last_mask/_last_segments/_last_raw_clusters, which detect()
        populates as a side effect rather than this recomputing them.

        If crop_to_roi, each panel is individually cropped+zoomed (see
        crop_to_roi()) before the labels are drawn on it -- done per-panel
        rather than once on the final composite, since the composite is a
        2x2 tile, not one coherent image the crop rectangle would apply to."""
        base = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = base.shape[:2]
        roi_polygon = self._roi_polygon(w, h)

        mask = self._last_mask
        if mask is not None:
            panel_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            if panel_mask.shape[:2] != (h, w):
                # mask_white() works on a copy downscaled by hough_scale --
                # resize back up so all four panels tile at the same size.
                panel_mask = cv2.resize(panel_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            panel_mask = np.zeros_like(base)

        # Dimmed backdrop for the segment/cluster overlays -- full
        # brightness would make the overlay colors hard to make out.
        dim = (base.astype(np.float32) * 0.35).astype(np.uint8)

        panel_segments = dim.copy()
        for seg in self._last_segments:
            (x1, y1), (x2, y2) = seg.points
            cv2.line(panel_segments, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

        panel_clusters = dim.copy()
        palette = [
            (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255),
            (255, 0, 255), (255, 255, 0), (128, 0, 255), (255, 128, 0),
        ]
        for i, group_segs in enumerate(self._last_raw_clusters):
            color = palette[i % len(palette)]
            for seg in group_segs:
                (x1, y1), (x2, y2) = seg.points
                cv2.line(panel_clusters, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

        panel_final = self.draw_debug(base, detections)  # already draws the ROI outline itself

        if crop_to_roi:
            # Cropped first, ROI outline skipped after -- once cropped, the
            # visible frame already approximates the ROI, and the outline's
            # coordinates would need re-deriving for the new crop+zoom
            # rather than just being reused as-is.
            panel_mask = self.crop_to_roi(panel_mask)
            panel_segments = self.crop_to_roi(panel_segments)
            panel_clusters = self.crop_to_roi(panel_clusters)
            panel_final = self.crop_to_roi(panel_final)
        else:
            for panel in (panel_mask, panel_segments, panel_clusters):
                cv2.polylines(panel, [roi_polygon], isClosed=True, color=(255, 128, 0), thickness=1)

        def label(panel: Any, text: str) -> Any:
            cv2.putText(panel, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            return panel

        label(panel_mask, 'mask')
        label(panel_segments, f'raw segments ({len(self._last_segments)})')
        label(panel_clusters, f'clusters ({len(self._last_raw_clusters)})')
        label(panel_final, f'final ({len(detections)})')

        top = np.hstack([panel_mask, panel_segments])
        bottom = np.hstack([panel_clusters, panel_final])
        return np.vstack([top, bottom])

    @staticmethod
    def _clip_segment_to_polygon(
        x1: float, y1: float, x2: float, y2: float, polygon: np.ndarray,
    ) -> Optional[Tuple[float, float, float, float]]:
        """Clip a line segment to a convex polygon (Cyrus-Beck), or None if
        none of it is inside."""
        dx, dy = x2 - x1, y2 - y1
        t_enter, t_exit = 0.0, 1.0
        n = len(polygon)
        for i in range(n):
            ex, ey = polygon[i]
            ex2, ey2 = polygon[(i + 1) % n]
            edge_dx, edge_dy = float(ex2 - ex), float(ey2 - ey)
            # Inward normal for _roi_polygon's clockwise (in image
            # coordinates) vertex order.
            nx, ny = edge_dy, -edge_dx
            denom = nx * dx + ny * dy
            num = nx * (ex - x1) + ny * (ey - y1)
            if abs(denom) < 1e-9:
                if num < 0:
                    return None  # parallel to this edge and outside it
                continue
            t = num / denom
            if denom < 0:
                t_enter = max(t_enter, t)
            else:
                t_exit = min(t_exit, t)
            if t_enter > t_exit:
                return None
        return (x1 + t_enter * dx, y1 + t_enter * dy, x1 + t_exit * dx, y1 + t_exit * dy)
