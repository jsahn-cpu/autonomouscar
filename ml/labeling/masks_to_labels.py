#!/usr/bin/env python3
"""Turns label_with_sam3.py's raw per-instance masks into one multi-class
training label per frame (ml/README.md step 3). Reads only
ml/data/sam3_raw/ -- never touches SAM3/GPU -- so every filter parameter
here can be re-tuned and re-run for free.

Classes (see labeling.yaml's class_ids): left_solid, center_dashed,
right_solid, lane_1, lane_2, background(0). SAM3 is only ever prompted with
generic concepts ("solid lane line", "dashed lane line", "road lane") --
it isn't reliable at spatial instructions like "the line on the right", so
left/right and lane-1/lane-2 assignment happens HERE instead:

  1. Per prompt (solid_line / dashed_line / lane_area), filter that
     prompt's own instances by score, ROI, and a shape filter appropriate
     to what it is (thin+long for lines, min-area for the lane surface).
  2. OR-merge the survivors of each prompt into one mask, then split that
     mask into connected components ("blobs") -- this is what turns
     SAM3's frequent oversegmentation (one physical line/curve reported as
     many small overlapping instances, see label_with_sam3.py runs) back
     into one blob per physical line/area.
  3. left_solid/right_solid AND lane_1/lane_2 are BOTH classified relative
     to the DASHED LINE's own fitted curve (see fit_dashed_boundary), not
     a fixed image-center column. A fixed column is wrong whenever the
     frame is skewed toward one side -- on a curving track, or e.g. near a
     corner where the camera happens to be pointed toward a parking area:
     the real second solid line can end up entirely on the "wrong" image
     half (or off-frame), while something that ISN'T that line (a parking
     line, say) can land on the expected side purely by image position and
     get mislabeled as if it were the real thing. Classifying against the
     dashed line's actual curve instead fixes the split-side-of-what
     problem -- it does NOT fix SAM3 having detected a non-lane-line
     object as a solid line in the first place; that's a detection-time
     failure mode, not a classification-time one, and needs eyeballing
     labels/tightening prompts to catch.
     - left_solid/right_solid: each blob's centroid (cx, cy) is compared
       against boundary(cy) (see classify_left_right).
     - lane_1/lane_2: split per-PIXEL against the curve (see
       split_area_by_dashed), not per-blob -- this also correctly handles
       a single continuous road-surface blob spanning across the dashed
       line, which a per-blob method can't split at all.
     Falls back to a constant image-center function for both if the
     dashed line wasn't detected in this frame at all (too few points to
     fit a line through).
  5. Compose the final label: lane_1/lane_2 written first, then
     center_dashed, then left_solid/right_solid LAST -- line classes still
     beat area classes everywhere, but solid now beats dashed at any
     overlap between the two (previously dashed was painted last and could
     silently overwrite a true solid-line pixel -- SAM3's dashed-line
     prompt was observed to also weakly fire on solid-line pixels
     sometimes, and the old order let that erase the whole solid line's
     label at any frame where it happened).
"""
import argparse
import json
import os
import pathlib
import random
import time
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image

# Each frame is CPU-bound (PNG decode) and processed in its own worker
# process below; cv2's internal thread pool per worker would oversubscribe
# the cores (workers x cv2-threads) -- same PyTorch/OpenCV gotcha as
# ml/training/dataset.py. One thread per process, workers are the only
# parallelism.
cv2.setNumThreads(0)

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def frame_id_to_session(frame_id: str) -> str:
    # collect_frames.py writes "<session>_<index:06d>_<stamp_ns>" -- index
    # and stamp_ns are always the last two, purely-numeric segments, so
    # rsplit lets a session name itself contain underscores safely.
    parts = frame_id.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else frame_id


def _bbox(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """(rmin, rmax, cmin, cmax) of True pixels (rmax/cmax exclusive), or None
    if the mask is empty. Used to skip full-resolution boolean ops on
    spatially-disjoint masks -- see dedup_by_iou / passes_line_shape_filter."""
    rows = np.any(mask, axis=1)
    if not rows.any():
        return None
    cols = np.any(mask, axis=0)
    r = np.where(rows)[0]
    c = np.where(cols)[0]
    return int(r[0]), int(r[-1]) + 1, int(c[0]), int(c[-1]) + 1


def load_prompt_instances(
    frame_dir: pathlib.Path, prompt_text: str, min_score: float, roi_top_ratio: float = 0.0
) -> List[Tuple[np.ndarray, float]]:
    with open(frame_dir / "meta.json") as f:
        meta = json.load(f)
    instances = []
    for inst in meta["prompts"].get(prompt_text, []):
        if inst["score"] < min_score:
            continue
        # cv2.imread (grayscale) decodes these mostly-black masks ~1.5x
        # faster than PIL here, and this is now the per-frame hot path.
        mask = cv2.imread(str(frame_dir / inst["mask_file"]), cv2.IMREAD_GRAYSCALE) > 0
        # ROI top crop folded in here (the mask is freshly owned, so this is
        # in place -- no extra full-frame copy per instance like the old
        # separate apply_roi_top_crop pass did).
        if roi_top_ratio > 0:
            mask[: int(mask.shape[0] * roi_top_ratio), :] = False
        instances.append((mask, inst["score"]))
    return instances


def any_mask_shape(frame_dir: pathlib.Path) -> Optional[Tuple[int, int]]:
    """Shape of any saved instance mask in this frame dir, regardless of
    prompt/score -- used only to size an all-background label when nothing
    survived filtering but the frame dir isn't literally empty."""
    with open(frame_dir / "meta.json") as f:
        meta = json.load(f)
    for prompt_instances in meta["prompts"].values():
        for inst in prompt_instances:
            arr = cv2.imread(str(frame_dir / inst["mask_file"]), cv2.IMREAD_GRAYSCALE)
            return arr.shape
    return None


def passes_line_shape_filter(mask: np.ndarray, min_aspect_ratio: float, min_length_px: float) -> bool:
    # Crop to the mask's bounding box first -- findContours on a full
    # 1920x1080 mostly-empty frame is far more work than on the small region
    # the instance actually occupies (the offset doesn't affect the
    # minAreaRect dimensions we care about).
    bb = _bbox(mask)
    if bb is None:
        return False
    sub = mask[bb[0]:bb[1], bb[2]:bb[3]]
    mask_u8 = sub.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest = max(contours, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(largest)
    long_side, short_side = max(w, h), max(min(w, h), 1e-3)
    return (long_side / short_side) >= min_aspect_ratio and long_side >= min_length_px


def passes_area_filter(mask: np.ndarray, min_area_px: float) -> bool:
    return int(np.count_nonzero(mask)) >= min_area_px


def iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return intersection / union if union else 0.0


def dedup_by_iou(
    instances: List[Tuple[np.ndarray, float]], iou_threshold: float
) -> List[Tuple[np.ndarray, float]]:
    """Keep the highest-scoring of each set of near-duplicate masks (IoU >
    threshold). The naive version ran `&` and `|` over the full 1920x1080
    frame for every O(n^2) pair, which dominated runtime once low score
    thresholds let ~40+ instances through per frame. This precomputes each
    mask's bounding box + pixel area once and, per pair, first checks
    whether the boxes even overlap (most line fragments are spatially
    disjoint -> IoU 0, skipped for free) and otherwise intersects only
    within the overlapping sub-box, never the whole frame."""
    order = sorted(range(len(instances)), key=lambda i: -instances[i][1])
    kept: List[Tuple[np.ndarray, float]] = []
    kept_meta: List[Tuple[np.ndarray, Tuple[int, int, int, int], int]] = []
    for i in order:
        mask, score = instances[i]
        bb = _bbox(mask)
        if bb is None:
            continue
        area = int(np.count_nonzero(mask[bb[0]:bb[1], bb[2]:bb[3]]))
        is_dup = False
        for kmask, kbb, karea in kept_meta:
            r0, r1 = max(bb[0], kbb[0]), min(bb[1], kbb[1])
            c0, c1 = max(bb[2], kbb[2]), min(bb[3], kbb[3])
            if r0 >= r1 or c0 >= c1:
                continue  # bounding boxes disjoint -> IoU 0
            inter = int(np.count_nonzero(mask[r0:r1, c0:c1] & kmask[r0:r1, c0:c1]))
            if inter == 0:
                continue
            union = area + karea - inter
            if union and inter / union > iou_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append((mask, score))
            kept_meta.append((mask, bb, area))
    return kept


def merge_masks(instances: List[Tuple[np.ndarray, float]], shape: Tuple[int, int]) -> np.ndarray:
    merged = np.zeros(shape, dtype=bool)
    for mask, _ in instances:
        merged |= mask
    return merged


def blobs_by_area(mask: np.ndarray, max_blobs: int) -> List[Tuple[np.ndarray, float, float]]:
    """Split a merged binary mask into up to max_blobs connected
    components (largest area first), pairing each with its centroid
    (cx, cy) -- this is what turns SAM3's oversegmented fragments of one
    physical line/area back into a single blob."""
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    components = [
        (labels == i, stats[i, cv2.CC_STAT_AREA], centroids[i][0], centroids[i][1])
        for i in range(1, num_labels)  # label 0 is background
    ]
    components.sort(key=lambda t: -t[1])
    components = components[:max_blobs]
    return [(blob_mask, cx, cy) for blob_mask, _, cx, cy in components]


def classify_left_right(blobs: List[Tuple[np.ndarray, float, float]], boundary):
    """Each blob goes to left/right by whether its centroid sits left or
    right of `boundary` AT THAT BLOB'S OWN ROW (boundary(cy), not a fixed
    image-center column) -- on a curving track (or a camera framing skewed
    toward one side, e.g. near a corner/parking area), a fixed column is
    wrong: a real solid line can end up entirely on the "wrong" image half,
    while something in the frame that ISN'T the missing line (a parking
    line, say) can land on the expected side purely by image position and
    get mislabeled as if it were the real thing. `boundary` is a callable
    x=f(y) -- pass a fitted dashed-line curve (see fit_dashed_boundary)
    when available, or a constant function as a fallback when it isn't.
    NOTE: this only fixes the left/right SPLIT -- it does not and cannot
    stop SAM3 from having detected a non-lane-line object as a solid line
    in the first place; that failure mode needs eyeballing labels/
    tightening prompts, not a smarter split rule.

    blobs is already area-sorted (largest first, see blobs_by_area), so if
    two blobs land on the same side only the larger one is kept."""
    left_mask, right_mask = None, None
    for mask, cx, cy in blobs:
        if cx < boundary(cy):
            if left_mask is None:
                left_mask = mask
        else:
            if right_mask is None:
                right_mask = mask
    return left_mask, right_mask


class _ClampedPoly:
    """Boundary x = f(y), but with y CLAMPED to the row range the dashed
    line was actually observed over before evaluating the polynomial -- so
    it is never extrapolated past its data. A degree-2 fit swings hard to
    one side the moment it's evaluated even slightly beyond its data, which
    was flipping the apparent curvature of the lane_1/lane_2 boundary in the
    near/far field of the frame where the dashed line wasn't detected (the
    lane area spans the whole frame height, the dashed line only a band in
    the middle). Outside [y_min, y_max] the boundary is just held flat at
    the nearest endpoint value instead. Callable with a scalar or an array
    of y (np.clip handles both), matching np.poly1d's interface."""

    def __init__(self, poly: np.poly1d, y_min: float, y_max: float) -> None:
        self._poly = poly
        self._y_min = y_min
        self._y_max = y_max

    def __call__(self, y):
        return self._poly(np.clip(y, self._y_min, self._y_max))


def fit_dashed_boundary(
    dashed_mask: np.ndarray, min_rows: int = 5, min_rows_quadratic: int = 10,
    min_span_frac: float = 0.25,
):
    """Fits x = f(y) through the dashed line's own per-row pixel centroid,
    to use as the lane_1/lane_2 split boundary instead of a fixed
    image-center column -- on a curving track the dashed line's actual
    x-position drifts a lot between the near and far field of the frame,
    so a fixed vertical split produces a visibly wrong lane_1/lane_2
    boundary there (confirmed on real training samples).

    Returns a _ClampedPoly (never extrapolated past the observed rows -- see
    that class), or None if fewer than min_rows distinct rows have a dashed
    pixel at all (callers fall back to the old image-center split).

    Degree is chosen conservatively to avoid a wrong-way curve (which is
    worse than a straight line -- it puts the lane_1/lane_2 boundary on the
    wrong side): degree 2 (a real curve) only when the dashed line has both
    >= min_rows_quadratic distinct rows AND spans >= min_span_frac of the
    frame height (a quadratic fit to points clustered in a thin band bends
    unpredictably), and additionally only if that parabola doesn't turn
    around WITHIN the observed band (a real forward-view lane line doesn't
    reverse direction -- a vertex inside the band means the curvature is
    noise, so fall back to a line). Degree 1 otherwise.
    """
    ys, xs = np.nonzero(dashed_mask)
    if ys.size == 0:
        return None
    # Vectorized per-row centroid (sum of x / count of x per row index) via
    # bincount instead of a Python-level per-pixel loop -- this runs once
    # per frame across the full dataset (tens of thousands of frames), and
    # a real 1920x1080 dashed mask can have thousands of pixels, so a pure-
    # Python loop here was a measurable slowdown.
    num_rows = dashed_mask.shape[0]
    row_sum = np.bincount(ys, weights=xs.astype(np.float64), minlength=num_rows)
    row_count = np.bincount(ys, minlength=num_rows)
    valid = row_count > 0
    if valid.sum() < min_rows:
        return None
    rows = np.nonzero(valid)[0]
    centroids = row_sum[valid] / row_count[valid]
    y_min, y_max = float(rows.min()), float(rows.max())

    span_ok = (y_max - y_min) >= min_span_frac * num_rows
    degree = 2 if (valid.sum() >= min_rows_quadratic and span_ok) else 1
    coeffs = np.polyfit(rows, centroids, deg=degree)
    if degree == 2:
        a, b, _ = coeffs
        vertex = -b / (2 * a) if a != 0 else float("inf")
        if y_min <= vertex <= y_max:  # parabola turns around inside the band -> noise
            coeffs = np.polyfit(rows, centroids, deg=1)
    return _ClampedPoly(np.poly1d(coeffs), y_min, y_max)


def split_area_by_dashed(
    area_mask: np.ndarray, boundary
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Per-PIXEL split of area_mask into lane_1 (left of the dashed line's
    fitted curve) / lane_2 (right of it) using fit_dashed_boundary's
    result. Deliberately per-pixel rather than per-blob (unlike
    classify_left_right) -- this correctly splits a single connected area
    blob that spans across the dashed line, not just already-separate
    blobs, which is the common case for a continuous road surface."""
    h, w = area_mask.shape
    boundary_x = boundary(np.arange(h))  # shape (h,)
    col_idx = np.broadcast_to(np.arange(w), (h, w))
    left = area_mask & (col_idx < boundary_x[:, None])
    right = area_mask & (col_idx >= boundary_x[:, None])
    return (left if left.any() else None), (right if right.any() else None)


def process_one_frame(frame_dir: pathlib.Path, params: dict) -> Optional[Tuple[str, str, bool]]:
    """Build and save one frame's label PNG. Fully self-contained (reads
    only this frame's sam3_raw dir, writes only its own labels/<id>.png) so
    it can run in a worker process -- see main()'s Pool. Returns
    (frame_id, session, is_empty), or None if the frame had no instances at
    all and its shape couldn't even be determined (nothing written)."""
    frame_id = frame_dir.name
    labels_dir = pathlib.Path(params["labels_dir"])
    prompts = params["prompts"]
    class_min_score = params["class_min_score"]
    class_ids = params["class_ids"]
    roi_top_ratio = params["roi_top_ratio"]
    line_min_aspect_ratio = params["line_min_aspect_ratio"]
    line_min_length_px = params["line_min_length_px"]
    lane_area_min_area_px = params["lane_area_min_area_px"]
    iou_dedup_threshold = params["iou_dedup_threshold"]
    max_blobs_per_class = params["max_blobs_per_class"]

    # ROI top crop is folded into the load (see load_prompt_instances) so
    # there's no separate full-frame copy per instance.
    solid = load_prompt_instances(frame_dir, prompts["solid_line"], class_min_score["solid_line"], roi_top_ratio)
    dashed = load_prompt_instances(frame_dir, prompts["dashed_line"], class_min_score["dashed_line"], roi_top_ratio)
    area = load_prompt_instances(frame_dir, prompts["lane_area"], class_min_score["lane_area"], roi_top_ratio)

    solid = [(m, s) for m, s in solid if passes_line_shape_filter(m, line_min_aspect_ratio, line_min_length_px)]
    dashed = [(m, s) for m, s in dashed if passes_line_shape_filter(m, line_min_aspect_ratio, line_min_length_px)]
    area = [(m, s) for m, s in area if passes_area_filter(m, lane_area_min_area_px)]

    solid = dedup_by_iou(solid, iou_dedup_threshold)
    dashed = dedup_by_iou(dashed, iou_dedup_threshold)
    area = dedup_by_iou(area, iou_dedup_threshold)

    is_empty = False
    if not (solid or dashed or area):
        is_empty = True
        shape = any_mask_shape(frame_dir)
        if shape is None:
            return None  # nothing to write, shape unknown -- caller counts as skipped
        label = np.zeros(shape, dtype=np.uint8)
    else:
        shape = (solid or dashed or area)[0][0].shape
        image_width = shape[1]

        solid_merged = merge_masks(solid, shape)
        dashed_merged = merge_masks(dashed, shape)
        area_merged = merge_masks(area, shape)

        # Left/right for BOTH solid lines and lane areas is decided
        # relative to the dashed line's own fitted curve, not a fixed
        # image-center column (see classify_left_right's docstring -- a
        # fixed column is wrong on a curve or a frame skewed toward one
        # side, e.g. near a corner/parking area). Falls back to a constant
        # image-center function only if the dashed line itself wasn't
        # detected in this frame.
        #
        # Fit the curve from dashed pixels that DON'T overlap solid_merged
        # -- when SAM3's dashed prompt weakly fires on solid-line pixels
        # (the same confusion that motivated solid beating dashed in the
        # paint order below), those pixels would otherwise pull the fitted
        # curve toward the solid line's own position instead of the real
        # dashed line's, which can flip the solid line's own left/right
        # classification against a curve that's partly fit from itself.
        dashed_boundary = fit_dashed_boundary(dashed_merged & ~solid_merged)
        boundary = dashed_boundary if dashed_boundary is not None else np.poly1d([image_width / 2])

        left_solid, right_solid = classify_left_right(blobs_by_area(solid_merged, max_blobs_per_class), boundary)

        if dashed_boundary is not None:
            lane_1, lane_2 = split_area_by_dashed(area_merged, dashed_boundary)
        else:
            lane_1, lane_2 = classify_left_right(blobs_by_area(area_merged, max_blobs_per_class), boundary)

        # Order matters: area classes first, then center_dashed, then
        # left_solid/right_solid last -- so line classes still beat area
        # classes everywhere, but solid also beats dashed at any overlap
        # between the two (see module docstring point 5).
        label = np.zeros(shape, dtype=np.uint8)
        if lane_1 is not None:
            label[lane_1] = class_ids["lane_1"]
        if lane_2 is not None:
            label[lane_2] = class_ids["lane_2"]
        if dashed_merged.any():
            label[dashed_merged] = class_ids["center_dashed"]
        if left_solid is not None:
            label[left_solid] = class_ids["left_solid"]
        if right_solid is not None:
            label[right_solid] = class_ids["right_solid"]
        if not label.any():
            is_empty = True

    Image.fromarray(label).save(labels_dir / f"{frame_id}.png")
    return (frame_id, frame_id_to_session(frame_id), is_empty)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(_REPO_ROOT / "ml" / "data" / "sam3_raw"))
    parser.add_argument("--labels-dir", default=str(_REPO_ROOT / "ml" / "data" / "labels"))
    parser.add_argument("--config", default=str(_REPO_ROOT / "ml" / "configs" / "labeling.yaml"))
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Parallel worker processes (each frame is independent). "
             "Default: os.cpu_count(). Use 1 for serial/debugging.",
    )
    parser.add_argument(
        "--min-score", type=float, default=None,
        help="Overrides labeling.yaml's class_min_score for ALL THREE "
             "classes uniformly (quick global strict/loose test). Default: "
             "use each class's own value from class_min_score.",
    )
    parser.add_argument(
        "--rejected-frames", default=str(_REPO_ROOT / "ml" / "data" / "rejected_frames.txt"),
        help="Optional file of frame_ids (one per line) to exclude entirely, "
             "hand-curated via review_labels_interactive.py.",
    )
    parser.add_argument("--splits-out", default=str(_REPO_ROOT / "ml" / "data" / "splits.json"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    class_min_score = config["class_min_score"]  # {solid_line, dashed_line, lane_area} -> threshold
    if args.min_score is not None:
        class_min_score = {k: args.min_score for k in class_min_score}
    roi_top_ratio = config["roi_top_ratio"]
    line_min_aspect_ratio = config["line_min_aspect_ratio"]
    line_min_length_px = config["line_min_length_px"]
    lane_area_min_area_px = config["lane_area_min_area_px"]
    iou_dedup_threshold = config["iou_dedup_threshold"]
    max_blobs_per_class = config["max_blobs_per_class"]
    class_ids = config["class_ids"]
    prompts = config["prompts"]  # {solid_line, dashed_line, lane_area} -> prompt text

    raw_dir = pathlib.Path(args.raw_dir)
    labels_dir = pathlib.Path(args.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rejected = set()
    rejected_path = pathlib.Path(args.rejected_frames)
    if rejected_path.exists():
        rejected = {line.strip() for line in rejected_path.read_text().splitlines() if line.strip()}
        print(f"Excluding {len(rejected)} hand-rejected frames from {rejected_path}")

    frame_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir() and d.name not in rejected)
    workers = args.workers if args.workers is not None else (os.cpu_count() or 1)
    print(f"Processing {len(frame_dirs)} frames with {workers} worker(s) "
          f"(class_min_score={class_min_score}, roi_top_ratio={roi_top_ratio})")

    params = {
        "labels_dir": str(labels_dir),
        "prompts": prompts,
        "class_min_score": class_min_score,
        "class_ids": class_ids,
        "roi_top_ratio": roi_top_ratio,
        "line_min_aspect_ratio": line_min_aspect_ratio,
        "line_min_length_px": line_min_length_px,
        "lane_area_min_area_px": lane_area_min_area_px,
        "iou_dedup_threshold": iou_dedup_threshold,
        "max_blobs_per_class": max_blobs_per_class,
    }
    worker_fn = partial(process_one_frame, params=params)

    empty_count = 0
    skipped_count = 0
    session_frames: Dict[str, List[str]] = {}
    total = len(frame_dirs)
    t0 = time.time()

    def handle_result(result):
        nonlocal empty_count, skipped_count
        if result is None:
            skipped_count += 1
            return
        frame_id, session, is_empty = result
        if is_empty:
            empty_count += 1
        session_frames.setdefault(session, []).append(frame_id)

    if workers == 1:
        results = map(worker_fn, frame_dirs)  # serial, no Pool overhead
    else:
        pool = Pool(workers)
        # chunksize batches frames per worker dispatch -- these are short
        # tasks, so a chunk amortizes the IPC/pickle overhead per frame.
        results = pool.imap_unordered(worker_fn, frame_dirs, chunksize=16)

    for i, result in enumerate(results, 1):
        handle_result(result)
        if i % 500 == 0 or i == total:
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (total - i) / rate if rate > 0 else 0
            print(f"  {i}/{total} ({100*i/total:.1f}%)  {rate:.0f} frames/s  ETA {eta/60:.1f} min",
                  flush=True)

    if workers != 1:
        pool.close()
        pool.join()

    if skipped_count:
        print(f"Skipped {skipped_count} frames with no instances at all (shape unknown)")
    if total:
        print(f"\nEmpty labels: {empty_count}/{total} ({100*empty_count/total:.1f}%)")
        if empty_count / total > 0.3:
            print("  -> over 30% empty: consider lowering min_score or checking the prompts.")

    rng = random.Random(args.seed)
    sessions = sorted(session_frames)
    splits = {"train": [], "val": []}
    if len(sessions) >= 2:
        rng.shuffle(sessions)
        n_val_sessions = max(1, round(len(sessions) * args.val_fraction))
        val_sessions = set(sessions[:n_val_sessions])
        for session, frame_ids in session_frames.items():
            bucket = "val" if session in val_sessions else "train"
            splits[bucket].extend(frame_ids)
        print(f"Session-level split: {len(val_sessions)}/{len(sessions)} sessions -> val")
    else:
        print(
            "WARNING: only one session found -- falling back to a frame-level "
            "random split instead of the intended session-level split. This "
            "likely overestimates val performance (train/val frames may be "
            "near-duplicates); add more sessions before trusting val metrics."
        )
        all_frame_ids = [fid for ids in session_frames.values() for fid in ids]
        rng.shuffle(all_frame_ids)
        n_val = max(1, round(len(all_frame_ids) * args.val_fraction))
        splits["val"] = all_frame_ids[:n_val]
        splits["train"] = all_frame_ids[n_val:]

    with open(args.splits_out, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Wrote {args.splits_out}: {len(splits['train'])} train / {len(splits['val'])} val")


if __name__ == "__main__":
    main()
