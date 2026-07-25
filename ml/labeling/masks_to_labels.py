#!/usr/bin/env python3
"""Turns label_with_sam3.py's raw per-instance masks into one binary
training label per frame (ml/README.md step 3). Reads only
ml/data/sam3_raw/ -- never touches SAM3/GPU -- so every filter parameter
here (score cutoff, ROI, shape) can be re-tuned and re-run for free.

Filtering, in order:
  1. --min-score: since label_with_sam3.py already applied labeling.yaml's
     confidence_threshold when it saved instances, this can only raise the
     effective threshold above that (anything scoring lower was never
     saved) -- it cannot recover instances below the original labeling
     threshold without re-running label_with_sam3.py itself. Defaults to
     the same value as labeling.yaml so it's a no-op unless overridden.
  2. ROI top-crop: zero out mask pixels above roi_top_ratio (ceiling/walls
     -- never floor). Deliberately does NOT reuse lane_detector.yaml's
     roi_left_ratio/roi_top_width_ratio/roi_bottom_width_ratio -- that
     narrower trapezoid (and the left-side cut) exists only because the
     Hough-based pipeline can't otherwise tell the dashed center line from
     the solid right boundary; a segmentation model has no such limitation
     and benefits from seeing every white marking in frame.
  3. Shape filter: reject instances whose minAreaRect long/short ratio or
     long-side length falls below the configured minimums (rejects blobby
     false positives -- hands, shoes, glare -- the same idea as
     LaneDetector's own aspect-ratio filter, just applied directly to a
     mask's contour instead of a Hough cluster).
  4. IoU dedup: instances (possibly from different prompts) whose masks
     overlap above iou_dedup_threshold are collapsed to the highest-scoring
     one before the final OR-merge, so duplicate prompt hits don't multiply
     unnecessarily in the output (harmless for the merge itself, just
     avoids wasted bookkeeping).

Also writes ml/data/splits.json (session-level train/val split) and prints
the fraction of frames left with an empty label after filtering -- a high
number is a signal to loosen thresholds or add prompts, not to hand-edit
labels.
"""
import argparse
import json
import pathlib
import random
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from PIL import Image

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def frame_id_to_session(frame_id: str) -> str:
    # collect_frames.py writes "<session>_<index:06d>_<stamp_ns>" -- index
    # and stamp_ns are always the last two, purely-numeric segments, so
    # rsplit lets a session name itself contain underscores safely.
    parts = frame_id.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else frame_id


def load_instances(frame_dir: pathlib.Path, min_score: float) -> List[Tuple[np.ndarray, float]]:
    with open(frame_dir / "meta.json") as f:
        meta = json.load(f)
    instances = []
    for prompt_instances in meta["prompts"].values():
        for inst in prompt_instances:
            if inst["score"] < min_score:
                continue
            mask = np.array(Image.open(frame_dir / inst["mask_file"])) > 0
            instances.append((mask, inst["score"]))
    return instances


def apply_roi_top_crop(mask: np.ndarray, roi_top_ratio: float) -> np.ndarray:
    top_row = int(mask.shape[0] * roi_top_ratio)
    mask = mask.copy()
    mask[:top_row, :] = False
    return mask


def passes_shape_filter(mask: np.ndarray, min_aspect_ratio: float, min_length_px: float) -> bool:
    mask_u8 = (mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return False
    largest = max(contours, key=cv2.contourArea)
    (_, _), (w, h), _ = cv2.minAreaRect(largest)
    long_side, short_side = max(w, h), max(min(w, h), 1e-3)
    return (long_side / short_side) >= min_aspect_ratio and long_side >= min_length_px


def iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.count_nonzero(a & b)
    union = np.count_nonzero(a | b)
    return intersection / union if union else 0.0


def dedup_by_iou(
    instances: List[Tuple[np.ndarray, float]], iou_threshold: float
) -> List[Tuple[np.ndarray, float]]:
    # highest score first, so a later lower-scoring near-duplicate gets
    # dropped in favor of the one already kept
    instances = sorted(instances, key=lambda t: -t[1])
    kept: List[Tuple[np.ndarray, float]] = []
    for mask, score in instances:
        if any(iou(mask, kept_mask) > iou_threshold for kept_mask, _ in kept):
            continue
        kept.append((mask, score))
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(_REPO_ROOT / "ml" / "data" / "sam3_raw"))
    parser.add_argument("--labels-dir", default=str(_REPO_ROOT / "ml" / "data" / "labels"))
    parser.add_argument("--config", default=str(_REPO_ROOT / "ml" / "configs" / "labeling.yaml"))
    parser.add_argument(
        "--min-score", type=float, default=None,
        help="Defaults to labeling.yaml's confidence_threshold (i.e. no "
             "additional filtering beyond what label_with_sam3.py already did).",
    )
    parser.add_argument(
        "--rejected-frames", default=str(_REPO_ROOT / "ml" / "data" / "rejected_frames.txt"),
        help="Optional file of frame_ids (one per line) to exclude entirely, "
             "hand-curated via review_labels.py's contact sheets.",
    )
    parser.add_argument("--splits-out", default=str(_REPO_ROOT / "ml" / "data" / "splits.json"))
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    min_score = args.min_score if args.min_score is not None else config["confidence_threshold"]
    roi_top_ratio = config["roi_top_ratio"]
    min_aspect_ratio = config["min_aspect_ratio"]
    min_length_px = config["min_length_px"]
    iou_dedup_threshold = config["iou_dedup_threshold"]

    raw_dir = pathlib.Path(args.raw_dir)
    labels_dir = pathlib.Path(args.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    rejected = set()
    rejected_path = pathlib.Path(args.rejected_frames)
    if rejected_path.exists():
        rejected = {line.strip() for line in rejected_path.read_text().splitlines() if line.strip()}
        print(f"Excluding {len(rejected)} hand-rejected frames from {rejected_path}")

    frame_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir() and d.name not in rejected)
    print(f"Processing {len(frame_dirs)} frames (min_score={min_score}, roi_top_ratio={roi_top_ratio})")

    empty_count = 0
    session_frames: Dict[str, List[str]] = {}
    for frame_dir in frame_dirs:
        frame_id = frame_dir.name
        instances = load_instances(frame_dir, min_score)
        instances = [(apply_roi_top_crop(m, roi_top_ratio), s) for m, s in instances]
        instances = [(m, s) for m, s in instances if passes_shape_filter(m, min_aspect_ratio, min_length_px)]
        instances = dedup_by_iou(instances, iou_dedup_threshold)

        if not instances:
            empty_count += 1
            # still write an all-background label -- a real "no marking
            # visible" frame is a legitimate (if sparse) training signal,
            # see labeling.yaml's comment on empty-label ratio monitoring.
            first_any = load_instances(frame_dir, min_score=0.0)
            shape = first_any[0][0].shape if first_any else None
            if shape is None:
                print(f"  WARNING: {frame_id} has no instances at all (any score) -- skipping, shape unknown")
                continue
            merged = np.zeros(shape, dtype=bool)
        else:
            merged = np.zeros_like(instances[0][0])
            for m, _ in instances:
                merged |= m

        Image.fromarray((merged.astype(np.uint8)) * 255).save(labels_dir / f"{frame_id}.png")
        session_frames.setdefault(frame_id_to_session(frame_id), []).append(frame_id)

    total = len(frame_dirs)
    if total:
        print(f"\nEmpty labels: {empty_count}/{total} ({100*empty_count/total:.1f}%)")
        if empty_count / total > 0.3:
            print("  -> over 30% empty: consider lowering min_score or adding prompts.")

    # Session-level train/val split (falls back to a frame-level split with
    # a warning if there's only one session -- can't split by session then).
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
