#!/usr/bin/env python3
"""Reports, per prompt, the score distribution of every instance SAM3
already captured into ml/data/sam3_raw/ -- so "is class_min_score too high?"
can be answered from data instead of guessed.

Unlike score_probe.py (which re-runs SAM3 on a few frames to sanity-check a
NEW prompt/session), this reads the already-saved raw instances and needs
no GPU: it just tallies meta.json scores across the whole dataset. The key
number it prints per class is how many instances sit in
[capture_threshold, class_min_score) -- i.e. captured by SAM3 but currently
thrown away by the filter. A large count there means lowering
class_min_score (and re-running masks_to_labels.py, which is free -- no SAM3
re-run) would recover real detections; a small count means the threshold
isn't what's losing your lines (look at the prompt/SAM3 detection itself).

    conda activate sam3   # (only needs numpy/pyyaml really)
    python score_distribution.py
"""
import argparse
import json
import pathlib

import numpy as np
import yaml

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_BUCKETS = [0.0, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70, 1.01]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", default=str(_REPO_ROOT / "ml" / "data" / "sam3_raw"))
    parser.add_argument("--config", default=str(_REPO_ROOT / "ml" / "configs" / "labeling.yaml"))
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only scan the first N frame dirs (quick check on a subset).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    prompts = config["prompts"]  # {solid_line, dashed_line, lane_area} -> text
    class_min_score = config["class_min_score"]
    capture_threshold = config.get("capture_threshold", 0.0)

    raw_dir = pathlib.Path(args.raw_dir)
    frame_dirs = sorted(d for d in raw_dir.iterdir() if d.is_dir())
    if args.limit is not None:
        frame_dirs = frame_dirs[: args.limit]
    print(f"Scanning {len(frame_dirs)} frame dirs under {raw_dir}\n")

    # per class_key -> list of scores, and per-frame max score
    scores = {k: [] for k in prompts}
    top_per_frame = {k: [] for k in prompts}
    frames_with_any = {k: 0 for k in prompts}

    for frame_dir in frame_dirs:
        meta_path = frame_dir / "meta.json"
        if not meta_path.exists():
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        for class_key, prompt_text in prompts.items():
            insts = meta["prompts"].get(prompt_text, [])
            frame_scores = [inst["score"] for inst in insts]
            scores[class_key].extend(frame_scores)
            if frame_scores:
                top_per_frame[class_key].append(max(frame_scores))
                frames_with_any[class_key] += 1

    n_frames = len(frame_dirs)
    for class_key, prompt_text in prompts.items():
        arr = np.array(scores[class_key])
        thr = class_min_score[class_key]
        print(f"=== {class_key}  (prompt={prompt_text!r}, class_min_score={thr}) ===")
        if arr.size == 0:
            print("  no instances captured at all -- SAM3 never detected this concept.\n")
            continue

        # per-instance bucket histogram
        print("  per-instance score histogram (all captured instances):")
        for lo, hi in zip(_BUCKETS[:-1], _BUCKETS[1:]):
            c = int(((arr >= lo) & (arr < hi)).sum())
            bar = "#" * min(60, c * 60 // max(arr.size, 1))
            print(f"    [{lo:.2f},{hi:.2f}) {c:7d}  {bar}")

        # the money number: captured but filtered out by the current threshold
        in_gap = int(((arr >= capture_threshold) & (arr < thr)).sum())
        kept = int((arr >= thr).sum())
        print(f"  captured-but-DROPPED by threshold [{capture_threshold:.2f},{thr:.2f}): "
              f"{in_gap} instances")
        print(f"  kept (>= {thr:.2f}): {kept} instances")

        # per-FRAME recall view: how many frames have their best instance
        # below the threshold (i.e. the whole class is lost in that frame)
        tops = np.array(top_per_frame[class_key])
        frames_lost = int((tops < thr).sum()) if tops.size else 0
        print(f"  frames where even the BEST instance is < {thr:.2f} "
              f"(class fully lost): {frames_lost}/{frames_with_any[class_key]} "
              f"frames-with-any  (of {n_frames} total)")
        # what threshold would keep the best instance in 90% of frames?
        if tops.size:
            p10 = float(np.percentile(tops, 10))
            print(f"  -> lowering class_min_score to ~{p10:.2f} would keep the best "
                  f"instance in ~90% of frames that have one")
        print()

    print("If a class shows many DROPPED instances / many fully-lost frames just "
          "below its threshold, lower class_min_score in labeling.yaml and rerun "
          "masks_to_labels.py (free -- no SAM3 rerun). If instead there are few "
          "captured instances at all, the threshold isn't the problem -- it's SAM3 "
          "detection/the prompt.")


if __name__ == "__main__":
    main()
