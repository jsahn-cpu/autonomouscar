#!/usr/bin/env python3
"""Cheap pre-check for a new labeling session: reports SAM3's raw score
distribution per prompt on a handful of sample frames, WITHOUT running the
expensive full-resolution mask interpolation. Run this before
label_with_sam3.py on any new session (different lighting/framing can shift
scores enough that the configured confidence_threshold/prompt set stop
being useful) -- see ml/README.md step 2 checklist.

Run inside the `sam3` conda env:
    conda activate sam3
    python score_probe.py --frames-dir ml/data/raw_frames --session <name> \
        --prompts "white lane line" "lane line" "road marking"
"""
import argparse
import pathlib
import random

import torch
from PIL import Image

from sam3_utils import autocast_ctx, load_model, make_processor, raw_scores_for_prompt

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", default=str(_REPO_ROOT / "ml" / "data" / "raw_frames"))
    parser.add_argument(
        "--session", default=None,
        help="Only probe frames whose filename starts with this session tag "
             "(as written by collect_frames.py --session-name). Default: all.",
    )
    parser.add_argument(
        "--num-samples", type=int, default=10,
        help="How many frames to randomly sample for the probe (default 10 "
             "-- this is meant to be a quick sanity check, not exhaustive).",
    )
    parser.add_argument(
        "--prompts", nargs="+",
        default=["solid lane line", "dashed lane line", "road lane"],
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames_dir = pathlib.Path(args.frames_dir)
    pattern = f"{args.session}_*.png" if args.session else "*.png"
    all_frames = sorted(frames_dir.glob(pattern))
    if not all_frames:
        raise SystemExit(f"No frames matching {pattern!r} under {frames_dir}")

    rng = random.Random(args.seed)
    sample = rng.sample(all_frames, k=min(args.num_samples, len(all_frames)))
    print(f"Probing {len(sample)}/{len(all_frames)} frames with prompts={args.prompts}")

    model = load_model()
    processor = make_processor(model)

    # prompt -> list of per-frame top score
    top_scores = {p: [] for p in args.prompts}
    with autocast_ctx():
        for frame_path in sample:
            image = Image.open(frame_path).convert("RGB")
            inference_state = processor.set_image(image)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            for prompt in args.prompts:
                scores = raw_scores_for_prompt(model, processor, inference_state, prompt)
                top_scores[prompt].append(float(scores.max()))

    print("\nPer-prompt top score across sampled frames (max/mean/min):")
    for prompt, scores in top_scores.items():
        print(
            f"  {prompt!r:25s} max={max(scores):.3f} "
            f"mean={sum(scores)/len(scores):.3f} min={min(scores):.3f}"
        )
    best_overall = max(top_scores, key=lambda p: sum(top_scores[p]) / len(top_scores[p]))
    print(f"\nBest-performing prompt on average: {best_overall!r}")
    print(
        "If these top scores are mostly well below your configured "
        "confidence_threshold (labeling.yaml, default 0.3), lower it or add "
        "prompts before running the full label_with_sam3.py batch."
    )


if __name__ == "__main__":
    main()
