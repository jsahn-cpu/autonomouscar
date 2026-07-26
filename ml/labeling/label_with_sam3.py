#!/usr/bin/env python3
"""Batch offline auto-labeling with SAM3 (ml/README.md step 2).

For every sampled frame, runs every configured prompt through SAM3 and
dumps each RAW instance mask + its score to disk -- filtering/merging is
deliberately NOT done here (see masks_to_labels.py) so that threshold/ROI/
shape-filter tuning never requires re-running SAM3 itself, which is the
expensive part on a 4GB GPU.

Run inside the `sam3` conda env:
    conda activate sam3
    python label_with_sam3.py --frames-dir ml/data/raw_frames --session <name>

Always run with a small --limit first (e.g. 20-30) to see the printed
per-frame timing and full-batch estimate before committing to the whole
session -- see ml/README.md step 2 checklist. Run score_probe.py first if
this is a new session (different lighting can shift the useful
confidence_threshold).
"""
import argparse
import json
import pathlib
import re
import time

import torch
import yaml
from PIL import Image

from sam3_utils import autocast_ctx, load_model, make_processor

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def slugify(prompt: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", prompt.lower()).strip("-")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames-dir", default=str(_REPO_ROOT / "ml" / "data" / "raw_frames"))
    parser.add_argument(
        "--session", default=None,
        help="Only label frames whose filename starts with this session tag. "
             "Default: all frames under --frames-dir.",
    )
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "ml" / "data" / "sam3_raw"))
    parser.add_argument("--config", default=str(_REPO_ROOT / "ml" / "configs" / "labeling.yaml"))
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only process the first N matching frames -- use this for a "
             "timing dry run before launching the full batch.",
    )
    parser.add_argument(
        "--skip-existing", action="store_true", default=True,
        help="Skip frames that already have an output directory (default "
             "on, so an interrupted batch can be safely re-run/resumed).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    # prompts is a {class_key: prompt_text} dict (see labeling.yaml) -- only
    # the text values are sent to SAM3; meta.json is keyed by prompt text,
    # which masks_to_labels.py routes on for its solid/dashed/area handling.
    prompts = list(config["prompts"].values())
    confidence_threshold = config["confidence_threshold"]

    frames_dir = pathlib.Path(args.frames_dir)
    pattern = f"{args.session}_*.png" if args.session else "*.png"
    frames = sorted(frames_dir.glob(pattern))
    if args.limit is not None:
        frames = frames[: args.limit]
    if not frames:
        raise SystemExit(f"No frames matching {pattern!r} under {frames_dir}")

    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.skip_existing:
        pending = [f for f in frames if not (out_dir / f.stem / "meta.json").exists()]
        skipped = len(frames) - len(pending)
        if skipped:
            print(f"Skipping {skipped} already-labeled frames (--skip-existing)")
        frames = pending

    print(
        f"Labeling {len(frames)} frames x {len(prompts)} prompts "
        f"(confidence_threshold={confidence_threshold}) -> {out_dir}"
    )
    if not frames:
        print("Nothing to do.")
        return

    model = load_model()
    processor = make_processor(model, confidence_threshold=confidence_threshold)

    per_frame_times = []
    t_start = time.time()
    with autocast_ctx():
        for i, frame_path in enumerate(frames):
            t0 = time.time()
            frame_id = frame_path.stem
            frame_out_dir = out_dir / frame_id
            frame_out_dir.mkdir(parents=True, exist_ok=True)

            image = Image.open(frame_path).convert("RGB")
            inference_state = processor.set_image(image)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            meta = {"frame": str(frame_path), "prompts": {}}
            instance_idx = 0
            for prompt in prompts:
                output = processor.set_text_prompt(state=inference_state, prompt=prompt)
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                prompt_meta = []
                mask_arr = masks.cpu().numpy() if torch.is_tensor(masks) else masks
                for k in range(mask_arr.shape[0]):
                    m = mask_arr[k]
                    if m.ndim == 3:
                        m = m[0]
                    score = float(scores[k])
                    fname = f"instance_{instance_idx}_{slugify(prompt)}_{score:.3f}.png"
                    Image.fromarray((m.astype("uint8")) * 255).save(frame_out_dir / fname)
                    prompt_meta.append(
                        {
                            "mask_file": fname,
                            "score": score,
                            "box": boxes[k].tolist() if torch.is_tensor(boxes) else list(boxes[k]),
                        }
                    )
                    instance_idx += 1
                meta["prompts"][prompt] = prompt_meta

            with open(frame_out_dir / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)

            dt = time.time() - t0
            per_frame_times.append(dt)
            n_instances = sum(len(v) for v in meta["prompts"].values())
            print(f"[{i+1}/{len(frames)}] {frame_id}: {n_instances} instances in {dt:.2f}s")

    total = time.time() - t_start
    avg = sum(per_frame_times) / len(per_frame_times)
    print(f"\nDone: {len(frames)} frames in {total:.1f}s (avg {avg:.2f}s/frame).")
    if args.limit is not None:
        all_frames_count = len(sorted(frames_dir.glob(
            f"{args.session}_*.png" if args.session else "*.png"
        )))
        est_total = avg * all_frames_count
        print(
            f"Estimated time for all {all_frames_count} frames in this "
            f"selection: {est_total/60:.1f} min. Re-run without --limit "
            f"(or with a larger one) once this looks acceptable."
        )


if __name__ == "__main__":
    main()
