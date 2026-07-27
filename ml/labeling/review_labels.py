#!/usr/bin/env python3
"""Builds contact-sheet grids (frame + label overlay, frame_id printed on
each tile) for quick eyeballing of masks_to_labels.py's output -- no GUI
labeling tool, just a directory of PNGs to flip through in an image viewer.

Note down any frame_id whose label looks clearly wrong (e.g. a whole
parking-marking blob classified as a lane line, left/right or lane_1/
lane_2 swapped, or nothing at all on a frame with obvious lines) into
ml/data/rejected_frames.txt (one frame_id per line) -- masks_to_labels.py
and the training Dataset both respect that file.

    python review_labels.py --labels-dir ml/data/labels
"""
import argparse
import pathlib

import cv2
import numpy as np
from PIL import Image, ImageDraw

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# BGR, index-matched to labeling.yaml's class_ids (0=background) -- same
# table as ml/training/train.py's _CLASS_COLORS, duplicated rather than
# imported since the two live in separate, independently-runnable packages
# (ml/labeling has no dependency on ml/training).
_CLASS_COLORS = np.array([
    [0, 0, 0],        # 0 background
    [0, 0, 255],       # 1 left_solid   (red)
    [0, 255, 255],     # 2 center_dashed (yellow)
    [255, 0, 0],       # 3 right_solid  (blue)
    [0, 180, 0],       # 4 lane_1       (dark green)
    [180, 0, 180],     # 5 lane_2       (purple)
], dtype=np.uint8)


def make_tile(frame_path: pathlib.Path, label_path: pathlib.Path, tile_size: int) -> Image.Image:
    frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)  # class index 0-5, not a 0/255 mask
    overlay = _CLASS_COLORS[label]
    blended = cv2.addWeighted(frame_bgr, 0.5, overlay, 0.5, 0)
    blended = cv2.resize(blended, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
    tile = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(tile)
    label_text = frame_path.stem
    draw.rectangle([0, tile_size - 14, len(label_text) * 6 + 4, tile_size], fill=(0, 0, 0))
    draw.text((2, tile_size - 13), label_text, fill=(255, 255, 0))
    return tile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-frames-dir", default=str(_REPO_ROOT / "ml" / "data" / "raw_frames"))
    parser.add_argument("--labels-dir", default=str(_REPO_ROOT / "ml" / "data" / "labels"))
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "ml" / "data" / "review"))
    parser.add_argument("--grid-size", type=int, default=4, help="NxN tiles per sheet")
    parser.add_argument("--tile-size", type=int, default=240, help="pixels per tile")
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Only review a random sample of this many labeled frames (default: all).",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = pathlib.Path(args.raw_frames_dir)
    labels_dir = pathlib.Path(args.labels_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_paths = sorted(labels_dir.glob("*.png"))
    if args.sample is not None:
        import random
        rng = random.Random(args.seed)
        label_paths = rng.sample(label_paths, k=min(args.sample, len(label_paths)))

    pairs = []
    for label_path in label_paths:
        frame_path = raw_dir / label_path.name
        if not frame_path.exists():
            print(f"  WARNING: no raw frame for label {label_path.name}, skipping")
            continue
        pairs.append((frame_path, label_path))

    if not pairs:
        raise SystemExit("No frame/label pairs found.")

    n = args.grid_size
    tiles_per_sheet = n * n
    print(f"Building {len(pairs)} tiles into {-(-len(pairs) // tiles_per_sheet)} sheet(s)...")

    for sheet_idx in range(0, len(pairs), tiles_per_sheet):
        sheet_pairs = pairs[sheet_idx : sheet_idx + tiles_per_sheet]
        sheet = Image.new("RGB", (n * args.tile_size, n * args.tile_size), (30, 30, 30))
        for i, (frame_path, label_path) in enumerate(sheet_pairs):
            tile = make_tile(frame_path, label_path, args.tile_size)
            row, col = divmod(i, n)
            sheet.paste(tile, (col * args.tile_size, row * args.tile_size))
        sheet_path = out_dir / f"sheet_{sheet_idx // tiles_per_sheet:03d}.png"
        sheet.save(sheet_path)
        print(f"  saved {sheet_path}")

    print(f"\nReview the sheets in {out_dir}, then list bad frame_ids "
          f"(one per line) in ml/data/rejected_frames.txt")


if __name__ == "__main__":
    main()
