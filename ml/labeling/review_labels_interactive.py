#!/usr/bin/env python3
"""Interactive one-key-per-frame label review -- shows each frame+label
overlay in a window and waits for a single keypress to judge it, instead
of eyeballing review_labels.py's static contact sheets and hand-typing
frame_ids into rejected_frames.txt.

  y / space : pass (frame is fine, just advance)
  n / r     : reject -- appends this frame_id to rejected_frames.txt AND
              deletes ml/data/sam3_raw/<frame_id>/ and ml/data/labels/
              <frame_id>.png (raw_frames/<frame_id>.png is NOT deleted --
              only the SAM3/label artifacts, so a rejection can still be
              regenerated from scratch later by rerunning label_with_sam3.py
              on that one frame if reconsidered)
  b         : back to the previous frame (to undo a misclick -- does NOT
              restore an already-deleted reject, see above)
  q / ESC   : quit -- progress is saved, rerun later to resume where you left off

Needs a display (X11/Wayland) -- run this at the machine's own screen, not
over a headless SSH session without X forwarding.

    python review_labels_interactive.py
    python review_labels_interactive.py --sample 500   # cap the session length
    python review_labels_interactive.py --restart       # ignore prior progress
"""
import argparse
import pathlib
import random
import shutil

import cv2
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Same table as review_labels.py/train.py -- duplicated for the same reason
# (ml/labeling has no dependency on ml/training).
_CLASS_COLORS = np.array([
    [0, 0, 0],        # 0 background
    [0, 0, 255],       # 1 left_solid   (red)
    [0, 255, 255],     # 2 center_dashed (yellow)
    [255, 0, 0],       # 3 right_solid  (blue)
    [0, 180, 0],       # 4 lane_1       (dark green)
    [180, 0, 180],     # 5 lane_2       (purple)
], dtype=np.uint8)

_WINDOW_NAME = "review (y=pass, n=reject, b=back, q=quit)"
_MAX_DISPLAY_DIM = 1280


def load_id_set(path: pathlib.Path) -> set:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def append_id(path: pathlib.Path, frame_id: str) -> None:
    with open(path, "a") as f:
        f.write(frame_id + "\n")


def build_display(frame_path: pathlib.Path, label_path: pathlib.Path, frame_id: str) -> np.ndarray:
    frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    label = cv2.imread(str(label_path), cv2.IMREAD_GRAYSCALE)
    overlay = _CLASS_COLORS[label]
    blended = cv2.addWeighted(frame_bgr, 0.5, overlay, 0.5, 0)

    h, w = blended.shape[:2]
    scale = _MAX_DISPLAY_DIM / max(h, w)
    if scale < 1.0:
        blended = cv2.resize(blended, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    cv2.rectangle(blended, (0, 0), (blended.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(blended, frame_id, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    return blended


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-frames-dir", default=str(_REPO_ROOT / "ml" / "data" / "raw_frames"))
    parser.add_argument("--labels-dir", default=str(_REPO_ROOT / "ml" / "data" / "labels"))
    parser.add_argument("--sam3-raw-dir", default=str(_REPO_ROOT / "ml" / "data" / "sam3_raw"))
    parser.add_argument("--rejected-frames", default=str(_REPO_ROOT / "ml" / "data" / "rejected_frames.txt"))
    parser.add_argument(
        "--reviewed-frames", default=str(_REPO_ROOT / "ml" / "data" / "reviewed_frames.txt"),
        help="Tracks every frame_id already judged (pass OR reject) so a "
             "rerun skips them and resumes where the last session left off.",
    )
    parser.add_argument(
        "--sample", type=int, default=None,
        help="Cap the session to this many (randomly ordered) frames, "
             "instead of all remaining unreviewed ones. Ignored if "
             "--every-n is set.",
    )
    parser.add_argument(
        "--every-n", type=int, default=None,
        help="Systematic spot-check instead of random sampling: take every "
             "Nth frame_id (in sorted/session order, not shuffled) -- gives "
             "even coverage across all sessions/times instead of whatever "
             "random.sample happens to pick. E.g. --every-n 5 reviews 1/5 "
             "of the dataset; raise N (20, 50...) if even that's too many.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--restart", action="store_true", help="Ignore reviewed_frames.txt and start over.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_dir = pathlib.Path(args.raw_frames_dir)
    labels_dir = pathlib.Path(args.labels_dir)
    sam3_raw_dir = pathlib.Path(args.sam3_raw_dir)
    rejected_path = pathlib.Path(args.rejected_frames)
    reviewed_path = pathlib.Path(args.reviewed_frames)

    reviewed = set() if args.restart else load_id_set(reviewed_path)
    rejected = load_id_set(rejected_path)

    frame_ids = sorted(p.stem for p in labels_dir.glob("*.png") if p.stem not in reviewed)
    if not frame_ids:
        print("Nothing left to review (see --restart to go through everything again).")
        return

    if args.every_n is not None:
        frame_ids = frame_ids[:: args.every_n]  # sorted order preserved -- see --every-n's help
    else:
        rng = random.Random(args.seed)
        rng.shuffle(frame_ids)
        if args.sample is not None:
            frame_ids = frame_ids[: args.sample]

    print(f"{len(frame_ids)} frames queued this session ({len(reviewed)} already reviewed before).")
    print("y/space=pass  n/r=reject  b=back  q=quit (progress is saved as you go)")

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    idx = 0
    n_rejected_this_session = 0
    while idx < len(frame_ids):
        frame_id = frame_ids[idx]
        frame_path = raw_dir / f"{frame_id}.png"
        label_path = labels_dir / f"{frame_id}.png"
        if not frame_path.exists():
            print(f"  WARNING: no raw frame for {frame_id}, skipping")
            idx += 1
            continue

        display = build_display(frame_path, label_path, frame_id)
        cv2.imshow(_WINDOW_NAME, display)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord('q'), 27):  # q or ESC
            break
        elif key == ord('b'):
            idx = max(0, idx - 1)
            continue
        elif key in (ord('n'), ord('r')):
            if frame_id not in rejected:
                append_id(rejected_path, frame_id)
                rejected.add(frame_id)
                n_rejected_this_session += 1
            if label_path.exists():
                label_path.unlink()
            frame_sam3_dir = sam3_raw_dir / frame_id
            if frame_sam3_dir.exists():
                shutil.rmtree(frame_sam3_dir)
        elif key not in (ord('y'), ord(' ')):
            continue  # unrecognized key -- redo this frame instead of silently advancing

        if frame_id not in reviewed:
            append_id(reviewed_path, frame_id)
            reviewed.add(frame_id)
        idx += 1

    cv2.destroyAllWindows()
    print(f"\nSession done: {idx} frames judged, {n_rejected_this_session} newly rejected.")
    print(f"Total reviewed so far: {len(reviewed)}. Rerun anytime to continue.")


if __name__ == "__main__":
    main()
