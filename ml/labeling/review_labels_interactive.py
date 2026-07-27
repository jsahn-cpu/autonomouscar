#!/usr/bin/env python3
"""Interactive one-key-per-frame label review -- shows each frame+label
overlay in a window and waits for a single keypress to judge it.

Whitelist model: the meaningful record is ml/data/verified_frames.txt, a
list of ONLY the frames whose label you eyeballed and approved. Training
uses that list directly (train.yaml: require_verified) -- anything not on
it is simply not used. Nothing is ever deleted, and there is no "rejected"
list: a frame you don't approve just doesn't get added to the whitelist.

  y / space : verify -- add this frame_id to verified_frames.txt (approved)
  n / s     : skip -- do NOT approve it (and if it was approved earlier,
              e.g. correcting a misclick after 'b', remove it from the
              whitelist). Nothing is deleted; the frame just stays off the
              training whitelist.
  b         : back to the previous frame (re-judge it; the last decision wins)
  q / ESC   : quit -- progress is saved, rerun later to resume where you left off

ml/data/reviewed_frames.txt separately tracks every frame you've SEEN
(verified or skipped) purely so a rerun resumes where you left off instead
of re-showing frames -- it is not used by training, only the whitelist is.

Needs a display (X11/Wayland) -- run this at the machine's own screen, not
over a headless SSH session without X forwarding.

    python review_labels_interactive.py
    python review_labels_interactive.py --every-n 20     # spot-check every 20th frame
    python review_labels_interactive.py --restart        # ignore prior progress
"""
import argparse
import pathlib
import random

import cv2
import numpy as np

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# Same table as train.py -- duplicated for the same reason
# (ml/labeling has no dependency on ml/training).
_CLASS_COLORS = np.array([
    [0, 0, 0],        # 0 background
    [0, 0, 255],       # 1 left_solid   (red)
    [0, 255, 255],     # 2 center_dashed (yellow)
    [255, 0, 0],       # 3 right_solid  (blue)
    [0, 180, 0],       # 4 lane_1       (dark green)
    [180, 0, 180],     # 5 lane_2       (purple)
], dtype=np.uint8)

_WINDOW_NAME = "review (y=verify, n/s=skip, b=back, q=quit)"
_MAX_DISPLAY_DIM = 1280


def load_id_set(path: pathlib.Path) -> set:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def append_id(path: pathlib.Path, frame_id: str) -> None:
    with open(path, "a") as f:
        f.write(frame_id + "\n")


def remove_id(path: pathlib.Path, frame_id: str) -> None:
    """Rewrite `path` without frame_id -- used to un-verify a frame when it's
    re-judged as skip after a 'b' (back). Order-preserving so the file stays
    readable/diffable."""
    if not path.exists():
        return
    kept = [ln for ln in path.read_text().splitlines() if ln.strip() and ln.strip() != frame_id]
    path.write_text("".join(ln + "\n" for ln in kept))


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
    parser.add_argument(
        "--verified-frames", default=str(_REPO_ROOT / "ml" / "data" / "verified_frames.txt"),
        help="The whitelist -- frame_ids approved (y). Training uses only "
             "these when train.yaml's require_verified is on.",
    )
    parser.add_argument(
        "--reviewed-frames", default=str(_REPO_ROOT / "ml" / "data" / "reviewed_frames.txt"),
        help="Tracks every frame_id already SEEN (verified or skipped) so a "
             "rerun skips them and resumes where the last session left off. "
             "Not used by training -- only the whitelist is.",
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
    verified_path = pathlib.Path(args.verified_frames)
    reviewed_path = pathlib.Path(args.reviewed_frames)

    reviewed = set() if args.restart else load_id_set(reviewed_path)
    verified = load_id_set(verified_path)

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

    print(f"{len(frame_ids)} frames queued this session ({len(reviewed)} already seen before).")
    print("y/space=verify  n/s=skip  b=back  q=quit (progress is saved as you go)")

    cv2.namedWindow(_WINDOW_NAME, cv2.WINDOW_NORMAL)
    idx = 0
    n_verified_this_session = 0
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
        elif key in (ord('y'), ord(' ')):
            # Verify -- add to the training whitelist. Nothing else is
            # touched; frames simply aren't used unless they're on this list.
            if frame_id not in verified:
                append_id(verified_path, frame_id)
                verified.add(frame_id)
                n_verified_this_session += 1
        elif key in (ord('n'), ord('s')):
            # Skip -- leave it OFF the whitelist. If it was verified earlier
            # (e.g. correcting a misclick after 'b'), un-verify it so the
            # last decision wins.
            if frame_id in verified:
                remove_id(verified_path, frame_id)
                verified.discard(frame_id)
                n_verified_this_session = max(0, n_verified_this_session - 1)
        else:
            continue  # unrecognized key -- redo this frame instead of silently advancing

        if frame_id not in reviewed:
            append_id(reviewed_path, frame_id)
            reviewed.add(frame_id)
        idx += 1

    cv2.destroyAllWindows()
    print(f"\nSession done: {idx} frames judged, {n_verified_this_session} newly verified.")
    print(f"Total verified (training whitelist): {len(verified)}. "
          f"Total seen: {len(reviewed)}. Rerun anytime to continue.")


if __name__ == "__main__":
    main()
