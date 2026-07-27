#!/usr/bin/env python3
"""Extract frames from recorded rosbag2 bags OFFLINE -- reads each bag's
CompressedImage messages directly (rosbag2_py) and writes every Nth as a
PNG, instead of `ros2 bag play` + a live subscriber (collect_frames.py).

Faster and more reliable than playback: no real-time pacing, no dropped
messages, no two-process dance. Output filenames match collect_frames.py's
convention (<session>_<index:06d>_<stamp_ns>.png) so the rest of the
pipeline (session-level splits etc.) treats them identically.

    source /opt/ros/humble/setup.bash
    python ml/labeling/extract_frames_from_bags.py --bags-dir data2 --every-n 2
"""
import argparse
import pathlib

import cv2
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bags-dir", default=str(_REPO_ROOT / "data2"),
                        help="Directory containing rosbag2_* bag dirs.")
    parser.add_argument("--output-dir", default=str(_REPO_ROOT / "ml" / "data" / "raw_frames"))
    parser.add_argument("--topic", default="/camera/front/image/compressed")
    parser.add_argument("--every-n", type=int, default=2,
                        help="Save 1 of every N messages (default 2).")
    return parser.parse_args()


def bag_dirs(bags_dir: pathlib.Path):
    # A bag is a directory containing metadata.yaml.
    return sorted(d for d in bags_dir.iterdir() if d.is_dir() and (d / "metadata.yaml").exists())


def extract_one(bag_dir: pathlib.Path, topic: str, every_n: int, out_dir: pathlib.Path) -> int:
    session = bag_dir.name
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                    output_serialization_format="cdr"),
    )
    reader.set_filter(rosbag2_py.StorageFilter(topics=[topic]))

    msg_idx = 0
    saved = 0
    while reader.has_next():
        _topic, data, t_ns = reader.read_next()
        if msg_idx % every_n == 0:
            msg = deserialize_message(data, CompressedImage)
            arr = np.frombuffer(bytes(msg.data), dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                fname = f"{session}_{saved:06d}_{t_ns}.png"
                cv2.imwrite(str(out_dir / fname), img)
                saved += 1
        msg_idx += 1
    return saved


def main() -> None:
    args = parse_args()
    bags_dir = pathlib.Path(args.bags_dir)
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bags = bag_dirs(bags_dir)
    if not bags:
        raise SystemExit(f"No bags (dirs with metadata.yaml) under {bags_dir}")
    print(f"Extracting from {len(bags)} bags on {args.topic!r} (every {args.every_n}th) -> {out_dir}")

    total = 0
    for bag in bags:
        n = extract_one(bag, args.topic, args.every_n, out_dir)
        total += n
        print(f"  {bag.name}: {n} frames")
    print(f"\nDone. {total} frames written. Output now holds "
          f"{len(list(out_dir.glob('*.png')))} PNGs total.")


if __name__ == "__main__":
    main()
