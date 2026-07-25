#!/usr/bin/env python3
"""Samples frames from one or more camera topics and dumps them to disk as
PNGs, for later offline SAM3 auto-labeling (see ml/README.md step 1).

Run with the ROS system Python (source install/setup.bash first) -- this is
a manual, one-off collection tool, not a persistent node, so it is not
registered as an autodrive_tools console_scripts entry point.

Works identically against a live camera or a `ros2 bag play`-ed recording,
since it's just a plain subscriber. Record with the stock CLI and play it
back through this script to extract frames from it, e.g.:
    ros2 bag record -o <bag> /camera/front/image_mono/compressed
    ros2 bag play <bag>   # in another terminal, while this script runs

Multiple --topic values are supported in one run (e.g. front+back cameras
recorded together in the same bag) -- each gets its own frame counter and,
when more than one topic is given, its own session suffix derived from the
topic name (so "/camera/front" + "/camera/back" under --session-name mybag
become sessions "mybag_front"/"mybag_back") so downstream session-level
train/val splitting still treats them as distinct recordings. With a single
topic (the common case), the session name is used as-is.
"""
import argparse
import pathlib
import sys

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def topic_suffix(topic: str) -> str:
    return topic.strip("/").split("/")[-1]


class _TopicCollector:
    """Per-topic frame counter + file writer -- one of these per --topic,
    so front/back (or any other simultaneous topics) don't share a frame
    index or collide on filenames."""

    def __init__(self, node: Node, topic: str, session: str, out_dir: pathlib.Path, every_n: int, max_frames) -> None:
        self._node = node
        self._session = session
        self._out_dir = out_dir
        self._every_n = every_n
        self._max_frames = max_frames
        self._seen_count = 0
        self.saved_count = 0
        node.create_subscription(CompressedImage, topic, self._on_image, 10)
        node.get_logger().info(f"Collecting from {topic} -> session={session!r}")

    def _on_image(self, msg: CompressedImage) -> None:
        # Decode the JPEG bytes directly with OpenCV instead of cv_bridge --
        # cv_bridge's compressed_imgmsg_to_cv2 always calls cvtColor2(im,
        # 'bgr8', desired_encoding) internally regardless of the actual
        # decoded channel count, which crashes on a genuinely single-channel
        # source (see lane_detector_node.py's use of 'passthrough' to work
        # around this for the live ROS path). Decoding raw bytes ourselves
        # sidesteps that entirely since there's no ROS message to build.
        #
        # Always decoded to grayscale here even for genuinely color sources
        # (e.g. real vehicle-recorded bags, which turned out to publish
        # plain color JPEGs unlike the live rig's grayscale mono topic) --
        # every later stage (SAM3 labeling, LaneDetector, the trained
        # model) is fine with either, and keeping raw_frames uniformly
        # single-channel avoids a mixed-format dataset downstream.
        self._seen_count += 1
        if (self._seen_count - 1) % self._every_n != 0:
            return

        arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            self._node.get_logger().warn("Failed to decode a frame, skipping")
            return

        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        filename = f"{self._session}_{self.saved_count:06d}_{stamp_ns}.png"
        cv2.imwrite(str(self._out_dir / filename), image)
        self.saved_count += 1
        if self.saved_count % 20 == 0:
            self._node.get_logger().info(f"[{self._session}] saved {self.saved_count} frames so far")

        if self._max_frames is not None and self.saved_count >= self._max_frames:
            self._node.get_logger().info(f"[{self._session}] reached --max-frames={self._max_frames}")
            raise SystemExit(0)


class FrameCollector(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("frame_collector")
        out_dir = pathlib.Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        multi = len(args.topic) > 1
        self._collectors = [
            _TopicCollector(
                self, topic,
                session=f"{args.session_name}_{topic_suffix(topic)}" if multi else args.session_name,
                out_dir=out_dir, every_n=args.every_n_frames, max_frames=args.max_frames,
            )
            for topic in args.topic
        ]

    def total_saved(self) -> int:
        return sum(c.saved_count for c in self._collectors)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-name", required=True,
        help="Tag embedded in output filenames -- also used later for "
             "session-level train/val splitting, so pick something that "
             "identifies this recording session (e.g. lighting/time-of-day, "
             "or a bag's name). With multiple --topic values, each gets "
             "this as a prefix plus its own topic-derived suffix.",
    )
    parser.add_argument(
        "--output-dir", default=str(_REPO_ROOT / "ml" / "data" / "raw_frames"),
        help="Directory to write PNGs into (default: ml/data/raw_frames "
             "under the repo root, regardless of current working directory).",
    )
    parser.add_argument(
        "--topic", nargs="+", default=["/camera/front/image_mono/compressed"],
        help="One or more topics to collect from in this same run (e.g. "
             "--topic /camera/front /camera/back for a bag recorded with "
             "both cameras at once).",
    )
    parser.add_argument(
        "--every-n-frames", type=int, default=30,
        help="Save 1 out of every N frames per topic (default 30 -- about "
             "1fps at the camera's 30Hz, since adjacent frames are "
             "near-duplicates for this purpose and just cost more SAM3 "
             "labeling time later).",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None,
        help="Stop each topic after saving this many frames (default: run "
             "until Ctrl-C, e.g. until a `ros2 bag play` finishes).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    node = FrameCollector(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit, ExternalShutdownException):
        # A `kill` on this process (e.g. collect_all_bags.sh moving on to
        # the next bag) delivers SIGTERM, which rclpy's own signal handler
        # reacts to by shutting the context down itself and raising this --
        # NOT the same as us calling rclpy.shutdown() below, so without
        # catching it here it propagates as an unhandled exception (nonzero
        # exit code, which trips `set -e` in collect_all_bags.sh and aborts
        # the whole loop after just one bag).
        pass
    finally:
        total = node.total_saved()
        try:
            node.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()
        # Plain print, not node.get_logger() -- the node/context may already
        # be torn down by this point (external shutdown case above), and
        # logging through a dead context just prints its own extra warning.
        print(f"Done -- saved {total} frames total")


if __name__ == "__main__":
    sys.exit(main())
