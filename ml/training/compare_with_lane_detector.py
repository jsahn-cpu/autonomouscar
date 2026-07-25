#!/usr/bin/env python3
"""Offline visual + quantitative comparison: the existing adaptiveThreshold
-based LaneDetector.mask_white() vs. a trained TinyUNet standing in for it,
run through the exact same downstream pipeline (Hough -> clustering ->
shape filter -> closed-shape/boundary filter -> Kalman tracking).

This is a comparison tool only -- it does NOT modify lane_detector.py,
lane_detector_node.py, or lane_detector.yaml, and does not touch the live
ROS pipeline. Swapping the trained model into the actual node is the next
step, out of scope here.

Run inside the `sam3` conda env (needs opencv-python + the trained
checkpoint; autodrive_perception.core.* has no rclpy dependency so it
imports fine here via a sys.path insert):
    conda activate sam3
    python compare_with_lane_detector.py --checkpoint ml/runs/my_run/best.pt --val-only
"""
import argparse
import csv
import json
import pathlib
import sys

import cv2
import numpy as np
import torch
import yaml

# autodrive_perception/core/*.py has no rclpy dependency (verified during
# planning), so it can be imported directly here without colcon/ROS.
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "src" / "autodrive" / "autodrive_perception"))
from autodrive_perception.core.lane_detector import LaneDetector  # noqa: E402
from autodrive_perception.core.lane_tracker import LaneTracker  # noqa: E402

from model import TinyUNet  # noqa: E402
from pidnet import PIDNetLite  # noqa: E402


def build_model_from_checkpoint_config(model_config: dict) -> torch.nn.Module:
    model_name = model_config.get("model", "tinyunet")
    if model_name == "pidnet":
        return PIDNetLite(base_channels=model_config.get("pidnet_base_channels", 32))
    return TinyUNet(base_channels=model_config["base_channels"])


class LearnedMaskLaneDetector(LaneDetector):
    """Same as LaneDetector, except mask_white() runs the trained model
    instead of adaptiveThreshold. Returns the same uint8 0/255 mask shape
    (downscaled by hough_scale, ROI-restricted) so every method inherited
    from LaneDetector (_find_segments, _cluster_collinear, ...) works
    completely unmodified on top of it."""

    def __init__(self, model: torch.nn.Module, model_image_size, device: torch.device, **kwargs) -> None:
        super().__init__(**kwargs)
        self._model = model
        self._model_image_size = model_image_size  # (H, W)
        self._device = device

    def mask_white(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape
        scaled_w = max(1, int(round(w * self._hough_scale)))
        scaled_h = max(1, int(round(h * self._hough_scale)))

        model_h, model_w = self._model_image_size
        inp = cv2.resize(gray, (model_w, model_h), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(inp).float().unsqueeze(0).unsqueeze(0) / 255.0
        with torch.no_grad():
            logits = self._model(tensor.to(self._device))
            prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_mask = (prob > 0.5).astype(np.uint8) * 255

        mask = cv2.resize(pred_mask, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)

        roi_polygon = self._roi_polygon(scaled_w, scaled_h)
        roi_mask = np.zeros_like(mask)
        cv2.fillPoly(roi_mask, [roi_polygon], 255)
        return cv2.bitwise_and(mask, roi_mask)


def frame_id_to_session(frame_id: str) -> str:
    parts = frame_id.rsplit("_", 2)
    return parts[0] if len(parts) == 3 else frame_id


def load_lane_detector_kwargs(params: dict) -> dict:
    keys = [
        "adaptive_block_size", "adaptive_c", "blur_kernel_size", "hough_threshold",
        "hough_min_line_length", "hough_max_line_gap", "hough_scale", "angle_tol_deg",
        "collinear_dist_tol_px", "max_segments", "roi_top_ratio", "roi_top_width_ratio",
        "roi_bottom_width_ratio", "roi_left_ratio", "filter_outside_track_boundary",
        "min_boundary_span_ratio", "min_line_aspect_ratio", "min_line_length_px", "corner_tol_px",
    ]
    return {k: params[k] for k in keys if k in params}


def load_lane_tracker_kwargs(params: dict) -> dict:
    # yaml key -> LaneTracker constructor arg (names diverge for the
    # smoothing_*/kalman_* prefixed ones, see lane_detector.yaml comments)
    mapping = {
        "roi_top_ratio": "roi_top_ratio",
        "smoothing_min_confirm_frames": "min_confirm_frames",
        "smoothing_max_missed_frames": "max_missed_frames",
        "kalman_process_noise": "process_noise",
        "kalman_measurement_noise": "measurement_noise",
        "kalman_initial_velocity_variance": "initial_velocity_variance",
        "kalman_gating_mahalanobis": "gating_mahalanobis",
        "parallel_angle_tol_deg": "parallel_angle_tol_deg",
        "spacing_tol_ratio": "spacing_tol_ratio",
        "min_consistent_group_size": "min_consistent_group_size",
        "min_projection_slope": "min_projection_slope",
        "max_abs_x_ratio": "max_abs_x_ratio",
    }
    return {arg: params[yaml_key] for yaml_key, arg in mapping.items() if yaml_key in params}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default=str(_REPO_ROOT / "ml" / "data"))
    parser.add_argument(
        "--lane-detector-config",
        default=str(_REPO_ROOT / "src/autodrive/autodrive_bringup/config/lane_detector.yaml"),
        help="Both detectors are built from these SAME tuned parameters -- "
             "the only difference between them is the mask source.",
    )
    parser.add_argument("--output-dir", default=None, help="default: <checkpoint's run dir>/compare")
    parser.add_argument(
        "--val-only", action="store_true",
        help="Only compare frames in splits.json's val bucket (recommended -- "
             "these were held out of training).",
    )
    parser.add_argument("--max-frames-per-session", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_config = ckpt["config"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_from_checkpoint_config(model_config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"Loaded {model_config.get('model', 'tinyunet')} checkpoint "
        f"(epoch={ckpt['epoch']}, val_iou={ckpt['val_iou']:.4f})"
    )

    with open(args.lane_detector_config) as f:
        yaml_params = yaml.safe_load(f)["lane_detector_node"]["ros__parameters"]
    detector_kwargs = load_lane_detector_kwargs(yaml_params)
    tracker_kwargs = load_lane_tracker_kwargs(yaml_params)

    baseline_detector = LaneDetector(**detector_kwargs)
    learned_detector = LearnedMaskLaneDetector(
        model, tuple(model_config["image_size"]), device, **detector_kwargs
    )
    baseline_tracker = LaneTracker(**tracker_kwargs)
    learned_tracker = LaneTracker(**tracker_kwargs)

    data_dir = pathlib.Path(args.data_dir)
    raw_dir = data_dir / "raw_frames"
    if args.val_only:
        with open(data_dir / "splits.json") as f:
            frame_ids = json.load(f)["val"]
    else:
        frame_ids = [p.stem for p in sorted(raw_dir.glob("*.png"))]

    by_session: dict = {}
    for fid in frame_ids:
        by_session.setdefault(frame_id_to_session(fid), []).append(fid)

    out_dir = pathlib.Path(args.output_dir) if args.output_dir else pathlib.Path(args.checkpoint).parent / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_rows = []
    for session, session_frame_ids in by_session.items():
        session_frame_ids = sorted(session_frame_ids)[: args.max_frames_per_session]
        for frame_id in session_frame_ids:
            image = cv2.imread(str(raw_dir / f"{frame_id}.png"), cv2.IMREAD_GRAYSCALE)
            if image is None:
                print(f"  WARNING: missing raw frame for {frame_id}, skipping")
                continue
            bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            baseline_dets = baseline_detector.detect(image)
            learned_dets = learned_detector.detect(image)

            baseline_debug = baseline_detector.draw_debug(bgr, baseline_dets)
            learned_debug = learned_detector.draw_debug(bgr, learned_dets)
            cv2.putText(baseline_debug, "baseline (adaptiveThreshold)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(learned_debug, "learned (TinyUNet)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            divider = np.full((bgr.shape[0], 4, 3), (255, 255, 255), dtype=np.uint8)
            side_by_side = np.hstack([baseline_debug, divider, learned_debug])
            cv2.imwrite(str(out_dir / f"{frame_id}.png"), side_by_side)

            for pipeline, dets, tracker in (
                ("baseline", baseline_dets, baseline_tracker),
                ("learned", learned_dets, learned_tracker),
            ):
                confirmed = tracker.update(dets, image.shape[0], image.shape[1])
                x_near = ";".join(f"{d['x_near']:.1f}" for d in confirmed)
                x_far = ";".join(f"{d['x_far']:.1f}" for d in confirmed)
                csv_rows.append(
                    {
                        "session": session, "frame_id": frame_id, "pipeline": pipeline,
                        "n_confirmed": len(confirmed), "x_near": x_near, "x_far": x_far,
                    }
                )

        print(f"session {session}: {len(session_frame_ids)} frames compared")

    csv_path = out_dir / "tracking_stability.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["session", "frame_id", "pipeline", "n_confirmed", "x_near", "x_far"])
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nSaved {len(csv_rows)//2} side-by-side comparisons + tracking CSV to {out_dir}")
    for pipeline in ("baseline", "learned"):
        counts = [r["n_confirmed"] for r in csv_rows if r["pipeline"] == pipeline]
        zero_frac = sum(1 for c in counts if c == 0) / len(counts) if counts else 0
        print(f"  {pipeline}: mean confirmed lines/frame={sum(counts)/len(counts):.2f}, "
              f"0-line frames={100*zero_frac:.1f}%")


if __name__ == "__main__":
    main()
