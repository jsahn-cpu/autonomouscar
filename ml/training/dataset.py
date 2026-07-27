"""torch Dataset for the SAM3-auto-labeled lane data (ml/README.md step 4).

No rclpy/ROS dependency -- pairs ml/data/raw_frames/<id>.png (color, see
ml/labeling/collect_frames.py) with ml/data/labels/<id>.png (single-channel,
pixel value = class index 0-5, see ml/labeling/masks_to_labels.py and
labeling.yaml's class_ids) for the frame_ids listed in a splits.json bucket,
respecting ml/data/rejected_frames.txt as a second, independent check (in
case it was edited after masks_to_labels.py last ran and splits.json wasn't
regenerated yet).
"""
import pathlib
import random
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Class indices are LEFT/RIGHT-POSITIONAL (see labeling.yaml's class_ids:
# 1=left_solid, 2=center_dashed, 3=right_solid, 4=lane_1, 5=lane_2, where
# lane_1/lane_2 are also just left/right of the dashed line -- see
# masks_to_labels.py). A horizontal flip augmentation swaps which physical
# side of the frame everything is on, so it must ALSO swap these class
# VALUES (1<->3, 4<->5, 2 and 0 unchanged) -- flipping the pixel array
# alone silently mislabels the flipped half of every flipped sample (the
# physically-right line ends up tagged left_solid), which the model then
# has to learn as if it were correct. Index i -> class after a flip.
_FLIP_CLASS_REMAP = np.array([0, 3, 2, 1, 5, 4], dtype=np.uint8)

# cv2 spawns its own internal thread pool for ops like resize/warpAffine by
# default. Inside a DataLoader worker subprocess (num_workers>0) that
# thread pool gets created PER WORKER, oversubscribing the CPU (N workers x
# cv2's own threads, all fighting for the same cores) -- a well-known
# PyTorch+OpenCV gotcha that silently turns "should be CPU-light" data
# loading into the actual training bottleneck. Disabling it lets
# DataLoader's own multiprocessing be the only parallelism in play.
cv2.setNumThreads(0)


def load_rejected_frames(path: pathlib.Path) -> set:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


class LaneSegDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        frame_ids: List[str],
        image_size: Tuple[int, int] = (288, 512),  # (H, W)
        augment: bool = False,
        crop_bottom_fraction: float = 1.0,
    ) -> None:
        data_dir = pathlib.Path(data_dir)
        self.raw_dir = data_dir / "raw_frames"
        self.labels_dir = data_dir / "labels"
        rejected = load_rejected_frames(data_dir / "rejected_frames.txt")
        self.frame_ids = [fid for fid in frame_ids if fid not in rejected]
        dropped = len(frame_ids) - len(self.frame_ids)
        if dropped:
            print(f"LaneSegDataset: dropped {dropped} rejected frame(s) from this split")
        self.image_size = image_size
        self.augment = augment
        # Keep only the bottom crop_bottom_fraction of each full-resolution
        # frame before resizing to image_size, instead of resizing the
        # whole frame down -- the far/ceiling/wall region is never useful
        # (SAM3 labels there are already zero via masks_to_labels.py's
        # roi_top_ratio) and including it just wastes resolution on
        # nothing; cropping first means the near-field thin lines that
        # actually matter get more effective pixels after the same resize.
        # 1.0 (default) keeps the full frame, matching the old behavior.
        self.crop_bottom_fraction = crop_bottom_fraction

    def __len__(self) -> int:
        return len(self.frame_ids)

    def _augment(self, image: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            label = np.ascontiguousarray(label[:, ::-1])
            label = _FLIP_CLASS_REMAP[label]

        # brightness/contrast/gamma jitter -- directly targets the lighting
        # robustness that motivated moving off adaptiveThreshold in the
        # first place, not a generic augmentation checklist item
        alpha = random.uniform(0.7, 1.3)  # contrast
        beta = random.uniform(-30, 30)  # brightness
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255)
        gamma = random.uniform(0.7, 1.4)
        image = 255.0 * (image / 255.0) ** gamma
        image = image.astype(np.uint8)

        # mild rotation (+-6 deg) -- kept small since a large rotation would
        # no longer resemble the camera's fixed forward-facing mount angle
        angle = random.uniform(-6, 6)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        # label border fill = 0 (background) -- rotation-exposed corners are
        # genuinely background, not any real class
        label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        return image, label

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_id = self.frame_ids[idx]
        # BGR, not converted to RGB -- deployment (cv2/ROS Image) will also
        # hand the model BGR frames, so keeping this consistent end-to-end
        # avoids a channel-order mismatch bug between training and inference.
        image = cv2.imread(str(self.raw_dir / f"{frame_id}.png"), cv2.IMREAD_COLOR)
        # single channel, pixel value = class index (0=background..5) --
        # NOT a 0/255 mask like the old binary setup, so no thresholding.
        label = cv2.imread(str(self.labels_dir / f"{frame_id}.png"), cv2.IMREAD_GRAYSCALE)
        if image is None or label is None:
            raise FileNotFoundError(f"Missing frame or label for {frame_id!r}")

        if self.crop_bottom_fraction < 1.0:
            full_h = image.shape[0]
            crop_top = int(round(full_h * (1.0 - self.crop_bottom_fraction)))
            image = image[crop_top:, :]
            label = label[crop_top:, :]

        h, w = self.image_size
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
        label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, label = self._augment(image, label)

        image_t = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0  # (3,H,W)
        label_t = torch.from_numpy(label.astype(np.int64))  # (H,W) class indices, for CrossEntropyLoss
        return image_t, label_t
