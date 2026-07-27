#!/usr/bin/env python3
"""Trains PIDNetLite on the SAM3-auto-labeled multi-class lane data
(ml/README.md step 4). Single candidate architecture -- see pidnet.py's
docstring for why PIDNet's boundary-aware design was picked over a plain
encoder-decoder for this class-boundary-heavy task (solid vs. dashed, left
vs. right, lane_1 vs. lane_2).

Run inside the `sam3` conda env (same one used for labeling -- torch is
already there):
    conda activate sam3
    python train.py --exp-name my_first_run

Reads ml/data/splits.json (written by masks_to_labels.py) for the train/val
frame_id lists. Saves the best-val-IoU checkpoint plus periodic qualitative
snapshots -- no tensorboard/wandb dependency, kept to what's already
installed.
"""
import argparse
import json
import pathlib
import shutil

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import LaneSegDataset
from losses import CEDiceLoss, PIDNetLoss, iou_score
from pidnet import PIDNetLite

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# BGR, index-matched to labeling.yaml's class_ids (0=background). Used only
# for the qualitative val snapshots below.
_CLASS_COLORS = np.array([
    [0, 0, 0],        # 0 background
    [0, 0, 255],       # 1 left_solid   (red)
    [0, 255, 255],     # 2 center_dashed (yellow)
    [255, 0, 0],       # 3 right_solid  (blue)
    [0, 180, 0],       # 4 lane_1       (dark green)
    [180, 0, 180],     # 5 lane_2       (purple)
], dtype=np.uint8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(_REPO_ROOT / "ml" / "data"))
    parser.add_argument("--config", default=str(_REPO_ROOT / "ml" / "configs" / "train.yaml"))
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--runs-dir", default=str(_REPO_ROOT / "ml" / "runs"))
    return parser.parse_args()


def colorize(label: np.ndarray) -> np.ndarray:
    return _CLASS_COLORS[label]


def save_snapshot(model, dataset, indices, device, out_path: pathlib.Path) -> None:
    model.eval()
    rows = []
    with torch.no_grad():
        for idx in indices:
            image_t, label_t = dataset[idx]
            logits = model(image_t.unsqueeze(0).to(device))
            pred = logits[0].argmax(dim=0).cpu().numpy().astype(np.uint8)

            image_bgr = (image_t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            gt = label_t.numpy().astype(np.uint8)

            gt_overlay = cv2.addWeighted(image_bgr, 0.5, colorize(gt), 0.5, 0)
            pred_overlay = cv2.addWeighted(image_bgr, 0.5, colorize(pred), 0.5, 0)

            rows.append(np.hstack([image_bgr, gt_overlay, pred_overlay]))
    grid = np.vstack(rows)
    cv2.imwrite(str(out_path), grid)
    model.train()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)

    run_dir = pathlib.Path(args.runs_dir) / args.exp_name
    (run_dir / "val_samples").mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "train_config.yaml")

    with open(pathlib.Path(args.data_dir) / "splits.json") as f:
        splits = json.load(f)

    image_size = tuple(config["image_size"])
    crop_bottom_fraction = config.get("crop_bottom_fraction", 1.0)
    require_reviewed = config.get("require_reviewed", False)
    train_ds = LaneSegDataset(
        args.data_dir, splits["train"], image_size=image_size, augment=True,
        crop_bottom_fraction=crop_bottom_fraction, require_reviewed=require_reviewed,
    )
    val_ds = LaneSegDataset(
        args.data_dir, splits["val"], image_size=image_size, augment=False,
        crop_bottom_fraction=crop_bottom_fraction, require_reviewed=require_reviewed,
    )
    print(f"train frames: {len(train_ds)}, val frames: {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise SystemExit("Empty train or val split -- run masks_to_labels.py first.")

    # persistent_workers on train_loader avoids respawning (and
    # re-importing torch/cv2 into) fresh worker processes every epoch.
    # val_loader intentionally does NOT get persistent_workers or the same
    # worker count -- it only runs briefly once per epoch, so keeping N
    # more processes alive and idle for the other 95% of the epoch just
    # competes with train_loader's own workers and the main process for
    # the same CPU cores.
    train_workers = config["num_workers"]
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=train_workers, drop_last=True, persistent_workers=train_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=min(2, train_workers),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = config["num_classes"]
    class_weights = config.get("class_weights")
    model = PIDNetLite(num_classes=num_classes, base_channels=config["base_channels"]).to(device)
    print(f"model: PIDNetLite ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")

    train_criterion = PIDNetLoss(
        num_classes=num_classes,
        ce_weight=config["ce_weight"],
        dice_weight=config["dice_weight"],
        aux_weight=config.get("aux_weight", 0.4),
        boundary_weight=config.get("boundary_weight", 20.0),
        class_weights=class_weights,
    )
    # val_loss always uses plain CEDiceLoss on the model's single main-output
    # tensor (eval mode) -- comparable across future architecture changes,
    # unlike the training loss which adds PIDNet's aux/boundary terms.
    val_criterion = CEDiceLoss(config["ce_weight"], config["dice_weight"], class_weights=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    snapshot_indices = list(range(min(config["num_val_snapshots"], len(val_ds))))

    best_iou = -1.0
    for epoch in range(config["epochs"]):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)  # (main, aux, boundary) tuple in train mode
            loss = train_criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_loss, val_iou_sum, n_val = 0.0, 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)  # model.eval() -> single tensor
                val_loss += val_criterion(logits, labels).item() * images.size(0)
                val_iou_sum += iou_score(logits, labels) * images.size(0)
                n_val += images.size(0)
        val_loss /= n_val
        val_iou = val_iou_sum / n_val

        print(
            f"epoch {epoch+1}/{config['epochs']} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_iou={val_iou:.4f}"
        )

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(
                {"model_state_dict": model.state_dict(), "epoch": epoch, "val_iou": val_iou, "config": config},
                run_dir / "best.pt",
            )
            print(f"  -> new best (val_iou={best_iou:.4f}), saved {run_dir/'best.pt'}")

        if (epoch + 1) % config["snapshot_every_n_epochs"] == 0 or epoch == config["epochs"] - 1:
            save_snapshot(model, val_ds, snapshot_indices, device, run_dir / "val_samples" / f"epoch_{epoch+1:04d}.png")

    print(f"\nDone. Best val_iou={best_iou:.4f}. Checkpoint: {run_dir/'best.pt'}")


if __name__ == "__main__":
    main()
