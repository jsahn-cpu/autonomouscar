#!/usr/bin/env python3
"""Trains the lane-marking segmenter on SAM3-auto-labeled data (ml/README.md
step 4). Supports two candidate architectures, selected via the config's
`model` key (default "tinyunet"): "tinyunet" (model.py, ~0.5M params, plain
CNN) or "pidnet" (pidnet.py, ~1.5M params, boundary-aware three-branch
design -- see its docstring for why that fits thin lane markings well).
Both are trained/compared the same way; use ml/training/compare_with_lane_detector.py
on each candidate's checkpoint to decide which one wins.

Run inside the `sam3` conda env (same one used for labeling -- torch is
already there):
    conda activate sam3
    python train.py --exp-name my_first_run
    python train.py --exp-name pidnet_run --config ../configs/train_pidnet.yaml

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
from losses import BCEDiceLoss, PIDNetLoss, iou_score
from model import TinyUNet
from pidnet import PIDNetLite


def build_model_and_criterion(config: dict, device: torch.device):
    """val_loss/val_iou always use plain BCEDiceLoss on the model's single
    main-output tensor (both architectures return that in eval mode) so the
    reported metric is directly comparable across candidates -- only the
    TRAINING loss differs (PIDNet's adds its aux/boundary terms)."""
    model_name = config.get("model", "tinyunet")
    if model_name == "pidnet":
        model = PIDNetLite(base_channels=config.get("pidnet_base_channels", 32)).to(device)
        train_criterion = PIDNetLoss(
            bce_weight=config["bce_weight"],
            dice_weight=config["dice_weight"],
            aux_weight=config.get("pidnet_aux_weight", 0.4),
            boundary_weight=config.get("pidnet_boundary_weight", 20.0),
        )
    elif model_name == "tinyunet":
        model = TinyUNet(base_channels=config["base_channels"]).to(device)
        train_criterion = BCEDiceLoss(config["bce_weight"], config["dice_weight"])
    else:
        raise ValueError(f"Unknown model {model_name!r} (expected 'tinyunet' or 'pidnet')")
    val_criterion = BCEDiceLoss(config["bce_weight"], config["dice_weight"])
    return model, train_criterion, val_criterion

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(_REPO_ROOT / "ml" / "data"))
    parser.add_argument("--config", default=str(_REPO_ROOT / "ml" / "configs" / "train.yaml"))
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--runs-dir", default=str(_REPO_ROOT / "ml" / "runs"))
    return parser.parse_args()


def save_snapshot(model, dataset, indices, device, out_path: pathlib.Path) -> None:
    model.eval()
    rows = []
    with torch.no_grad():
        for idx in indices:
            image_t, label_t = dataset[idx]
            logits = model(image_t.unsqueeze(0).to(device))
            pred = (torch.sigmoid(logits)[0, 0] > 0.5).cpu().numpy()

            image = (image_t[0].numpy() * 255).astype(np.uint8)
            gt = (label_t[0].numpy() * 255).astype(np.uint8)
            image_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

            gt_overlay = image_bgr.copy()
            gt_overlay[gt > 127] = (0, 255, 0)  # green = ground truth (SAM3-derived label)
            gt_overlay = cv2.addWeighted(image_bgr, 0.5, gt_overlay, 0.5, 0)

            pred_overlay = image_bgr.copy()
            pred_overlay[pred] = (0, 0, 255)  # red = model prediction
            pred_overlay = cv2.addWeighted(image_bgr, 0.5, pred_overlay, 0.5, 0)

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
    train_ds = LaneSegDataset(
        args.data_dir, splits["train"], image_size=image_size, augment=True,
        crop_bottom_fraction=crop_bottom_fraction,
    )
    val_ds = LaneSegDataset(
        args.data_dir, splits["val"], image_size=image_size, augment=False,
        crop_bottom_fraction=crop_bottom_fraction,
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
    # the same CPU cores (this is what made an earlier run slower, not
    # faster, after giving both loaders num_workers=8 + persistent=True:
    # 16 total worker processes on a 16-core machine left nothing for the
    # main training process itself).
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
    model, train_criterion, val_criterion = build_model_and_criterion(config, device)
    print(f"model: {config.get('model', 'tinyunet')} ({sum(p.numel() for p in model.parameters())/1e6:.2f}M params)")
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
            outputs = model(images)  # single tensor for TinyUNet; (main, aux, boundary) tuple for PIDNet in train mode
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
                logits = model(images)  # model.eval() -> always a single tensor, both architectures
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
