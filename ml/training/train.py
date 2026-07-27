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
import time

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import LaneSegDataset
from losses import CEDiceLoss, PIDNetLoss, update_iou_stats
from pidnet import PIDNetLite

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

# BGR, index-matched to labeling.yaml's class_ids (0=background). Used only
# for the qualitative val snapshots below.
_CLASS_COLORS = np.array([
    [0, 0, 0],        # 0 background
    [0, 0, 255],       # 1 solid_line  (red)
    [0, 255, 255],     # 2 dashed_line (yellow)
    [0, 180, 0],       # 3 lane_area   (green)
], dtype=np.uint8)

# For the per-class IoU line -- index-matched to class_ids above.
_CLASS_NAMES = ["bg", "solid", "dashed", "lane_area"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=str(_REPO_ROOT / "ml" / "data"))
    parser.add_argument("--config", default=str(_REPO_ROOT / "ml" / "configs" / "train.yaml"))
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--runs-dir", default=str(_REPO_ROOT / "ml" / "runs"))
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from this exp-name's last.pt (model+optimizer+scheduler"
             "+epoch), e.g. after an interrupted run, instead of starting fresh.",
    )
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
    require_verified = config.get("require_verified", False)
    train_ds = LaneSegDataset(
        args.data_dir, splits["train"], image_size=image_size, augment=True,
        crop_bottom_fraction=crop_bottom_fraction, require_verified=require_verified,
    )
    val_ds = LaneSegDataset(
        args.data_dir, splits["val"], image_size=image_size, augment=False,
        crop_bottom_fraction=crop_bottom_fraction, require_verified=require_verified,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    # Fixed input size every batch -> let cuDNN autotune the fastest conv
    # algorithms once instead of re-picking per shape. Pure speed win here.
    torch.backends.cudnn.benchmark = use_cuda

    train_workers = config["num_workers"]
    # pin_memory + non_blocking .to(device) below overlaps the host->GPU
    # copy with compute; prefetch_factor keeps each worker a few batches
    # ahead so the GPU isn't left waiting on the (CPU-bound) augmentation.
    loader_common = dict(pin_memory=use_cuda)
    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True,
        num_workers=train_workers, drop_last=True, persistent_workers=train_workers > 0,
        prefetch_factor=4 if train_workers > 0 else None, **loader_common,
    )
    val_workers = min(2, train_workers)
    val_loader = DataLoader(
        val_ds, batch_size=config["batch_size"], shuffle=False,
        num_workers=val_workers, **loader_common,
    )

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
    ).to(device)
    # val_loss always uses plain CEDiceLoss on the model's single main-output
    # tensor (eval mode) -- comparable across future architecture changes,
    # unlike the training loss which adds PIDNet's aux/boundary terms.
    val_criterion = CEDiceLoss(config["ce_weight"], config["dice_weight"], class_weights=class_weights).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # Mixed precision (fp16 autocast + loss scaling) -- big throughput win on
    # the 3090's tensor cores, and the loss scaler keeps fp16 gradients from
    # underflowing. Disabled automatically on CPU. Toggle via config: amp.
    use_amp = config.get("amp", True) and use_cuda
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = config.get("grad_clip")  # None -> off

    start_epoch = 0
    best_iou = -1.0
    last_ckpt = run_dir / "last.pt"
    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if use_amp and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_iou = ckpt.get("best_iou", -1.0)
        print(f"Resumed from {last_ckpt} at epoch {start_epoch} (best_iou so far {best_iou:.4f})")

    snapshot_indices = list(range(min(config["num_val_snapshots"], len(val_ds))))
    total_epochs = config["epochs"]
    n_train_batches = len(train_loader)

    for epoch in range(start_epoch, total_epochs):
        model.train()
        train_loss = 0.0
        n_seen = 0
        epoch_t0 = time.time()
        for i, (images, labels) in enumerate(train_loader, 1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", enabled=use_amp):
                outputs = model(images)  # (main, aux, boundary) tuple in train mode
                loss = train_criterion(outputs, labels)
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            n_seen += images.size(0)
            if i % 50 == 0 or i == n_train_batches:
                elapsed = time.time() - epoch_t0
                ips = n_seen / elapsed if elapsed > 0 else 0
                print(
                    f"  epoch {epoch+1}/{total_epochs}  batch {i}/{n_train_batches}  "
                    f"loss={train_loss/n_seen:.4f}  {ips:.0f} img/s",
                    flush=True,
                )
        train_loss /= len(train_ds)
        scheduler.step()

        model.eval()
        val_loss, n_val = 0.0, 0
        inter = torch.zeros(num_classes, device=device)
        union = torch.zeros(num_classes, device=device)
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", enabled=use_amp):
                    logits = model(images)  # model.eval() -> single tensor
                    val_loss += val_criterion(logits, labels).item() * images.size(0)
                update_iou_stats(logits.float(), labels, inter, union)
                n_val += images.size(0)
        val_loss /= n_val

        inter_np, union_np = inter.cpu().numpy(), union.cpu().numpy()
        per_class_iou = np.where(union_np > 0, inter_np / np.maximum(union_np, 1), np.nan)
        fg = per_class_iou[1:]
        val_iou = float(np.nanmean(fg)) if np.isfinite(fg).any() else 1.0
        per_class_str = "  ".join(
            f"{_CLASS_NAMES[c]}={per_class_iou[c]:.2f}" for c in range(1, num_classes)
        )
        lr_now = scheduler.get_last_lr()[0]
        epoch_time = time.time() - epoch_t0

        print(
            f"epoch {epoch+1}/{total_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_iou={val_iou:.4f} "
            f"lr={lr_now:.2e} ({epoch_time:.0f}s)\n"
            f"  per-class IoU: {per_class_str}"
        )

        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if use_amp else None,
            "epoch": epoch,
            "val_iou": val_iou,
            "best_iou": best_iou,
            "config": config,
        }
        torch.save(ckpt, last_ckpt)  # always, so --resume can pick up
        if val_iou > best_iou:
            best_iou = val_iou
            ckpt["best_iou"] = best_iou
            torch.save(ckpt, run_dir / "best.pt")
            print(f"  -> new best (val_iou={best_iou:.4f}), saved {run_dir/'best.pt'}")

        if (epoch + 1) % config["snapshot_every_n_epochs"] == 0 or epoch == total_epochs - 1:
            save_snapshot(model, val_ds, snapshot_indices, device, run_dir / "val_samples" / f"epoch_{epoch+1:04d}.png")

    print(f"\nDone. Best val_iou={best_iou:.4f}. Checkpoint: {run_dir/'best.pt'}")


if __name__ == "__main__":
    main()
