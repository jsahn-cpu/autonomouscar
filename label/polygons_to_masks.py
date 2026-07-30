"""ONE-OFF: SSE polygon JSON -> single-channel class-mask PNG (for training).

Reads label/datasets/{train,val,test}/labels/*.json (SSE polygon annotations)
and rasterises each into datasets/{split}/labels_mask/<name>.png, a uint8
single-channel mask whose pixel value is the class id.

classIndex (from the SSE export, confirmed by the objects' `label` field and
by each class's mean x position) -> mask class id:
    classIndex 1  left_solid_line   -> 1
    classIndex 0  center_dashed_line-> 2
    classIndex 2  right_solid_line  -> 3
    (everything else)                background 0

num_classes for training = 4 (bg + 3 lines). Mask size = the paired image's
size (1920x1080). Throwaway: lives in label/, not the ml/ pipeline.

Run:  python3 label/polygons_to_masks.py
"""
import glob
import json
import os
import pathlib

import cv2
import numpy as np
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
DATASETS = HERE / "datasets"
SPLITS = ["train", "val", "test"]

# SSE classIndex -> mask class id (background 0)
CLASS_MAP = {1: 1, 0: 2, 2: 3}   # left_solid=1, center_dashed=2, right_solid=3
CLASS_NAME = {1: "left_solid(1)", 2: "center_dashed(2)", 3: "right_solid(3)"}


def image_size_for(split: str, name: str):
    """(w, h) of the paired image, so the mask matches exactly."""
    img_path = DATASETS / split / "images" / f"{name}.png"
    if not img_path.exists():
        return None
    return Image.open(img_path).size  # (w, h)


def convert_split(split: str) -> dict:
    labels_dir = DATASETS / split / "labels"
    out_dir = DATASETS / split / "labels_mask"
    out_dir.mkdir(exist_ok=True)
    stats = {"files": 0, "no_image": 0, "px": {1: 0, 2: 0, 3: 0}, "obj_skipped": 0}

    for jf in sorted(glob.glob(str(labels_dir / "*.json"))):
        name = pathlib.Path(jf).stem
        size = image_size_for(split, name)
        if size is None:
            stats["no_image"] += 1
            continue
        w, h = size
        mask = np.zeros((h, w), dtype=np.uint8)
        data = json.load(open(jf))
        for obj in data.get("objects", []):
            ci = obj.get("classIndex")
            if ci not in CLASS_MAP:
                stats["obj_skipped"] += 1
                continue
            cid = CLASS_MAP[ci]
            pts = np.array([[p["x"], p["y"]] for p in obj.get("polygon", [])],
                           dtype=np.float32)
            if len(pts) < 3:
                stats["obj_skipped"] += 1
                continue
            cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], int(cid))
        Image.fromarray(mask).save(out_dir / f"{name}.png")
        stats["files"] += 1
        for cid in (1, 2, 3):
            stats["px"][cid] += int((mask == cid).sum())
    return stats


def consolidate() -> None:
    """Gather every split's images + masks into ONE training-ready dataset
    folder matching ml/training/dataset.py's layout:
        label/lane_dataset/raw_frames/<id>.png   (color image)
        label/lane_dataset/labels/<id>.png       (class mask 0-3)
        label/lane_dataset/splits.json           ({"train":[ids],"val":..,"test":..})
    """
    import shutil
    out = HERE / "lane_dataset"
    (out / "raw_frames").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    splits = {}
    for split in SPLITS:
        ids = []
        mask_dir = DATASETS / split / "labels_mask"
        for mp in sorted(glob.glob(str(mask_dir / "*.png"))):
            name = pathlib.Path(mp).stem
            img = DATASETS / split / "images" / f"{name}.png"
            if not img.exists():
                continue
            shutil.copyfile(img, out / "raw_frames" / f"{name}.png")
            shutil.copyfile(mp, out / "labels" / f"{name}.png")
            ids.append(name)
        splits[split] = ids
    with open(out / "splits.json", "w") as f:
        json.dump(splits, f, indent=1)
    print(f"\n[consolidated] -> {out}")
    print(f"    raw_frames/ + labels/ : {sum(len(v) for v in splits.values())} pairs")
    print(f"    splits.json : train={len(splits['train'])} "
          f"val={len(splits['val'])} test={len(splits['test'])}")


def main() -> None:
    for split in SPLITS:
        if not (DATASETS / split / "labels").is_dir():
            print(f"[skip] {split}: no labels dir")
            continue
        s = convert_split(split)
        total = sum(s["px"].values()) or 1
        print(f"[{split}] wrote {s['files']} masks -> {DATASETS/split/'labels_mask'}"
              f"  (no_image={s['no_image']}, skipped_obj={s['obj_skipped']})")
        for cid in (1, 2, 3):
            print(f"    {CLASS_NAME[cid]:20s} pixels={s['px'][cid]:>12,} "
                  f"({100*s['px'][cid]/total:.1f}% of foreground)")
    consolidate()


if __name__ == "__main__":
    main()
