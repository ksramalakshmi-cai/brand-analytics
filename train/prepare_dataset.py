"""
Prepare a YOLO-format training dataset from your brand-analytics pipeline outputs.

Input sources (pick one or combine):
  A) Crops folder from pipeline outputs  (outputs/<run>/crops/<brand>/<image>.jpg)
  B) Annotated frames + detections_detail.csv  (bounding boxes already known)

Output structure:
  train/
    data/
      images/
        train/   *.jpg
        val/     *.jpg
      labels/
        train/   *.txt  (YOLO format: class_id cx cy w h, normalised)
        val/     *.txt
      dataset.yaml

Usage:
  # From crops directory (generates synthetic full-image bbox annotations)
  python train/prepare_dataset.py --crops outputs/videoplayback/crops --out train/data

  # From detections CSV + frames
  python train/prepare_dataset.py \
      --csv outputs/videoplayback/detections_detail.csv \
      --frames-dir outputs/videoplayback/frames \
      --out train/data
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


def _safe_name(label: str) -> str:
    import re
    return re.sub(r"[^\w\-]+", "_", label.lower()).strip("_")


# --------------------------------------------------------------------------- #
# Source A: build from crops directory                                         #
# --------------------------------------------------------------------------- #

def from_crops(crops_dir: Path, out_dir: Path, val_split: float = 0.15) -> Dict[str, int]:
    """
    Each sub-folder under crops_dir is a brand name.
    Every image in that sub-folder becomes a full-image training sample
    with a single centered bounding box (cx=0.5, cy=0.5, w=1.0, h=1.0).
    """
    brand_dirs = sorted([d for d in crops_dir.iterdir() if d.is_dir()])
    if not brand_dirs:
        raise ValueError(f"No brand sub-directories found in {crops_dir}")

    class_map: Dict[str, int] = {d.name: i for i, d in enumerate(brand_dirs)}
    samples: List[Tuple[Path, str]] = []

    for brand_dir in brand_dirs:
        for img_path in brand_dir.glob("*.jpg"):
            samples.append((img_path, brand_dir.name))
        for img_path in brand_dir.glob("*.png"):
            samples.append((img_path, brand_dir.name))

    print(f"  Found {len(samples)} crop images across {len(class_map)} brands")
    _write_dataset(samples, class_map, out_dir, val_split, annotation_mode="full")
    return class_map


# --------------------------------------------------------------------------- #
# Source B: build from detections_detail.csv + frame images                   #
# --------------------------------------------------------------------------- #

def from_csv(
    csv_path: Path,
    frames_root: Path,
    out_dir: Path,
    val_split: float = 0.15,
) -> Dict[str, int]:
    """
    Parse detections_detail.csv to get per-frame bounding boxes, then
    copy the corresponding annotated frames.
    """
    rows = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise ValueError(f"No rows found in {csv_path}")

    labels = sorted(set(r["label"] for r in rows))
    class_map = {l: i for i, l in enumerate(labels)}

    frame_annotations: Dict[str, List[dict]] = {}
    for row in rows:
        key = row["frame_index"]
        frame_annotations.setdefault(key, []).append(row)

    samples: List[Tuple[Path, str, List[dict]]] = []
    for frame_idx, dets in frame_annotations.items():
        ts = dets[0]["timestamp_sec"]
        fname_pattern = f"frame_{int(frame_idx):06d}_t{float(ts):.1f}s.jpg"

        img_path = None
        for brand_dir in frames_root.iterdir():
            candidate = brand_dir / fname_pattern
            if candidate.exists():
                img_path = candidate
                break
        if img_path is None:
            candidate = frames_root / fname_pattern
            if candidate.exists():
                img_path = candidate

        if img_path:
            samples.append((img_path, frame_idx, dets))

    print(f"  Found {len(samples)} frames with detections")
    _write_dataset_csv(samples, class_map, out_dir, val_split)
    return class_map


# --------------------------------------------------------------------------- #
# Writers                                                                      #
# --------------------------------------------------------------------------- #

def _write_dataset(
    samples: List[Tuple[Path, str]],
    class_map: Dict[str, int],
    out_dir: Path,
    val_split: float,
    annotation_mode: str = "full",
) -> None:
    random.shuffle(samples)
    split_idx = max(1, int(len(samples) * (1 - val_split)))
    splits = {"train": samples[:split_idx], "val": samples[split_idx:]}

    for split, items in splits.items():
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, brand in items:
            dst_img = img_dir / img_path.name
            shutil.copy2(img_path, dst_img)

            cls_id = class_map[brand]
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            with open(lbl_path, "w") as f:
                f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")

    _write_yaml(out_dir, class_map)
    print(f"  Dataset written to: {out_dir.resolve()}")
    print(f"  Classes ({len(class_map)}): {list(class_map.keys())}")
    print(f"  Train: {len(splits['train'])}  Val: {len(splits['val'])}")


def _write_dataset_csv(
    samples,
    class_map: Dict[str, int],
    out_dir: Path,
    val_split: float,
) -> None:
    random.shuffle(samples)
    split_idx = max(1, int(len(samples) * (1 - val_split)))
    split_map = {
        "train": samples[:split_idx],
        "val": samples[split_idx:],
    }

    for split, items in split_map.items():
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path, frame_idx, dets in items:
            dst_img = img_dir / img_path.name
            shutil.copy2(img_path, dst_img)

            lbl_path = lbl_dir / (img_path.stem + ".txt")
            with open(lbl_path, "w") as f:
                for row in dets:
                    cls_id = class_map[row["label"]]
                    cx = (float(row["nx1"]) + float(row["nx2"])) / 2
                    cy = (float(row["ny1"]) + float(row["ny2"])) / 2
                    bw = float(row["nx2"]) - float(row["nx1"])
                    bh = float(row["ny2"]) - float(row["ny1"])
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    _write_yaml(out_dir, class_map)
    print(f"  Dataset written to: {out_dir.resolve()}")
    print(f"  Classes ({len(class_map)}): {list(class_map.keys())}")
    print(f"  Train: {len(split_map['train'])}  Val: {len(split_map['val'])}")


def _write_yaml(out_dir: Path, class_map: Dict[str, int]) -> None:
    yaml_path = out_dir / "dataset.yaml"
    lines = [
        f"path: {out_dir.resolve()}",
        "train: images/train",
        "val:   images/val",
        "",
        f"nc: {len(class_map)}",
        f"names: {list(class_map.keys())}",
    ]
    with open(yaml_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  dataset.yaml → {yaml_path}")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from pipeline outputs")
    parser.add_argument("--crops", type=str, default="",
                        help="Path to crops root dir (outputs/<run>/crops/)")
    parser.add_argument("--csv", type=str, default="",
                        help="Path to detections_detail.csv")
    parser.add_argument("--frames-dir", type=str, default="",
                        help="Path to annotated frames dir (used with --csv)")
    parser.add_argument("--out", type=str, default="train/data",
                        help="Output dataset directory (default: train/data)")
    parser.add_argument("--val-split", type=float, default=0.15,
                        help="Fraction of data to use for validation (default: 0.15)")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.crops:
        print(f"[Mode] Building from crops: {args.crops}")
        class_map = from_crops(Path(args.crops), out_dir, args.val_split)
    elif args.csv and args.frames_dir:
        print(f"[Mode] Building from CSV + frames: {args.csv}")
        class_map = from_csv(Path(args.csv), Path(args.frames_dir), out_dir, args.val_split)
    else:
        parser.print_help()
        return

    print("\nDone. Run training with:")
    print(f"  python train/train.py --data {out_dir / 'dataset.yaml'}\n")


if __name__ == "__main__":
    main()
