"""
Fine-tune YOLOv8 on your custom logo dataset.

Steps:
  1. Prepare dataset:
       python train/prepare_dataset.py --crops outputs/<run>/crops --out train/data

  2. Train:
       python train/train.py --data train/data/dataset.yaml --epochs 100

  3. Use the fine-tuned model in the pipeline:
       python pipeline.py --input video.mp4 --model train/runs/train/weights/best.pt

Typical choices for base model:
  yolov8n.pt  — nano  (fastest, lowest accuracy)
  yolov8s.pt  — small
  yolov8m.pt  — medium
  yolov8l.pt  — large
  yolov8x.pt  — xlarge (slowest, highest accuracy)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8 for logo detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic fine-tune with 100 epochs
  python train/train.py --data train/data/dataset.yaml --epochs 100

  # Use a larger base model, higher resolution
  python train/train.py --data train/data/dataset.yaml \\
      --model yolov8m.pt --epochs 150 --img-size 1280

  # Resume interrupted training
  python train/train.py --resume train/runs/train/weights/last.pt
        """,
    )
    parser.add_argument("--data", type=str, default="train/data/dataset.yaml",
                        help="Path to dataset.yaml (default: train/data/dataset.yaml)")
    parser.add_argument("--model", type=str, default="yolov8n.pt",
                        help="Base YOLOv8 weights to fine-tune from (default: yolov8n.pt)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Training epochs (default: 100)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Training image size (default: 640)")
    parser.add_argument("--batch", type=int, default=16,
                        help="Batch size (-1 for auto, default: 16)")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate (default: 0.01)")
    parser.add_argument("--lrf", type=float, default=0.01,
                        help="Final learning rate fraction (default: 0.01)")
    parser.add_argument("--device", type=str, default="",
                        help="Training device: '', 'cpu', '0', '0,1' (default: auto)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--project", type=str, default="train/runs",
                        help="Output directory (default: train/runs)")
    parser.add_argument("--name", type=str, default="train",
                        help="Run name (default: train)")
    parser.add_argument("--patience", type=int, default=30,
                        help="Early stopping patience (default: 30)")
    parser.add_argument("--freeze", type=int, default=0,
                        help="Number of backbone layers to freeze (default: 0 = fine-tune all)")
    parser.add_argument("--resume", type=str, default="",
                        help="Resume from last.pt checkpoint path")
    parser.add_argument("--augment", action="store_true",
                        help="Enable extra augmentations (mosaic, mixup, etc.)")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained base weights (default: True)")
    args = parser.parse_args()

    from ultralytics import YOLO

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"[ERROR] Dataset config not found: {data_path}")
        print("  Run: python train/prepare_dataset.py first")
        sys.exit(1)

    if args.resume:
        print(f"[Resume] Loading checkpoint: {args.resume}")
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        print(f"[Train] Base model : {args.model}")
        print(f"[Train] Dataset    : {data_path}")
        print(f"[Train] Epochs     : {args.epochs}")
        print(f"[Train] Image size : {args.img_size}")
        print(f"[Train] Batch      : {args.batch}")
        print(f"[Train] Device     : {args.device or 'auto'}")
        print(f"[Train] Output     : {args.project}/{args.name}")

        model = YOLO(args.model)

        train_kwargs = dict(
            data=str(data_path),
            epochs=args.epochs,
            imgsz=args.img_size,
            batch=args.batch,
            lr0=args.lr0,
            lrf=args.lrf,
            patience=args.patience,
            workers=args.workers,
            project=args.project,
            name=args.name,
            exist_ok=True,
            pretrained=args.pretrained,
            verbose=True,
        )
        if args.device:
            train_kwargs["device"] = args.device
        if args.freeze:
            train_kwargs["freeze"] = args.freeze
        if args.augment:
            train_kwargs.update(dict(mosaic=1.0, mixup=0.1, copy_paste=0.1))

        results = model.train(**train_kwargs)

    best_path = Path(args.project) / args.name / "weights" / "best.pt"
    print(f"\n{'='*60}")
    print(f"  Training complete")
    print(f"  Best weights : {best_path.resolve()}")
    print(f"")
    print(f"  Use in pipeline:")
    print(f"    python pipeline.py --input video.mp4 --model {best_path}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    main()
