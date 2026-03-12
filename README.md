# Brand Logo Detection & Visibility Pipeline

Detect logos/brands in images and videos, and produce structured analytics: position, size, screen-time, frame-level bounding boxes, and cropped logo exports.

## Project Structure

```
brand-analytics/
├── pipeline.py              # CLI entry-point
├── config.py                # PipelineConfig dataclass
├── requirements.txt
├── src/
│   ├── frame_extractor.py   # Video/image frame sampling
│   ├── logo_detector.py     # YOLOv8 detection wrapper
│   ├── brand_tracker.py     # Aggregation & analytics
│   └── visualizer.py        # Bounding-box drawing & export
├── outputs/                 # Generated per-run results
└── sample/                  # Drop test files here
```

## Setup

```bash
pip install -r requirements.txt
```

> The default model (`yolov8n.pt`) auto-downloads on first run.
> For logo-specific detection, train or obtain a YOLO model on a logo dataset
> (e.g. LogoDet-3K, FlickrLogos-32, OpenLogo) and pass it via `--model`.

## Quick Start

```bash
# Run on a video (sample 1 frame/sec)
python pipeline.py --input video.mp4

# Run on an image
python pipeline.py --input logo_photo.jpg

# Custom model + higher sample rate
python pipeline.py --input ad_clip.mp4 --model logo_best.pt --fps 2 --conf 0.3

# Force CPU
python pipeline.py --input video.mp4 --device cpu
```

## CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input, -i` | *(required)* | Path to image or video |
| `--model, -m` | `yolov8n.pt` | YOLO model weights path |
| `--fps` | `1.0` | Frames per second to sample from video |
| `--conf` | `0.25` | Detection confidence threshold |
| `--iou` | `0.45` | NMS IoU threshold |
| `--img-size` | `640` | Inference resolution (model rescales internally) |
| `--device` | auto | `cpu`, `0`, `cuda:0`, etc. |
| `--output, -o` | `outputs/` | Root output directory |
| `--no-frames` | off | Skip saving annotated frames |
| `--no-crops` | off | Skip saving cropped logo images |

## Output Structure

Each run creates a sub-directory named after the input file:

```
outputs/<input_stem>/
├── brand_summary.json       # Per-brand aggregated analytics
├── brand_summary.csv        # Same data in CSV
├── detections_detail.csv    # Every detection in every frame
├── frames/                  # Annotated frames with bounding boxes
│   ├── frame_000000_t0.0s.jpg
│   ├── frame_000001_t1.0s.jpg
│   └── ...
└── crops/                   # Cropped logo regions
    ├── frame_000000_det0_nike.jpg
    └── ...
```

### brand_summary.json

```json
{
  "media": {
    "path": "video.mp4",
    "is_video": true,
    "width": 1920,
    "height": 1080,
    "duration_sec": 30.0
  },
  "brands": {
    "nike": {
      "total_detections": 12,
      "total_screen_time_sec": 10.0,
      "visibility_pct": 33.33,
      "avg_area_pct": 5.42,
      "avg_position_nx": 0.75,
      "avg_position_ny": 0.85,
      "dominant_quadrant": "bottom-right"
    }
  }
}
```

## How It Handles Different Media

| Concern | Approach |
|---------|----------|
| **Resolution** | Detection runs at `--img-size` internally (default 640px); bounding boxes are mapped back to original resolution. Normalised coordinates (0-1) are always included for resolution-independent analysis. |
| **Aspect ratio** | YOLO letterbox-pads non-square inputs automatically; no cropping occurs. |
| **Video duration** | Frames are sampled at `--fps` rate regardless of total length. A 10s and a 10-min video both work — you just get more frames for longer content. |
| **Image input** | Treated as a single frame (timestamp = 0). All outputs still generated. |

## Using a Custom Logo Model

The pipeline accepts any YOLO-compatible `.pt` weights file. To train your own:

```bash
# 1. Prepare dataset in YOLO format (images + labels)
# 2. Train
yolo detect train data=logo_dataset.yaml model=yolov8n.pt epochs=100

# 3. Use the best weights
python pipeline.py --input video.mp4 --model runs/detect/train/weights/best.pt
```

## Troubleshooting: PaddleOCR segfault on CPU

On some CPU-only environments (e.g. AWS without GPU), PaddleOCR can hit an Intel MKL segfault. Try:

1. **Use the wrapper script** (sets thread limits before Python starts):
   ```bash
   ./run_ocr.sh --input videoplayback.mp4 --mode ocr
   ```

2. **Set env vars yourself** before running:
   ```bash
   MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 pipeline.py --input videoplayback.mp4 --mode ocr
   ```

3. **Use the CPU-only Paddle build** if you installed the default (GPU) one:
   ```bash
   pip uninstall paddlepaddle
   pip install paddlepaddle
   ```
   (Official CPU wheel is typically named `paddlepaddle`; GPU is `paddlepaddle-gpu`.)

## Programmatic Usage

```python
from config import PipelineConfig
from pipeline import run_pipeline

config = PipelineConfig(
    input_path="video.mp4",
    model_path="logo_best.pt",
    fps=1.0,
    confidence_threshold=0.3,
    output_dir="my_results",
)

summaries = run_pipeline(config)

for brand, stats in summaries.items():
    print(f"{brand}: visible for {stats.total_screen_time_sec}s")
```
