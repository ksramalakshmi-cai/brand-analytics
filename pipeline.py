"""
Main pipeline: ties together frame extraction, detection, OCR, tracking,
and visualisation.

Three modes:
  detect  — YOLO object detection only (no OCR)
  ocr     — OCR-only full-frame text scanning (no YOLO model needed)
  both    — YOLO detection + OCR text matching on detected regions

Usage:
    python pipeline.py --input video.mp4 --mode detect
    python pipeline.py --input video.mp4 --mode ocr
    python pipeline.py --input video.mp4 --mode both
"""
from __future__ import annotations

# Reduce PaddleOCR/Intel MKL segfaults on CPU — must be set before any paddle import
import os as _os
for _k in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    _os.environ.setdefault(_k, "1")
_os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import argparse
import sys
import time
from pathlib import Path
from typing import List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tqdm import tqdm

from config import PipelineConfig
from src.frame_extractor import extract_frames, get_media_info
from src.logo_detector import Detection
from src.brand_tracker import BrandTracker
from src.visualizer import Visualizer


def _load_labels(config: PipelineConfig) -> List[str]:
    """Collect target labels from both --labels and --labels-file."""
    labels: List[str] = list(config.target_labels)

    if config.labels_file:
        p = Path(config.labels_file)
        if p.exists():
            with open(p) as f:
                for line in f:
                    line = line.strip().rstrip(",").strip()
                    if line and not line.startswith("#"):
                        labels.append(line)
        else:
            print(f"[WARN] Labels file not found: {p}")

    seen = set()
    deduped = []
    for l in labels:
        key = l.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(l)
    return deduped


def run_pipeline(config: PipelineConfig) -> dict:
    """Execute the full pipeline and return the brand summaries."""
    input_path = Path(config.input_path)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        sys.exit(1)

    mode = config.mode
    media = get_media_info(input_path)
    target_labels = _load_labels(config)

    use_detect = mode in ("detect", "both")
    use_ocr = mode in ("ocr", "both")
    use_reference = config.detector == "reference"

    if use_reference and config.logos_dir:
        ref_root = Path(config.logos_dir)
        if ref_root.is_dir():
            derived = [
                d.name for d in sorted(ref_root.iterdir())
                if d.is_dir() and not d.name.startswith(".")
            ]
            for name in derived:
                if name.lower() not in {t.lower() for t in target_labels}:
                    target_labels.append(name)

    if use_ocr and not target_labels:
        print("[ERROR] OCR mode requires target labels. Use --labels or --labels-file.")
        sys.exit(1)

    # ---- Header ----
    print(f"\n{'='*60}")
    print(f"  Brand Visibility Pipeline")
    print(f"{'='*60}")
    print(f"  Input      : {media.path}")
    print(f"  Type       : {'Video' if media.is_video else 'Image'}")
    print(f"  Resolution : {media.width}x{media.height}")
    if media.is_video:
        print(f"  Duration   : {media.duration_sec:.1f}s")
        print(f"  Native FPS : {media.native_fps:.2f}")
        print(f"  Total Frms : {media.total_frames}")
    print(f"  Mode       : {mode}")
    print(f"  Detector   : {config.detector}")
    if use_detect and not use_reference:
        print(f"  Model      : {config.model_path}")
    if use_reference:
        print(f"  Logos dir  : {config.logos_dir}")
        print(f"  CLIP model : {config.clip_model} ({config.clip_pretrained})")
        print(f"  Sim thresh : {config.similarity_threshold}")
    print(f"  Sample FPS : {config.fps}")
    if target_labels:
        print(f"  Labels     : {len(target_labels)} brands")
    if use_ocr:
        print(f"  OCR backend: {config.ocr_backend}")
        print(f"  OCR boost  : +{config.ocr_confidence_boost:.0%} on text match")
    print(f"{'='*60}\n")

    run_dir = config.resolve_output_dir()

    # ---- Init components (only what's needed) ----
    detector = None
    if use_reference:
        if not config.logos_dir:
            print("[ERROR] --logos-dir is required when --detector reference is used.")
            sys.exit(1)
        from src.reference_matcher import ReferenceMatcher
        detector = ReferenceMatcher(
            logos_dir=config.logos_dir,
            model_name=config.clip_model,
            pretrained=config.clip_pretrained,
            similarity_threshold=config.similarity_threshold,
            device=config.device,
            img_size=config.img_size,
            exclusion_coverage=config.ocr_exclusion_coverage,
        )
    elif use_detect:
        from src.logo_detector import LogoDetector
        detector = LogoDetector(config)

    ocr_reader = None
    if use_ocr:
        import torch
        from src.ocr_reader import OCRReader
        use_gpu = torch.cuda.is_available() and config.device != "cpu"
        ocr_reader = OCRReader(
            target_labels=target_labels,
            languages=config.ocr_languages,
            gpu=use_gpu,
            match_threshold=config.ocr_match_threshold,
            confidence_boost=config.ocr_confidence_boost,
            backend=config.ocr_backend,
        )

    tracker = BrandTracker(sample_fps=config.fps, target_labels=target_labels)
    visualizer = Visualizer(config)

    if media.is_video:
        estimated_frames = int(media.duration_sec * config.fps) + 1
    else:
        estimated_frames = 1

    # ---- Frame loop ----
    t0 = time.time()
    detection_count = 0
    ocr_match_count = 0

    for frame_meta in tqdm(
        extract_frames(input_path, sample_fps=config.fps),
        total=estimated_frames,
        desc="Processing",
        unit="frame",
    ):
        detections: List[Detection] = []

        if mode == "ocr":
            # OCR-only: scan full frame for text brands
            detections = ocr_reader.scan_frame(frame_meta.frame)
            ocr_match_count += len(detections)

        elif mode == "both" and use_reference and detector is not None:
            # OCR-first strategy: OCR scans full frame, CLIP fills the gaps
            ocr_dets = ocr_reader.scan_frame(frame_meta.frame)
            ocr_match_count += len(ocr_dets)

            exclusion_zones = [
                (d.x1, d.y1, d.x2, d.y2) for d in ocr_dets
            ]
            clip_dets = detector.detect(
                frame_meta.frame, exclusion_zones=exclusion_zones,
            )
            detections = ocr_dets + clip_dets

        elif mode == "both" and not use_reference and detector is not None:
            # YOLO + OCR verify: detector first, OCR boosts matches
            detections = detector.detect(frame_meta.frame)
            for det in detections:
                ocr_result = ocr_reader.process_crop(
                    frame_meta.frame,
                    det.x1, det.y1, det.x2, det.y2,
                )
                det.ocr_text = ocr_result["ocr_text"]
                det.ocr_matched_label = ocr_result["ocr_matched_label"]
                det.ocr_match_ratio = ocr_result["ocr_match_ratio"]
                det.original_confidence = det.confidence

                if det.ocr_matched_label:
                    det.confidence = min(1.0, det.confidence + ocr_result["ocr_confidence_boost"])
                    det.label = det.ocr_matched_label
                    ocr_match_count += 1

        elif mode == "detect" and detector is not None:
            detections = detector.detect(frame_meta.frame)

        detection_count += len(detections)

        tracker.add(
            frame_index=frame_meta.index,
            timestamp_sec=frame_meta.timestamp_sec,
            detections=detections,
            frame_width=frame_meta.width,
            frame_height=frame_meta.height,
        )

        # Visualization: for YOLO+OCR mode show only OCR-matched; otherwise all
        if mode == "both" and not use_reference:
            vis_detections = [d for d in detections if d.ocr_matched_label]
        else:
            vis_detections = detections

        if vis_detections:
            visualizer.save_annotated_frame(
                frame_meta.frame, vis_detections,
                frame_meta.index, frame_meta.timestamp_sec,
                run_dir,
            )
            visualizer.save_cropped_logos(
                frame_meta.frame, vis_detections,
                frame_meta.index, run_dir,
            )

    elapsed = time.time() - t0

    # ---- Save reports ----
    if config.save_report_csv:
        tracker.save_detail_csv(run_dir / "detections_detail.csv")
        tracker.save_summary_csv(run_dir / "brand_summary.csv", media_info=media)

    if config.save_report_json:
        tracker.save_summary_json(run_dir / "brand_summary.json", media_info=media)

    # ---- Print results ----
    summaries = tracker.summarise()

    if target_labels:
        target_lower = {t.lower() for t in target_labels}
        display_summaries = {
            label: s for label, s in summaries.items()
            if label.lower() in target_lower
        }
    else:
        display_summaries = summaries

    print(f"\n{'='*60}")
    print(f"  Results — mode: {mode}")
    print(f"{'='*60}")
    print(f"  Time elapsed    : {elapsed:.1f}s")
    print(f"  Frames analysed : {estimated_frames}")
    print(f"  Total detections: {detection_count}")
    if use_ocr:
        print(f"  OCR text matches: {ocr_match_count}")
    if target_labels:
        print(f"  Matched brands  : {len(display_summaries)} / {len(target_labels)} targets")
    else:
        print(f"  Unique labels   : {len(display_summaries)}")
    print()

    for label, s in sorted(display_summaries.items(), key=lambda x: -x[1].total_screen_time_sec):
        vis_pct = (
            f"{s.total_screen_time_sec / media.duration_sec * 100:.1f}%"
            if media.is_video and media.duration_sec > 0 else "N/A"
        )
        print(f"  [{label}]")
        print(f"    Detections      : {s.total_detections}")
        print(f"    Screen time     : {s.total_screen_time_sec:.1f}s ({vis_pct} of video)")
        print(f"    Avg size        : {s.avg_area_pct:.2f}% of frame")
        print(f"    Size range      : {s.min_area_pct:.2f}% - {s.max_area_pct:.2f}%")
        print(f"    Avg position    : ({s.avg_position_nx:.2f}, {s.avg_position_ny:.2f})")
        print(f"    Dominant region : {s.dominant_quadrant}")
        print(f"    Avg confidence  : {s.avg_confidence:.3f}")
        print()

    if target_labels:
        matched = {l.lower() for l in display_summaries}
        missing = [t for t in target_labels if t.lower() not in matched]
        if missing:
            print(f"  Brands NOT detected: {', '.join(missing)}")
            print()

    print(f"  Outputs saved to: {run_dir.resolve()}")
    print(f"{'='*60}\n")

    return summaries


def main():
    parser = argparse.ArgumentParser(
        description="Brand Logo Detection & Visibility Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  detect  YOLO object detection only (default)
  ocr     OCR-only — scans full frames for text, matches against --labels
  both    YOLO detection + OCR text matching on detected regions

Examples:
  python pipeline.py --input video.mp4 --mode detect
  python pipeline.py --input video.mp4 --mode ocr --labels-file labels.txt
  python pipeline.py --input video.mp4 --mode both --labels "coca cola,nike"
        """,
    )
    parser.add_argument("--input", "-i", required=True, help="Path to image or video file")
    parser.add_argument(
        "--mode", choices=["detect", "ocr", "both"], default="detect",
        help="Pipeline mode: detect (YOLO only), ocr (OCR only), both (YOLO+OCR)",
    )
    parser.add_argument("--model", "-m", default="yolov8n.pt", help="YOLO model weights (default: yolov8n.pt)")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames per second to sample (default: 5.0)")
    parser.add_argument("--conf", type=float, default=0.8, help="Confidence threshold (default: 0.8)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--device", default="", help="Device: 'cpu', '0', 'cuda:0', etc. (default: auto)")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory (default: outputs/)")
    parser.add_argument("--no-frames", action="store_true", help="Skip saving annotated frames")
    parser.add_argument("--no-crops", action="store_true", help="Skip saving cropped logos")
    parser.add_argument(
        "--detector", default="yolo",
        choices=["yolo", "reference"],
        help="Detection backend (default: yolo)",
    )

    ref_group = parser.add_argument_group("Reference matching (CLIP) options")
    ref_group.add_argument("--logos-dir", type=str, default="",
                           help="Path to reference logos directory (required for --detector reference)")
    ref_group.add_argument("--clip-model", type=str, default="ViT-B-32",
                           help="CLIP model architecture (default: ViT-B-32)")
    ref_group.add_argument("--clip-pretrained", type=str, default="openai",
                           help="CLIP pretrained weights tag (default: openai)")
    ref_group.add_argument("--similarity-threshold", type=float, default=0.75,
                           help="Min cosine similarity for a reference match (default: 0.75)")
    ref_group.add_argument("--ocr-exclusion-coverage", type=float, default=1.0,
                           help="Skip CLIP patches overlapping OCR hits by this fraction (default: 0.3)")

    label_group = parser.add_argument_group("Label / OCR options")
    label_group.add_argument(
        "--labels", type=str, default="",
        help="Comma-separated brand names to match (e.g. 'coca cola,nike,adidas')",
    )
    label_group.add_argument(
        "--labels-file", type=str, default="labels.txt",
        help="Path to text file with one brand name per line (default: labels.txt)",
    )
    label_group.add_argument(
        "--ocr-lang", type=str, default="en",
        help="OCR language(s), comma-separated (default: en)",
    )
    label_group.add_argument(
        "--ocr-match-threshold", type=float, default=0.8,
        help="Min fuzzy-match ratio to count as brand match (default: 0.5)",
    )
    label_group.add_argument(
        "--ocr-boost", type=float, default=0.15,
        help="Confidence boost added on OCR text match in 'both' mode (default: 0.15)",
    )
    label_group.add_argument(
        "--ocr-backend", choices=["paddle", "easyocr"], default="easyocr",
        help="OCR engine: easyocr (default, avoids Paddle segfault on CPU) or paddle",
    )

    args = parser.parse_args()

    target_labels = [l.strip() for l in args.labels.split(",") if l.strip()] if args.labels else []

    config = PipelineConfig(
        input_path=args.input,
        mode=args.mode,
        fps=args.fps,
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        img_size=args.img_size,
        device=args.device,
        output_dir=args.output,
        save_annotated_frames=not args.no_frames,
        save_cropped_logos=not args.no_crops,
        target_labels=target_labels,
        labels_file=args.labels_file,
        ocr_backend=args.ocr_backend,
        ocr_languages=[l.strip() for l in args.ocr_lang.split(",")],
        ocr_match_threshold=args.ocr_match_threshold,
        ocr_confidence_boost=args.ocr_boost,
        detector=args.detector,
        logos_dir=args.logos_dir,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        similarity_threshold=args.similarity_threshold,
        ocr_exclusion_coverage=args.ocr_exclusion_coverage,
    )

    run_pipeline(config)


if __name__ == "__main__":
    main()
