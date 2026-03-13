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
import csv
import json
import sys
import time
from pathlib import Path
from typing import Dict, Any
from typing import List, Optional

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tqdm import tqdm

from config import PipelineConfig, LabelConfig
from src.frame_extractor import extract_frames, get_media_info
from src.logo_detector import Detection
from src.brand_tracker import BrandTracker
from src.visualizer import Visualizer

_VALID_METHODS = {"ocr", "detector", "both"}


def _load_label_config(config: PipelineConfig) -> LabelConfig:
    """Build a LabelConfig from --labels, --labels-file (.yaml or .txt).

    YAML format (per-brand method):
        google: ocr
        emirates: detector
        budweiser: both

    Plain-text format (backward compatible, all default to 'both'):
        google
        emirates
        budweiser
    """
    lc = LabelConfig()

    # 1) --labels CLI flag  →  all default to "both"
    for raw in config.target_labels:
        name = raw.strip()
        if name:
            lc.both.append(name)

    # 2) --labels-file
    if config.labels_file:
        p = Path(config.labels_file)
        if p.exists():
            if p.suffix.lower() in (".yaml", ".yml"):
                _parse_yaml_labels(p, lc)
            else:
                _parse_txt_labels(p, lc)
        else:
            print(f"[WARN] Labels file not found: {p}")

    return lc


def _parse_yaml_labels(path: Path, lc: LabelConfig) -> None:
    import yaml
    with open(path) as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        return
    for name, method in data.items():
        name = str(name).strip()
        method = str(method).strip().lower()
        if not name or method not in _VALID_METHODS:
            continue
        if method == "ocr":
            lc.ocr.append(name)
        elif method == "detector":
            lc.detector.append(name)
        else:
            lc.both.append(name)


def _parse_txt_labels(path: Path, lc: LabelConfig) -> None:
    """Legacy plain-text file: one brand per line, all assigned 'both'."""
    with open(path) as f:
        for line in f:
            line = line.strip().rstrip(",").strip()
            if line and not line.startswith("#"):
                lc.both.append(line)


def _merge_fcs_into_brand_summary_json(path: Path, brands_result: Dict[str, Any]) -> None:
    """Add metric_inputs and fcs_breakdown to each brand in existing brand_summary.json."""
    with open(path) as f:
        payload = json.load(f)
    for label, data in brands_result.items():
        if label in payload.get("brands", {}):
            payload["brands"][label]["metric_inputs"] = data.get("metric_inputs", {})
            payload["brands"][label]["fcs_breakdown"] = data.get("fcs_breakdown", {})
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)


def _merge_fcs_into_brand_summary_csv(path: Path, brands_result: Dict[str, Any]) -> None:
    """Add fcs_score column to existing brand_summary.csv."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or []) + ["fcs_score"]
    for row in rows:
        label = row.get("label", "")
        fcs = brands_result.get(label, {}).get("fcs_breakdown", {})
        row["fcs_score"] = fcs.get("fcs_score", "")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_pipeline(
    config: PipelineConfig,
    label_config: Optional[LabelConfig] = None,
) -> dict:
    """Execute the full pipeline and return structured results.

    Parameters
    ----------
    config : PipelineConfig
    label_config : LabelConfig, optional
        If provided, used directly (API path).  Otherwise built from
        ``config.target_labels`` / ``config.labels_file`` (CLI path).
    """
    input_path = Path(config.input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    mode = config.mode
    media = get_media_info(input_path)

    if label_config is None:
        label_config = _load_label_config(config)

    use_detect = mode in ("detect", "both")
    use_ocr = mode in ("ocr", "both")
    use_reference = config.detector == "reference"

    all_labels = label_config.all_labels

    if use_ocr and not all_labels:
        raise ValueError("OCR mode requires target labels. Use --labels or --labels-file.")

    # Resolve per-mode eligible label sets
    if mode == "both":
        ocr_labels = label_config.ocr_eligible
        det_labels_set = label_config.detector_eligible_set
    else:
        ocr_labels = all_labels
        det_labels_set = {n.lower() for n in all_labels}

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
    if all_labels:
        n_ocr = len(label_config.ocr)
        n_det = len(label_config.detector)
        n_both = len(label_config.both)
        print(f"  Labels     : {len(all_labels)} brands "
              f"(ocr:{n_ocr}  detector:{n_det}  both:{n_both})")
    if use_ocr:
        print(f"  OCR backend: {config.ocr_backend}")
        print(f"  OCR boost  : +{config.ocr_confidence_boost:.0%} on text match")
    print(f"{'='*60}\n")

    run_dir = config.resolve_output_dir()

    # ---- Init components (only what's needed) ----
    detector = None
    if use_reference:
        if not config.logos_dir:
            raise ValueError("--logos-dir is required when --detector reference is used.")
        from src.reference_matcher import ReferenceMatcher
        det_brand_names = list(label_config.detector_eligible_set) if all_labels else None
        detector = ReferenceMatcher(
            logos_dir=config.logos_dir,
            model_name=config.clip_model,
            pretrained=config.clip_pretrained,
            similarity_threshold=config.similarity_threshold,
            device=config.device,
            img_size=config.img_size,
            exclusion_coverage=config.ocr_exclusion_coverage,
            brand_filter=det_brand_names,
            patch_scales=config.clip_patch_scales,
            stride_ratio=config.clip_stride_ratio,
            refine=config.clip_refine,
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
            target_labels=ocr_labels,
            languages=config.ocr_languages,
            gpu=use_gpu,
            match_threshold=config.ocr_match_threshold,
            confidence_boost=config.ocr_confidence_boost,
            backend=config.ocr_backend,
            deepseek_api_key=config.deepseek_api_key,
            deepseek_base_url=config.deepseek_base_url,
            deepseek_model=config.deepseek_model,
        )

    tracker = BrandTracker(sample_fps=config.fps, target_labels=all_labels)
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
            detections = ocr_reader.scan_frame(frame_meta.frame)
            ocr_match_count += len(detections)

        elif mode == "both" and use_reference and detector is not None:
            # OCR-first: scan for ocr-eligible brands
            ocr_dets = ocr_reader.scan_frame(frame_meta.frame)
            ocr_match_count += len(ocr_dets)

            # CLIP fills uncovered areas, filtered to detector-eligible brands
            exclusion_zones = [
                (d.x1, d.y1, d.x2, d.y2) for d in ocr_dets
            ]
            clip_dets = detector.detect(
                frame_meta.frame, exclusion_zones=exclusion_zones,
            )
            clip_dets = [
                d for d in clip_dets
                if d.label.lower() in det_labels_set
            ]
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
                    det.source = "ocr+detector"
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

    # ---- Build results ----
    summaries = tracker.summarise()

    if all_labels:
        target_lower = {t.lower() for t in all_labels}
        display_summaries = {
            label: s for label, s in summaries.items()
            if label.lower() in target_lower
        }
    else:
        display_summaries = summaries

    matched = {l.lower() for l in display_summaries}
    missing = [t for t in all_labels if t.lower() not in matched] if all_labels else []
    if not all_labels:
        target_lower = set()

    filtered_records = [
        r for r in tracker.records
        if not target_lower or r.label.lower() in target_lower
    ]
    frames_with_target_brands = len({r.frame_index for r in filtered_records})
    frame_coverage_pct = round(
        (frames_with_target_brands / estimated_frames) * 100, 2
    ) if estimated_frames else 0.0

    def _safe_div(num: float, den: float) -> float:
        return num / den if den and den > 0 else 0.0

    brands_result = {}
    for label, s in display_summaries.items():
        vis_pct = (
            round(s.total_screen_time_sec / media.duration_sec * 100, 2)
            if media.is_video and media.duration_sec > 0 else None
        )
        label_records = [r for r in filtered_records if r.label.lower() == label.lower()]
        source_set = sorted({r.source for r in label_records if r.source})
        if source_set:
            source_tokens = set()
            for src in source_set:
                source_tokens.update(part for part in src.split("+") if part)
            if source_tokens == {"ocr"}:
                source = "ocr"
            elif source_tokens == {"reference"}:
                source = "reference"
            elif source_tokens == {"detector"}:
                source = "detector"
            else:
                source = "+".join(sorted(source_tokens))
        else:
            source = "unknown"
        frame_counts = {}
        for r in label_records:
            frame_counts[r.frame_index] = frame_counts.get(r.frame_index, 0) + 1
        max_objects = max(frame_counts.values()) if frame_counts else 0
        roi_duration = float(s.total_screen_time_sec)
        total_duration = float(media.duration_sec) if media.duration_sec > 0 else roi_duration
        if label_records and config.fps > 0:
            tmin = min(r.timestamp_sec for r in label_records)
            tmax = max(r.timestamp_sec for r in label_records)
            object_presence_duration = max(0.0, (tmax - tmin) + (1.0 / config.fps))
        else:
            object_presence_duration = 0.0

        prominence_weightage = 0.35 * _safe_div(roi_duration, object_presence_duration)
        frequency_weightage = 0.15 * _safe_div(float(max_objects), 5.0)
        duration_weightage = 0.2 * _safe_div(roi_duration, total_duration)
        quality_weightage = 0.3
        fcs_score = prominence_weightage + frequency_weightage + duration_weightage + quality_weightage
        brands_result[label] = {
            "label": label,
            "source": source,
            "total_detections": s.total_detections,
            "total_screen_time_sec": s.total_screen_time_sec,
            "visibility_pct": vis_pct,
            "avg_confidence": s.avg_confidence,
            "avg_area_pct": s.avg_area_pct,
            "min_area_pct": s.min_area_pct,
            "max_area_pct": s.max_area_pct,
            "avg_position_nx": s.avg_position_nx,
            "avg_position_ny": s.avg_position_ny,
            "dominant_quadrant": s.dominant_quadrant,
            "frames_visible": s.frames_visible,
            "timestamps_visible": s.timestamps_visible,
            "first_seen_timestamp_sec": s.timestamps_visible[0] if s.timestamps_visible else None,
            "last_seen_timestamp_sec": s.timestamps_visible[-1] if s.timestamps_visible else None,
            "metric_inputs": {
                "roi_duration": round(roi_duration, 4),
                "object_presence_duration": round(object_presence_duration, 4),
                "max_objects": max_objects,
                "total_duration": round(total_duration, 4),
            },
            "fcs_breakdown": {
                "prominence_weightage": round(prominence_weightage, 6),
                "frequency_weightage": round(frequency_weightage, 6),
                "duration_weightage": round(duration_weightage, 6),
                "quality_weightage": round(quality_weightage, 6),
                "fcs_score": round(fcs_score, 6),
            },
        }

    detection_details = [
        {
            "frame_number": r.frame_index,
            "timestamp_sec": r.timestamp_sec,
            "label": r.label,
            "source": r.source,
            "confidence": r.confidence,
            "coordinates_px": {
                "x1": r.x1, "y1": r.y1, "x2": r.x2, "y2": r.y2,
                "width_px": r.width_px, "height_px": r.height_px,
            },
            "coordinates_normalized": {
                "x1": r.nx1, "y1": r.ny1, "x2": r.nx2, "y2": r.ny2,
            },
            "space_occupied": {
                "area_px": r.area_px,
                "area_pct": r.area_pct,
            },
            "region": r.position_quadrant,
        }
        for r in filtered_records
    ]

    result = {
        "status": "success",
        "media": {
            "path": str(media.path),
            "is_video": media.is_video,
            "width": media.width,
            "height": media.height,
            "duration_sec": media.duration_sec,
            "native_fps": media.native_fps,
        },
        "elapsed_sec": round(elapsed, 2),
        "frames_analysed": estimated_frames,
        "total_detections": detection_count,
        "ocr_matches": ocr_match_count,
        "brands_detected": len(display_summaries),
        "brands_requested": len(all_labels),
        "summary_metrics": {
            "frames_with_target_brands": frames_with_target_brands,
            "frame_coverage_pct": frame_coverage_pct,
            "total_unique_target_brands_seen": len(display_summaries),
        },
        "brands": brands_result,
        "detection_details": detection_details,
        "brands_not_detected": missing,
        "output_dir": str(run_dir.resolve()),
    }

    # Add FCS and metric_inputs into existing brand_summary files
    if config.save_report_json:
        _merge_fcs_into_brand_summary_json(run_dir / "brand_summary.json", brands_result)
    if config.save_report_csv:
        _merge_fcs_into_brand_summary_csv(run_dir / "brand_summary.csv", brands_result)

    # ---- Eval inference arm (Gemini ground-truth comparison → MLflow) ----
    if media.is_video:
        try:
            from src.eval_inference import run_eval
            video_name = Path(config.input_path).stem
            eval_rows = run_eval(
                video_path=str(input_path),
                video_name=video_name,
                detection_details=detection_details,
                total_frames=estimated_frames,
                sample_fps=config.fps,
            )
            if eval_rows is not None:
                result["eval_results"] = eval_rows
        except Exception as eval_exc:
            print(f"  [eval] Eval step failed (non-fatal): {eval_exc}")

    # ---- Console summary (CLI usage) ----
    print(f"\n{'='*60}")
    print(f"  Results — mode: {mode}")
    print(f"{'='*60}")
    print(f"  Time elapsed    : {elapsed:.1f}s")
    print(f"  Frames analysed : {estimated_frames}")
    print(f"  Total detections: {detection_count}")
    if use_ocr:
        print(f"  OCR text matches: {ocr_match_count}")
    if all_labels:
        print(f"  Matched brands  : {len(display_summaries)} / {len(all_labels)} targets")
    else:
        print(f"  Unique labels   : {len(display_summaries)}")
    print()

    for label, s in sorted(display_summaries.items(), key=lambda x: -x[1].total_screen_time_sec):
        vis_pct_str = (
            f"{s.total_screen_time_sec / media.duration_sec * 100:.1f}%"
            if media.is_video and media.duration_sec > 0 else "N/A"
        )
        fcs = brands_result.get(label, {}).get("fcs_breakdown", {})
        fcs_score = fcs.get("fcs_score", 0)
        print(f"  [{label}]")
        print(f"    Detections      : {s.total_detections}")
        print(f"    Screen time     : {s.total_screen_time_sec:.1f}s ({vis_pct_str} of video)")
        print(f"    Avg size        : {s.avg_area_pct:.2f}% of frame")
        print(f"    Size range      : {s.min_area_pct:.2f}% - {s.max_area_pct:.2f}%")
        print(f"    Avg position    : ({s.avg_position_nx:.2f}, {s.avg_position_ny:.2f})")
        print(f"    Dominant region : {s.dominant_quadrant}")
        print(f"    Avg confidence  : {s.avg_confidence:.3f}")
        print(f"    FCS score       : {fcs_score:.4f}")
        print()

    if missing:
        print(f"  Brands NOT detected: {', '.join(missing)}")
        print()

    print(f"  Outputs saved to: {run_dir.resolve()}")
    print(f"{'='*60}\n")

    return result


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
        "--mode", choices=["detect", "ocr", "both"], default="both",
        help="Pipeline mode: detect (YOLO only), ocr (OCR only), both (YOLO+OCR)",
    )
    parser.add_argument("--model", "-m", default="yolov8n.pt", help="YOLO model weights (default: yolov8n.pt)")
    parser.add_argument("--fps", type=float, default=5.0, help="Frames per second to sample (default: 5.0)")
    parser.add_argument("--conf", type=float, default=0.8, help="Confidence threshold (default: 0.8)")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold for NMS (default: 0.45)")
    parser.add_argument("--img-size", type=int, default=640, help="Inference image size (default: 640)")
    parser.add_argument("--device", default="cuda", help="Device: 'cpu', '0', 'cuda:0', etc. (default: auto)")
    parser.add_argument("--output", "-o", default="outputs", help="Output directory (default: outputs/)")
    parser.add_argument("--no-frames", action="store_true", help="Skip saving annotated frames")
    parser.add_argument("--no-crops", action="store_true", help="Skip saving cropped logos")
    parser.add_argument(
        "--fast", action="store_true",
        help="Faster run: fewer CLIP patches, no refinement (~0.5s/frame target). Frames/crops still saved unless --no-frames/--no-crops.",
    )
    parser.add_argument(
        "--detector", default="reference",
        choices=["yolo", "reference"],
        help="Detection backend (default: yolo)",
    )

    ref_group = parser.add_argument_group("Reference matching (CLIP) options")
    ref_group.add_argument("--logos-dir", type=str, default="logos",
                           help="Path to reference logos directory (required for --detector reference)")
    ref_group.add_argument("--clip-model", type=str, default="ViT-B-32",
                           help="CLIP model architecture (default: ViT-B-32)")
    ref_group.add_argument("--clip-pretrained", type=str, default="openai",
                           help="CLIP pretrained weights tag (default: openai)")
    ref_group.add_argument("--similarity-threshold", type=float, default=0.8,
                           help="Min cosine similarity for a reference match (default: 0.75)")
    ref_group.add_argument("--ocr-exclusion-coverage", type=float, default=1.0,
                           help="Skip CLIP patches overlapping OCR hits by this fraction (default: 0.3)")
    ref_group.add_argument("--clip-patch-scales", type=int, nargs="*", default=None,
                           help="CLIP patch scales in px (default: 96 256). More scales = slower.")
    ref_group.add_argument("--clip-stride-ratio", type=float, default=None,
                           help="CLIP stride as fraction of scale (default: 0.65). Lower = more patches, slower.")
    ref_group.add_argument("--clip-refine", action="store_true",
                           help="Refine detection boxes (slower, slightly more accurate)")

    label_group = parser.add_argument_group("Label / OCR options")
    label_group.add_argument(
        "--labels", type=str, default="",
        help="Comma-separated brand names to match (e.g. 'coca cola,nike,adidas')",
    )
    label_group.add_argument(
        "--labels-file", type=str, default="labels.yaml",
        help="Path to labels YAML/text file (default: labels.yaml)",
    )
    label_group.add_argument(
        "--ocr-lang", type=str, default="en",
        help="OCR language(s), comma-separated (default: en)",
    )
    label_group.add_argument(
        "--ocr-match-threshold", type=float, default=0.7,
        help="Min fuzzy-match ratio to count as brand match (default: 0.5)",
    )
    label_group.add_argument(
        "--ocr-boost", type=float, default=0.15,
        help="Confidence boost added on OCR text match in 'both' mode (default: 0.15)",
    )
    label_group.add_argument(
        "--ocr-backend", choices=["easyocr", "paddle", "deepseek"], default="easyocr",
        help="OCR engine: easyocr (default), paddle, or deepseek (VLM, needs API key)",
    )

    ds_group = parser.add_argument_group("Vision-LLM OCR options")
    ds_group.add_argument("--deepseek-api-key", type=str, default="",
                          help="API key (or set GOOGLE_API_KEY / DEEPSEEK_API_KEY env var)")
    ds_group.add_argument("--deepseek-base-url", type=str,
                          default="https://generativelanguage.googleapis.com/v1beta/openai/",
                          help="OpenAI-compatible vision API base URL")
    ds_group.add_argument("--deepseek-model", type=str, default="gemini-2.0-flash",
                          help="Vision model name (default: gemini-2.0-flash)")

    args = parser.parse_args()

    target_labels = [l.strip() for l in args.labels.split(",") if l.strip()] if args.labels else []

    clip_scales = args.clip_patch_scales if args.clip_patch_scales is not None else [96, 256]
    clip_stride = args.clip_stride_ratio if args.clip_stride_ratio is not None else 0.65
    clip_refine = args.clip_refine and not args.fast

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
        deepseek_api_key=args.deepseek_api_key,
        deepseek_base_url=args.deepseek_base_url,
        deepseek_model=args.deepseek_model,
        detector=args.detector,
        logos_dir=args.logos_dir,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        similarity_threshold=args.similarity_threshold,
        ocr_exclusion_coverage=args.ocr_exclusion_coverage,
        clip_patch_scales=clip_scales,
        clip_stride_ratio=clip_stride,
        clip_refine=clip_refine,
    )

    try:
        run_pipeline(config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
