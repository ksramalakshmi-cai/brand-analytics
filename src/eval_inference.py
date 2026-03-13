"""
Evaluation inference arm: samples random frames from a processed video,
sends them to a multimodal Gemini model for ground-truth brand identification,
and logs comparison tables to MLflow.

MLflow table columns:
  frame_index | timestamp_sec | brands_seen_gemini | brands_extracted_ocr | score
  score = (matched / total_gemini) * 100
"""
from __future__ import annotations

import base64
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np


_GEMINI_MODEL = os.getenv("BA_EVAL_MODEL", "gemini-3-flash")
_MLFLOW_TRACKING_URI = os.getenv("BA_MLFLOW_TRACKING_URI", "http://34.55.97.182:5000")
_MLFLOW_EXPERIMENT = os.getenv("BA_MLFLOW_EXPERIMENT", "brand-visibility-evals")
_EVAL_FRAME_COUNT = int(os.getenv("BA_EVAL_FRAME_COUNT", "10"))

_DEFAULT_BRAND_PROMPT = """\
You are a brand recognition expert. Look at this image carefully and list ALL \
brand names, logos, and company names that are visible — even partially.
Return ONLY a JSON array of brand name strings, lowercase. No explanation.
Example: ["nike", "coca cola", "emirates"]
If no brands are visible return an empty array: []"""


def _build_prompt(target_logos: Optional[List[str]] = None) -> str:
    if not target_logos:
        return _DEFAULT_BRAND_PROMPT
    labels_str = json.dumps([s.strip().lower() for s in target_logos if s.strip()])
    return (
        f"You are a brand recognition expert. Consider ONLY these brands: {labels_str}.\n"
        "Look at this image and identify which of these brands (logos or text) are visible — even partially.\n"
        "Return ONLY a JSON array of brand name strings from the list above, lowercase. "
        "Do not include brands outside this list. No explanation.\n"
        'If none of these brands are visible return an empty array: []'
    )


def _image_to_base64(image: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def _call_gemini(image: np.ndarray, api_key: str, prompt: str) -> List[str]:
    """Send a single frame to Gemini and return list of brand names."""
    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(_GEMINI_MODEL)

    b64 = _image_to_base64(image)
    response = model.generate_content(
        [
            prompt,
            {"mime_type": "image/jpeg", "data": b64},
        ],
        generation_config={"temperature": 0.0, "max_output_tokens": 1024},
    )

    raw = response.text.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        items = json.loads(raw)
        if isinstance(items, list):
            return [str(b).strip().lower() for b in items if str(b).strip()]
    except (json.JSONDecodeError, TypeError):
        pass
    return []


def _sample_frame_indices(total_frames: int, count: int) -> List[int]:
    """Pick `count` evenly-spread random frame indices."""
    if total_frames <= count:
        return list(range(total_frames))
    step = total_frames / count
    indices = []
    for i in range(count):
        lo = int(i * step)
        hi = int((i + 1) * step) - 1
        indices.append(random.randint(lo, max(lo, hi)))
    return sorted(set(indices))


def _extract_specific_frames(
    video_path: str, indices: List[int], sample_fps: float,
) -> Dict[int, Tuple[np.ndarray, float]]:
    """Return {logical_index: (frame_bgr, timestamp_sec)} for the requested indices."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(native_fps / sample_fps))
    target_set = set(indices)
    results: Dict[int, Tuple[np.ndarray, float]] = {}

    raw_frame = 0
    logical = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if raw_frame % frame_interval == 0:
            if logical in target_set:
                ts = round(raw_frame / native_fps, 3)
                results[logical] = (frame, ts)
                if len(results) == len(target_set):
                    break
            logical += 1
        raw_frame += 1

    cap.release()
    return results


def _ocr_brands_for_frame(
    frame_index: int,
    detection_details: List[Dict[str, Any]],
) -> Set[str]:
    """Collect unique brand labels the pipeline detected for a given frame."""
    return {
        d["label"].lower()
        for d in detection_details
        if d.get("frame_number") == frame_index
    }


def run_eval(
    video_path: str,
    video_name: str,
    detection_details: List[Dict[str, Any]],
    total_frames: int,
    sample_fps: float = 5.0,
    api_key: Optional[str] = None,
    target_logos: Optional[List[str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """Run the Gemini eval arm and log results to MLflow.

    Returns the eval table rows, or None if prerequisites are missing.
    """
    key = api_key or os.getenv("GOOGLE_API_KEY", "")
    if not key:
        print("  [eval] Skipping eval: GOOGLE_API_KEY not set")
        return None

    try:
        import mlflow
        import google.generativeai  # noqa: F401
    except ImportError as exc:
        print(f"  [eval] Skipping eval: {exc}")
        return None

    prompt = _build_prompt(target_logos)

    indices = _sample_frame_indices(total_frames, _EVAL_FRAME_COUNT)
    frames = _extract_specific_frames(video_path, indices, sample_fps)

    if not frames:
        print("  [eval] No frames could be extracted for eval")
        return None

    labels_info = f" for {target_logos}" if target_logos else ""
    print(f"  [eval] Evaluating {len(frames)} frames with {_GEMINI_MODEL}{labels_info} ...")

    table_rows: List[Dict[str, Any]] = []

    for idx in sorted(frames.keys()):
        frame_bgr, timestamp = frames[idx]
        gemini_brands = _call_gemini(frame_bgr, key, prompt)
        ocr_brands = _ocr_brands_for_frame(idx, detection_details)

        gemini_set = set(gemini_brands)
        matched = gemini_set & ocr_brands
        total_gt = len(gemini_set)
        score = round((len(matched) / total_gt) * 100, 2) if total_gt > 0 else 100.0

        table_rows.append({
            "frame_index": idx,
            "timestamp_sec": timestamp,
            "brands_seen_gemini": ", ".join(sorted(gemini_set)) or "(none)",
            "brands_extracted_ocr": ", ".join(sorted(ocr_brands)) or "(none)",
            "matched": ", ".join(sorted(matched)) or "(none)",
            "total_gemini": total_gt,
            "total_ocr": len(ocr_brands),
            "score": score,
        })

    avg_score = (
        round(sum(r["score"] for r in table_rows) / len(table_rows), 2)
        if table_rows else 0.0
    )

    # Log to MLflow
    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    mlflow.set_experiment(_MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=video_name):
        mlflow.log_param("video_name", video_name)
        mlflow.log_param("video_path", video_path)
        mlflow.log_param("eval_model", _GEMINI_MODEL)
        mlflow.log_param("target_logos", ", ".join(target_logos) if target_logos else "(all)")
        mlflow.log_param("frames_evaluated", len(table_rows))
        mlflow.log_param("total_frames_pipeline", total_frames)

        mlflow.log_metric("avg_eval_score", avg_score)
        mlflow.log_metric("frames_evaluated", len(table_rows))

        for i, row in enumerate(table_rows):
            mlflow.log_metric("frame_score", row["score"], step=i)
            mlflow.log_metric("brands_gemini_count", row["total_gemini"], step=i)
            mlflow.log_metric("brands_ocr_count", row["total_ocr"], step=i)

        # Log the full table — try artifact first, fall back to tags
        try:
            import pandas as pd
            df = pd.DataFrame(table_rows)
            mlflow.log_table(data=df, artifact_file="eval_results.json")
        except Exception:
            table_json = json.dumps(table_rows, default=str)
            # MLflow tags have a 5000-char limit; truncate if needed
            if len(table_json) <= 5000:
                mlflow.set_tag("eval_table", table_json)
            else:
                mlflow.set_tag("eval_table_truncated", table_json[:4990] + "...]")
            for i, row in enumerate(table_rows):
                prefix = f"frame_{i}"
                mlflow.set_tag(f"{prefix}_idx", str(row["frame_index"]))
                mlflow.set_tag(f"{prefix}_ts", str(row["timestamp_sec"]))
                mlflow.set_tag(f"{prefix}_gemini", row["brands_seen_gemini"])
                mlflow.set_tag(f"{prefix}_ocr", row["brands_extracted_ocr"])
                mlflow.set_tag(f"{prefix}_score", str(row["score"]))

    print(f"  [eval] Logged to MLflow experiment='{_MLFLOW_EXPERIMENT}' "
          f"run='{video_name}' avg_score={avg_score}%")

    return table_rows
