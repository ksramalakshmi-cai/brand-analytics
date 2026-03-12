"""
FastAPI application exposing the Brand Analytics pipeline as a REST API.

Start the server:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health            — liveness check
    GET  /labels            — list all brands from labels.yaml with their methods
    POST /analyze           — run pipeline on a server-side file (JSON body)
    POST /analyze/upload    — upload a file and run the pipeline
"""
from __future__ import annotations

import os as _os
for _k in (
    "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_k, "1")
_os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import LabelConfig, PipelineConfig
from pipeline import _parse_yaml_labels, run_pipeline

# ---------------------------------------------------------------------------
# Server defaults — override via environment variables
# ---------------------------------------------------------------------------
_DEFAULT_LABELS_FILE = _os.getenv("BA_LABELS_FILE", "labels.yaml")
_DEFAULT_LOGOS_DIR = _os.getenv("BA_LOGOS_DIR", "logos")
_DEFAULT_DETECTOR = _os.getenv("BA_DETECTOR", "reference")
_DEFAULT_MODE = _os.getenv("BA_MODE", "both")
_DEFAULT_OCR_BACKEND = _os.getenv("BA_OCR_BACKEND", "easyocr")
_DEFAULT_FPS = float(_os.getenv("BA_FPS", "5.0"))
_DEFAULT_DEVICE = _os.getenv("BA_DEVICE", "")
_DEFAULT_CLIP_MODEL = _os.getenv("BA_CLIP_MODEL", "ViT-B-32")
_DEFAULT_CLIP_PRETRAINED = _os.getenv("BA_CLIP_PRETRAINED", "openai")
_DEFAULT_SIMILARITY = float(_os.getenv("BA_SIMILARITY_THRESHOLD", "0.75"))
_DEFAULT_OUTPUT_DIR = _os.getenv("BA_OUTPUT_DIR", "outputs")

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Brand Analytics API",
    description="Brand logo detection and visibility analysis pipeline",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------
class AnalyzeRequest(BaseModel):
    labels: List[str] = Field(
        ..., min_length=1,
        description="Brand names to detect (must be defined in labels.yaml)",
    )
    input_path: Optional[str] = Field(
        None, description="Path to a video or image file on the server",
    )
    video_url: Optional[str] = Field(
        None, description="HTTP/HTTPS URL of the video or image to analyze",
    )
    mode: Optional[str] = Field(
        None, description="Pipeline mode: detect | ocr | both (default: server setting)",
    )
    fps: Optional[float] = Field(
        None, description="Frames-per-second sampling rate",
    )
    detector: Optional[str] = Field(
        None, description="Detector backend: yolo | reference",
    )


class BrandResult(BaseModel):
    label: str
    source: str
    total_detections: int
    total_screen_time_sec: float
    visibility_pct: Optional[float] = None
    avg_confidence: float
    avg_area_pct: float
    min_area_pct: float
    max_area_pct: float
    avg_position_nx: float
    avg_position_ny: float
    dominant_quadrant: str
    frames_visible: List[int]
    timestamps_visible: List[float]
    first_seen_timestamp_sec: Optional[float] = None
    last_seen_timestamp_sec: Optional[float] = None


class AnalyzeResponse(BaseModel):
    status: str
    media: Dict[str, Any]
    elapsed_sec: float
    frames_analysed: int
    total_detections: int
    ocr_matches: int
    brands_detected: int
    brands_requested: int
    summary_metrics: Dict[str, Any]
    brands: Dict[str, BrandResult]
    detection_details: List[Dict[str, Any]]
    brands_not_detected: List[str]
    output_dir: str


class LabelInfo(BaseModel):
    name: str
    method: str


class LabelsResponse(BaseModel):
    total: int
    labels: List[LabelInfo]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_full_label_config() -> LabelConfig:
    """Load every brand from the server-side labels YAML."""
    lc = LabelConfig()
    p = Path(_DEFAULT_LABELS_FILE)
    if p.exists() and p.suffix.lower() in (".yaml", ".yml"):
        _parse_yaml_labels(p, lc)
    return lc


def _build_label_config(requested: List[str]) -> LabelConfig:
    """Build a LabelConfig for the *requested* subset, inheriting
    per-brand detection methods from labels.yaml."""
    full = _load_full_label_config()

    method_map: Dict[str, tuple] = {}
    for name in full.ocr:
        method_map[name.lower()] = (name, "ocr")
    for name in full.detector:
        method_map[name.lower()] = (name, "detector")
    for name in full.both:
        method_map[name.lower()] = (name, "both")

    lc = LabelConfig()
    unknown: List[str] = []

    for req in requested:
        key = req.strip().lower()
        if not key:
            continue
        if key in method_map:
            orig_name, method = method_map[key]
            getattr(lc, method).append(orig_name)
        else:
            unknown.append(req)

    if unknown:
        raise HTTPException(
            status_code=422,
            detail={
                "error": f"Labels not found in {_DEFAULT_LABELS_FILE}",
                "unknown_labels": unknown,
                "available_labels": full.all_labels,
            },
        )

    return lc


def _make_config(
    input_path: str,
    mode: Optional[str] = None,
    fps: Optional[float] = None,
    detector: Optional[str] = None,
) -> PipelineConfig:
    return PipelineConfig(
        input_path=input_path,
        mode=mode or _DEFAULT_MODE,
        fps=fps or _DEFAULT_FPS,
        detector=detector or _DEFAULT_DETECTOR,
        logos_dir=_DEFAULT_LOGOS_DIR,
        ocr_backend=_DEFAULT_OCR_BACKEND,
        device=_DEFAULT_DEVICE,
        clip_model=_DEFAULT_CLIP_MODEL,
        clip_pretrained=_DEFAULT_CLIP_PRETRAINED,
        similarity_threshold=_DEFAULT_SIMILARITY,
        output_dir=_DEFAULT_OUTPUT_DIR,
        labels_file="",
        target_labels=[],
    )


def _download_media_from_url(video_url: str) -> str:
    """Download URL media into a temporary local file and return its path."""
    parsed = urllib.parse.urlparse(video_url)
    if parsed.scheme not in ("http", "https"):
        raise HTTPException(422, detail="video_url must use http or https")

    suffix = Path(parsed.path).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with urllib.request.urlopen(video_url) as resp:
            shutil.copyfileobj(resp, tmp)
        tmp.close()
        return tmp.name
    except Exception as exc:
        Path(tmp.name).unlink(missing_ok=True)
        raise HTTPException(400, detail=f"Failed to download video_url: {exc}") from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/labels", response_model=LabelsResponse)
def list_labels():
    """List all brands defined in labels.yaml with their detection methods."""
    full = _load_full_label_config()
    items = [
        LabelInfo(name=name, method=full.method_for(name))
        for name in full.all_labels
    ]
    return LabelsResponse(total=len(items), labels=items)


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    """Run the pipeline on a file already present on the server.

    The request body only needs ``labels`` (brand names) and ``input_path``.
    Labels are cross-referenced against *labels.yaml* to determine each
    brand's detection method (ocr / detector / both).
    """
    if not req.input_path and not req.video_url:
        raise HTTPException(422, detail="Provide either input_path or video_url")
    label_config = _build_label_config(req.labels)
    local_path = req.input_path
    downloaded_tmp_path: Optional[str] = None
    if req.video_url:
        downloaded_tmp_path = _download_media_from_url(req.video_url)
        local_path = downloaded_tmp_path

    if not local_path:
        raise HTTPException(422, detail="No usable input path provided")

    input_path = Path(local_path)
    if not input_path.exists():
        raise HTTPException(404, detail=f"Input file not found: {local_path}")

    try:
        config = _make_config(
            input_path=str(local_path),
            mode=req.mode,
            fps=req.fps,
            detector=req.detector,
        )
        result = run_pipeline(config, label_config=label_config)
        return result
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    finally:
        if downloaded_tmp_path:
            Path(downloaded_tmp_path).unlink(missing_ok=True)


@app.post("/analyze/upload", response_model=AnalyzeResponse)
def analyze_upload(
    file: UploadFile = File(..., description="Video or image file to analyze"),
    labels: str = Form(..., description="Comma-separated brand names"),
    mode: Optional[str] = Form(default=None),
    fps: Optional[float] = Form(default=None),
    detector: Optional[str] = Form(default=None),
):
    """Upload a media file and run the pipeline.

    ``labels`` is a comma-separated string of brand names that must exist
    in *labels.yaml*.
    """
    label_list = [l.strip() for l in labels.split(",") if l.strip()]
    if not label_list:
        raise HTTPException(422, detail="At least one label is required")

    label_config = _build_label_config(label_list)

    suffix = Path(file.filename or "upload").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()

        config = _make_config(
            input_path=tmp.name,
            mode=mode,
            fps=fps,
            detector=detector,
        )

        try:
            result = run_pipeline(config, label_config=label_config)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(400, detail=str(exc)) from exc

        return result
    finally:
        Path(tmp.name).unlink(missing_ok=True)
