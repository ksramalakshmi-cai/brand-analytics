"""
Brand Analytics API
===================

Start:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints
---------
Logo Registration:
    POST   /logos                 — register a new logo (multipart: images + metadata)
    GET    /logos                 — list registered logos
    GET    /logos/{logo_id}       — get logo detail
    PATCH  /logos/{logo_id}       — update metadata / add reference images
    DELETE /logos/{logo_id}       — remove logo and its reference images

Video Processing (async):
    POST   /process               — submit video for background analysis
    GET    /jobs/{job_id}         — poll job status & results

Stats:
    GET    /stats                 — aggregated brand-visibility stats
    GET    /stats?logo_ids=a,b    — filter to specific logos

Misc:
    GET    /health
"""
from __future__ import annotations

import os as _os

for _k in (
    "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_k, "1")
_os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import json
import logging
import shutil
import sys
import tempfile
import time
import traceback
import urllib.parse
import urllib.request
import uuid
from collections.abc import AsyncGenerator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import db
from config import LabelConfig, PipelineConfig
from pipeline import run_pipeline
from src.frame_extractor import extract_frames

log = logging.getLogger("brand_analytics")

# ── Server defaults (override via env) ────────────────────────────────────
_LOGOS_DIR = _os.getenv("BA_LOGOS_DIR", str(_SCRIPT_DIR / "logos"))
_DETECTOR = _os.getenv("BA_DETECTOR", "reference")
_MODE = _os.getenv("BA_MODE", "both")
_OCR_BACKEND = _os.getenv("BA_OCR_BACKEND", "easyocr")
_FPS = float(_os.getenv("BA_FPS", "5.0"))
_DEVICE = _os.getenv("BA_DEVICE", "")
_CLIP_MODEL = _os.getenv("BA_CLIP_MODEL", "ViT-B-32")
_CLIP_PRETRAINED = _os.getenv("BA_CLIP_PRETRAINED", "openai")
_SIMILARITY = float(_os.getenv("BA_SIMILARITY_THRESHOLD", "0.75"))
_OUTPUT_DIR = _os.getenv("BA_OUTPUT_DIR", "outputs")
_S3_BUCKET = _os.getenv("BA_S3_BUCKET", "")
_S3_PREFIX = _os.getenv("BA_S3_PREFIX", "annotated-videos/")
_MAX_WORKERS = int(_os.getenv("BA_MAX_WORKERS", "2"))

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

_executor = ThreadPoolExecutor(max_workers=_MAX_WORKERS)

db.init_db()
Path(_LOGOS_DIR).mkdir(parents=True, exist_ok=True)

@asynccontextmanager
async def _lifespan(application: FastAPI) -> AsyncGenerator[None, None]:
    db.init_db()
    yield

# ── FastAPI app ───────────────────────────────────────────────────────────
app = FastAPI(
    title="Brand Analytics API",
    description="Logo detection, video processing and brand-visibility analytics",
    version="2.0.0",
    lifespan=_lifespan,
)


# ── Pydantic models ──────────────────────────────────────────────────────

class LogoOut(BaseModel):
    logo_id: str
    name: str
    detection_method: str
    reference_count: int
    created_at: str
    updated_at: str


class ProcessRequest(BaseModel):
    url: str = Field(..., description="Video URL (CloudFront / S3 presigned / HTTPS)")
    video_id: str = Field(..., description="Unique post / video identifier (dedup key)")
    engagements: int = Field(default=0, description="Engagement count for weighting")
    target_logos: List[str] = Field(
        ..., min_length=1,
        description="List of registered logo_id values to detect",
    )


class JobOut(BaseModel):
    job_id: str
    video_id: str
    status: str
    message: Optional[str] = None


class JobDetailOut(BaseModel):
    job_id: str
    video_id: str
    video_url: str
    engagements: int
    status: str
    target_logos: List[str]
    error_message: Optional[str] = None
    annotated_video_s3_url: Optional[str] = None
    frames_analysed: Optional[int] = None
    duration_sec: Optional[float] = None
    processing_time_sec: Optional[float] = None
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    result: Optional[Dict[str, Any]] = None


class LogoStatOut(BaseModel):
    logo_id: str
    name: str
    detection_method: str
    aggregate: Dict[str, Any]
    per_video: List[Dict[str, Any]]


# =====================================================================
# 1. LOGO REGISTRATION
# =====================================================================

@app.post("/logos", response_model=LogoOut, status_code=201)
async def register_logo(
    logo_id: str = Form(..., description="Unique logo identifier (e.g. 'emirates')"),
    name: str = Form(..., description="Display name"),
    detection_method: str = Form(default="both", description="ocr | detector | both"),
    images: List[UploadFile] = File(default=[], description="Reference logo images"),
):
    if detection_method not in ("ocr", "detector", "both"):
        raise HTTPException(422, "detection_method must be ocr, detector, or both")
    if db.get_logo(logo_id):
        raise HTTPException(409, f"Logo '{logo_id}' already exists")

    logo_dir = Path(_LOGOS_DIR) / logo_id
    logo_dir.mkdir(parents=True, exist_ok=True)
    saved = _save_uploaded_images(images, logo_dir)

    logo = db.create_logo(logo_id, name, detection_method, reference_count=saved)
    return logo


@app.get("/logos", response_model=List[LogoOut])
def list_logos():
    return db.list_logos()


@app.get("/logos/{logo_id}", response_model=LogoOut)
def get_logo(logo_id: str):
    logo = db.get_logo(logo_id)
    if not logo:
        raise HTTPException(404, f"Logo '{logo_id}' not found")
    return logo


@app.patch("/logos/{logo_id}", response_model=LogoOut)
async def update_logo(
    logo_id: str,
    name: Optional[str] = Form(default=None),
    detection_method: Optional[str] = Form(default=None),
    images: List[UploadFile] = File(default=[], description="Additional reference images"),
):
    existing = db.get_logo(logo_id)
    if not existing:
        raise HTTPException(404, f"Logo '{logo_id}' not found")
    if detection_method and detection_method not in ("ocr", "detector", "both"):
        raise HTTPException(422, "detection_method must be ocr, detector, or both")

    kwargs: Dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if detection_method is not None:
        kwargs["detection_method"] = detection_method

    logo_dir = Path(_LOGOS_DIR) / logo_id
    logo_dir.mkdir(parents=True, exist_ok=True)
    added = _save_uploaded_images(images, logo_dir)
    if added:
        ref_count = len([p for p in logo_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTS])
        kwargs["reference_count"] = ref_count

    return db.update_logo(logo_id, **kwargs)


@app.delete("/logos/{logo_id}", status_code=204)
def delete_logo(logo_id: str):
    if not db.delete_logo(logo_id):
        raise HTTPException(404, f"Logo '{logo_id}' not found")
    logo_dir = Path(_LOGOS_DIR) / logo_id
    if logo_dir.is_dir():
        shutil.rmtree(logo_dir, ignore_errors=True)


# =====================================================================
# 2. VIDEO PROCESSING (async)
# =====================================================================

@app.post("/process", response_model=JobOut, status_code=202)
def submit_video(req: ProcessRequest):
    existing = db.get_job_by_video_id(req.video_id)
    if existing:
        return JobOut(
            job_id=existing["job_id"],
            video_id=req.video_id,
            status=existing["status"],
            message="Duplicate video_id — returning existing job",
        )

    unknown = [lid for lid in req.target_logos if not db.get_logo(lid)]
    if unknown:
        raise HTTPException(422, detail={
            "error": "Unregistered logo_id(s)",
            "unknown": unknown,
            "registered": [l["logo_id"] for l in db.list_logos()],
        })

    job_id = uuid.uuid4().hex
    db.create_job(job_id, req.video_id, req.url, req.engagements, req.target_logos)

    _executor.submit(_process_video_job, job_id)

    return JobOut(
        job_id=job_id,
        video_id=req.video_id,
        status="pending",
        message="Video processing queued",
    )


@app.get("/jobs/{job_id}", response_model=JobDetailOut)
def get_job(job_id: str):
    job = db.get_job(job_id)
    if not job:
        raise HTTPException(404, f"Job '{job_id}' not found")
    return job


# =====================================================================
# 3. STATS
# =====================================================================

@app.get("/stats", response_model=List[LogoStatOut])
def get_stats(logo_ids: Optional[str] = None):
    """Aggregated visibility stats.  ``?logo_ids=a,b`` to filter."""
    id_list = [s.strip() for s in logo_ids.split(",") if s.strip()] if logo_ids else None
    rows = db.get_logo_stats(id_list)
    out: List[LogoStatOut] = []
    for r in rows:
        per_video = db.get_per_video_breakdown(r["logo_id"])
        avg_fcs = r.get("avg_fcs_score", 0)
        total_play = r.get("total_play_count", 0)
        log.info("[stats] %s: avg_fcs_score=%.4f total_play_count=%s", r["logo_id"], avg_fcs, total_play)
        print(f"  [stats] {r['logo_id']}: avg_fcs_score={avg_fcs:.4f}  total_play_count={total_play}")
        out.append(LogoStatOut(
            logo_id=r["logo_id"],
            name=r["name"],
            detection_method=r["detection_method"],
            aggregate={
                "total_videos": r["total_videos"],
                "total_detections": r["total_detections"],
                "total_screen_time_sec": r["total_screen_time_sec"],
                "avg_screen_time_per_video_sec": r["avg_screen_time_per_video_sec"],
                "avg_visibility_pct": r["avg_visibility_pct"],
                "avg_confidence": r["avg_confidence"],
                "avg_area_pct": r["avg_area_pct"],
                "avg_fcs_score": avg_fcs,
                "total_play_count": total_play,
                "total_engagements": r["total_engagements"],
                "weighted_visibility_score": r["weighted_visibility_score"],
            },
            per_video=per_video,
        ))
    # Save metrics to file
    _save_stats_metrics(out)
    return out


def _save_stats_metrics(stats_out: List[LogoStatOut]) -> None:
    """Write stats response to outputs/stats_export.json for persistence."""
    out_dir = Path(_os.getenv("BA_OUTPUT_DIR", str(_SCRIPT_DIR / "outputs")))
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stats_export.json"
    payload = [
        {
            "logo_id": s.logo_id,
            "name": s.name,
            "avg_fcs_score": s.aggregate.get("avg_fcs_score", 0),
            "total_play_count": s.aggregate.get("total_play_count", 0),
            "aggregate": s.aggregate,
            "per_video": s.per_video,
        }
        for s in stats_out
    ]
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    log.info("Stats metrics saved to %s", path)


@app.get("/health")
def health():
    return {"status": "ok"}


# =====================================================================
# INTERNAL HELPERS
# =====================================================================

def _save_uploaded_images(images: List[UploadFile], dest_dir: Path) -> int:
    saved = 0
    for img in images:
        if not img.filename:
            continue
        ext = Path(img.filename).suffix.lower()
        if ext not in _IMAGE_EXTS:
            continue
        out_path = dest_dir / img.filename
        with open(out_path, "wb") as f:
            shutil.copyfileobj(img.file, f)
        saved += 1
    return saved


def _download_video(url: str) -> str:
    """Download video from URL to a temp file. Returns local path."""
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme}")
    suffix = Path(parsed.path).suffix or ".mp4"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        with urllib.request.urlopen(url) as resp:
            shutil.copyfileobj(resp, tmp)
        tmp.close()
        return tmp.name
    except Exception as exc:
        Path(tmp.name).unlink(missing_ok=True)
        raise RuntimeError(f"Failed to download video: {exc}") from exc


def _build_label_config_from_db(logo_ids: List[str]) -> LabelConfig:
    """Build LabelConfig from DB-registered logos."""
    lc = LabelConfig()
    for lid in logo_ids:
        logo = db.get_logo(lid)
        if not logo:
            continue
        method = logo["detection_method"]
        name = logo["name"]
        if method == "ocr":
            lc.ocr.append(name)
        elif method == "detector":
            lc.detector.append(name)
        else:
            lc.both.append(name)
    return lc


def _name_to_logo_id(target_logos: List[str]) -> Dict[str, str]:
    """Map display-name (lowercase) → logo_id for storage."""
    mapping: Dict[str, str] = {}
    for lid in target_logos:
        logo = db.get_logo(lid)
        if logo:
            mapping[logo["name"].lower()] = lid
    return mapping


def _create_annotated_video(
    input_path: str, detection_details: List[Dict], sample_fps: float, output_path: str,
) -> None:
    """Re-read video and overlay detection bounding boxes into a new video file."""
    det_map: Dict[int, List[Dict]] = {}
    for d in detection_details:
        det_map.setdefault(d["frame_number"], []).append(d)

    writer = None
    for frame_meta in extract_frames(input_path, sample_fps=sample_fps):
        frame = frame_meta.frame.copy()
        h, w = frame.shape[:2]

        if writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(output_path, fourcc, sample_fps, (w, h))

        recs = det_map.get(frame_meta.index, [])
        for r in recs:
            coords = r.get("coordinates_px", {})
            x1, y1 = coords.get("x1", 0), coords.get("y1", 0)
            x2, y2 = coords.get("x2", 0), coords.get("y2", 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{r['label']} {r['confidence']:.2f}"
            (tw, th_), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th_ - 8), (x1 + tw + 4, y1), (0, 255, 0), -1)
            cv2.putText(
                frame, label_text, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
            )

        writer.write(frame)

    if writer:
        writer.release()


def _upload_to_s3(local_path: str, s3_key: str) -> str:
    """Upload file to S3 if configured. Returns the S3 URL or empty string."""
    if not _S3_BUCKET:
        return ""
    try:
        import boto3
        s3 = boto3.client("s3")
        s3.upload_file(local_path, _S3_BUCKET, s3_key)
        return f"s3://{_S3_BUCKET}/{s3_key}"
    except Exception as exc:
        log.warning("S3 upload failed: %s", exc)
        return ""


# ── Background worker ─────────────────────────────────────────────────────

def _process_video_job(job_id: str) -> None:
    """Runs in a background thread."""
    job = db.get_job(job_id)
    if not job:
        return

    video_id = job["video_id"]
    video_url = job["video_url"]
    target_logos = job["target_logos"]
    engagements = job["engagements"]
    local_path: Optional[str] = None

    try:
        db.update_job_status(job_id, "processing")
        t0 = time.time()

        local_path = _download_video(video_url)
        label_config = _build_label_config_from_db(target_logos)
        name_to_id = _name_to_logo_id(target_logos)

        config = PipelineConfig(
            input_path=local_path,
            mode=_MODE,
            fps=_FPS,
            detector=_DETECTOR,
            logos_dir=_LOGOS_DIR,
            ocr_backend=_OCR_BACKEND,
            device=_DEVICE,
            clip_model=_CLIP_MODEL,
            clip_pretrained=_CLIP_PRETRAINED,
            similarity_threshold=_SIMILARITY,
            output_dir=_OUTPUT_DIR,
            labels_file="",
            target_labels=[],
        )

        result = run_pipeline(config, label_config=label_config)
        elapsed = round(time.time() - t0, 2)

        # Build detection rows for DB, mapping display name → logo_id
        det_rows: List[Dict[str, Any]] = []
        for d in result.get("detection_details", []):
            logo_id = name_to_id.get(d["label"].lower(), d["label"])
            coords = d.get("coordinates_px", {})
            ncoords = d.get("coordinates_normalized", {})
            space = d.get("space_occupied", {})
            det_rows.append({
                "logo_id": logo_id,
                "frame_number": d["frame_number"],
                "timestamp_sec": d["timestamp_sec"],
                "x1": coords.get("x1"), "y1": coords.get("y1"),
                "x2": coords.get("x2"), "y2": coords.get("y2"),
                "nx1": ncoords.get("x1"), "ny1": ncoords.get("y1"),
                "nx2": ncoords.get("x2"), "ny2": ncoords.get("y2"),
                "width_px": coords.get("width_px"), "height_px": coords.get("height_px"),
                "area_px": space.get("area_px"), "area_pct": space.get("area_pct"),
                "confidence": d["confidence"],
                "source": d["source"],
                "region": d["region"],
            })

        db.store_detections(job_id, video_id, det_rows)

        # Create annotated video
        annotated_path = str(Path(_OUTPUT_DIR) / f"{video_id}_annotated.mp4")
        try:
            _create_annotated_video(
                local_path, result.get("detection_details", []), _FPS, annotated_path,
            )
        except Exception as ve:
            log.warning("Annotated video creation failed: %s", ve)
            annotated_path = ""

        s3_url = ""
        if annotated_path and Path(annotated_path).exists():
            s3_key = f"{_S3_PREFIX}{video_id}_annotated.mp4"
            s3_url = _upload_to_s3(annotated_path, s3_key)

        # Recompute aggregate stats for every target logo
        for lid in target_logos:
            try:
                db.recompute_logo_stats(lid)
            except Exception as se:
                log.warning("Stats recomputation failed for %s: %s", lid, se)

        db.update_job_status(
            job_id, "completed",
            result_json=json.dumps(result, default=str),
            annotated_video_s3_url=s3_url or None,
            frames_analysed=result.get("frames_analysed"),
            duration_sec=result.get("media", {}).get("duration_sec"),
            processing_time_sec=elapsed,
            completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        )

    except Exception as exc:
        log.exception("Job %s failed", job_id)
        db.update_job_status(
            job_id, "failed",
            error_message=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if local_path:
            Path(local_path).unlink(missing_ok=True)
