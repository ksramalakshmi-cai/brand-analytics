"""
MongoDB database layer for Brand Analytics.

Collections (in DB ``brand_visibility_analytics``):
    logos            — registered brand logos
    jobs             — async video-processing jobs
    detections       — per-frame detection tuples
    logo_stats       — aggregated visibility stats per logo (materialised)

Connection string is read from env ``BA_MONGO_URI`` (falls back to the
Atlas cluster below).  The database name is ``brand_visibility_analytics``.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from pymongo import MongoClient, ASCENDING

_MONGO_URI = os.getenv(
    "BA_MONGO_URI",
    "mongodb+srv://consumaengg:teamconsuma0506@cluster0.j0nfcwm.mongodb.net/ra"
    "?retryWrites=true&w=majority",
)
_DB_NAME = os.getenv("BA_MONGO_DB", "brand_visibility_analytics")

PROMINENCE_WEIGHT = 0.35
FREQUENCY_WEIGHT = 0.15
DURATION_WEIGHT = 0.2
QUALITY_WEIGHT = 0.3

_client: MongoClient = MongoClient(_MONGO_URI)
_db = _client[_DB_NAME]

logos_col = _db["logos"]
jobs_col = _db["jobs"]
detections_col = _db["detections"]
logo_stats_col = _db["logo_stats"]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def init_db() -> None:
    """Create indexes (idempotent)."""
    logos_col.create_index("logo_id", unique=True)
    jobs_col.create_index("job_id", unique=True)
    jobs_col.create_index("video_id", unique=True)
    detections_col.create_index([("job_id", ASCENDING)])
    detections_col.create_index([("logo_id", ASCENDING)])
    detections_col.create_index([("video_id", ASCENDING)])
    logo_stats_col.create_index("logo_id", unique=True)


def _safe_div(num: float, den: float) -> float:
    return num / den if den and den > 0 else 0.0


def _fcs_breakdown(
    roi_duration: float,
    object_presence_duration: float,
    max_objects: int,
    total_duration: float,
) -> Dict[str, float]:
    prominence = PROMINENCE_WEIGHT * _safe_div(roi_duration, object_presence_duration)
    frequency = FREQUENCY_WEIGHT * _safe_div(max_objects, 5.0)
    duration = DURATION_WEIGHT * _safe_div(roi_duration, total_duration)
    quality = QUALITY_WEIGHT
    total = prominence + frequency + duration + quality
    return {
        "prominence_weightage": round(prominence, 6),
        "frequency_weightage": round(frequency, 6),
        "duration_weightage": round(duration, 6),
        "quality_weightage": round(quality, 6),
        "fcs_score": round(total, 6),
    }


def _strip_id(doc: Optional[Dict]) -> Optional[Dict[str, Any]]:
    """Remove MongoDB ``_id`` field from a document."""
    if doc is None:
        return None
    doc.pop("_id", None)
    return doc


# ── Logo CRUD ─────────────────────────────────────────────────────────────

def create_logo(
    logo_id: str, name: str, detection_method: str = "both",
    reference_count: int = 0, s3_path: str = "",
) -> Dict[str, Any]:
    now = _now()
    logos_col.insert_one({
        "logo_id": logo_id,
        "name": name,
        "detection_method": detection_method,
        "reference_count": reference_count,
        "s3_path": s3_path,
        "created_at": now,
        "updated_at": now,
    })
    logo_stats_col.update_one(
        {"logo_id": logo_id},
        {"$setOnInsert": {"logo_id": logo_id}},
        upsert=True,
    )
    return get_logo(logo_id)  # type: ignore[return-value]


def get_logo(logo_id: str) -> Optional[Dict[str, Any]]:
    return _strip_id(logos_col.find_one({"logo_id": logo_id}))


def list_logos() -> List[Dict[str, Any]]:
    return [_strip_id(d) for d in logos_col.find().sort("created_at", ASCENDING)]  # type: ignore[misc]


def update_logo(logo_id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
    allowed = {"name", "detection_method", "reference_count", "s3_path"}
    updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not updates:
        return get_logo(logo_id)
    updates["updated_at"] = _now()
    logos_col.update_one({"logo_id": logo_id}, {"$set": updates})
    return get_logo(logo_id)


def delete_logo(logo_id: str) -> bool:
    result = logos_col.delete_one({"logo_id": logo_id})
    if result.deleted_count:
        jobs_col.update_many({}, {"$pull": {"target_logos": logo_id}})
        detections_col.delete_many({"logo_id": logo_id})
        logo_stats_col.delete_one({"logo_id": logo_id})
        return True
    return False


# ── Job CRUD ──────────────────────────────────────────────────────────────

def create_job(
    job_id: str, video_id: str, video_url: str, engagements: int, target_logo_ids: List[str],
) -> Dict[str, Any]:
    now = _now()
    jobs_col.insert_one({
        "job_id": job_id,
        "video_id": video_id,
        "video_url": video_url,
        "engagements": engagements,
        "target_logos": target_logo_ids,
        "status": "pending",
        "error_message": None,
        "result_json": None,
        "annotated_video_s3_url": None,
        "frames_analysed": None,
        "duration_sec": None,
        "processing_time_sec": None,
        "created_at": now,
        "updated_at": now,
        "completed_at": None,
    })
    return get_job(job_id)  # type: ignore[return-value]


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    doc = _strip_id(jobs_col.find_one({"job_id": job_id}))
    if not doc:
        return None
    if doc.get("result_json"):
        doc["result"] = json.loads(doc["result_json"])
    else:
        doc["result"] = None
    return doc


def get_job_by_video_id(video_id: str) -> Optional[Dict[str, Any]]:
    doc = jobs_col.find_one({"video_id": video_id})
    if not doc:
        return None
    return get_job(doc["job_id"])


def update_job_status(job_id: str, status: str, **kwargs: Any) -> None:
    allowed_keys = {
        "error_message", "annotated_video_s3_url", "frames_analysed",
        "duration_sec", "processing_time_sec", "completed_at", "result_json",
    }
    updates: Dict[str, Any] = {"status": status, "updated_at": _now()}
    for k, v in kwargs.items():
        if k in allowed_keys:
            updates[k] = v
    jobs_col.update_one({"job_id": job_id}, {"$set": updates})


# ── Detections ────────────────────────────────────────────────────────────

def store_detections(job_id: str, video_id: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    docs = []
    for d in rows:
        docs.append({
            "job_id": job_id,
            "video_id": video_id,
            "logo_id": d["logo_id"],
            "frame_number": d["frame_number"],
            "timestamp_sec": d["timestamp_sec"],
            "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
            "nx1": d["nx1"], "ny1": d["ny1"], "nx2": d["nx2"], "ny2": d["ny2"],
            "width_px": d["width_px"], "height_px": d["height_px"],
            "area_px": d["area_px"], "area_pct": d["area_pct"],
            "confidence": d["confidence"],
            "source": d["source"],
            "region": d["region"],
        })
    detections_col.insert_many(docs)


# ── Stats ─────────────────────────────────────────────────────────────────

def recompute_logo_stats(logo_id: str) -> None:
    """Re-materialise aggregate stats for a logo from completed jobs."""
    pipeline_jobs = list(jobs_col.find({
        "target_logos": logo_id,
        "status": "completed",
    }))

    total_videos = len(pipeline_jobs)
    total_engagements = sum(j.get("engagements") or 0 for j in pipeline_jobs)
    total_play_count = total_engagements

    agg_result = list(detections_col.aggregate([
        {"$match": {"logo_id": logo_id}},
        {"$group": {
            "_id": None,
            "cnt": {"$sum": 1},
            "avg_conf": {"$avg": "$confidence"},
            "avg_area": {"$avg": "$area_pct"},
        }},
    ]))
    if agg_result:
        total_detections = agg_result[0]["cnt"]
        avg_confidence = round(agg_result[0]["avg_conf"] or 0, 4)
        avg_area_pct = round(agg_result[0]["avg_area"] or 0, 2)
    else:
        total_detections = 0
        avg_confidence = 0.0
        avg_area_pct = 0.0

    total_screen_time = 0.0
    visibility_pcts: list = []
    weighted_vis = 0.0
    fcs_scores: List[float] = []

    for j in pipeline_jobs:
        dur = j.get("duration_sec") or 0
        eng = j.get("engagements") or 0
        fa = j.get("frames_analysed") or 1
        jid = j["job_id"]

        fc_result = list(detections_col.aggregate([
            {"$match": {"job_id": jid, "logo_id": logo_id}},
            {"$group": {"_id": "$frame_number"}},
            {"$count": "c"},
        ]))
        fc = fc_result[0]["c"] if fc_result else 0

        ts_result = list(detections_col.aggregate([
            {"$match": {"job_id": jid, "logo_id": logo_id}},
            {"$group": {
                "_id": None,
                "min_ts": {"$min": "$timestamp_sec"},
                "max_ts": {"$max": "$timestamp_sec"},
            }},
        ]))
        min_ts = ts_result[0]["min_ts"] if ts_result else None
        max_ts = ts_result[0]["max_ts"] if ts_result else None

        max_objs_result = list(detections_col.aggregate([
            {"$match": {"job_id": jid, "logo_id": logo_id}},
            {"$group": {"_id": "$frame_number", "c": {"$sum": 1}}},
            {"$group": {"_id": None, "max_objects": {"$max": "$c"}}},
        ]))
        max_objects = int(max_objs_result[0]["max_objects"]) if max_objs_result else 0

        if fa > 0 and dur > 0:
            interval = dur / fa
            st = fc * interval
            total_screen_time += st
            vis_pct = st / dur * 100
            visibility_pcts.append(vis_pct)
            weighted_vis += vis_pct * eng

            if min_ts is not None and max_ts is not None:
                object_presence_duration = max(0.0, float(max_ts) - float(min_ts) + interval)
            else:
                object_presence_duration = 0.0
            fcs = _fcs_breakdown(
                roi_duration=st,
                object_presence_duration=object_presence_duration,
                max_objects=max_objects,
                total_duration=dur,
            )["fcs_score"]
            fcs_scores.append(fcs)

    avg_st = (total_screen_time / total_videos) if total_videos else 0
    avg_vis = (sum(visibility_pcts) / len(visibility_pcts)) if visibility_pcts else 0
    avg_fcs = (sum(fcs_scores) / len(fcs_scores)) if fcs_scores else 0

    logo_stats_col.update_one(
        {"logo_id": logo_id},
        {"$set": {
            "logo_id": logo_id,
            "total_videos": total_videos,
            "total_detections": total_detections,
            "total_screen_time_sec": round(total_screen_time, 2),
            "avg_screen_time_per_video_sec": round(avg_st, 2),
            "avg_visibility_pct": round(avg_vis, 2),
            "avg_confidence": avg_confidence,
            "avg_area_pct": avg_area_pct,
            "avg_fcs_score": round(avg_fcs, 6),
            "total_play_count": int(total_play_count),
            "total_engagements": total_engagements,
            "weighted_visibility_score": round(weighted_vis, 4),
            "updated_at": _now(),
        }},
        upsert=True,
    )


def get_logo_stats(logo_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    query: Dict[str, Any] = {}
    if logo_ids:
        query["logo_id"] = {"$in": logo_ids}
    rows = list(logo_stats_col.find(query))
    result: List[Dict[str, Any]] = []
    for r in rows:
        r.pop("_id", None)
        logo = get_logo(r["logo_id"])
        if not logo:
            continue
        r["name"] = logo["name"]
        r["detection_method"] = logo["detection_method"]
        for field in (
            "total_videos", "total_detections", "total_screen_time_sec",
            "avg_screen_time_per_video_sec", "avg_visibility_pct",
            "avg_confidence", "avg_area_pct", "avg_fcs_score",
            "total_play_count", "total_engagements", "weighted_visibility_score",
        ):
            r.setdefault(field, 0)
        result.append(r)
    return result


def get_per_video_breakdown(logo_id: str) -> List[Dict[str, Any]]:
    """Per-video stats for a single logo."""
    pipeline_jobs = list(jobs_col.find({
        "target_logos": logo_id,
        "status": "completed",
    }).sort("created_at", -1))

    breakdown: List[Dict[str, Any]] = []
    for j in pipeline_jobs:
        dur = j.get("duration_sec") or 0
        fa = j.get("frames_analysed") or 1
        jid = j["job_id"]

        agg_result = list(detections_col.aggregate([
            {"$match": {"job_id": jid, "logo_id": logo_id}},
            {"$group": {
                "_id": None,
                "cnt": {"$sum": 1},
                "frame_cnt": {"$addToSet": "$frame_number"},
                "avg_conf": {"$avg": "$confidence"},
                "avg_area": {"$avg": "$area_pct"},
            }},
        ]))
        if agg_result:
            cnt = agg_result[0]["cnt"]
            fc = len(agg_result[0]["frame_cnt"])
            avg_conf = agg_result[0]["avg_conf"] or 0
            avg_area = agg_result[0]["avg_area"] or 0
        else:
            cnt = 0
            fc = 0
            avg_conf = 0
            avg_area = 0

        interval = (dur / fa) if (fa and dur) else 0
        st = round(fc * interval, 2)
        vis = round(st / dur * 100, 2) if dur else 0

        ts_result = list(detections_col.aggregate([
            {"$match": {"job_id": jid, "logo_id": logo_id}},
            {"$group": {
                "_id": None,
                "min_ts": {"$min": "$timestamp_sec"},
                "max_ts": {"$max": "$timestamp_sec"},
            }},
        ]))
        min_ts = ts_result[0]["min_ts"] if ts_result else None
        max_ts = ts_result[0]["max_ts"] if ts_result else None
        object_presence_duration = (
            max(0.0, float(max_ts) - float(min_ts) + interval)
            if min_ts is not None and max_ts is not None else 0.0
        )

        max_objs_result = list(detections_col.aggregate([
            {"$match": {"job_id": jid, "logo_id": logo_id}},
            {"$group": {"_id": "$frame_number", "c": {"$sum": 1}}},
            {"$group": {"_id": None, "max_objects": {"$max": "$c"}}},
        ]))
        max_objects = int(max_objs_result[0]["max_objects"]) if max_objs_result else 0

        fcs = _fcs_breakdown(
            roi_duration=st,
            object_presence_duration=object_presence_duration,
            max_objects=max_objects,
            total_duration=dur,
        )

        breakdown.append({
            "video_id": j["video_id"],
            "job_id": jid,
            "detections": cnt,
            "frames_visible": fc,
            "screen_time_sec": st,
            "visibility_pct": vis,
            "avg_confidence": round(avg_conf, 4),
            "avg_area_pct": round(avg_area, 2),
            "engagements": j.get("engagements") or 0,
            "play_count": j.get("engagements") or 0,
            "metric_inputs": {
                "roi_duration": round(st, 4),
                "object_presence_duration": round(object_presence_duration, 4),
                "max_objects": max_objects,
                "total_duration": round(float(dur), 4),
            },
            "fcs_breakdown": fcs,
        })
    return breakdown
