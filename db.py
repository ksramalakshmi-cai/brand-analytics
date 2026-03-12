"""
SQLite database layer for Brand Analytics.

Tables:
    logos            — registered brand logos
    jobs             — async video-processing jobs
    job_target_logos — M:N link between jobs and target logos
    detections       — per-frame detection tuples
    logo_stats       — aggregated visibility stats per logo (materialised)
"""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

_DB_PATH = os.getenv("BA_DB_PATH", "brand_analytics.db")


@contextmanager
def get_connection():
    conn = sqlite3.connect(_DB_PATH, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS logos (
                logo_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                detection_method TEXT DEFAULT 'both'
                    CHECK(detection_method IN ('ocr','detector','both')),
                reference_count INTEGER DEFAULT 0,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY,
                video_id TEXT NOT NULL UNIQUE,
                video_url TEXT NOT NULL,
                engagements INTEGER DEFAULT 0,
                status TEXT DEFAULT 'pending'
                    CHECK(status IN ('pending','processing','completed','failed')),
                error_message TEXT,
                result_json TEXT,
                annotated_video_s3_url TEXT,
                frames_analysed INTEGER,
                duration_sec REAL,
                processing_time_sec REAL,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now')),
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS job_target_logos (
                job_id TEXT REFERENCES jobs(job_id) ON DELETE CASCADE,
                logo_id TEXT REFERENCES logos(logo_id) ON DELETE CASCADE,
                PRIMARY KEY (job_id, logo_id)
            );

            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                video_id TEXT NOT NULL,
                logo_id TEXT NOT NULL,
                frame_number INTEGER NOT NULL,
                timestamp_sec REAL NOT NULL,
                x1 INTEGER, y1 INTEGER, x2 INTEGER, y2 INTEGER,
                nx1 REAL, ny1 REAL, nx2 REAL, ny2 REAL,
                width_px INTEGER, height_px INTEGER,
                area_px INTEGER,
                area_pct REAL,
                confidence REAL,
                source TEXT,
                region TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_det_job  ON detections(job_id);
            CREATE INDEX IF NOT EXISTS idx_det_logo ON detections(logo_id);
            CREATE INDEX IF NOT EXISTS idx_det_vid  ON detections(video_id);

            CREATE TABLE IF NOT EXISTS logo_stats (
                logo_id TEXT PRIMARY KEY REFERENCES logos(logo_id) ON DELETE CASCADE,
                total_videos INTEGER DEFAULT 0,
                total_detections INTEGER DEFAULT 0,
                total_screen_time_sec REAL DEFAULT 0,
                avg_screen_time_per_video_sec REAL DEFAULT 0,
                avg_visibility_pct REAL DEFAULT 0,
                avg_confidence REAL DEFAULT 0,
                avg_area_pct REAL DEFAULT 0,
                total_engagements INTEGER DEFAULT 0,
                weighted_visibility_score REAL DEFAULT 0,
                updated_at TEXT DEFAULT (datetime('now'))
            );
        """)


# ── Logo CRUD ─────────────────────────────────────────────────────────────

def create_logo(
    logo_id: str, name: str, detection_method: str = "both", reference_count: int = 0,
) -> Dict[str, Any]:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO logos (logo_id, name, detection_method, reference_count) "
            "VALUES (?, ?, ?, ?)",
            (logo_id, name, detection_method, reference_count),
        )
        conn.execute("INSERT OR IGNORE INTO logo_stats (logo_id) VALUES (?)", (logo_id,))
    return get_logo(logo_id)  # type: ignore[return-value]


def get_logo(logo_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM logos WHERE logo_id = ?", (logo_id,)).fetchone()
        return dict(row) if row else None


def list_logos() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        return [dict(r) for r in conn.execute("SELECT * FROM logos ORDER BY created_at").fetchall()]


def update_logo(logo_id: str, **kwargs: Any) -> Optional[Dict[str, Any]]:
    allowed = {"name", "detection_method", "reference_count"}
    updates = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    if not updates:
        return get_logo(logo_id)
    sets = ", ".join(f"{k} = ?" for k in updates)
    vals = list(updates.values()) + [logo_id]
    with get_connection() as conn:
        conn.execute(f"UPDATE logos SET {sets}, updated_at = datetime('now') WHERE logo_id = ?", vals)
    return get_logo(logo_id)


def delete_logo(logo_id: str) -> bool:
    with get_connection() as conn:
        return conn.execute("DELETE FROM logos WHERE logo_id = ?", (logo_id,)).rowcount > 0


# ── Job CRUD ──────────────────────────────────────────────────────────────

def create_job(
    job_id: str, video_id: str, video_url: str, engagements: int, target_logo_ids: List[str],
) -> Dict[str, Any]:
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO jobs (job_id, video_id, video_url, engagements) VALUES (?, ?, ?, ?)",
            (job_id, video_id, video_url, engagements),
        )
        for lid in target_logo_ids:
            conn.execute(
                "INSERT INTO job_target_logos (job_id, logo_id) VALUES (?, ?)", (job_id, lid),
            )
    return get_job(job_id)  # type: ignore[return-value]


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
        if not row:
            return None
        job = dict(row)
        targets = conn.execute(
            "SELECT logo_id FROM job_target_logos WHERE job_id = ?", (job_id,),
        ).fetchall()
        job["target_logos"] = [r["logo_id"] for r in targets]
        if job.get("result_json"):
            job["result"] = json.loads(job["result_json"])
        else:
            job["result"] = None
        return job


def get_job_by_video_id(video_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT job_id FROM jobs WHERE video_id = ?", (video_id,)).fetchone()
        if not row:
            return None
        return get_job(row["job_id"])


def update_job_status(job_id: str, status: str, **kwargs: Any) -> None:
    extras = ""
    vals: list = [status]
    for k, v in kwargs.items():
        if k in (
            "error_message", "annotated_video_s3_url", "frames_analysed",
            "duration_sec", "processing_time_sec", "completed_at", "result_json",
        ):
            extras += f", {k} = ?"
            vals.append(v)
    vals.append(job_id)
    with get_connection() as conn:
        conn.execute(
            f"UPDATE jobs SET status = ?, updated_at = datetime('now'){extras} WHERE job_id = ?",
            vals,
        )


# ── Detections ────────────────────────────────────────────────────────────

def store_detections(job_id: str, video_id: str, rows: List[Dict[str, Any]]) -> None:
    with get_connection() as conn:
        conn.executemany(
            """INSERT INTO detections
               (job_id, video_id, logo_id, frame_number, timestamp_sec,
                x1, y1, x2, y2, nx1, ny1, nx2, ny2,
                width_px, height_px, area_px, area_pct,
                confidence, source, region)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [
                (
                    job_id, video_id,
                    d["logo_id"], d["frame_number"], d["timestamp_sec"],
                    d["x1"], d["y1"], d["x2"], d["y2"],
                    d["nx1"], d["ny1"], d["nx2"], d["ny2"],
                    d["width_px"], d["height_px"], d["area_px"], d["area_pct"],
                    d["confidence"], d["source"], d["region"],
                )
                for d in rows
            ],
        )


# ── Stats ─────────────────────────────────────────────────────────────────

def recompute_logo_stats(logo_id: str) -> None:
    """Re-materialise aggregate stats for a logo from completed jobs."""
    with get_connection() as conn:
        jobs = conn.execute("""
            SELECT j.job_id, j.video_id, j.engagements, j.duration_sec, j.frames_analysed
            FROM jobs j JOIN job_target_logos jt ON j.job_id = jt.job_id
            WHERE jt.logo_id = ? AND j.status = 'completed'
        """, (logo_id,)).fetchall()

        total_videos = len(jobs)
        total_engagements = sum(r["engagements"] or 0 for r in jobs)

        agg = conn.execute("""
            SELECT COUNT(*) AS cnt,
                   AVG(confidence) AS avg_conf,
                   AVG(area_pct) AS avg_area
            FROM detections WHERE logo_id = ?
        """, (logo_id,)).fetchone()

        total_detections = agg["cnt"] or 0
        avg_confidence = round(agg["avg_conf"] or 0, 4)
        avg_area_pct = round(agg["avg_area"] or 0, 2)

        total_screen_time = 0.0
        visibility_pcts: list = []
        weighted_vis = 0.0

        for j in jobs:
            dur = j["duration_sec"] or 0
            eng = j["engagements"] or 0
            fa = j["frames_analysed"] or 1

            fc = conn.execute(
                "SELECT COUNT(DISTINCT frame_number) AS c FROM detections "
                "WHERE job_id = ? AND logo_id = ?",
                (j["job_id"], logo_id),
            ).fetchone()["c"]

            if fa > 0 and dur > 0:
                interval = dur / fa
                st = fc * interval
                total_screen_time += st
                vis_pct = st / dur * 100
                visibility_pcts.append(vis_pct)
                weighted_vis += vis_pct * eng

        avg_st = (total_screen_time / total_videos) if total_videos else 0
        avg_vis = (sum(visibility_pcts) / len(visibility_pcts)) if visibility_pcts else 0

        conn.execute("""
            INSERT INTO logo_stats
                (logo_id, total_videos, total_detections, total_screen_time_sec,
                 avg_screen_time_per_video_sec, avg_visibility_pct,
                 avg_confidence, avg_area_pct,
                 total_engagements, weighted_visibility_score, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,datetime('now'))
            ON CONFLICT(logo_id) DO UPDATE SET
                total_videos=excluded.total_videos,
                total_detections=excluded.total_detections,
                total_screen_time_sec=excluded.total_screen_time_sec,
                avg_screen_time_per_video_sec=excluded.avg_screen_time_per_video_sec,
                avg_visibility_pct=excluded.avg_visibility_pct,
                avg_confidence=excluded.avg_confidence,
                avg_area_pct=excluded.avg_area_pct,
                total_engagements=excluded.total_engagements,
                weighted_visibility_score=excluded.weighted_visibility_score,
                updated_at=excluded.updated_at
        """, (
            logo_id, total_videos, total_detections, round(total_screen_time, 2),
            round(avg_st, 2), round(avg_vis, 2),
            avg_confidence, avg_area_pct,
            total_engagements, round(weighted_vis, 4),
        ))


def get_logo_stats(logo_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        if logo_ids:
            ph = ",".join("?" for _ in logo_ids)
            rows = conn.execute(
                f"SELECT ls.*, l.name, l.detection_method "
                f"FROM logo_stats ls JOIN logos l ON ls.logo_id = l.logo_id "
                f"WHERE ls.logo_id IN ({ph})", logo_ids,
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ls.*, l.name, l.detection_method "
                "FROM logo_stats ls JOIN logos l ON ls.logo_id = l.logo_id",
            ).fetchall()
        return [dict(r) for r in rows]


def get_per_video_breakdown(logo_id: str) -> List[Dict[str, Any]]:
    """Per-video stats for a single logo."""
    with get_connection() as conn:
        jobs = conn.execute("""
            SELECT j.job_id, j.video_id, j.engagements, j.duration_sec, j.frames_analysed
            FROM jobs j JOIN job_target_logos jt ON j.job_id = jt.job_id
            WHERE jt.logo_id = ? AND j.status = 'completed'
            ORDER BY j.created_at DESC
        """, (logo_id,)).fetchall()

        breakdown: List[Dict[str, Any]] = []
        for j in jobs:
            dur = j["duration_sec"] or 0
            fa = j["frames_analysed"] or 1
            agg = conn.execute("""
                SELECT COUNT(*) AS cnt,
                       COUNT(DISTINCT frame_number) AS frame_cnt,
                       AVG(confidence) AS avg_conf,
                       AVG(area_pct) AS avg_area
                FROM detections WHERE job_id = ? AND logo_id = ?
            """, (j["job_id"], logo_id)).fetchone()

            fc = agg["frame_cnt"] or 0
            interval = (dur / fa) if (fa and dur) else 0
            st = round(fc * interval, 2)
            vis = round(st / dur * 100, 2) if dur else 0

            breakdown.append({
                "video_id": j["video_id"],
                "job_id": j["job_id"],
                "detections": agg["cnt"] or 0,
                "frames_visible": fc,
                "screen_time_sec": st,
                "visibility_pct": vis,
                "avg_confidence": round(agg["avg_conf"] or 0, 4),
                "avg_area_pct": round(agg["avg_area"] or 0, 2),
                "engagements": j["engagements"] or 0,
            })
        return breakdown
