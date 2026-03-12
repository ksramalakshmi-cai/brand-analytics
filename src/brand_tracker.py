"""
Aggregates per-frame detections into brand-level visibility analytics.

Outputs:
  - Per-brand summary: total screen time, average position/size, frame ranges
  - Per-frame detail: every detection with absolute + normalised coordinates
"""

from __future__ import annotations

import json
import csv
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from src.frame_extractor import MediaInfo
from src.logo_detector import Detection


@dataclass
class FrameRecord:
    """One detection in one frame."""
    frame_index: int
    timestamp_sec: float
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    nx1: float
    ny1: float
    nx2: float
    ny2: float
    width_px: int
    height_px: int
    area_px: int
    area_pct: float
    frame_width: int
    frame_height: int
    position_quadrant: str  # e.g. "top-left", "center"
    ocr_text: str = ""
    ocr_matched_label: Optional[str] = None
    ocr_match_ratio: float = 0.0
    original_confidence: float = 0.0


@dataclass
class BrandSummary:
    """Aggregated visibility stats for a single brand/label."""
    label: str
    total_detections: int = 0
    frames_visible: List[int] = field(default_factory=list)
    timestamps_visible: List[float] = field(default_factory=list)
    total_screen_time_sec: float = 0.0
    avg_confidence: float = 0.0
    avg_area_pct: float = 0.0
    avg_position_nx: float = 0.0  # normalised center x
    avg_position_ny: float = 0.0  # normalised center y
    min_area_pct: float = 0.0
    max_area_pct: float = 0.0
    dominant_quadrant: str = ""


def _quadrant(nx_center: float, ny_center: float) -> str:
    col = "left" if nx_center < 0.33 else ("center" if nx_center < 0.66 else "right")
    row = "top" if ny_center < 0.33 else ("middle" if ny_center < 0.66 else "bottom")
    return f"{row}-{col}"


class BrandTracker:
    """Collects detections across frames and computes analytics."""

    def __init__(self, sample_fps: float = 1.0, target_labels: Optional[List[str]] = None):
        self.sample_fps = sample_fps
        self.target_labels = target_labels or []
        self._target_lower = {t.lower() for t in self.target_labels}
        self.records: List[FrameRecord] = []
        self._brand_detections: Dict[str, List[FrameRecord]] = {}

    def add(
        self,
        frame_index: int,
        timestamp_sec: float,
        detections: List[Detection],
        frame_width: int,
        frame_height: int,
    ) -> None:
        for det in detections:
            nx_center = (det.nx1 + det.nx2) / 2
            ny_center = (det.ny1 + det.ny2) / 2

            rec = FrameRecord(
                frame_index=frame_index,
                timestamp_sec=timestamp_sec,
                label=det.label,
                confidence=det.confidence,
                x1=det.x1, y1=det.y1, x2=det.x2, y2=det.y2,
                nx1=det.nx1, ny1=det.ny1, nx2=det.nx2, ny2=det.ny2,
                width_px=det.width_px, height_px=det.height_px,
                area_px=det.area_px, area_pct=det.area_pct,
                frame_width=frame_width, frame_height=frame_height,
                position_quadrant=_quadrant(nx_center, ny_center),
                ocr_text=det.ocr_text,
                ocr_matched_label=det.ocr_matched_label,
                ocr_match_ratio=det.ocr_match_ratio,
                original_confidence=det.original_confidence,
            )
            self.records.append(rec)
            self._brand_detections.setdefault(det.label, []).append(rec)

    def _is_target(self, label: str) -> bool:
        """Check if a label matches one of the target brands (or pass all if no targets)."""
        if not self._target_lower:
            return True
        return label.lower() in self._target_lower

    def summarise(self) -> Dict[str, BrandSummary]:
        summaries: Dict[str, BrandSummary] = {}
        interval = 1.0 / self.sample_fps if self.sample_fps > 0 else 1.0

        for label, recs in self._brand_detections.items():
            if not self._is_target(label):
                continue

            frames = sorted(set(r.frame_index for r in recs))
            timestamps = sorted(set(r.timestamp_sec for r in recs))
            confs = [r.confidence for r in recs]
            areas = [r.area_pct for r in recs]
            nxs = [(r.nx1 + r.nx2) / 2 for r in recs]
            nys = [(r.ny1 + r.ny2) / 2 for r in recs]
            quads = [r.position_quadrant for r in recs]

            quad_counts: Dict[str, int] = {}
            for q in quads:
                quad_counts[q] = quad_counts.get(q, 0) + 1
            dominant = max(quad_counts, key=quad_counts.get) if quad_counts else ""

            summaries[label] = BrandSummary(
                label=label,
                total_detections=len(recs),
                frames_visible=frames,
                timestamps_visible=timestamps,
                total_screen_time_sec=round(len(frames) * interval, 2),
                avg_confidence=round(sum(confs) / len(confs), 4),
                avg_area_pct=round(sum(areas) / len(areas), 2),
                avg_position_nx=round(sum(nxs) / len(nxs), 4),
                avg_position_ny=round(sum(nys) / len(nys), 4),
                min_area_pct=round(min(areas), 2),
                max_area_pct=round(max(areas), 2),
                dominant_quadrant=dominant,
            )

        return summaries

    # ------------------------------------------------------------------ I/O

    def save_detail_csv(self, path: Path) -> None:
        filtered = [r for r in self.records if self._is_target(r.label)]
        if not filtered:
            return
        fieldnames = list(FrameRecord.__dataclass_fields__.keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for rec in filtered:
                writer.writerow(asdict(rec))

    def save_summary_json(self, path: Path, media_info: Optional[MediaInfo] = None) -> None:
        summaries = self.summarise()
        filtered = [r for r in self.records if self._is_target(r.label)]
        payload = {
            "media": asdict(media_info) if media_info else {},
            "target_labels": self.target_labels,
            "brands": {
                label: {
                    **asdict(s),
                    "visibility_pct": round(
                        s.total_screen_time_sec / media_info.duration_sec * 100, 2
                    ) if media_info and media_info.duration_sec > 0 else None,
                }
                for label, s in summaries.items()
            },
            "total_frames_analysed": len(set(r.frame_index for r in self.records)) if self.records else 0,
            "frames_with_target_brands": len(set(r.frame_index for r in filtered)) if filtered else 0,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2, default=str)

    def save_summary_csv(self, path: Path, media_info: Optional[MediaInfo] = None) -> None:
        summaries = self.summarise()
        if not summaries:
            return
        fieldnames = [
            "label", "total_detections", "total_screen_time_sec",
            "visibility_pct", "avg_confidence", "avg_area_pct",
            "min_area_pct", "max_area_pct",
            "avg_position_nx", "avg_position_ny", "dominant_quadrant",
            "frames_visible", "timestamps_visible",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for label, s in summaries.items():
                vis_pct = (
                    round(s.total_screen_time_sec / media_info.duration_sec * 100, 2)
                    if media_info and media_info.duration_sec > 0 else None
                )
                writer.writerow({
                    "label": s.label,
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
                    "frames_visible": ";".join(str(f) for f in s.frames_visible),
                    "timestamps_visible": ";".join(str(t) for t in s.timestamps_visible),
                })
