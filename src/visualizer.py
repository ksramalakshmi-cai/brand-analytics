"""
Draw bounding boxes on frames and export annotated images + cropped logos.
"""

from __future__ import annotations

import re

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.logo_detector import Detection
from config import PipelineConfig

# Distinct colours for up to 20 brands; cycles after that
_PALETTE: List[Tuple[int, int, int]] = [
    (0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255),
    (255, 0, 255), (255, 255, 0), (128, 0, 255), (0, 128, 255),
    (255, 128, 0), (0, 255, 128), (128, 255, 0), (255, 0, 128),
    (64, 224, 208), (255, 165, 0), (75, 0, 130), (220, 20, 60),
    (0, 191, 255), (50, 205, 50), (255, 99, 71), (186, 85, 211),
]


class Visualizer:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self._color_map: Dict[str, Tuple[int, int, int]] = dict(config.label_colors)
        self._color_idx = 0

    def _get_color(self, label: str) -> Tuple[int, int, int]:
        if label not in self._color_map:
            self._color_map[label] = _PALETTE[self._color_idx % len(_PALETTE)]
            self._color_idx += 1
        return self._color_map[label]

    def draw_detections(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        timestamp_sec: Optional[float] = None,
    ) -> np.ndarray:
        """Return a copy of the frame with bounding boxes and labels drawn."""
        annotated = frame.copy()
        thickness = self.config.bbox_thickness
        font_scale = self.config.font_scale

        for det in detections:
            color = self._get_color(det.label)
            cv2.rectangle(annotated, (det.x1, det.y1), (det.x2, det.y2), color, thickness)

            conf_str = f"{det.confidence:.2f}"
            if det.ocr_matched_label:
                conf_str += " OCR"
            text = f"{det.label} {conf_str}"

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            cv2.rectangle(
                annotated,
                (det.x1, det.y1 - th - 8),
                (det.x1 + tw + 4, det.y1),
                color, -1,
            )
            cv2.putText(
                annotated, text,
                (det.x1 + 2, det.y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
            )

            if det.ocr_text:
                ocr_label = f'OCR: "{det.ocr_text[:40]}"'
                (otw, oth), _ = cv2.getTextSize(ocr_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, 1)
                cv2.rectangle(
                    annotated,
                    (det.x1, det.y2),
                    (det.x1 + otw + 4, det.y2 + oth + 8),
                    (40, 40, 40), -1,
                )
                cv2.putText(
                    annotated, ocr_label,
                    (det.x1 + 2, det.y2 + oth + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (200, 200, 200), 1,
                )

        if timestamp_sec is not None:
            ts_text = f"t={timestamp_sec:.1f}s"
            cv2.putText(
                annotated, ts_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
            )

        return annotated

    @staticmethod
    def _safe_dirname(label: str) -> str:
        """Convert a label into a filesystem-safe directory name."""
        return re.sub(r"[^\w\-]+", "_", label.lower()).strip("_")

    def save_annotated_frame(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_index: int,
        timestamp_sec: float,
        output_dir: Path,
    ) -> List[Path]:
        """Save annotated frame into a subdirectory for each detected brand."""
        if not self.config.save_annotated_frames or not detections:
            return []

        annotated = self.draw_detections(frame, detections, timestamp_sec)
        fname = f"frame_{frame_index:06d}_t{timestamp_sec:.1f}s.jpg"
        saved: List[Path] = []
        seen_labels: set = set()

        for det in detections:
            key = det.label.lower()
            if key in seen_labels:
                continue
            seen_labels.add(key)
            brand_dir = output_dir / "frames" / self._safe_dirname(det.label)
            brand_dir.mkdir(parents=True, exist_ok=True)
            path = brand_dir / fname
            cv2.imwrite(str(path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
            saved.append(path)

        return saved

    def save_cropped_logos(
        self,
        frame: np.ndarray,
        detections: List[Detection],
        frame_index: int,
        output_dir: Path,
    ) -> List[Path]:
        if not self.config.save_cropped_logos:
            return []
        paths = []
        for i, det in enumerate(detections):
            crop = frame[det.y1:det.y2, det.x1:det.x2]
            if crop.size == 0:
                continue
            brand_dir = output_dir / "crops" / self._safe_dirname(det.label)
            brand_dir.mkdir(parents=True, exist_ok=True)
            path = brand_dir / f"frame_{frame_index:06d}_det{i}.jpg"
            cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            paths.append(path)
        return paths
