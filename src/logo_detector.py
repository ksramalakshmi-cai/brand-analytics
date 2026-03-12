"""
Logo detection wrapper around Ultralytics YOLO.

Accepts any YOLO-compatible weights — swap in a logo-specific model
(e.g. trained on LogoDet-3K or OpenLogo) by changing config.model_path.
"""

from __future__ import annotations

import numpy as np
import torch
from dataclasses import dataclass
from typing import List, Optional

from ultralytics import YOLO

from config import PipelineConfig

_CUDA_AVAILABLE = torch.cuda.is_available()


@dataclass
class Detection:
    """A single detected logo / object in a frame."""
    class_id: int
    label: str
    confidence: float
    # absolute pixel coordinates
    x1: int
    y1: int
    x2: int
    y2: int
    # normalised coordinates (0-1) — resolution-independent
    nx1: float
    ny1: float
    nx2: float
    ny2: float
    # derived metrics
    width_px: int
    height_px: int
    area_px: int
    area_pct: float  # % of total frame area
    # OCR fields (populated by OCRReader when enabled)
    ocr_text: str = ""
    ocr_matched_label: Optional[str] = None
    ocr_match_ratio: float = 0.0
    original_confidence: float = 0.0  # pre-boost confidence
    source: str = "detector"  # detector | reference | ocr


class LogoDetector:
    """Thin wrapper that loads a YOLO model once and runs inference."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        if config.device:
            device = config.device
        else:
            device = "cuda:0" if _CUDA_AVAILABLE else "cpu"
        self.model = YOLO(config.model_path)
        self.model.to(device)

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run inference on a single BGR frame and return detections."""
        h, w = frame.shape[:2]
        frame_area = h * w

        results = self.model.predict(
            source=frame,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            imgsz=self.config.img_size,
            verbose=False,
        )

        detections: List[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id = int(box.cls[0])
                label = self.model.names.get(cls_id, str(cls_id))
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                bw = x2 - x1
                bh = y2 - y1

                detections.append(Detection(
                    class_id=cls_id,
                    label=label,
                    confidence=conf,
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    nx1=round(x1 / w, 4),
                    ny1=round(y1 / h, 4),
                    nx2=round(x2 / w, 4),
                    ny2=round(y2 / h, 4),
                    width_px=bw,
                    height_px=bh,
                    area_px=bw * bh,
                    area_pct=round((bw * bh) / frame_area * 100, 2),
                    source="detector",
                ))

        return detections
