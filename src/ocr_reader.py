"""
OCR-based text extraction for logo/brand detection using PaddleOCR.

Two operating modes:
  1. **Crop mode** (used with YOLO) — reads text inside a bounding box and
     fuzzy-matches it against known labels to boost confidence.
  2. **Full-frame mode** (OCR-only pipeline) — scans the entire frame for
     text, matches against labels, and returns Detection objects with the
     bounding boxes that PaddleOCR found.
"""

from __future__ import annotations

import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

# Prevent Intel MKL segfaults on CPU-only machines — must be set before
# PaddlePaddle is imported anywhere.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import cv2
import numpy as np

_reader = None  # lazy singleton


def _get_reader(lang: str, gpu: bool):
    """Lazy-init PaddleOCR (heavy to create, so we cache it)."""
    global _reader
    if _reader is None:
        from paddleocr import PaddleOCR
        try:
            _reader = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=gpu)
        except (TypeError, ValueError):
            _reader = PaddleOCR(use_angle_cls=True, lang=lang)
    return _reader


def _normalise(text: str) -> str:
    """Lower-case, collapse whitespace, strip non-alphanumeric."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _best_match(
    ocr_text: str,
    target_labels: List[str],
    threshold: float,
) -> Tuple[Optional[str], float]:
    """
    Return (matched_label, similarity_ratio) if the best fuzzy match is
    above *threshold*, otherwise (None, 0.0).

    Matching strategy:
      1. Exact substring — if the normalised target appears inside the
         normalised OCR text (or vice-versa), ratio = 1.0.
      2. SequenceMatcher ratio — longest-common-subsequence based.
    """
    norm_ocr = _normalise(ocr_text)
    if not norm_ocr:
        return None, 0.0

    best_label: Optional[str] = None
    best_ratio: float = 0.0

    for label in target_labels:
        norm_label = _normalise(label)
        if not norm_label:
            continue

        if norm_label in norm_ocr or norm_ocr in norm_label:
            return label, 1.0

        ratio = SequenceMatcher(None, norm_ocr, norm_label).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_label = label

    if best_ratio >= threshold:
        return best_label, best_ratio
    return None, 0.0


def _parse_paddle_result(result) -> List[Tuple[list, str, float]]:
    """
    Normalise PaddleOCR output into a flat list of (bbox, text, confidence).

    Handles multiple PaddleOCR output formats:
      - Legacy: [[[(bbox, (text, conf)), ...], ...]]
      - v3+:    [[(bbox, (text, conf)), ...]]  or  [None]
      - v5:     may return dicts with 'rec_text', 'rec_score', 'dt_polys'
    """
    entries: List[Tuple[list, str, float]] = []
    if result is None:
        return entries

    # v5+ dict-based output: list of dicts with 'rec_text', 'rec_score', etc.
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        for item in result:
            text = item.get("rec_text", "") or item.get("text", "")
            conf = float(item.get("rec_score", 0) or item.get("score", 0))
            bbox = item.get("dt_polys", None) or item.get("boxes", None)
            if text and bbox is not None:
                if hasattr(bbox, "tolist"):
                    bbox = bbox.tolist()
                entries.append((bbox, text, conf))
        return entries

    # Legacy / v3 list-of-pages format
    for page in result:
        if page is None:
            continue
        if isinstance(page, dict):
            text = page.get("rec_text", "") or page.get("text", "")
            conf = float(page.get("rec_score", 0) or page.get("score", 0))
            bbox = page.get("dt_polys", None) or page.get("boxes", None)
            if text and bbox is not None:
                if hasattr(bbox, "tolist"):
                    bbox = bbox.tolist()
                entries.append((bbox, text, conf))
            continue
        for item in page:
            if isinstance(item, dict):
                text = item.get("rec_text", "") or item.get("text", "")
                conf = float(item.get("rec_score", 0) or item.get("score", 0))
                bbox = item.get("dt_polys", None) or item.get("boxes", None)
                if text and bbox is not None:
                    if hasattr(bbox, "tolist"):
                        bbox = bbox.tolist()
                    entries.append((bbox, text, conf))
            else:
                bbox = item[0]
                text = item[1][0]
                conf = float(item[1][1])
                entries.append((bbox, text, conf))
    return entries


class OCRReader:
    """Wraps PaddleOCR and provides brand-label matching."""

    def __init__(
        self,
        target_labels: List[str],
        languages: Optional[List[str]] = None,
        gpu: bool = True,
        match_threshold: float = 0.5,
        confidence_boost: float = 0.15,
    ):
        self.target_labels = target_labels
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.match_threshold = match_threshold
        self.confidence_boost = confidence_boost
        paddle_lang = self.languages[0] if self.languages else "en"
        self._reader = _get_reader(paddle_lang, self.gpu)

    # -------------------------------------------------------------- helpers

    def read_text(self, image: np.ndarray) -> str:
        """Run OCR on a BGR image and return the concatenated text."""
        if image.size == 0:
            return ""
        result = self._reader.ocr(image)
        entries = _parse_paddle_result(result)
        texts = [text for _, text, conf in entries if conf > 0.2]
        return " ".join(texts)

    def match_label(self, ocr_text: str) -> Tuple[Optional[str], float]:
        """Fuzzy-match OCR text against target labels."""
        return _best_match(ocr_text, self.target_labels, self.match_threshold)

    # ----------------------------------------- mode 1: augment YOLO crops

    def process_crop(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> Dict:
        """
        Crop, OCR, match — used when YOLO already provides bounding boxes.
        """
        crop = frame[max(0, y1):y2, max(0, x1):x2]
        ocr_text = self.read_text(crop)
        matched_label, ratio = self.match_label(ocr_text)

        boost = self.confidence_boost if matched_label else 0.0

        return {
            "ocr_text": ocr_text,
            "ocr_matched_label": matched_label,
            "ocr_match_ratio": round(ratio, 4),
            "ocr_confidence_boost": boost,
        }

    # ----------------------------------------- mode 2: full-frame OCR scan

    def scan_frame(self, frame: np.ndarray) -> list:
        """
        Run OCR on the entire frame.  For every text region whose content
        fuzzy-matches a target label, return a Detection object.
        """
        from src.logo_detector import Detection

        h, w = frame.shape[:2]
        frame_area = h * w
        result = self._reader.ocr(frame)
        raw_results = _parse_paddle_result(result)

        detections: list = []

        for bbox, text, ocr_conf in raw_results:
            if ocr_conf < 0.2:
                continue

            matched_label, ratio = self.match_label(text)
            if matched_label is None:
                continue

            xs = [int(pt[0]) for pt in bbox]
            ys = [int(pt[1]) for pt in bbox]
            x1, x2 = max(0, min(xs)), min(w, max(xs))
            y1, y2 = max(0, min(ys)), min(h, max(ys))
            bw = x2 - x1
            bh = y2 - y1
            if bw <= 0 or bh <= 0:
                continue

            confidence = round(ocr_conf * ratio, 4)

            detections.append(Detection(
                class_id=-1,
                label=matched_label,
                confidence=confidence,
                x1=x1, y1=y1, x2=x2, y2=y2,
                nx1=round(x1 / w, 4),
                ny1=round(y1 / h, 4),
                nx2=round(x2 / w, 4),
                ny2=round(y2 / h, 4),
                width_px=bw,
                height_px=bh,
                area_px=bw * bh,
                area_pct=round((bw * bh) / frame_area * 100, 2),
                ocr_text=text,
                ocr_matched_label=matched_label,
                ocr_match_ratio=ratio,
                original_confidence=0.0,
            ))

        return detections
