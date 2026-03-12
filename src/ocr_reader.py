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

import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

_reader = None  # lazy singleton


def _get_reader(lang: str, gpu: bool):
    """Lazy-init PaddleOCR (heavy to create, so we cache it)."""
    global _reader
    if _reader is None:
        from paddleocr import PaddleOCR
        _reader = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=gpu, show_log=False)
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
    PaddleOCR returns [page_results] where each page is a list of
    (bbox, (text, conf)) tuples — or None if nothing was detected.
    """
    entries: List[Tuple[list, str, float]] = []
    if result is None:
        return entries
    for page in result:
        if page is None:
            continue
        for item in page:
            bbox = item[0]           # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text = item[1][0]        # str
            conf = float(item[1][1]) # float
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
        result = self._reader.ocr(image, cls=True)
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
        result = self._reader.ocr(frame, cls=True)
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
