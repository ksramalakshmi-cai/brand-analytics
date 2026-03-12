"""
OCR-based text extraction for logo/brand detection.

Supports three backends (set via config or --ocr-backend):
  - easyocr   Default; avoids Paddle segfaults on CPU-only machines.
  - paddle    PaddleOCR (faster on GPU; can segfault on some CPU environments).
  - deepseek  DeepSeek vision-language model via OpenAI-compatible API.
              Best at reading distorted/artistic text; requires DEEPSEEK_API_KEY.

Two operating modes:
  1. Crop mode (used with YOLO) — read text inside a bounding box, fuzzy-match to labels.
  2. Full-frame mode (OCR-only pipeline) — scan entire frame, return Detection objects for matches.
"""

from __future__ import annotations

import base64
import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Only set for Paddle backend; must be before paddle is ever imported.
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")


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
    """Return (matched_label, similarity_ratio) if best fuzzy match >= threshold else (None, 0.0)."""
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
    """Normalise PaddleOCR output to list of (bbox, text, confidence)."""
    entries: List[Tuple[list, str, float]] = []
    if result is None:
        return entries

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


def _image_to_data_url(image: np.ndarray, max_side: int = 1024) -> str:
    """Encode a BGR numpy image as a base64 data-URL (JPEG)."""
    h, w = image.shape[:2]
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    _, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


_DEEPSEEK_SCAN_PROMPT = """\
You are an OCR engine. Analyse this image and return ALL visible text.
For each distinct text block return a JSON object with:
  "text": the exact text string,
  "confidence": your confidence 0.0-1.0,
  "bbox": [left, top, right, bottom] as fractions of image width/height (0.0-1.0).
Return ONLY a JSON array, no markdown fences, no commentary."""

_DEEPSEEK_READ_PROMPT = "Extract ALL visible text from this image. Return only the raw text, nothing else."


class _DeepSeekVLM:
    """Thin wrapper around an OpenAI-compatible vision endpoint (DeepSeek default)."""

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
    ):
        from openai import OpenAI
        key = api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        if not key:
            raise ValueError(
                "DeepSeek API key required. Set DEEPSEEK_API_KEY env var "
                "or pass --deepseek-api-key."
            )
        self._client = OpenAI(api_key=key, base_url=base_url)
        self._model = model

    def _call(self, prompt: str, image: np.ndarray) -> str:
        data_url = _image_to_data_url(image)
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=2048,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    def read_text(self, image: np.ndarray) -> str:
        return self._call(_DEEPSEEK_READ_PROMPT, image)

    def scan_frame_raw(self, image: np.ndarray) -> List[Dict]:
        """Return list of {"text", "confidence", "bbox"} dicts."""
        raw = self._call(_DEEPSEEK_SCAN_PROMPT, image)
        # Strip markdown fences if the model wraps the JSON anyway
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        try:
            items = json.loads(raw)
            if isinstance(items, list):
                return items
        except (json.JSONDecodeError, TypeError):
            pass
        return []


class OCRReader:
    """Single interface for OCR + brand matching; supports easyocr, paddle, and deepseek backends."""

    def __init__(
        self,
        target_labels: List[str],
        languages: Optional[List[str]] = None,
        gpu: bool = True,
        match_threshold: float = 0.5,
        confidence_boost: float = 0.15,
        backend: str = "easyocr",
        deepseek_api_key: str = "",
        deepseek_base_url: str = "https://api.deepseek.com",
        deepseek_model: str = "deepseek-chat",
    ):
        self.target_labels = target_labels
        self.languages = languages or ["en"]
        self.gpu = gpu
        self.match_threshold = match_threshold
        self.confidence_boost = confidence_boost
        self._backend = backend.lower()

        if self._backend == "easyocr":
            import easyocr
            self._reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        elif self._backend == "paddle":
            from paddleocr import PaddleOCR
            lang = self.languages[0] if self.languages else "en"
            try:
                self._reader = PaddleOCR(use_angle_cls=True, lang=lang, use_gpu=self.gpu)
            except (TypeError, ValueError):
                self._reader = PaddleOCR(use_angle_cls=True, lang=lang)
        elif self._backend == "deepseek":
            self._vlm = _DeepSeekVLM(
                api_key=deepseek_api_key,
                base_url=deepseek_base_url,
                model=deepseek_model,
            )
        else:
            raise ValueError(f"Unknown OCR backend: {self._backend!r}. Use easyocr, paddle, or deepseek.")

    def match_label(self, ocr_text: str) -> Tuple[Optional[str], float]:
        """Fuzzy-match OCR text against target labels."""
        return _best_match(ocr_text, self.target_labels, self.match_threshold)

    def read_text(self, image: np.ndarray) -> str:
        """Run OCR on a BGR image and return concatenated text."""
        if image.size == 0:
            return ""

        if self._backend == "easyocr":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            results = self._reader.readtext(gray, detail=1, paragraph=False)
            texts = [r[1] for r in results if r[2] > 0.2]
            return " ".join(texts)

        if self._backend == "deepseek":
            return self._vlm.read_text(image)

        result = self._reader.ocr(image)
        entries = _parse_paddle_result(result)
        texts = [text for _, text, conf in entries if conf > 0.2]
        return " ".join(texts)

    def process_crop(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> Dict:
        """Crop, OCR, match — used when YOLO provides bounding boxes."""
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

    def scan_frame(self, frame: np.ndarray) -> list:
        """Run OCR on full frame; return list of Detection for text that matches target labels."""
        from src.logo_detector import Detection

        h, w = frame.shape[:2]
        frame_area = h * w

        if self._backend == "deepseek":
            return self._scan_frame_deepseek(frame, h, w, frame_area)

        raw_results: List[Tuple[list, str, float]] = []

        if self._backend == "easyocr":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            results = self._reader.readtext(gray, detail=1, paragraph=False)
            for r in results:
                bbox, text, conf = r[0], r[1], r[2]
                raw_results.append((bbox, text, float(conf)))
        else:
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

    def _scan_frame_deepseek(
        self, frame: np.ndarray, h: int, w: int, frame_area: int,
    ) -> list:
        """DeepSeek-specific scan: VLM returns text + normalised bboxes."""
        from src.logo_detector import Detection

        items = self._vlm.scan_frame_raw(frame)
        detections: list = []

        for item in items:
            text = item.get("text", "")
            ocr_conf = float(item.get("confidence", 0.5))
            bbox_norm = item.get("bbox", [0.0, 0.0, 1.0, 1.0])

            if ocr_conf < 0.2 or not text:
                continue

            matched_label, ratio = self.match_label(text)
            if matched_label is None:
                continue

            if len(bbox_norm) == 4:
                nl, nt, nr, nb = [float(v) for v in bbox_norm]
            else:
                nl, nt, nr, nb = 0.0, 0.0, 1.0, 1.0

            x1 = max(0, min(int(nl * w), w))
            y1 = max(0, min(int(nt * h), h))
            x2 = max(0, min(int(nr * w), w))
            y2 = max(0, min(int(nb * h), h))
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
                nx1=round(nl, 4),
                ny1=round(nt, 4),
                nx2=round(nr, 4),
                ny2=round(nb, 4),
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
