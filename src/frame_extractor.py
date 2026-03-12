"""
Extract frames from video at a configurable FPS, or load a single image.
Handles arbitrary resolutions and durations gracefully.
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Generator


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v"}


@dataclass
class FrameMeta:
    """Metadata attached to every extracted frame."""
    index: int           # sequential counter (0-based)
    timestamp_sec: float # seconds into the video (0 for images)
    source_path: str
    width: int
    height: int
    frame: np.ndarray    # BGR numpy array


@dataclass
class MediaInfo:
    """Summary of the source media."""
    path: str
    is_video: bool
    width: int
    height: int
    total_frames: int
    native_fps: float
    duration_sec: float


def _detect_media_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in VIDEO_EXTENSIONS:
        return "video"
    raise ValueError(f"Unsupported file type: {ext}")


def get_media_info(path: str | Path) -> MediaInfo:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input not found: {path}")

    media_type = _detect_media_type(path)

    if media_type == "image":
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"Failed to read image: {path}")
        h, w = img.shape[:2]
        return MediaInfo(
            path=str(path), is_video=False,
            width=w, height=h,
            total_frames=1, native_fps=0, duration_sec=0,
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {path}")

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    duration = total / native_fps if native_fps > 0 else 0
    cap.release()

    return MediaInfo(
        path=str(path), is_video=True,
        width=w, height=h,
        total_frames=total, native_fps=native_fps,
        duration_sec=duration,
    )


def extract_frames(
    path: str | Path,
    sample_fps: float = 1.0,
) -> Generator[FrameMeta, None, None]:
    """
    Yield FrameMeta objects.

    For images: yields a single frame.
    For videos: yields one frame every 1/sample_fps seconds.
    """
    path = Path(path)
    media_type = _detect_media_type(path)

    if media_type == "image":
        img = cv2.imread(str(path))
        if img is None:
            raise IOError(f"Failed to read image: {path}")
        h, w = img.shape[:2]
        yield FrameMeta(
            index=0, timestamp_sec=0.0,
            source_path=str(path), width=w, height=h, frame=img,
        )
        return

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"Failed to open video: {path}")

    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_interval = max(1, round(native_fps / sample_fps))

    frame_number = 0
    emitted = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_number % frame_interval == 0:
            h, w = frame.shape[:2]
            timestamp = frame_number / native_fps
            yield FrameMeta(
                index=emitted,
                timestamp_sec=round(timestamp, 3),
                source_path=str(path),
                width=w, height=h,
                frame=frame,
            )
            emitted += 1

        frame_number += 1

    cap.release()
