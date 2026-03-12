"""
Amazon Rekognition backend for brand/logo detection.

Supports two Rekognition APIs:
  - detect_labels   : General object/scene detection (built-in, no training needed).
  - custom_labels   : Custom Labels project inference (trained logo model in Rekognition).

The output is the same Detection dataclass used by the YOLO backend, so the
rest of the pipeline (tracker, visualiser, reports) is unchanged.

Setup:
  1. Configure AWS credentials:
       export AWS_ACCESS_KEY_ID=...
       export AWS_SECRET_ACCESS_KEY=...
       export AWS_DEFAULT_REGION=us-east-1
     Or use an IAM role / AWS profile.

  2. For custom_labels mode, start your Custom Labels model project first:
       aws rekognition start-project-version \
           --project-version-arn <arn> \
           --min-inference-units 1

  3. Run:
       python pipeline.py --input video.mp4 --mode detect \
           --detector rekognition-labels
       python pipeline.py --input video.mp4 --mode detect \
           --detector rekognition-custom \
           --rekognition-arn <project-version-arn>
"""

from __future__ import annotations

import io
import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from src.logo_detector import Detection


def _frame_to_bytes(frame: np.ndarray) -> bytes:
    """Encode a BGR numpy frame to JPEG bytes for Rekognition."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return buf.tobytes()


class RekognitionDetector:
    """
    Rekognition-based logo/label detector.

    mode = "labels"   → DetectLabels API (general purpose, no training)
    mode = "custom"   → DetectCustomLabels API (requires a trained Custom Labels project)
    """

    def __init__(
        self,
        mode: str = "labels",
        project_version_arn: Optional[str] = None,
        target_labels: Optional[List[str]] = None,
        confidence_threshold: float = 0.5,
        region: str = "us-east-1",
        aws_profile: Optional[str] = None,
    ):
        import boto3

        self.mode = mode
        self.project_version_arn = project_version_arn
        self.target_labels = [t.lower() for t in (target_labels or [])]
        self.confidence_threshold = confidence_threshold

        session = boto3.Session(profile_name=aws_profile) if aws_profile else boto3.Session()
        self.client = session.client("rekognition", region_name=region)

        if mode == "custom" and not project_version_arn:
            raise ValueError(
                "project_version_arn is required for rekognition-custom mode.\n"
                "Start your Custom Labels project version first and pass --rekognition-arn."
            )

    def _is_target(self, label: str) -> bool:
        if not self.target_labels:
            return True
        return label.lower() in self.target_labels

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Run Rekognition on a frame and return Detection objects."""
        h, w = frame.shape[:2]
        frame_area = h * w
        image_bytes = _frame_to_bytes(frame)

        if self.mode == "custom":
            return self._detect_custom(image_bytes, h, w, frame_area)
        return self._detect_labels(image_bytes, h, w, frame_area)

    def _detect_labels(
        self, image_bytes: bytes, h: int, w: int, frame_area: int
    ) -> List[Detection]:
        response = self.client.detect_labels(
            Image={"Bytes": image_bytes},
            MinConfidence=self.confidence_threshold * 100,
        )
        detections: List[Detection] = []

        for label in response.get("Labels", []):
            name = label["Name"]
            conf = label["Confidence"] / 100.0

            if not self._is_target(name):
                continue

            instances = label.get("Instances", [])
            if not instances:
                # Label without bounding box — synthesize full-frame bbox
                detections.append(self._make_detection(
                    name, conf, 0, 0, w, h, h, w, frame_area
                ))
                continue

            for inst in instances:
                bb = inst.get("BoundingBox", {})
                inst_conf = inst.get("Confidence", label["Confidence"]) / 100.0
                x1 = int(bb["Left"] * w)
                y1 = int(bb["Top"] * h)
                x2 = int((bb["Left"] + bb["Width"]) * w)
                y2 = int((bb["Top"] + bb["Height"]) * h)
                detections.append(self._make_detection(
                    name, inst_conf, x1, y1, x2, y2, h, w, frame_area
                ))

        return detections

    def _detect_custom(
        self, image_bytes: bytes, h: int, w: int, frame_area: int
    ) -> List[Detection]:
        response = self.client.detect_custom_labels(
            ProjectVersionArn=self.project_version_arn,
            Image={"Bytes": image_bytes},
            MinConfidence=self.confidence_threshold * 100,
        )
        detections: List[Detection] = []

        for label in response.get("CustomLabels", []):
            name = label["Name"]
            conf = label["Confidence"] / 100.0

            if not self._is_target(name):
                continue

            geo = label.get("Geometry", {})
            bb = geo.get("BoundingBox", None)
            if bb:
                x1 = int(bb["Left"] * w)
                y1 = int(bb["Top"] * h)
                x2 = int((bb["Left"] + bb["Width"]) * w)
                y2 = int((bb["Top"] + bb["Height"]) * h)
            else:
                x1, y1, x2, y2 = 0, 0, w, h

            detections.append(self._make_detection(
                name, conf, x1, y1, x2, y2, h, w, frame_area
            ))

        return detections

    @staticmethod
    def _make_detection(
        label: str, conf: float,
        x1: int, y1: int, x2: int, y2: int,
        h: int, w: int, frame_area: int,
    ) -> Detection:
        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        return Detection(
            class_id=-1,
            label=label,
            confidence=round(conf, 4),
            x1=x1, y1=y1, x2=x2, y2=y2,
            nx1=round(x1 / w, 4), ny1=round(y1 / h, 4),
            nx2=round(x2 / w, 4), ny2=round(y2 / h, 4),
            width_px=bw, height_px=bh,
            area_px=bw * bh,
            area_pct=round((bw * bh) / frame_area * 100, 2),
        )
