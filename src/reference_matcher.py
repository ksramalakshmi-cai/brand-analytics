"""
CLIP-based reference logo matcher.

Given a directory of reference logo images (one subfolder per brand),
this module finds visually similar regions in video frames without any
training.  Folder names map directly to brand labels used throughout
the pipeline.

Directory layout expected:
    logos/
      coca_cola/
        logo1.png
        logo2.jpg
      emirates/
        crest.png

Usage through the pipeline:
    python pipeline.py --input video.mp4 --mode detect \
        --detector reference --logos-dir logos/
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.logo_detector import Detection

_CUDA_AVAILABLE = torch.cuda.is_available()
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

Box = Tuple[int, int, int, int]


class ReferenceMatcher:
    """
    Detects logos by comparing multi-scale frame patches against
    pre-computed CLIP embeddings of reference images.

    Supports *exclusion zones*: when OCR has already found brands in
    certain frame regions, those areas can be skipped to avoid
    duplicate work and give OCR priority.
    """

    def __init__(
        self,
        logos_dir: str,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        similarity_threshold: float = 0.75,
        device: str = "",
        img_size: int = 640,
        patch_scales: Optional[List[int]] = None,
        stride_ratio: float = 0.5,
        batch_size: int = 64,
        nms_iou: float = 0.45,
        exclusion_coverage: float = 0.3,
        refine: bool = True,
        brand_filter: Optional[List[str]] = None,
    ):
        import open_clip

        self.similarity_threshold = similarity_threshold
        self.img_size = img_size
        self.patch_scales = patch_scales or [48, 96, 160, 256]
        self.stride_ratio = stride_ratio
        self.batch_size = batch_size
        self.nms_iou = nms_iou
        self.exclusion_coverage = exclusion_coverage
        self.refine = refine

        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda:0" if _CUDA_AVAILABLE else "cpu")

        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained, device=self.device,
        )
        self.model.eval()

        self.brand_names: List[str] = []
        self.brand_embeddings: torch.Tensor = torch.empty(0)
        self._load_references(logos_dir, brand_filter)

    # --------------------------------------------------------------------- #
    # Reference loading                                                      #
    # --------------------------------------------------------------------- #

    def _load_references(
        self, logos_dir: str, brand_filter: Optional[List[str]] = None,
    ) -> None:
        root = Path(logos_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Logos directory not found: {root}")

        all_dirs = sorted(
            d for d in root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        if not all_dirs:
            raise ValueError(f"No brand sub-directories in {root}")

        if brand_filter:
            allowed = {n.lower() for n in brand_filter}
            brand_dirs = [d for d in all_dirs if d.name.lower() in allowed]
            skipped = [d.name for d in all_dirs if d.name.lower() not in allowed]
            if skipped:
                print(f"  [ReferenceMatcher] Skipping dirs not in labels: {skipped}")
        else:
            brand_dirs = all_dirs

        if not brand_dirs:
            raise ValueError(
                f"No matching brand directories in {root} for labels: {brand_filter}"
            )

        names: List[str] = []
        embeddings: List[torch.Tensor] = []

        for brand_dir in brand_dirs:
            imgs = [
                p for p in brand_dir.iterdir()
                if p.suffix.lower() in _IMAGE_EXTS
            ]
            if not imgs:
                continue

            brand_embs = []
            for img_path in imgs:
                emb = self._embed_pil(img_path)
                if emb is not None:
                    brand_embs.append(emb)

            if not brand_embs:
                continue

            avg_emb = torch.stack(brand_embs).mean(dim=0)
            avg_emb = F.normalize(avg_emb, dim=-1)

            names.append(brand_dir.name)
            embeddings.append(avg_emb)

        if not names:
            raise ValueError(f"No valid reference images found under {root}")

        self.brand_names = names
        self.brand_embeddings = torch.stack(embeddings)  # (N_brands, D)
        print(f"  [ReferenceMatcher] Loaded {len(names)} brands: {names}")

    def _embed_pil(self, path: Path) -> Optional[torch.Tensor]:
        from PIL import Image
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            return None
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
        return F.normalize(emb.squeeze(0), dim=-1)

    # --------------------------------------------------------------------- #
    # Geometry helpers                                                        #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _intersection_area(a: Box, b: Box) -> int:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        return max(0, ix2 - ix1) * max(0, iy2 - iy1)

    @staticmethod
    def _box_area(b: Box) -> int:
        return max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    @classmethod
    def _is_covered(
        cls,
        patch_box: Box,
        exclusion_zones: List[Box],
        coverage_threshold: float,
    ) -> bool:
        """True if the patch is sufficiently covered by any exclusion zone."""
        patch_area = cls._box_area(patch_box)
        if patch_area <= 0:
            return True
        for zone in exclusion_zones:
            inter = cls._intersection_area(patch_box, zone)
            if inter / patch_area >= coverage_threshold:
                return True
        return False

    @staticmethod
    def _iou(a: Box, b: Box) -> float:
        ix1 = max(a[0], b[0])
        iy1 = max(a[1], b[1])
        ix2 = min(a[2], b[2])
        iy2 = min(a[3], b[3])
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    # --------------------------------------------------------------------- #
    # Multi-scale patch extraction                                            #
    # --------------------------------------------------------------------- #

    @classmethod
    def _extract_patches(
        cls,
        frame: np.ndarray,
        scales: List[int],
        stride_ratio: float,
        exclusion_zones: Optional[List[Box]] = None,
        coverage_threshold: float = 0.3,
    ) -> Tuple[List[np.ndarray], List[Box]]:
        """Return (crops, boxes) where boxes are in frame coords.
        Patches overlapping exclusion zones are skipped."""
        h, w = frame.shape[:2]
        crops: List[np.ndarray] = []
        boxes: List[Box] = []

        for scale in scales:
            if scale > min(h, w):
                continue
            stride = max(1, int(scale * stride_ratio))
            for y in range(0, h - scale + 1, stride):
                for x in range(0, w - scale + 1, stride):
                    box: Box = (x, y, x + scale, y + scale)
                    if exclusion_zones and cls._is_covered(
                        box, exclusion_zones, coverage_threshold
                    ):
                        continue
                    crops.append(frame[y : y + scale, x : x + scale])
                    boxes.append(box)

        return crops, boxes

    # --------------------------------------------------------------------- #
    # Batch CLIP embedding                                                    #
    # --------------------------------------------------------------------- #

    def _embed_crops(self, crops: List[np.ndarray]) -> torch.Tensor:
        """Encode a list of BGR numpy crops through CLIP."""
        from PIL import Image

        all_embs: List[torch.Tensor] = []

        for i in range(0, len(crops), self.batch_size):
            batch_crops = crops[i : i + self.batch_size]
            tensors = []
            for crop in batch_crops:
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                tensors.append(self.preprocess(pil_img))
            batch = torch.stack(tensors).to(self.device)

            with torch.no_grad():
                embs = self.model.encode_image(batch)
            embs = F.normalize(embs, dim=-1)
            all_embs.append(embs)

        return torch.cat(all_embs, dim=0)  # (N_patches, D)

    # --------------------------------------------------------------------- #
    # NMS                                                                     #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _nms(
        boxes: List[Box],
        scores: List[float],
        labels: List[int],
        iou_threshold: float,
    ) -> List[int]:
        """Per-class greedy NMS. Returns indices to keep."""
        if not boxes:
            return []

        from collections import defaultdict
        by_class: Dict[int, List[int]] = defaultdict(list)
        for idx, lbl in enumerate(labels):
            by_class[lbl].append(idx)

        keep: List[int] = []
        for cls_indices in by_class.values():
            cls_indices.sort(key=lambda i: -scores[i])
            alive = list(cls_indices)
            while alive:
                best = alive.pop(0)
                keep.append(best)
                bx1, by1, bx2, by2 = boxes[best]
                remaining = []
                for other in alive:
                    ox1, oy1, ox2, oy2 = boxes[other]
                    ix1 = max(bx1, ox1)
                    iy1 = max(by1, oy1)
                    ix2 = min(bx2, ox2)
                    iy2 = min(by2, oy2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    area_b = (bx2 - bx1) * (by2 - by1)
                    area_o = (ox2 - ox1) * (oy2 - oy1)
                    union = area_b + area_o - inter
                    iou = inter / union if union > 0 else 0.0
                    if iou < iou_threshold:
                        remaining.append(other)
                alive = remaining
        return keep

    # --------------------------------------------------------------------- #
    # Bbox refinement — zoom in on coarse matches for tighter boxes          #
    # --------------------------------------------------------------------- #

    def _refine_box(
        self,
        proc_frame: np.ndarray,
        coarse_box: Box,
        brand_idx: int,
        coarse_score: float,
    ) -> Tuple[Box, float]:
        """
        Given a coarse patch match, search within a local window using
        smaller patches to tighten the bounding box.
        """
        h, w = proc_frame.shape[:2]
        cx1, cy1, cx2, cy2 = coarse_box
        coarse_size = max(cx2 - cx1, cy2 - cy1)

        if coarse_size <= 48:
            return coarse_box, coarse_score

        pad = coarse_size // 4
        rx1 = max(0, cx1 - pad)
        ry1 = max(0, cy1 - pad)
        rx2 = min(w, cx2 + pad)
        ry2 = min(h, cy2 + pad)
        region = proc_frame[ry1:ry2, rx1:rx2]

        fine_scales = [s for s in [coarse_size // 3, coarse_size // 2] if s >= 24]
        if not fine_scales:
            return coarse_box, coarse_score

        fine_stride = max(1, fine_scales[0] // 3)
        rh, rw = region.shape[:2]
        fine_crops: List[np.ndarray] = []
        fine_boxes: List[Box] = []

        for scale in fine_scales:
            if scale > min(rh, rw):
                continue
            for y in range(0, rh - scale + 1, fine_stride):
                for x in range(0, rw - scale + 1, fine_stride):
                    fine_crops.append(region[y : y + scale, x : x + scale])
                    fine_boxes.append((rx1 + x, ry1 + y, rx1 + x + scale, ry1 + y + scale))

        if not fine_crops:
            return coarse_box, coarse_score

        fine_embs = self._embed_crops(fine_crops)
        brand_emb = self.brand_embeddings[brand_idx].unsqueeze(0)  # (1, D)
        sims = (fine_embs @ brand_emb.T).squeeze(-1)  # (N,)

        best_idx = int(sims.argmax())
        best_score = float(sims[best_idx])

        if best_score >= self.similarity_threshold:
            return fine_boxes[best_idx], best_score
        return coarse_box, coarse_score

    # --------------------------------------------------------------------- #
    # Main detect method                                                      #
    # --------------------------------------------------------------------- #

    def detect(
        self,
        frame: np.ndarray,
        exclusion_zones: Optional[List[Box]] = None,
    ) -> List[Detection]:
        """
        Find reference-matched logos in a single BGR frame.

        Args:
            frame: BGR numpy array.
            exclusion_zones: list of (x1, y1, x2, y2) boxes in original
                frame coordinates to skip (e.g. regions already found by OCR).
        """
        orig_h, orig_w = frame.shape[:2]
        frame_area = orig_h * orig_w

        scale_factor = self.img_size / max(orig_h, orig_w)
        if scale_factor < 1.0:
            proc_w = int(orig_w * scale_factor)
            proc_h = int(orig_h * scale_factor)
            proc_frame = cv2.resize(frame, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            proc_frame = frame
            proc_w, proc_h = orig_w, orig_h
            scale_factor = 1.0

        # Scale exclusion zones to processed coordinates
        proc_exclusions: Optional[List[Box]] = None
        if exclusion_zones:
            proc_exclusions = [
                (
                    int(z[0] * scale_factor),
                    int(z[1] * scale_factor),
                    int(z[2] * scale_factor),
                    int(z[3] * scale_factor),
                )
                for z in exclusion_zones
            ]

        crops, boxes = self._extract_patches(
            proc_frame, self.patch_scales, self.stride_ratio,
            exclusion_zones=proc_exclusions,
            coverage_threshold=self.exclusion_coverage,
        )
        if not crops:
            return []

        patch_embs = self._embed_crops(crops)  # (N_patches, D)

        # Cosine similarity: (N_patches, N_brands)
        sims = patch_embs @ self.brand_embeddings.T

        # Gather matches above threshold
        match_boxes: List[Box] = []
        match_scores: List[float] = []
        match_labels: List[int] = []

        for patch_idx in range(sims.shape[0]):
            for brand_idx in range(sims.shape[1]):
                score = float(sims[patch_idx, brand_idx])
                if score >= self.similarity_threshold:
                    match_boxes.append(boxes[patch_idx])
                    match_scores.append(score)
                    match_labels.append(brand_idx)

        keep = self._nms(match_boxes, match_scores, match_labels, self.nms_iou)

        # Refine kept detections for tighter bounding boxes
        refined_boxes: List[Box] = []
        refined_scores: List[float] = []
        refined_labels: List[int] = []
        for idx in keep:
            box = match_boxes[idx]
            score = match_scores[idx]
            brand_idx = match_labels[idx]

            if self.refine:
                box, score = self._refine_box(proc_frame, box, brand_idx, score)

            refined_boxes.append(box)
            refined_scores.append(score)
            refined_labels.append(brand_idx)

        # Second NMS pass after refinement (boxes may have shifted)
        if self.refine and refined_boxes:
            keep2 = self._nms(refined_boxes, refined_scores, refined_labels, self.nms_iou)
        else:
            keep2 = list(range(len(refined_boxes)))

        detections: List[Detection] = []
        for idx in keep2:
            bx1, by1, bx2, by2 = refined_boxes[idx]
            # Scale back to original frame coordinates
            x1 = max(0, min(int(bx1 / scale_factor), orig_w))
            y1 = max(0, min(int(by1 / scale_factor), orig_h))
            x2 = max(0, min(int(bx2 / scale_factor), orig_w))
            y2 = max(0, min(int(by2 / scale_factor), orig_h))
            brand_name = self.brand_names[refined_labels[idx]]
            conf = refined_scores[idx]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            detections.append(Detection(
                class_id=refined_labels[idx],
                label=brand_name,
                confidence=round(conf, 4),
                x1=x1, y1=y1, x2=x2, y2=y2,
                nx1=round(x1 / orig_w, 4),
                ny1=round(y1 / orig_h, 4),
                nx2=round(x2 / orig_w, 4),
                ny2=round(y2 / orig_h, 4),
                width_px=bw,
                height_px=bh,
                area_px=bw * bh,
                area_pct=round((bw * bh) / frame_area * 100, 2),
            ))

        return detections
