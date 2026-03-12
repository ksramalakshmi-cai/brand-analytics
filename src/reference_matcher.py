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


class ReferenceMatcher:
    """
    Detects logos by comparing multi-scale frame patches against
    pre-computed CLIP embeddings of reference images.
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
    ):
        import open_clip

        self.similarity_threshold = similarity_threshold
        self.img_size = img_size
        self.patch_scales = patch_scales or [64, 128, 256, 384]
        self.stride_ratio = stride_ratio
        self.batch_size = batch_size
        self.nms_iou = nms_iou

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
        self._load_references(logos_dir)

    # --------------------------------------------------------------------- #
    # Reference loading                                                      #
    # --------------------------------------------------------------------- #

    def _load_references(self, logos_dir: str) -> None:
        root = Path(logos_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"Logos directory not found: {root}")

        brand_dirs = sorted(
            d for d in root.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )
        if not brand_dirs:
            raise ValueError(f"No brand sub-directories in {root}")

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
    # Multi-scale patch extraction                                            #
    # --------------------------------------------------------------------- #

    @staticmethod
    def _extract_patches(
        frame: np.ndarray,
        scales: List[int],
        stride_ratio: float,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
        """Return (crops, boxes) where boxes are in original frame coords."""
        h, w = frame.shape[:2]
        crops: List[np.ndarray] = []
        boxes: List[Tuple[int, int, int, int]] = []

        for scale in scales:
            if scale > min(h, w):
                continue
            stride = max(1, int(scale * stride_ratio))
            for y in range(0, h - scale + 1, stride):
                for x in range(0, w - scale + 1, stride):
                    crops.append(frame[y : y + scale, x : x + scale])
                    boxes.append((x, y, x + scale, y + scale))

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
        boxes: List[Tuple[int, int, int, int]],
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
    # Main detect method                                                      #
    # --------------------------------------------------------------------- #

    def detect(self, frame: np.ndarray) -> List[Detection]:
        """
        Find reference-matched logos in a single BGR frame.
        Returns Detection objects compatible with the rest of the pipeline.
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

        crops, boxes = self._extract_patches(proc_frame, self.patch_scales, self.stride_ratio)
        if not crops:
            return []

        patch_embs = self._embed_crops(crops)  # (N_patches, D)

        # Cosine similarity: (N_patches, N_brands)
        sims = patch_embs @ self.brand_embeddings.T

        # Gather matches above threshold
        match_boxes: List[Tuple[int, int, int, int]] = []
        match_scores: List[float] = []
        match_labels: List[int] = []

        for patch_idx in range(sims.shape[0]):
            for brand_idx in range(sims.shape[1]):
                score = float(sims[patch_idx, brand_idx])
                if score >= self.similarity_threshold:
                    bx1, by1, bx2, by2 = boxes[patch_idx]
                    # Scale back to original frame coordinates
                    ox1 = int(bx1 / scale_factor)
                    oy1 = int(by1 / scale_factor)
                    ox2 = int(bx2 / scale_factor)
                    oy2 = int(by2 / scale_factor)
                    ox1 = max(0, min(ox1, orig_w))
                    oy1 = max(0, min(oy1, orig_h))
                    ox2 = max(0, min(ox2, orig_w))
                    oy2 = max(0, min(oy2, orig_h))
                    match_boxes.append((ox1, oy1, ox2, oy2))
                    match_scores.append(score)
                    match_labels.append(brand_idx)

        keep = self._nms(match_boxes, match_scores, match_labels, self.nms_iou)

        detections: List[Detection] = []
        for idx in keep:
            x1, y1, x2, y2 = match_boxes[idx]
            brand_name = self.brand_names[match_labels[idx]]
            conf = match_scores[idx]
            bw = max(1, x2 - x1)
            bh = max(1, y2 - y1)

            detections.append(Detection(
                class_id=match_labels[idx],
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
