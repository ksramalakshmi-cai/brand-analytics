from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class LabelConfig:
    """Per-brand method routing parsed from labels.yaml.

    Each brand is assigned a method:
      ocr      — OCR is the primary detector for this brand (text-heavy logos)
      detector — CLIP/YOLO is the primary detector (visual logos)
      both     — use both methods
    """
    ocr: List[str] = field(default_factory=list)
    detector: List[str] = field(default_factory=list)
    both: List[str] = field(default_factory=list)

    @property
    def all_labels(self) -> List[str]:
        """Every brand name, deduplicated, preserving order."""
        seen: Set[str] = set()
        out: List[str] = []
        for name in self.ocr + self.detector + self.both:
            key = name.lower()
            if key not in seen:
                seen.add(key)
                out.append(name)
        return out

    @property
    def ocr_eligible(self) -> List[str]:
        """Brands that OCR should try to find (ocr + both)."""
        return self._dedup(self.ocr + self.both)

    @property
    def detector_eligible(self) -> List[str]:
        """Brands that the detector should try to find (detector + both)."""
        return self._dedup(self.detector + self.both)

    @property
    def detector_eligible_set(self) -> Set[str]:
        return {n.lower() for n in self.detector_eligible}

    @property
    def ocr_eligible_set(self) -> Set[str]:
        return {n.lower() for n in self.ocr_eligible}

    @staticmethod
    def _dedup(names: List[str]) -> List[str]:
        seen: Set[str] = set()
        out: List[str] = []
        for n in names:
            key = n.lower()
            if key not in seen:
                seen.add(key)
                out.append(n)
        return out

    def method_for(self, label: str) -> str:
        """Return the assigned method for a label ('ocr', 'detector', or 'both')."""
        low = label.lower()
        if low in {n.lower() for n in self.ocr}:
            return "ocr"
        if low in {n.lower() for n in self.detector}:
            return "detector"
        return "both"


@dataclass
class PipelineConfig:
    # --- Input ---
    input_path: str = ""

    # --- Frame Extraction ---
    fps: float = 1.0  # frames per second to sample

    # --- Detection ---
    model_path: str = "yolov8n.pt"  # default model; swap for a logo-trained model
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = ""  # "" = auto (CUDA if available, else CPU)
    img_size: int = 640  # inference resolution (model rescales internally)

    # --- Mode ---
    # "detect"  = YOLO only
    # "ocr"     = OCR only (no model needed)
    # "both"    = YOLO + OCR
    mode: str = "detect"

    # --- Detector backend ---
    # "yolo"      = local YOLOv8 model (default)
    # "reference" = CLIP-based reference logo matching
    detector: str = "yolo"

    # --- Reference matching (CLIP) ---
    logos_dir: str = ""                   # path to reference logos directory
    clip_model: str = "ViT-B-32"         # open_clip model architecture
    clip_pretrained: str = "openai"       # pretrained weights tag
    similarity_threshold: float = 0.75   # min cosine similarity to count as a match
    ocr_exclusion_coverage: float = 0.3  # skip CLIP patches that overlap OCR hits by this fraction
    # Speed vs quality: fewer scales + larger stride + no refine ≈ faster (e.g. ~0.5s/frame target)
    clip_patch_scales: List[int] = field(default_factory=lambda: [96, 256])
    clip_stride_ratio: float = 0.65
    clip_refine: bool = False

    # --- OCR ---
    # "easyocr" | "paddle" | "deepseek"
    ocr_backend: str = "easyocr"
    target_labels: List[str] = field(default_factory=list)
    labels_file: str = ""  # path to a text file with one brand name per line

    # --- DeepSeek VLM OCR ---
    deepseek_api_key: str = ""                         # or set DEEPSEEK_API_KEY env var
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"              # any OpenAI-compatible vision model
    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    ocr_match_threshold: float = 0.5   # min fuzzy-match ratio to count as a hit
    ocr_confidence_boost: float = 0.15  # added to detection conf on label match

    # --- Output ---
    output_dir: str = "outputs"
    save_annotated_frames: bool = True
    save_cropped_logos: bool = True
    save_report_csv: bool = True
    save_report_json: bool = True

    # --- Visualisation ---
    bbox_thickness: int = 2
    font_scale: float = 0.6
    label_colors: dict = field(default_factory=dict)  # brand_name -> (B, G, R)

    def resolve_output_dir(self, input_path: Optional[str] = None) -> Path:
        """Create a run-specific output directory derived from the input filename."""
        stem = Path(input_path or self.input_path).stem
        run_dir = Path(self.output_dir) / stem
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "frames").mkdir(exist_ok=True)
        (run_dir / "crops").mkdir(exist_ok=True)
        return run_dir
