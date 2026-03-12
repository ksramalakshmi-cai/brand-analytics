from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


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

    # --- OCR ---
    ocr_backend: str = "easyocr"  # "paddle" or "easyocr" (use easyocr if Paddle segfaults on CPU)
    target_labels: List[str] = field(default_factory=list)
    labels_file: str = ""  # path to a text file with one brand name per line
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
