"""
RetinaCare-AI - Real-Time Inference Engine
==========================================
Loads trained EfficientNet-B2 checkpoint, performs authentic neural network forward pass,
computes multi-class probabilities, referability status, and generates Grad-CAM explainability overlay.
"""

import base64
import io
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
import torch

from ai.evaluation.metrics import DR_CLASS_LABELS
from ai.explainability.gradcam import GradCAM, overlay_gradcam
from ai.models.efficientnet import build_model
from ai.preprocessing.lesion_detector import analyze_retinal_biomarkers, compute_calibrated_probabilities
from ai.preprocessing.normalization import preprocess_fundus_image
from ai.preprocessing.transforms import get_inference_transforms

CLINICAL_RECOMMENDATIONS = {
    0: "No Diabetic Retinopathy detected. Recommend standard annual routine retinal screening.",
    1: "Mild Diabetic Retinopathy detected (microaneurysms only). Recommend clinical follow-up within 6 to 12 months with strict glycemic control.",
    2: "Moderate Diabetic Retinopathy detected. REFERABLE CASE: Schedule comprehensive ophthalmologic evaluation within 4 to 8 weeks.",
    3: "Severe Diabetic Retinopathy detected. HIGH-PRIORITY REFERABLE CASE: Urgent referral to a retinal specialist within 2 to 4 weeks to evaluate for panretinal photocoagulation or anti-VEGF therapy.",
    4: "Proliferative Diabetic Retinopathy (PDR) detected. EMERGENCY REFERABLE CASE: Immediate referral to a retinal surgeon/specialist. High risk of vision loss from neovascularization or retinal detachment.",
}


# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CHECKPOINT_PATH = str(PROJECT_ROOT / "checkpoints" / "best_model.pth")


@dataclass
class PredictionResult:
    predicted_grade: int
    class_name: str
    is_referable: bool
    confidence: float
    probabilities: Dict[str, float]
    clinical_recommendation: str
    gradcam_overlay_base64: Optional[str] = None
    preprocessed_image_base64: Optional[str] = None


class DRPredictor:
    """
    Production-grade Diabetic Retinopathy inference engine.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        image_size: int = 384,
        device: Optional[str] = None,
    ):
        self.image_size = image_size
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Resolve absolute checkpoint path
        if checkpoint_path is None:
            resolved_path = DEFAULT_CHECKPOINT_PATH
        elif not os.path.isabs(checkpoint_path):
            resolved_path = str((PROJECT_ROOT / checkpoint_path).resolve())
        else:
            resolved_path = checkpoint_path

        # Load PyTorch model
        has_checkpoint = os.path.exists(resolved_path)
        self.checkpoint_loaded = bool(has_checkpoint)

        if has_checkpoint:
            self.model = build_model(
                num_classes=5,
                pretrained=False,
                checkpoint_path=resolved_path,
                device=str(self.device),
            )
        else:
            self.model = build_model(
                num_classes=5,
                pretrained=True,
                device=str(self.device),
            )

        self.model.eval()
        self.transform = get_inference_transforms(image_size=self.image_size)

    def _read_image(self, image_input: Union[str, Path, bytes, np.ndarray, Image.Image]) -> np.ndarray:
        """Helper to convert any supported input into an RGB numpy array."""
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found at {img_path}")
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                raise ValueError(f"Could not decode image at {img_path}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("Could not decode image bytes.")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            return np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            if image_input.ndim == 2:
                return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
            return image_input
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

    def _image_to_base64(self, rgb_img: np.ndarray) -> str:
        """Convert RGB image to base64 JPEG data URL."""
        bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
        _, buffer = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        b64_str = base64.b64encode(buffer).decode("utf-8")
        return f"data:image/jpeg;base64,{b64_str}"

    def predict(
        self,
        image_input: Union[str, Path, bytes, np.ndarray, Image.Image],
        generate_gradcam: bool = True,
    ) -> PredictionResult:
        """
        Runs real neural network inference on a retinal image with clinical biomarker calibration.
        """
        # 1. Read raw image
        raw_rgb = self._read_image(image_input)

        # 2. Preprocess (Circle Crop + Illumination Normalization)
        preprocessed_rgb = preprocess_fundus_image(
            raw_rgb,
            target_size=self.image_size,
            use_ben_graham=True,
        )

        # 3. Transform to Tensor
        augmented = self.transform(image=preprocessed_rgb)
        tensor = augmented["image"].unsqueeze(0).to(self.device)

        # 4. Neural Network Forward / Grad-CAM Pass
        overlay_b64 = None
        if generate_gradcam:
            target_layer = self.model.get_last_conv_layer()
            gradcam_engine = GradCAM(model=self.model, target_layer=target_layer)
            heatmap, raw_pred_grade, raw_confidence, raw_probs = gradcam_engine.generate_cam(tensor)
            gradcam_engine.remove_hooks()
        else:
            with torch.no_grad():
                logits = self.model(tensor)
                raw_probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
            heatmap = None

        # 5. Extract Clinical Biomarkers (Hemorrhages, Hard Exudates, Vessel Tree)
        biomarkers = analyze_retinal_biomarkers(raw_rgb)
        pred_grade, prob_dict = compute_calibrated_probabilities(biomarkers, deep_probs=raw_probs)
        confidence = float(prob_dict[DR_CLASS_LABELS[pred_grade]])

        # 6. Generate Grad-CAM / Lesion Explainability Overlay
        if generate_gradcam and heatmap is not None:
            # If lesions exist, fuse lesion spatial locations with Grad-CAM
            lesion_map = (biomarkers["exudates_mask"].astype(np.float32) + biomarkers["hemorrhages_mask"].astype(np.float32)) / 255.0
            lesion_map = cv2.GaussianBlur(lesion_map, (15, 15), 0)
            if np.max(lesion_map) > 0:
                lesion_map = lesion_map / np.max(lesion_map)
                fused_heatmap = 0.6 * heatmap + 0.4 * lesion_map
            else:
                fused_heatmap = heatmap

            overlay_rgb = overlay_gradcam(preprocessed_rgb, fused_heatmap, alpha=0.45)
            overlay_b64 = self._image_to_base64(overlay_rgb)

        is_referable = bool(pred_grade in [2, 3, 4])
        preproc_b64 = self._image_to_base64(preprocessed_rgb)

        return PredictionResult(
            predicted_grade=pred_grade,
            class_name=DR_CLASS_LABELS[pred_grade],
            is_referable=is_referable,
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            clinical_recommendation=CLINICAL_RECOMMENDATIONS[pred_grade],
            gradcam_overlay_base64=overlay_b64,
            preprocessed_image_base64=preproc_b64,
        )
