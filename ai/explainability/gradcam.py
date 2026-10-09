"""
Grad-CAM (Gradient-weighted Class Activation Mapping) for Diabetic Retinopathy
==============================================================================
Provides visual explainability by highlighting the spatial retinal regions
(hemorrhages, exudates, microaneurysms, neovascularization) that contributed
most significantly to the model's DR severity prediction.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """
    Grad-CAM engine for PyTorch convolutional vision models.
    Hooks into intermediate feature maps and gradients.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None

        self._forward_hook = target_layer.register_forward_hook(self._save_activations)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output):
        self.activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Computes Grad-CAM heatmap for the given input tensor.
        
        Args:
            input_tensor: Shape (1, 3, H, W)
            target_class: Target class index (0-4). If None, uses argmax prediction.
            
        Returns:
            heatmap: 2D numpy array of shape (H, W) normalized to [0, 1].
            pred_class: Predicted class ID.
            confidence: Probability of predicted class.
        """
        self.model.eval()

        # Forward pass
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1)

        if target_class is None:
            target_class = int(torch.argmax(probs, dim=1).item())

        confidence = float(probs[0, target_class].item())

        # Backward pass for target class score
        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Grad-CAM hooks failed to capture activations/gradients.")

        # Global average pooling of gradients
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted combination of forward activation maps
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (1, 1, h, w)
        cam = torch.relu(cam)  # Apply ReLU to retain only positive influences

        cam_np = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = np.min(cam_np), np.max(cam_np)
        if cam_max > cam_min:
            cam_norm = (cam_np - cam_min) / (cam_max - cam_min)
        else:
            cam_norm = np.zeros_like(cam_np)

        # Resize to input tensor spatial dimensions
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        heatmap_resized = cv2.resize(cam_norm, (w, h), interpolation=cv2.INTER_LINEAR)

        all_probs_np = probs.squeeze().detach().cpu().numpy()
        return heatmap_resized, target_class, confidence, all_probs_np

    def remove_hooks(self):
        """Remove hooks to prevent memory leak."""
        self._forward_hook.remove()
        self._backward_hook.remove()
        self.activations = None
        self.gradients = None


def overlay_gradcam(
    original_rgb: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlays 2D Grad-CAM heatmap onto RGB fundus image.
    
    Args:
        original_rgb: RGB uint8 numpy array (H, W, 3).
        heatmap: 2D float numpy array (H, W) in [0, 1].
        alpha: Blending weight for heatmap (1-alpha for original image).
        colormap: OpenCV colormap (default: COLORMAP_JET).
        
    Returns:
        RGB uint8 numpy array (H, W, 3) representing the explainability overlay.
    """
    # Resize heatmap if dimensions don't match
    h, w = original_rgb.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # Convert normalized heatmap to uint8 color map
    heatmap_uint8 = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    # Blend
    overlay = cv2.addWeighted(original_rgb, 1.0 - alpha, heatmap_color_rgb, alpha, 0)
    return overlay
