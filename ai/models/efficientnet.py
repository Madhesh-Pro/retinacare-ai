"""
EfficientNet-B2 Architecture for Diabetic Retinopathy Classification
====================================================================
Uses pretrained EfficientNet-B2 backbone with a customized clinical classifier head.
Designed for 5-class severity grading (0: No DR, 1: Mild, 2: Moderate, 3: Severe, 4: Proliferative)
and layer access for Grad-CAM explainability.
"""

from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B2_Weights, efficientnet_b2


class EfficientNetB2_DR(nn.Module):
    """
    Diabetic Retinopathy Classification Network based on EfficientNet-B2.
    """

    def __init__(
        self,
        num_classes: int = 5,
        pretrained: bool = True,
        dropout_rate: float = 0.4,
    ):
        super().__init__()
        self.num_classes = num_classes

        # Load EfficientNet-B2 backbone
        weights = EfficientNet_B2_Weights.DEFAULT if pretrained else None
        self.backbone = efficientnet_b2(weights=weights)

        in_features = self.backbone.classifier[1].in_features  # 1408 for EfficientNet-B2

        # Replace default classifier with clinical fine-tuning head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate, inplace=True),
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(inplace=True),
            nn.Dropout(p=dropout_rate * 0.75, inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Tensor of shape (B, 3, H, W)
        Returns:
            Logits tensor of shape (B, 5)
        """
        return self.backbone(x)

    def get_last_conv_layer(self) -> nn.Module:
        """
        Returns the final convolutional block in features (features[8])
        for Grad-CAM activation and gradient extraction.
        """
        return self.backbone.features[-1]


def build_model(
    num_classes: int = 5,
    pretrained: bool = True,
    checkpoint_path: Optional[str] = None,
    device: str = "cpu",
) -> EfficientNetB2_DR:
    """
    Model builder helper. Optionally loads saved weights from checkpoint.
    """
    model = EfficientNetB2_DR(num_classes=num_classes, pretrained=pretrained)

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        else:
            model.load_state_dict(state)

    model.to(device)
    return model
