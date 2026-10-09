"""
Data Augmentation & Transformation Pipelines
=============================================
Provides train, validation, and test augmentation transforms
using Albumentations and PyTorch torchvision normalization.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# Standard ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 512) -> A.Compose:
    """
    Robust clinical data augmentation pipeline for training EfficientNet-B2:
    - Flips & Rotations (Retinal images are rotational invariants)
    - Shift, Scale, Affine
    - Color Jitter & Brightness adjustments
    - CoarseDropout / Cutout
    - ImageNet Normalization & Tensor conversion
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.08,
            scale_limit=0.08,
            rotate_limit=180,
            border_mode=0,
            p=0.6,
        ),
        A.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.05,
            p=0.5,
        ),
        A.CoarseDropout(
            max_holes=6,
            max_height=int(image_size * 0.08),
            max_width=int(image_size * 0.08),
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.4,
        ),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 512) -> A.Compose:
    """
    Deterministic transformation for validation, testing, and real-time inference.
    """
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ])


def get_inference_transforms(image_size: int = 512) -> A.Compose:
    """Alias for validation transforms during deployment."""
    return get_val_transforms(image_size=image_size)
