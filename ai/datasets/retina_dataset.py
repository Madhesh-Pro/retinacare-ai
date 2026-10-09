"""
PyTorch Dataset & DataLoader for Diabetic Retinopathy Fundus Images
===================================================================
"""

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from ai.preprocessing.transforms import get_inference_transforms, get_train_transforms, get_val_transforms


class RetinaDataset(Dataset):
    """
    PyTorch Dataset for Diabetic Retinopathy classification.
    Loads fundus images and applies Albumentations augmentations.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        images_dir: str,
        transform: Optional[Callable] = None,
        is_test: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.is_test = is_test

        # Check required columns
        if "id_code" not in self.df.columns:
            raise ValueError("DataFrame must contain 'id_code' column.")

    def __len__(self) -> int:
        return len(self.df)

    def _get_image_path(self, id_code: str) -> Path:
        # Check direct path or extensions
        p = self.images_dir / id_code
        if p.is_file():
            return p
        for ext in [".png", ".jpg", ".jpeg", ".PNG", ".JPG"]:
            candidate = self.images_dir / f"{id_code}{ext}"
            if candidate.is_file():
                return candidate
        return self.images_dir / f"{id_code}.png"

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.df.iloc[idx]
        id_code = str(row["id_code"]).strip()

        img_path = self._get_image_path(id_code)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found on disk: {img_path}")

        # Read RGB image
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise IOError(f"Failed to decode image buffer: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        # Apply Albumentations transform
        if self.transform is not None:
            augmented = self.transform(image=rgb)
            img_tensor = augmented["image"]
        else:
            # Fallback to default inference transform
            default_t = get_inference_transforms(image_size=rgb.shape[0])
            img_tensor = default_t(image=rgb)["image"]

        if not self.is_test and "diagnosis" in row:
            label = torch.tensor(int(row["diagnosis"]), dtype=torch.long)
            return img_tensor, label, id_code
        else:
            return img_tensor, torch.tensor(-1, dtype=torch.long), id_code


def create_dataloader(
    df: pd.DataFrame,
    images_dir: str,
    transform: Optional[Callable],
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 2,
    is_test: bool = False,
) -> DataLoader:
    """Helper to instantiate PyTorch DataLoader."""
    dataset = RetinaDataset(
        df=df,
        images_dir=images_dir,
        transform=transform,
        is_test=is_test,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
