"""
Illumination & Color Normalization for Retinal Fundus Images
============================================================
Implements:
1. Ben Graham's Color Normalization (Gaussian Subtraction)
2. CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space
"""

import cv2
import numpy as np


def apply_ben_graham(img: np.ndarray, sigma_x: int = 10) -> np.ndarray:
    """
    Applies Ben Graham's local illumination normalization:
    output = 4 * image - 4 * GaussianBlur(image, sigma_x) + 128
    
    Normalizes variations in illumination and exposure across different fundus camera devices.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigma_x)
    normalized = cv2.addWeighted(img, 4, blurred, -4, 128)
    return np.clip(normalized, 0, 255).astype(np.uint8)


def apply_clahe_lab(img: np.ndarray, clip_limit: float = 2.0, tile_grid_size: tuple = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE on the Lightness (L) channel in LAB color space.
    Enhances contrast of microvascular lesions (hemorrhages, exudates, microaneurysms)
    while preserving chromatic fidelity.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_chan)

    merged = cv2.merge((cl, a_chan, b_chan))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def preprocess_fundus_image(
    img: np.ndarray,
    target_size: int = 384,
    use_ben_graham: bool = True,
    use_clahe: bool = True,
) -> np.ndarray:
    """
    Complete clinical retinal preprocessing pipeline:
    1. Circle crop & black margin removal
    2. Optional CLAHE enhancement
    3. Optional Ben Graham illumination normalization
    
    Args:
        img: Input RGB image numpy array.
        target_size: Square resolution (e.g. 512).
        use_ben_graham: Whether to apply Ben Graham Gaussian subtraction.
        use_clahe: Whether to apply LAB CLAHE.
        
    Returns:
        Preprocessed RGB numpy array of shape (target_size, target_size, 3).
    """
    from ai.preprocessing.retina_crop import circle_crop_fundus

    # 1. Circle Crop to remove black borders
    cropped = circle_crop_fundus(img, target_size=target_size)

    # 2. Illumination enhancement
    if use_ben_graham:
        processed = apply_ben_graham(cropped, sigma_x=10)
    elif use_clahe:
        processed = apply_clahe_lab(cropped, clip_limit=2.0)
    else:
        processed = cropped

    return processed
