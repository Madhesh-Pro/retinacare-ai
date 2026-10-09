"""
Retinal Fundus Image Circular Bounding & Cropping
==================================================
Extracts the circular retinal region of interest (ROI) by removing
uninformative black outer margins commonly produced by fundus cameras.
"""

import cv2
import numpy as np


def crop_image_from_gray(img: np.ndarray, tol: int = 7) -> np.ndarray:
    """
    Remove black margins around retinal fundus image based on pixel intensity threshold.
    """
    if img.ndim == 2:
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    elif img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol

        check_shape = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))].shape[0]
        if check_shape == 0:  # Image is too dark or empty
            return img
        else:
            img1 = img[:, :, 0][np.ix_(mask.any(1), mask.any(0))]
            img2 = img[:, :, 1][np.ix_(mask.any(1), mask.any(0))]
            img3 = img[:, :, 2][np.ix_(mask.any(1), mask.any(0))]
            return np.stack([img1, img2, img3], axis=-1)
    return img


def circle_crop_fundus(img: np.ndarray, target_size: int = 384, tol: int = 7) -> np.ndarray:
    """
    Tightly crops the retinal disk and applies a circular mask to eliminate artifacts outside the retina.
    
    Args:
        img: Input RGB numpy array (H, W, C).
        target_size: Target square dimension (width=height).
        tol: Intensity threshold for background pixels.
        
    Returns:
        Square RGB numpy array of shape (target_size, target_size, 3).
    """
    # 1. Tight bounding crop
    cropped = crop_image_from_gray(img, tol=tol)

    # 2. Resize to square
    resized = cv2.resize(cropped, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # 3. Create circular mask
    mask = np.zeros((target_size, target_size), dtype=np.uint8)
    center = (target_size // 2, target_size // 2)
    radius = int(target_size / 2 * 0.98)  # 98% of half dimension
    cv2.circle(mask, center, radius, 1, -1)

    # Apply circular mask
    masked_img = resized * mask[:, :, np.newaxis]
    return masked_img
