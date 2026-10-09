#!/usr/bin/env python3
"""
RetinaCare-AI - Dataset Preparation & Stratified Splitting Script
================================================================
1. Crops circular retina region and strips non-informative dark borders.
2. Applies Ben Graham's illumination normalization / CLAHE.
3. Resizes images to standard resolution (e.g. 512x512).
4. Generates stratified Train / Validation / Test splits preserving class ratios.
5. Saves processed images to data/processed/images and split CSVs.
"""

import argparse
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

from ai.preprocessing.normalization import preprocess_fundus_image
from scripts.inspect_dataset import find_image_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RetinaCare-Prep")


def process_single_image(
    src_path: str,
    dst_path: str,
    image_size: int = 512,
    use_ben_graham: bool = True,
) -> Tuple[str, bool, str]:
    """
    Worker function to process and save a single retinal image.
    """
    try:
        bgr_img = cv2.imread(src_path)
        if bgr_img is None:
            return (src_path, False, "Failed to read image with OpenCV")

        rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        processed_rgb = preprocess_fundus_image(
            rgb_img,
            target_size=image_size,
            use_ben_graham=use_ben_graham,
        )

        processed_bgr = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        cv2.imwrite(dst_path, processed_bgr)
        return (src_path, True, "")
    except Exception as e:
        return (src_path, False, str(e))


def prepare_dataset(
    csv_path: str,
    raw_images_dir: str,
    output_dir: str = "data/processed",
    image_size: int = 512,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    num_workers: int = 4,
    use_ben_graham: bool = True,
):
    csv_p = Path(csv_path).resolve()
    raw_img_p = Path(raw_images_dir).resolve()
    out_p = Path(output_dir).resolve()
    processed_img_dir = out_p / "images"
    processed_img_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Starting RetinaCare-AI Dataset Preparation Pipeline")
    logger.info(f"  CSV Path        : {csv_p}")
    logger.info(f"  Raw Images Dir  : {raw_img_p}")
    logger.info(f"  Output Dir      : {out_p}")
    logger.info(f"  Target Size     : {image_size}x{image_size}")
    logger.info(f"  Ben Graham Norm : {use_ben_graham}")
    logger.info("=" * 70)

    if not csv_p.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_p}")
    if not raw_img_p.exists():
        raise FileNotFoundError(f"Raw images dir not found: {raw_img_p}")

    df = pd.read_csv(csv_p)
    logger.info(f"Loaded {len(df)} records from CSV.")

    # Match existing images
    valid_records = []
    tasks = []

    for _, row in df.iterrows():
        id_code = str(row["id_code"]).strip()
        diagnosis = int(row["diagnosis"])
        src_file = find_image_file(raw_img_p, id_code)
        if src_file and src_file.exists():
            dst_file = processed_img_dir / f"{id_code}.png"
            valid_records.append({
                "id_code": id_code,
                "diagnosis": diagnosis,
                "file_path": str(dst_file.relative_to(out_p.parent.parent)),
                "is_referable": int(diagnosis in [2, 3, 4]),
            })
            tasks.append((str(src_file), str(dst_file), image_size, use_ben_graham))

    if not valid_records:
        raise ValueError("No matching image files found to process!")

    valid_df = pd.DataFrame(valid_records)
    logger.info(f"Matched {len(valid_df)} valid images for preprocessing.")

    # Preprocessing batch with thread pool
    logger.info(f"Preprocessing images using {num_workers} workers...")
    failed_count = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_single_image, *task) for task in tasks
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing", unit="img"):
            _, success, err = f.result()
            if not success:
                failed_count += 1
                logger.warning(f"Failed to process image: {err}")

    logger.info(f"Preprocessing complete. Successfully processed: {len(tasks) - failed_count}/{len(tasks)}")

    # Stratified Train / Validation / Test Splitting
    logger.info("Generating stratified train / validation / test splits...")
    
    # 1. Split Train+Val vs Test
    split_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(split_test.split(valid_df, valid_df["diagnosis"]))
    
    train_val_df = valid_df.iloc[train_val_idx].reset_index(drop=True)
    test_df = valid_df.iloc[test_idx].reset_index(drop=True)

    # 2. Split Train vs Val from Train+Val
    val_adjusted_size = val_size / (1.0 - test_size)
    split_val = StratifiedShuffleSplit(n_splits=1, test_size=val_adjusted_size, random_state=seed)
    train_idx, val_idx = next(split_val.split(train_val_df, train_val_df["diagnosis"]))

    train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
    val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

    # Save split CSVs
    train_csv_path = out_p / "train_split.csv"
    val_csv_path = out_p / "val_split.csv"
    test_csv_path = out_p / "test_split.csv"

    train_df.to_csv(train_csv_path, index=False)
    val_df.to_csv(val_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    logger.info(f"Saved stratified splits:")
    logger.info(f"  - Train Set : {len(train_df):>6} images ({len(train_df)/len(valid_df)*100:.1f}%) -> {train_csv_path}")
    logger.info(f"  - Val Set   : {len(val_df):>6} images ({len(val_df)/len(valid_df)*100:.1f}%) -> {val_csv_path}")
    logger.info(f"  - Test Set  : {len(test_df):>6} images ({len(test_df)/len(valid_df)*100:.1f}%) -> {test_csv_path}")

    # Summary table
    print("\n" + "=" * 65)
    print("       STRATIFIED SPLIT CLASS DISTRIBUTION SUMMARY")
    print("=" * 65)
    print(f" {'Grade':<6} | {'Train':<8} | {'Val':<8} | {'Test':<8} | {'Total':<8}")
    print(f" {'-'*6} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
    for g in range(5):
        tr_c = (train_df['diagnosis'] == g).sum()
        vl_c = (val_df['diagnosis'] == g).sum()
        ts_c = (test_df['diagnosis'] == g).sum()
        tot = tr_c + vl_c + ts_c
        print(f"  {g:<5} | {tr_c:>7} | {vl_c:>7} | {ts_c:>7} | {tot:>7}")
    print("=" * 65 + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description="RetinaCare-AI Dataset Preprocessing & Stratified Splitting")
    parser.add_argument("--csv-path", type=str, default="data/raw/train.csv", help="Path to raw train.csv")
    parser.add_argument("--raw-images-dir", type=str, default="data/raw/train_images", help="Path to raw train_images directory")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Path to save processed images and split CSVs")
    parser.add_argument("--image-size", type=int, default=512, help="Target image size (width & height)")
    parser.add_argument("--val-size", type=float, default=0.15, help="Validation set fraction")
    parser.add_argument("--test-size", type=float, default=0.15, help="Test set fraction")
    parser.add_argument("--workers", type=int, default=4, help="Worker threads for processing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible stratified splitting")
    parser.add_argument("--no-ben-graham", action="store_true", help="Disable Ben Graham illumination normalization")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        prepare_dataset(
            csv_path=args.csv_path,
            raw_images_dir=args.raw_images_dir,
            output_dir=args.output_dir,
            image_size=args.image_size,
            val_size=args.val_size,
            test_size=args.test_size,
            seed=args.seed,
            num_workers=args.workers,
            use_ben_graham=not args.no_ben_graham,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error during dataset preparation: {e}")
        sys.exit(1)
