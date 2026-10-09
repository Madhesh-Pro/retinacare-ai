#!/usr/bin/env python3
"""
RetinaCare-AI - Dataset Inspection Script
==========================================
Performs comprehensive dataset inspection for Diabetic Retinopathy screening (APTOS 2019 format).

Checks performed:
1. Total images in CSV and image directory.
2. Class distribution and imbalance analysis (Classes 0-4 + Referable DR split).
3. Missing images (CSV entries with no file on disk) and orphan images (files not in CSV).
4. Corrupt / unreadable images (PIL & OpenCV integrity verification).
5. Image dimensions, channels, aspect ratios, min/max/mean/median resolution.
6. Duplicate detection (Duplicate CSV IDs and identical file content via MD5 hashing).
7. Structured JSON report export and clean console output.

All dataset paths are configurable via CLI arguments or configs/dataset_config.yaml.
"""

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import yaml
from tqdm import tqdm

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RetinaCare-Inspector")

# DR Class Label Mapping according to ICDR clinical scale
DR_CLASS_NAMES = {
    0: "No Diabetic Retinopathy (No DR)",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy (PDR)",
}

REFERABLE_CLASSES = {2, 3, 4}
NON_REFERABLE_CLASSES = {0, 1}
SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".bmp", ".tiff"}


@dataclass
class ImageMetadata:
    id_code: str
    file_path: str
    file_name: str
    file_size_bytes: int
    is_valid: bool
    error_msg: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    aspect_ratio: Optional[float] = None
    md5_hash: Optional[str] = None
    diagnosis: Optional[int] = None


@dataclass
class InspectionReport:
    timestamp: str
    csv_path: str
    images_dir: str
    total_csv_records: int = 0
    total_image_files_on_disk: int = 0
    total_matched_images: int = 0
    missing_images_count: int = 0
    missing_images_list: List[str] = field(default_factory=list)
    orphan_images_count: int = 0
    orphan_images_list: List[str] = field(default_factory=list)
    corrupt_images_count: int = 0
    corrupt_images_list: List[Dict[str, str]] = field(default_factory=list)
    duplicate_csv_ids_count: int = 0
    duplicate_csv_ids_list: List[str] = field(default_factory=list)
    duplicate_image_files_count: int = 0
    duplicate_image_groups: List[Dict[str, Any]] = field(default_factory=list)
    class_distribution: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    referable_summary: Dict[str, Any] = field(default_factory=dict)
    dimension_stats: Dict[str, Any] = field(default_factory=dict)
    channel_distribution: Dict[str, int] = field(default_factory=dict)
    top_resolutions: List[Dict[str, Any]] = field(default_factory=list)


def load_config(config_path: Optional[str]) -> Dict[str, Any]:
    """Load configuration from YAML file if available."""
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Could not parse config file '{config_path}': {e}. Using defaults/CLI args.")
    return {}


def calculate_md5(file_path: Path, chunk_size: int = 65536) -> str:
    """Compute MD5 checksum of a file for exact duplicate content detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def inspect_single_image(
    file_path: Path,
    id_code: str,
    diagnosis: Optional[int],
    compute_hash: bool = True,
) -> ImageMetadata:
    """
    Thoroughly inspect a single image file:
    - Verifies PIL opening & integrity.
    - Decodes pixel matrix using OpenCV to verify against partial/corrupt buffers.
    - Computes dimensions, channels, and MD5 hash.
    """
    file_size = file_path.stat().st_size
    meta = ImageMetadata(
        id_code=id_code,
        file_path=str(file_path),
        file_name=file_path.name,
        file_size_bytes=file_size,
        is_valid=False,
        diagnosis=diagnosis,
    )

    if file_size == 0:
        meta.error_msg = "Zero-byte empty file"
        return meta

    try:
        # Step 1: Verify using PIL
        with Image.open(file_path) as pil_img:
            pil_img.verify()

        # Step 2: Full pixel decode via OpenCV
        cv_img = cv2.imread(str(file_path), cv2.IMREAD_UNCHANGED)
        if cv_img is None:
            meta.error_msg = "OpenCV failed to decode image buffer (corrupted file format or headers)"
            return meta

        if cv_img.ndim == 2:
            height, width = cv_img.shape
            channels = 1
        elif cv_img.ndim == 3:
            height, width, channels = cv_img.shape
        else:
            meta.error_msg = f"Unexpected image dimensions shape: {cv_img.shape}"
            return meta

        if height <= 0 or width <= 0:
            meta.error_msg = f"Invalid image resolution: width={width}, height={height}"
            return meta

        meta.is_valid = True
        meta.width = width
        meta.height = height
        meta.channels = channels
        meta.aspect_ratio = round(width / height, 4)

        if compute_hash:
            meta.md5_hash = calculate_md5(file_path)

    except Exception as exc:
        meta.error_msg = f"Exception during validation: {str(exc)}"
        meta.is_valid = False

    return meta


def find_image_file(images_dir: Path, id_code: str) -> Optional[Path]:
    """
    Locate image file on disk matching id_code.
    Handles id_code with or without explicit extension.
    """
    # Direct match if id_code contains extension
    direct_path = images_dir / id_code
    if direct_path.is_file():
        return direct_path

    # Try supported extensions
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        candidate = images_dir / f"{id_code}{ext}"
        if candidate.is_file():
            return candidate

    return None


def run_dataset_inspection(
    csv_path: str,
    images_dir: str,
    output_report: Optional[str] = None,
    num_workers: int = 4,
    check_duplicates: bool = True,
) -> InspectionReport:
    """
    Main dataset inspection pipeline.
    """
    csv_p = Path(csv_path).resolve()
    images_p = Path(images_dir).resolve()

    from datetime import datetime
    report = InspectionReport(
        timestamp=datetime.now().isoformat(),
        csv_path=str(csv_p),
        images_dir=str(images_p),
    )

    logger.info("=" * 70)
    logger.info("Starting RetinaCare-AI Dataset Inspection")
    logger.info(f"  CSV Path   : {csv_p}")
    logger.info(f"  Images Dir : {images_p}")
    logger.info("=" * 70)

    # 1. Validate paths
    if not csv_p.exists():
        raise FileNotFoundError(
            f"Dataset CSV file not found at: {csv_p}\n"
            f"Please check the path or ensure 'train.csv' is placed in '{images_p.parent}'"
        )

    if not images_p.exists() or not images_p.is_dir():
        raise FileNotFoundError(
            f"Images directory not found at: {images_p}\n"
            f"Please check the path or ensure 'train_images' directory exists."
        )

    # 2. Read and validate CSV
    logger.info("Reading dataset CSV metadata...")
    try:
        df = pd.read_csv(csv_p)
    except Exception as e:
        raise ValueError(f"Failed to read CSV file '{csv_p}': {e}")

    report.total_csv_records = len(df)
    logger.info(f"Found {len(df)} records in CSV.")

    # Check required columns
    required_cols = {"id_code", "diagnosis"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in CSV: {missing_cols}. Found columns: {list(df.columns)}"
        )

    # Check duplicate ID codes in CSV
    duplicated_mask = df["id_code"].duplicated(keep=False)
    if duplicated_mask.any():
        dup_ids = df.loc[duplicated_mask, "id_code"].unique().tolist()
        report.duplicate_csv_ids_count = len(dup_ids)
        report.duplicate_csv_ids_list = [str(x) for x in dup_ids]
        logger.warning(f"Detected {len(dup_ids)} duplicate ID codes in CSV!")

    # Check invalid diagnosis values
    valid_diagnosis_values = {0, 1, 2, 3, 4}
    invalid_diag_rows = df[~df["diagnosis"].isin(valid_diagnosis_values)]
    if not invalid_diag_rows.empty:
        logger.warning(
            f"Found {len(invalid_diag_rows)} rows with non-standard diagnosis values: "
            f"{invalid_diag_rows['diagnosis'].unique()}"
        )

    # 3. Scan Image Files on Disk
    logger.info("Scanning image directory...")
    all_disk_files = [
        f for f in images_p.iterdir()
        if f.is_file() and f.suffix in SUPPORTED_IMAGE_EXTENSIONS
    ]
    report.total_image_files_on_disk = len(all_disk_files)
    logger.info(f"Found {len(all_disk_files)} image files on disk.")

    disk_files_by_name = {f.name: f for f in all_disk_files}
    disk_files_by_stem = {f.stem: f for f in all_disk_files}

    # 4. Cross-reference CSV with Disk Files
    matched_tasks: List[Tuple[Path, str, int]] = []
    missing_images: List[str] = []
    matched_file_paths: Set[Path] = set()

    for _, row in df.iterrows():
        raw_id = str(row["id_code"]).strip()
        diag = int(row["diagnosis"]) if pd.notna(row["diagnosis"]) and str(row["diagnosis"]).isdigit() else None

        img_file = find_image_file(images_p, raw_id)
        if img_file and img_file.exists():
            matched_tasks.append((img_file, raw_id, diag))
            matched_file_paths.add(img_file)
        else:
            missing_images.append(raw_id)

    report.total_matched_images = len(matched_tasks)
    report.missing_images_count = len(missing_images)
    report.missing_images_list = missing_images

    # Orphan images: files on disk not present in CSV
    orphan_files = [f.name for f in all_disk_files if f not in matched_file_paths]
    report.orphan_images_count = len(orphan_files)
    report.orphan_images_list = orphan_files

    logger.info(f"Matched images: {len(matched_tasks)} | Missing images: {len(missing_images)} | Orphan files: {len(orphan_files)}")

    # 5. Multithreaded Image Integrity & Dimension Inspection
    logger.info(f"Inspecting image integrity and dimensions using {num_workers} workers...")
    inspected_meta: List[ImageMetadata] = []

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(inspect_single_image, path, id_c, diag, check_duplicates): id_c
            for path, id_c, diag in matched_tasks
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Inspecting Images", unit="img"):
            inspected_meta.append(future.result())

    # 6. Analyze Corrupt Images
    corrupt_meta = [m for m in inspected_meta if not m.is_valid]
    valid_meta = [m for m in inspected_meta if m.is_valid]

    report.corrupt_images_count = len(corrupt_meta)
    report.corrupt_images_list = [
        {"id_code": m.id_code, "file_path": m.file_path, "error": m.error_msg or "Unknown error"}
        for m in corrupt_meta
    ]

    # 7. Dimension & Resolution Statistics
    if valid_meta:
        widths = [m.width for m in valid_meta if m.width is not None]
        heights = [m.height for m in valid_meta if m.height is not None]
        aspect_ratios = [m.aspect_ratio for m in valid_meta if m.aspect_ratio is not None]
        channels_list = [m.channels for m in valid_meta if m.channels is not None]

        report.dimension_stats = {
            "valid_images_evaluated": len(valid_meta),
            "width": {
                "min": int(np.min(widths)),
                "max": int(np.max(widths)),
                "mean": float(np.round(np.mean(widths), 2)),
                "median": float(np.median(widths)),
                "std": float(np.round(np.std(widths), 2)),
            },
            "height": {
                "min": int(np.min(heights)),
                "max": int(np.max(heights)),
                "mean": float(np.round(np.mean(heights), 2)),
                "median": float(np.median(heights)),
                "std": float(np.round(np.std(heights), 2)),
            },
            "aspect_ratio": {
                "min": float(np.min(aspect_ratios)),
                "max": float(np.max(aspect_ratios)),
                "mean": float(np.round(np.mean(aspect_ratios), 3)),
                "median": float(np.median(aspect_ratios)),
            },
        }

        channel_counts = Counter(channels_list)
        report.channel_distribution = {
            "RGB (3 channels)": channel_counts.get(3, 0),
            "Grayscale (1 channel)": channel_counts.get(1, 0),
            "RGBA / Alpha (4 channels)": channel_counts.get(4, 0),
            "Other": sum(count for ch, count in channel_counts.items() if ch not in (1, 3, 4)),
        }

        # Top resolutions
        res_pairs = Counter([f"{m.width}x{m.height}" for m in valid_meta])
        report.top_resolutions = [
            {"resolution": res, "count": count, "percentage": round((count / len(valid_meta)) * 100, 2)}
            for res, count in res_pairs.most_common(5)
        ]

    # 8. Duplicate Image File Content Detection (via MD5)
    if check_duplicates:
        hash_map: Dict[str, List[ImageMetadata]] = {}
        for m in valid_meta:
            if m.md5_hash:
                hash_map.setdefault(m.md5_hash, []).append(m)

        duplicate_groups = []
        for h, items in hash_map.items():
            if len(items) > 1:
                duplicate_groups.append({
                    "md5_hash": h,
                    "duplicate_count": len(items),
                    "id_codes": [item.id_code for item in items],
                    "file_paths": [item.file_path for item in items],
                })

        report.duplicate_image_files_count = sum(len(g["id_codes"]) - 1 for g in duplicate_groups)
        report.duplicate_image_groups = duplicate_groups

    # 9. Class Distribution & Imbalance Analysis
    total_valid_records = len(df[df["diagnosis"].isin(valid_diagnosis_values)])
    class_counts = df["diagnosis"].value_counts().to_dict()

    for class_id in range(5):
        count = int(class_counts.get(class_id, 0))
        pct = round((count / total_valid_records * 100), 2) if total_valid_records > 0 else 0.0
        report.class_distribution[class_id] = {
            "class_id": class_id,
            "class_name": DR_CLASS_NAMES.get(class_id, "Unknown"),
            "count": count,
            "percentage": pct,
            "is_referable": class_id in REFERABLE_CLASSES,
        }

    # Referable vs Non-referable summary
    non_ref_count = sum(class_counts.get(c, 0) for c in NON_REFERABLE_CLASSES)
    ref_count = sum(class_counts.get(c, 0) for c in REFERABLE_CLASSES)
    non_ref_pct = round((non_ref_count / total_valid_records * 100), 2) if total_valid_records > 0 else 0.0
    ref_pct = round((ref_count / total_valid_records * 100), 2) if total_valid_records > 0 else 0.0
    imbalance_ratio = round(non_ref_count / ref_count, 2) if ref_count > 0 else "N/A"

    report.referable_summary = {
        "non_referable": {
            "classes": list(NON_REFERABLE_CLASSES),
            "label": "Non-Referable DR (Grade 0 & 1)",
            "count": non_ref_count,
            "percentage": non_ref_pct,
        },
        "referable": {
            "classes": list(REFERABLE_CLASSES),
            "label": "Referable DR (Grade 2, 3 & 4)",
            "count": ref_count,
            "percentage": ref_pct,
        },
        "non_ref_to_ref_ratio": f"{imbalance_ratio}:1" if isinstance(imbalance_ratio, (int, float)) else "N/A",
    }

    # 10. Save JSON report if output path provided
    if output_report:
        out_p = Path(output_report)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        logger.info(f"Detailed inspection report saved to: {out_p.resolve()}")

    return report


def print_formatted_report(report: InspectionReport) -> None:
    """
    Renders clean, professional ASCII tables and summary dashboard to the terminal.
    """
    sep = "=" * 82
    sub_sep = "-" * 82

    print("\n" + sep)
    print(" " * 22 + "RETINACARE-AI DATASET INSPECTION REPORT")
    print(sep)
    print(f"Timestamp        : {report.timestamp}")
    print(f"Dataset CSV Path : {report.csv_path}")
    print(f"Images Directory : {report.images_dir}")
    print(sub_sep)
    print(" 1. DATASET OVERVIEW & INTEGRITY")
    print(sub_sep)
    print(f"  * Total CSV Rows               : {report.total_csv_records:,}")
    print(f"  * Total Images Found on Disk   : {report.total_image_files_on_disk:,}")
    print(f"  * Matched Images for Training  : {report.total_matched_images:,}")
    print(f"  * Missing Images (in CSV only) : {report.missing_images_count:,}" + (" [ALERT: Images missing!]" if report.missing_images_count > 0 else " [OK]"))
    print(f"  * Orphan Images (on disk only) : {report.orphan_images_count:,}" + (" [NOTE: Unlabeled images]" if report.orphan_images_count > 0 else " [OK]"))
    print(f"  * Corrupted / Unreadable Files : {report.corrupt_images_count:,}" + (" [ALERT: Corrupt files detected!]" if report.corrupt_images_count > 0 else " [OK]"))
    print(f"  * Duplicate CSV ID Codes       : {report.duplicate_csv_ids_count:,}" + (" [ALERT: Duplicate entries!]" if report.duplicate_csv_ids_count > 0 else " [OK]"))
    print(f"  * Exact Duplicate Image Files  : {report.duplicate_image_files_count:,}" + (" [ALERT: Duplicate images!]" if report.duplicate_image_files_count > 0 else " [OK]"))

    print("\n" + sub_sep)
    print(" 2. CLASS DISTRIBUTION (APTOS 2019 Scale)")
    print(sub_sep)
    print(f" {'ID':<3} | {'Class Name':<42} | {'Count':<8} | {'Percent':<8} | {'Clinical Category':<15}")
    print(f" {'-'*3} | {'-'*42} | {'-'*8} | {'-'*8} | {'-'*15}")
    for cid in range(5):
        c_info = report.class_distribution.get(cid, {})
        c_name = c_info.get("class_name", "Unknown")
        c_count = c_info.get("count", 0)
        c_pct = c_info.get("percentage", 0.0)
        c_ref = "Referable DR" if c_info.get("is_referable") else "Non-Referable"
        print(f"  {cid:<2} | {c_name:<42} | {c_count:>7,} | {c_pct:>7.2f}% | {c_ref:<15}")

    print("\n" + sub_sep)
    print(" 3. CLINICAL REFERABILITY BREAKDOWN")
    print(sub_sep)
    ref_sum = report.referable_summary
    non_ref = ref_sum.get("non_referable", {})
    ref = ref_sum.get("referable", {})
    print(f"  * Non-Referable DR (Grade 0 & 1) : {non_ref.get('count', 0):,} ({non_ref.get('percentage', 0.0):.2f}%)")
    print(f"  * Referable DR     (Grade 2, 3, 4): {ref.get('count', 0):,} ({ref.get('percentage', 0.0):.2f}%)")
    print(f"  * Non-Referable to Referable Ratio: {ref_sum.get('non_ref_to_ref_ratio', 'N/A')}")

    if report.dimension_stats:
        print("\n" + sub_sep)
        print(" 4. IMAGE DIMENSIONS & RESOLUTION PROFILE")
        print(sub_sep)
        w = report.dimension_stats.get("width", {})
        h = report.dimension_stats.get("height", {})
        ar = report.dimension_stats.get("aspect_ratio", {})

        print(f"  * Evaluated Images    : {report.dimension_stats.get('valid_images_evaluated', 0):,}")
        print(f"  * Width  (px)         : Min = {w.get('min')}, Max = {w.get('max')}, Mean = {w.get('mean')}, Median = {w.get('median')}, Std = {w.get('std')}")
        print(f"  * Height (px)         : Min = {h.get('min')}, Max = {h.get('max')}, Mean = {h.get('mean')}, Median = {h.get('median')}, Std = {h.get('std')}")
        print(f"  * Aspect Ratio (W/H)  : Min = {ar.get('min')}, Max = {ar.get('max')}, Mean = {ar.get('mean')}, Median = {ar.get('median')}")

        print("\n  * Color Channel Profile:")
        for ch_name, count in report.channel_distribution.items():
            print(f"      - {ch_name:<28}: {count:,}")

        if report.top_resolutions:
            print("\n  * Top Image Resolutions:")
            for item in report.top_resolutions:
                print(f"      - {item['resolution']:<15}: {item['count']:>6,} images ({item['percentage']:>5.2f}%)")

    # Alerts section
    alerts = []
    if report.missing_images_count > 0:
        alerts.append(f"[ALERT] {report.missing_images_count} images referenced in CSV do not exist on disk.")
    if report.corrupt_images_count > 0:
        alerts.append(f"[ALERT] {report.corrupt_images_count} images are unreadable or corrupted.")
    if report.duplicate_csv_ids_count > 0:
        alerts.append(f"[ALERT] {report.duplicate_csv_ids_count} duplicate ID codes detected in train.csv.")
    if report.duplicate_image_files_count > 0:
        alerts.append(f"[ALERT] {report.duplicate_image_files_count} duplicate image content files detected.")

    print("\n" + sub_sep)
    print(" 5. QUALITY & HEALTH SUMMARY")
    print(sub_sep)
    if not alerts:
        print("  [OK] All dataset integrity checks PASSED without corruption or missing references.")
    else:
        for alert in alerts:
            print(f"  {alert}")

    print(sep + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="RetinaCare-AI Dataset Inspector for Diabetic Retinopathy (APTOS 2019)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv-path",
        type=str,
        default=None,
        help="Path to dataset metadata CSV (train.csv)",
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=None,
        help="Path to retinal images folder (train_images/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/dataset_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="results/dataset_inspection_report.json",
        help="Path to save detailed JSON inspection report",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent worker threads for file reading",
    )
    parser.add_argument(
        "--skip-duplicates",
        action="store_true",
        help="Skip computing MD5 hashes for duplicate detection (faster on huge datasets)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load configuration file if present
    cfg = load_config(args.config)
    ds_cfg = cfg.get("dataset", {})
    insp_cfg = cfg.get("inspection", {})

    # Priority: CLI arguments -> config file -> sensible defaults
    csv_path = (
        args.csv_path
        or ds_cfg.get("raw_csv_path")
        or "data/raw/train.csv"
    )
    images_dir = (
        args.images_dir
        or ds_cfg.get("raw_images_dir")
        or "data/raw/train_images"
    )
    output_report = (
        args.output_report
        or insp_cfg.get("report_filename")
        or "results/dataset_inspection_report.json"
    )
    num_workers = args.workers or insp_cfg.get("num_workers", 4)
    check_duplicates = not args.skip_duplicates

    try:
        report = run_dataset_inspection(
            csv_path=csv_path,
            images_dir=images_dir,
            output_report=output_report,
            num_workers=num_workers,
            check_duplicates=check_duplicates,
        )
        print_formatted_report(report)
    except FileNotFoundError as fnf_err:
        logger.error(str(fnf_err))
        print("\n" + "=" * 75)
        print("[GUIDE] HOW TO GET STARTED:")
        print("  1. Place the APTOS 2019 dataset in:")
        print(f"     - CSV file    : {os.path.abspath(csv_path)}")
        print(f"     - Image folder: {os.path.abspath(images_dir)}")
        print("  2. Or pass custom paths using:")
        print("     python scripts/inspect_dataset.py --csv-path /path/to/train.csv --images-dir /path/to/train_images")
        print("=" * 75 + "\n")
        sys.exit(1)
    except Exception as err:
        logger.exception(f"Unexpected error during dataset inspection: {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
