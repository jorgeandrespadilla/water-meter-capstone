"""
Crop odometer regions from full water meter images using YOLO OBB labels.

Reads images from data/annotations/images/ and their corresponding OBB
labels from data/annotations/obb/, crops each OBB region, rotates it to
be axis-aligned, and saves to data/annotations/ocr-crops/.

Skips images with no label, empty labels, or multiple bounding boxes.

Usage:
    python scripts/data/02_crop_odometers.py --force
"""

import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.cropping import OBB_CROP_PADDING, crop_obb
from utils.logging import setup_logger
from utils.orientation import ensure_landscape

ANNOTATIONS_DIR = PROJECT_ROOT / "data" / "annotations"
IMAGES_DIR = ANNOTATIONS_DIR / "images"
LABELS_DIR = ANNOTATIONS_DIR / "obb"
OUTPUT_DIR = ANNOTATIONS_DIR / "ocr-crops"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}


def parse_obb_line(line: str) -> tuple[int, np.ndarray] | None:
    """Parse a single YOLO OBB label line.

    Format: class_id x1 y1 x2 y2 x3 y3 x4 y4 (normalized coordinates).
    Returns (class_id, points_4x2) or None on failure.
    """
    parts = line.strip().split()
    if len(parts) != 9:
        return None
    try:
        class_id = int(parts[0])
        coords = [float(x) for x in parts[1:]]
    except ValueError:
        return None
    points = np.array(coords, dtype=np.float32).reshape(4, 2)
    return class_id, points


def crop_odometers(args: argparse.Namespace) -> None:
    log = setup_logger("02_crop")

    if not IMAGES_DIR.exists():
        log.error(f"Images directory not found: {IMAGES_DIR}")
        return
    if not LABELS_DIR.exists():
        log.error(f"OBB labels directory not found: {LABELS_DIR}")
        return

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        if not args.force:
            log.error(f"Output directory already has files: {OUTPUT_DIR}")
            log.error("Use --force to overwrite.")
            return
        log.info(f"Clearing existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_files = sorted(
        f for f in IMAGES_DIR.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )

    total_found = len(image_files)
    cropped_count = 0
    skipped_no_label = 0
    skipped_multi_bbox = 0
    skipped_degenerate = 0
    error_count = 0

    log.info(f"Found {total_found} images in {IMAGES_DIR}")

    for img_path in image_files:
        label_path = LABELS_DIR / f"{img_path.stem}.txt"

        # Check for label file
        if not label_path.exists():
            skipped_no_label += 1
            log.debug(f"No label: {img_path.name}")
            continue

        # Read and filter blank lines
        lines = [
            ln for ln in label_path.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

        if not lines:
            skipped_no_label += 1
            log.debug(f"Empty label: {img_path.name}")
            continue

        if len(lines) > 1:
            skipped_multi_bbox += 1
            log.warning(f"Multiple bounding boxes ({len(lines)}): {img_path.name}")
            continue

        # Parse OBB
        parsed = parse_obb_line(lines[0])
        if parsed is None:
            error_count += 1
            log.error(f"Failed to parse OBB label: {img_path.name}")
            continue

        _class_id, points = parsed

        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            error_count += 1
            log.error(f"Could not read image: {img_path}")
            continue

        # Crop OBB region
        crop = crop_obb(
            img, points,
            padding=args.padding,
            coord_format="normalized",
        )
        if crop is None:
            skipped_degenerate += 1
            log.warning(f"Degenerate crop: {img_path.name}")
            continue

        crop = ensure_landscape(crop)

        # Save crop
        out_path = OUTPUT_DIR / img_path.name
        ext = img_path.suffix.lower()
        if ext in {".jpg", ".jpeg"}:
            cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            cv2.imwrite(str(out_path), crop)
        cropped_count += 1

    # Summary
    log.info("")
    log.info("=== Summary ===")
    log.info(f"Total images found:    {total_found}")
    log.info(f"Cropped successfully:  {cropped_count}")
    log.info(f"Skipped (no label):    {skipped_no_label}")
    log.info(f"Skipped (multi-bbox):  {skipped_multi_bbox}")
    log.info(f"Skipped (degenerate):  {skipped_degenerate}")
    log.info(f"Errors:                {error_count}")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info("")
    log.info("Next step: python scripts/data/build_ocr.py --force")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop odometer regions from detection dataset using OBB labels"
    )
    parser.add_argument(
        "--padding", type=int, default=OBB_CROP_PADDING,
        help=f"Pixels of padding around the OBB crop (default: {OBB_CROP_PADDING})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output directory",
    )
    args = parser.parse_args()
    crop_odometers(args)
