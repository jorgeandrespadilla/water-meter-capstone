"""
Build the OCR training dataset from annotations/.

Reads:
  - data/annotations/metadata.csv      (image, background, n_digits, has_decimal, group, split)
  - data/annotations/ocr-labels.csv    (image, reading, rotation_applied)
  - data/annotations/ocr-crops/        (1199 cropped odometer images)

Writes:
  - data/ocr/images/{train,val,test}/   rotated + masked crops
  - data/ocr/{train,val,test}.txt       PaddleOCR label files (images/<name>\t<label>)
  - data/ocr/mask_decisions.csv         per-image masking decision report

Processing per image:
  1. Apply rotation (0/90/180/270) from ocr-labels.csv
  2. Apply red masking on images with decimals (unless --no-mask)
  3. Save processed image
  4. Write integer-only label (decimal part stripped)

Usage:
  python scripts/data/build_ocr.py --force
  python scripts/data/build_ocr.py --force --no-mask
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.logging import setup_logger

DATA = ROOT / "data"
ANNOTATIONS = DATA / "annotations"
METADATA = ANNOTATIONS / "metadata.csv"
OCR_LABELS = ANNOTATIONS / "ocr-labels.csv"
SRC_CROPS = ANNOTATIONS / "ocr-crops"
OUTPUT = DATA / "ocr"

SPLITS = ["train", "val", "test"]

VALID_ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}

logger = setup_logger("build_ocr")


def apply_rotation(image: np.ndarray, degrees: int) -> np.ndarray:
    """Rotate image by 0/90/180/270 degrees."""
    rotation_code = VALID_ROTATIONS.get(degrees)
    if rotation_code is None:
        return image
    return cv2.rotate(image, rotation_code)


def load_metadata():
    """Load metadata.csv -> dict[image] = split."""
    splits = {}
    with open(METADATA, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            splits[r["image"]] = r["split"]
    return splits


def load_ocr_labels():
    """Load ocr-labels.csv -> dict[image] = {reading, rotation_applied}."""
    labels = {}
    with open(OCR_LABELS, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            labels[r["image"]] = {
                "reading": r["reading"].strip(),
                "rotation_applied": r.get("rotation_applied", "0").strip(),
            }
    return labels


def integer_label(reading: str) -> str:
    """Strip decimal part: '9747.219' -> '9747', '07205' -> '07205'."""
    return reading.split(".")[0]


def has_decimal(reading: str) -> bool:
    return "." in reading


def build(force: bool, no_mask: bool):
    # Lazy import — only needed if masking is actually applied
    from utils.masking import apply_red_mask

    if not METADATA.exists():
        logger.error(f"{METADATA} not found. Run 03_generate_metadata.py first.")
        sys.exit(1)
    if not OCR_LABELS.exists():
        logger.error(f"{OCR_LABELS} not found.")
        sys.exit(1)

    if OUTPUT.exists():
        if not force:
            logger.error(f"{OUTPUT} already exists. Use --force to overwrite.")
            sys.exit(1)
        shutil.rmtree(OUTPUT)

    # Create split directories
    for split in SPLITS:
        (OUTPUT / "images" / split).mkdir(parents=True)

    split_map = load_metadata()
    ocr_labels = load_ocr_labels()

    counts = {s: 0 for s in SPLITS}
    rotated_count = 0
    masked_count = 0
    errors = []
    split_entries = {s: [] for s in SPLITS}
    mask_decisions = []

    for image, split in sorted(split_map.items()):
        if image not in ocr_labels:
            errors.append(f"No OCR label for: {image}")
            continue

        src_path = SRC_CROPS / image
        if not src_path.exists():
            errors.append(f"Crop not found: {src_path}")
            continue

        label_info = ocr_labels[image]
        reading = label_info["reading"]

        if not reading:
            errors.append(f"Empty reading for: {image}")
            continue

        img = cv2.imread(str(src_path))
        if img is None:
            errors.append(f"Could not read image: {src_path}")
            continue

        # 1. Apply rotation
        rotation_str = label_info["rotation_applied"]
        degrees = 0
        if rotation_str and rotation_str != "0":
            try:
                degrees = int(rotation_str)
            except ValueError:
                errors.append(f"Invalid rotation '{rotation_str}' for {image}")
                continue
            if degrees not in VALID_ROTATIONS:
                errors.append(f"Invalid rotation {degrees} for {image}")
                continue

        img = apply_rotation(img, degrees)
        if degrees != 0:
            rotated_count += 1

        # 2. Apply red masking for images with decimal digits
        if has_decimal(reading) and not no_mask:
            result = apply_red_mask(img)
            img = result.image
            masked_count += 1
            logger.debug(f"  {image}: {result.reason}")

            # Record decision for CSV report
            mask_decisions.append({
                "image": image,
                "split": split,
                "reason": result.reason,
                "red_pct": f"{result.stats.get('red_pixel_pct', 0):.2f}",
                "median_a": f"{result.stats.get('median_a', 0):.0f}",
            })

        # 3. Save processed image
        dst_path = OUTPUT / "images" / split / image
        cv2.imwrite(str(dst_path), img)

        # 4. Record label entry (integer-only)
        label = integer_label(reading)
        split_entries[split].append((image, label))
        counts[split] += 1

    # Write PaddleOCR split files
    for split in SPLITS:
        split_file = OUTPUT / f"{split}.txt"
        with open(split_file, "w", encoding="utf-8") as f:
            for image, label in split_entries[split]:
                f.write(f"images/{split}/{image}\t{label}\n")

    # Write mask decisions CSV
    if mask_decisions:
        csv_path = OUTPUT / "mask_decisions.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "image", "split", "reason", "red_pct", "median_a",
            ])
            writer.writeheader()
            writer.writerows(mask_decisions)
        logger.info(f"  Mask decisions: {csv_path}")

    # Summary
    total = sum(counts.values())
    logger.info(f"Built OCR dataset: {OUTPUT}")
    logger.info(f"  Total: {total}")
    for s in SPLITS:
        logger.info(f"  {s}: {counts[s]}")
    logger.info(f"  Rotated: {rotated_count}")
    logger.info(f"  Masking: {'disabled' if no_mask else 'enabled'}")

    if masked_count > 0:
        logger.info(f"  Red-masked: {masked_count}")

    if errors:
        logger.error(f"ERRORS ({len(errors)}):")
        for e in errors:
            logger.error(f"  {e}")
        sys.exit(1)

    expected = len(split_map)
    if total != expected:
        logger.warning(f"Expected {expected} images, got {total}")
        sys.exit(1)

    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Build OCR training dataset from annotations/."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing data/ocr/ directory.",
    )
    parser.add_argument(
        "--no-mask", action="store_true",
        help="Skip red masking — output raw rotated crops.",
    )
    args = parser.parse_args()
    build(args.force, args.no_mask)


if __name__ == "__main__":
    main()
