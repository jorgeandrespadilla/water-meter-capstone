#!/usr/bin/env python3
"""
Organize Classification Images

Reads data/classification/labels.csv and copies images from data/cleaned/images
into class subdirectories.

Output: data/classification/{valid,invalid,ambiguous}/

Usage:
    python scripts/data/01_organize_classification.py
"""

import csv
import shutil
import sys
from pathlib import Path
from collections import Counter

# =============================================================================
# Configuration
# =============================================================================
# Base directory (repo root)
BASE_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE_DIR))
from utils.logging import setup_logger

DATA_DIR = BASE_DIR / "data"

INPUT_CSV = DATA_DIR / "classification" / "labels.csv"
SOURCE_DIR = DATA_DIR / "cleaned" / "images"
OUTPUT_DIR = DATA_DIR / "classification"

CLASSIFICATION_CATEGORIES = {"valid", "invalid", "ambiguous"}


def main():
    logger = setup_logger("01_organize_classification")

    logger.info("Organizing classification images...")

    # Validate inputs
    if not INPUT_CSV.exists():
        logger.error(f"CSV not found: {INPUT_CSV}")
        logger.error("   Export and place classification labels first.")
        return 1

    if not SOURCE_DIR.exists():
        logger.error(f"Source images not found: {SOURCE_DIR}")
        return 1

    # Check if output directory exists
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        logger.warning(f"Output directory already exists: {OUTPUT_DIR}")
        logger.info("Existing images will be kept, new ones will be copied.")
        response = input("Continue? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            logger.info("Cancelled.")
            return 0

    # Load CSV
    logger.info(f"Loading: {INPUT_CSV}")
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        data = list(reader)

    logger.info(f"Found {len(data)} entries")

    # Create output directories
    for class_name in CLASSIFICATION_CATEGORIES:
        (OUTPUT_DIR / class_name).mkdir(parents=True, exist_ok=True)

    # Copy images
    logger.info("Copying images...")

    success = 0
    skipped = 0
    errors = 0

    for item in data:
        filename = item["filename"]
        class_name = item["class"]

        src = SOURCE_DIR / filename
        dst = OUTPUT_DIR / class_name / filename

        if not src.exists():
            errors += 1
            continue

        if dst.exists():
            skipped += 1
            continue

        try:
            shutil.copy2(src, dst)
            success += 1
        except Exception as e:
            logger.error(f"Error copying {filename}: {e}")
            errors += 1

    # Summary
    logger.info(f"Copied: {success}, Skipped: {skipped}, Errors: {errors}")

    # Final counts
    logger.info("Result:")
    for class_name in sorted(CLASSIFICATION_CATEGORIES):
        class_dir = OUTPUT_DIR / class_name
        count = len(list(class_dir.glob("*.jpg")))
        logger.info(f"  {class_dir}: {count} images")

    logger.info(f"Images organized in: {OUTPUT_DIR}")
    logger.info("Done!")
    return 0


if __name__ == "__main__":
    exit(main())
