"""
Build the OBB (Oriented Bounding Box) training dataset from annotations/.

Reads:
  - data/annotations/metadata.csv   (image, ..., split)
  - data/annotations/images/         (1199 source images)
  - data/annotations/obb/            (1199 YOLO OBB label files)

Writes:
  - data/obb/images/{train,val,test}/
  - data/obb/labels/{train,val,test}/
  - data/obb/data.yaml

Usage:
  python scripts/build/build_obb.py
  python scripts/build/build_obb.py --force   # overwrite existing
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from utils.logging import setup_logger

DATA = ROOT / "data"
ANNOTATIONS = DATA / "annotations"
METADATA = ANNOTATIONS / "metadata.csv"
SRC_IMAGES = ANNOTATIONS / "images"
SRC_LABELS = ANNOTATIONS / "obb"
OUTPUT = DATA / "obb"

SPLITS = ["train", "val", "test"]
DATA_YAML = """\
path: .
train: images/train
val: images/val
test: images/test

names:
  0: Odometer
"""

logger = setup_logger("build_obb")


def load_metadata():
    """Load metadata.csv and return list of (image, split) tuples."""
    rows = []
    with open(METADATA, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r["image"], r["split"]))
    return rows


def build(force: bool):
    if not METADATA.exists():
        logger.error(f"{METADATA} not found. Run 03_generate_metadata.py first.")
        sys.exit(1)

    if OUTPUT.exists():
        if not force:
            logger.error(f"{OUTPUT} already exists. Use --force to overwrite.")
            sys.exit(1)
        shutil.rmtree(OUTPUT)

    # Create split directories
    for split in SPLITS:
        (OUTPUT / "images" / split).mkdir(parents=True)
        (OUTPUT / "labels" / split).mkdir(parents=True)

    rows = load_metadata()
    counts = {s: 0 for s in SPLITS}
    errors = []

    for image, split in rows:
        stem = Path(image).stem

        src_img = SRC_IMAGES / image
        src_lbl = SRC_LABELS / f"{stem}.txt"

        if not src_img.exists():
            errors.append(f"Image not found: {src_img}")
            continue
        if not src_lbl.exists():
            errors.append(f"Label not found: {src_lbl}")
            continue

        shutil.copy2(src_img, OUTPUT / "images" / split / image)
        shutil.copy2(src_lbl, OUTPUT / "labels" / split / f"{stem}.txt")
        counts[split] += 1

    # Write data.yaml
    (OUTPUT / "data.yaml").write_text(DATA_YAML)

    # Summary
    total = sum(counts.values())
    logger.info(f"Built OBB dataset: {OUTPUT}")
    logger.info(f"  Total: {total}")
    for s in SPLITS:
        logger.info(f"  {s}: {counts[s]}")

    if errors:
        logger.error(f"ERRORS ({len(errors)}):")
        for e in errors:
            logger.error(f"  {e}")
        sys.exit(1)

    # Sanity check
    expected = len(rows)
    if total != expected:
        logger.warning(f"Expected {expected} files, got {total}")
        sys.exit(1)

    logger.info("Done.")


def main():
    parser = argparse.ArgumentParser(
        description="Build OBB training dataset from annotations/."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing data/obb/ directory.",
    )
    args = parser.parse_args()
    build(args.force)


if __name__ == "__main__":
    main()
