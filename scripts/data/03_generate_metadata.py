"""
Generate annotations/metadata.csv with stratified split.

Reads:
  - data/annotations/ocr-labels.csv   (image, reading, rotation_applied)
  - data/annotations/bgblack.txt      (manual annotation: images with black background)

Writes:
  - data/annotations/metadata.csv

Columns: image, background, n_digits, has_decimal, group, split

Usage:
  # Default (seed=42, 80/10/10):
  python scripts/data/03_generate_metadata.py

  # Custom seed:
  python scripts/data/03_generate_metadata.py --seed 123

  # Custom ratios:
  python scripts/data/03_generate_metadata.py --train-ratio 0.7 --val-ratio 0.15
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from utils.logging import setup_logger

ROOT = PROJECT_ROOT / "data"
ANNOTATIONS = ROOT / "annotations"
OCR_LABELS = ANNOTATIONS / "ocr-labels.csv"
BGBLACK_FILE = ANNOTATIONS / "bgblack.txt"
OUTPUT = ANNOTATIONS / "metadata.csv"

logger = setup_logger("03_generate_metadata")


def load_bgblack_set():
    if not BGBLACK_FILE.exists():
        logger.error(f"{BGBLACK_FILE} not found.")
        sys.exit(1)
    with open(BGBLACK_FILE) as f:
        return {line.strip() for line in f if line.strip()}


def parse_reading(reading_str):
    """Extract n_digits (integer part) and has_decimal from a reading string."""
    if "." in reading_str:
        integer_part = reading_str.split(".")[0]
        has_decimal = True
    else:
        integer_part = reading_str
        has_decimal = False
    return len(integer_part), has_decimal


def build_rows(bgblack_set):
    rows = []
    with open(OCR_LABELS) as f:
        reader = csv.DictReader(f)
        for r in reader:
            image = r["image"]
            n_digits, has_decimal = parse_reading(r["reading"])
            background = "bgblack" if image in bgblack_set else "bgwhite"
            dec_str = "dec" if has_decimal else "nodec"
            group = f"{background}_{n_digits}d_{dec_str}"
            rows.append({
                "image": image,
                "background": background,
                "n_digits": n_digits,
                "has_decimal": has_decimal,
                "group": group,
            })
    return rows


def stratified_split(rows, seed, train_ratio, val_ratio):
    """Apply stratified split. Handle small groups manually."""
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        logger.error(f"train_ratio + val_ratio = {train_ratio + val_ratio} >= 1.0")
        sys.exit(1)

    val_test_ratio = 1.0 - train_ratio
    val_of_remaining = val_ratio / val_test_ratio

    group_counts = Counter(r["group"] for r in rows)

    stratifiable = [r for r in rows if group_counts[r["group"]] >= 3]
    manual = [r for r in rows if group_counts[r["group"]] < 3]

    groups = [r["group"] for r in stratifiable]
    train_rows, temp_rows, _, temp_groups = train_test_split(
        stratifiable, groups,
        test_size=val_test_ratio, stratify=groups, random_state=seed,
    )
    val_rows, test_rows = train_test_split(
        temp_rows,
        test_size=(1.0 - val_of_remaining), stratify=temp_groups, random_state=seed,
    )

    for r in train_rows:
        r["split"] = "train"
    for r in val_rows:
        r["split"] = "val"
    for r in test_rows:
        r["split"] = "test"

    # Manual assignment for groups with <3 samples
    small_groups = {}
    for r in manual:
        small_groups.setdefault(r["group"], []).append(r)
    for members in small_groups.values():
        if len(members) == 1:
            members[0]["split"] = "train"
        elif len(members) == 2:
            members[0]["split"] = "train"
            members[1]["split"] = "val"

    all_rows = train_rows + val_rows + test_rows + manual
    all_rows.sort(key=lambda r: r["image"])
    return all_rows


def write_csv(rows, output):
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["image", "background", "n_digits", "has_decimal", "group", "split"]
        )
        writer.writeheader()
        writer.writerows(rows)


def log_summary(rows):
    n = len(rows)
    logger.info(f"Total rows: {n}")

    split_counts = Counter(r["split"] for r in rows)
    logger.info("Split distribution:")
    for s in ["train", "val", "test"]:
        c = split_counts.get(s, 0)
        logger.info(f"  {s}: {c} ({c / n * 100:.1f}%)")

    group_counts = Counter(r["group"] for r in rows)
    logger.info(f"Groups ({len(group_counts)} unique):")
    for group, count in sorted(group_counts.items(), key=lambda x: -x[1]):
        splits = Counter(r["split"] for r in rows if r["group"] == group)
        split_str = ", ".join(f"{s}={splits.get(s, 0)}" for s in ["train", "val", "test"])
        logger.info(f"  {group}: {count} ({split_str})")

    bg = Counter(r["background"] for r in rows)
    dec = Counter(r["has_decimal"] for r in rows)
    logger.info(f"Background: bgwhite={bg['bgwhite']}, bgblack={bg.get('bgblack', 0)}")
    logger.info(f"Decimal: dec={dec[True]}, nodec={dec[False]}")

    groups_no_val = [g for g in group_counts if not any(r["group"] == g and r["split"] == "val" for r in rows)]
    groups_no_test = [g for g in group_counts if not any(r["group"] == g and r["split"] == "test" for r in rows)]
    if groups_no_val:
        logger.warning(f"Groups with 0 val samples: {groups_no_val}")
    if groups_no_test:
        logger.warning(f"Groups with 0 test samples: {groups_no_test}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotations/metadata.csv with stratified split."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8).")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Val split ratio (default: 0.1).")
    args = parser.parse_args()

    bgblack = load_bgblack_set()
    logger.info(f"Background flags: {BGBLACK_FILE} ({len(bgblack)} images)")
    logger.info(f"OCR labels: {OCR_LABELS}")
    logger.info(f"Seed: {args.seed}, Split: {args.train_ratio}/{args.val_ratio}/{1.0 - args.train_ratio - args.val_ratio:.1f}")

    rows = build_rows(bgblack)
    rows = stratified_split(rows, args.seed, args.train_ratio, args.val_ratio)
    write_csv(rows, OUTPUT)

    logger.info(f"Wrote {OUTPUT}")
    log_summary(rows)


if __name__ == "__main__":
    main()
