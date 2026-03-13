"""
Benchmark OCR latency on odometer crops.

Measures three things separately:
1. Model load time
2. Warmup inference time(s)
3. Effective inference latency per image (recognition only)

This script is intended for OCR crops where detection is already solved and we
want to isolate the recognition stage, matching how the project pipeline uses
the OCR recognizer after cropping/preprocessing.

Examples:
    python scripts/tools/benchmark_ocr_latency.py samples/ocr --limit 10

    python scripts/tools/benchmark_ocr_latency.py samples/ocr --limit 20 --mask

    python scripts/tools/benchmark_ocr_latency.py data/ocr/images/test ^
        --model-dir models/ocr-reader/model ^
        --model-name en_PP-OCRv4_mobile_rec ^
        --save-csv logs/ocr_latency.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Skip online model source checks during local benchmarks.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

DEFAULT_OCR_MODEL_DIR = (
    PROJECT_ROOT / "models" / "ocr-reader" / "model"
)
DEFAULT_OCR_MODEL_NAME = "en_PP-OCRv4_mobile_rec"

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp"}
_DIGITS_ONLY = re.compile(r"\D")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark OCR recognition latency on odometer crops.",
    )
    parser.add_argument(
        "input",
        help="Path to a single OCR crop or a directory of OCR crops.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to benchmark.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warmup runs before measuring the dataset (default: 1).",
    )
    parser.add_argument(
        "--mask",
        action="store_true",
        help="Apply LAB red-digit masking before OCR.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_OCR_MODEL_NAME,
        help=f"OCR model name for paddlex.create_model (default: {DEFAULT_OCR_MODEL_NAME}).",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_OCR_MODEL_DIR),
        help="Directory containing the OCR inference model.",
    )
    parser.add_argument(
        "--save-csv",
        metavar="FILE",
        help="Optional CSV path for per-image timings.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Hide per-image rows and print only the summary.",
    )
    return parser.parse_args()


def collect_images(input_path: Path, limit: int | None) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Not an image file: {input_path}")
        images = [input_path]
    elif input_path.is_dir():
        images = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise ValueError(f"No images found in directory: {input_path}")
    else:
        raise ValueError(f"Path not found: {input_path}")

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be > 0")
        images = images[:limit]

    return images


def load_image(image_path: Path, mask: bool):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if not mask:
        return image

    from utils.masking import apply_red_mask

    return apply_red_mask(image, include_stats=False).image


def extract_rec_output(res) -> tuple[str, float]:
    rec_text = ""
    rec_score = 0.0

    if hasattr(res, "json"):
        data = res.json
        if "res" in data:
            rec_text = data["res"].get("rec_text", "")
            rec_score = data["res"].get("rec_score", 0.0)
        else:
            rec_text = data.get("rec_text", "")
            rec_score = data.get("rec_score", 0.0)
    elif isinstance(res, dict):
        rec_text = res.get("rec_text", "")
        rec_score = res.get("rec_score", 0.0)
    else:
        rec_text = getattr(res, "rec_text", "")
        rec_score = getattr(res, "rec_score", 0.0)

    return str(rec_text), float(rec_score)


def digits_only(text: str) -> str:
    return _DIGITS_ONLY.sub("", text)


def run_rec_only(model, image) -> tuple[str, float, float]:
    start = time.perf_counter()
    output = model.predict(input=image, batch_size=1)

    raw_text = ""
    rec_score = 0.0
    for res in output:
        raw_text, rec_score = extract_rec_output(res)
        break

    elapsed_ms = (time.perf_counter() - start) * 1000
    return raw_text, rec_score, elapsed_ms


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def print_summary(
    rows: list[dict],
    load_s: float,
    warmup_ms: list[float],
    *,
    model_name: str,
    model_dir: Path,
    mask: bool,
) -> None:
    timings = [row["time_ms"] for row in rows]
    with_text = sum(1 for row in rows if row["text"])

    summary = {
        "images": len(rows),
        "with_text": with_text,
        "empty": len(rows) - with_text,
        "load_s": round(load_s, 3),
        "warmup_ms": [round(v, 1) for v in warmup_ms],
        "mean_ms": round(statistics.mean(timings), 1),
        "median_ms": round(statistics.median(timings), 1),
        "p90_ms": round(percentile(timings, 0.90), 1),
        "p95_ms": round(percentile(timings, 0.95), 1),
        "min_ms": round(min(timings), 1),
        "max_ms": round(max(timings), 1),
    }

    print("\n" + "=" * 72)
    print("OCR LATENCY BENCHMARK")
    print("-" * 72)
    print(f"Model name:         {model_name}")
    print(f"Model dir:          {model_dir}")
    print(f"Masking:            {'ON' if mask else 'OFF'}")
    print(f"Images measured:    {summary['images']}")
    print(f"Images with text:   {summary['with_text']}")
    print(f"Model load time:    {summary['load_s']:.3f} s")
    if warmup_ms:
        print(f"Warmup runs:        {', '.join(f'{v:.1f} ms' for v in summary['warmup_ms'])}")
    else:
        print("Warmup runs:        none")
    print("-" * 72)
    print(f"Mean latency:       {summary['mean_ms']:.1f} ms")
    print(f"Median latency:     {summary['median_ms']:.1f} ms")
    print(f"P90 latency:        {summary['p90_ms']:.1f} ms")
    print(f"P95 latency:        {summary['p95_ms']:.1f} ms")
    print(f"Min latency:        {summary['min_ms']:.1f} ms")
    print(f"Max latency:        {summary['max_ms']:.1f} ms")
    print("=" * 72)
    print("JSON summary:")
    print(json.dumps(summary, ensure_ascii=False))


def save_csv(rows: list[dict], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "file",
                "text",
                "raw_text",
                "confidence",
                "time_ms",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    model_dir = Path(args.model_dir)

    try:
        image_paths = collect_images(input_path, args.limit)
    except ValueError as exc:
        print(f"Error: {exc}")
        sys.exit(1)

    from paddlex import create_model

    print(f"Loading OCR model: {args.model_name}")
    print(f"Model dir: {model_dir}")
    load_start = time.perf_counter()
    model = create_model(
        model_name=args.model_name,
        model_dir=str(model_dir),
    )
    load_s = time.perf_counter() - load_start

    loaded_images = []
    for image_path in image_paths:
        try:
            loaded_images.append((image_path, load_image(image_path, args.mask)))
        except ValueError as exc:
            print(f"Skipping {image_path.name}: {exc}")

    if not loaded_images:
        print("Error: no readable images to benchmark.")
        sys.exit(1)

    warmup_ms = []
    for idx in range(args.warmup):
        _, image = loaded_images[idx % len(loaded_images)]
        _, _, elapsed_ms = run_rec_only(model, image)
        warmup_ms.append(elapsed_ms)

    rows = []
    for image_path, image in loaded_images:
        raw_text, rec_score, elapsed_ms = run_rec_only(model, image)
        row = {
            "file": image_path.name,
            "text": digits_only(raw_text),
            "raw_text": raw_text,
            "confidence": round(rec_score, 4),
            "time_ms": round(elapsed_ms, 1),
        }
        rows.append(row)
        if not args.quiet:
            print(json.dumps(row, ensure_ascii=False))

    print_summary(
        rows,
        load_s,
        warmup_ms,
        model_name=args.model_name,
        model_dir=model_dir,
        mask=args.mask,
    )

    if args.save_csv:
        csv_path = Path(args.save_csv)
        save_csv(rows, csv_path)
        print(f"Saved CSV: {csv_path}")


if __name__ == "__main__":
    main()
