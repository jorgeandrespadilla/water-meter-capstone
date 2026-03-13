"""Evaluate the full water meter pipeline on a held-out split.

Runs the end-to-end pipeline over raw images from `data/obb/images/<split>`,
compares predictions against integer OCR labels from `data/ocr/<split>.txt`,
and exports:

- `predictions.csv`: per-image results
- `failures.csv`: subset of non-matching or unreadable cases
- `summary.json`: machine-readable aggregate metrics
- `summary.md`: human-readable report for documentation

The primary comparison uses the normalized integer reading (digits only,
without leading zeros). A secondary strict match is also reported to quantify
cases where the business value is correct but the zero padding differs.

Examples:
    python scripts/eval/evaluate_end_to_end.py

    python scripts/eval/evaluate_end_to_end.py --split test --threshold 0.7

    python scripts/eval/evaluate_end_to_end.py --split val --limit 20
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Avoid online source checks when loading local Paddle models.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from app.pipeline import WaterMeterPipeline
from scripts.tools.eval.evaluate_pilot import (
    digits_only,
    normalize_business_reading,
    safe_mean,
    safe_median,
    safe_percentile,
    safe_rate,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the full pipeline on a held-out split.",
    )
    parser.add_argument(
        "--data-root",
        default=str(PROJECT_ROOT / "data"),
        help="Dataset root containing `obb/`, `ocr/` and `annotations/`.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Held-out split to evaluate (default: test).",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "logs" / "end_to_end_eval"),
        help="Directory where evaluation artifacts will be written.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Validation threshold passed to the pipeline (default: 0.7).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Evaluate only the first N rows (for smoke tests).",
    )
    parser.add_argument(
        "--images",
        default=None,
        help="Comma-separated list of image names to evaluate (e.g. 00060.jpg,00076.jpg).",
    )
    return parser.parse_args()


def load_split_labels(labels_path: Path, selected_images: set[str] | None) -> list[dict]:
    rows: list[dict] = []
    with labels_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rel_path, expected_raw = line.split("\t", 1)
            image_name = Path(rel_path).name
            if selected_images and image_name not in selected_images:
                continue
            rows.append({
                "image": image_name,
                "expected_raw": expected_raw.strip(),
            })
    return rows


def load_metadata(metadata_path: Path) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    with metadata_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            metadata[str(row["image"]).strip()] = {
                "background": str(row.get("background", "")).strip(),
                "n_digits": str(row.get("n_digits", "")).strip(),
                "has_decimal": str(row.get("has_decimal", "")).strip(),
                "group": str(row.get("group", "")).strip(),
                "split": str(row.get("split", "")).strip(),
            }
    return metadata


def classify_outcome(
    *,
    image_found: bool,
    status: str,
    predicted_norm: str | None,
    business_match: bool,
) -> str:
    if not image_found:
        return "missing_image"
    if status == "no_detection":
        return "no_detection"
    if predicted_norm is None:
        return "no_reading"
    if business_match:
        return "match"
    return "mismatch"


def subgroup_stats(rows: list[dict], key: str) -> dict[str, dict]:
    grouped = defaultdict(lambda: {"total": 0, "matches": 0, "readable": 0})
    for row in rows:
        subgroup = row.get(key) or "unknown"
        grouped[subgroup]["total"] += 1
        grouped[subgroup]["matches"] += int(row["business_match"])
        grouped[subgroup]["readable"] += int(row["predicted_norm"] is not None)

    return {
        str(name): {
            "total": stats["total"],
            "match_rate": safe_rate(stats["matches"], stats["total"]),
            "readable_rate": safe_rate(stats["readable"], stats["total"]),
        }
        for name, stats in sorted(grouped.items(), key=lambda item: str(item[0]))
    }


def build_summary(predictions: list[dict], args: argparse.Namespace) -> dict:
    total_rows = len(predictions)
    found_rows = [row for row in predictions if row["image_found"]]
    matched_rows = [row for row in found_rows if row["business_match"]]
    strict_matched_rows = [row for row in found_rows if row["strict_match"]]
    readable_rows = [row for row in found_rows if row["predicted_norm"] is not None]
    valid_rows = [row for row in found_rows if row["status"] == "valid"]
    review_rows = [row for row in found_rows if row["status"] == "needs_review"]
    no_detection_rows = [row for row in found_rows if row["status"] == "no_detection"]
    valid_correct_rows = [row for row in valid_rows if row["business_match"]]
    valid_incorrect_rows = [row for row in valid_rows if not row["business_match"]]
    correct_but_review_rows = [
        row for row in found_rows
        if row["status"] != "valid" and row["business_match"]
    ]

    timings_total = [row["total_ms"] for row in found_rows if row["total_ms"] is not None]
    timings_detection = [row["detection_ms"] for row in found_rows if row["detection_ms"] is not None]
    timings_pre = [row["preprocessing_ms"] for row in found_rows if row["preprocessing_ms"] is not None]
    timings_ocr = [row["ocr_ms"] for row in found_rows if row["ocr_ms"] is not None]

    status_counts = Counter(row["status"] for row in found_rows)
    outcome_counts = Counter(row["outcome"] for row in predictions)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "data_root": str(args.data_root),
        "split": args.split,
        "threshold": args.threshold,
        "total_rows": total_rows,
        "images_found": len(found_rows),
        "images_missing": total_rows - len(found_rows),
        "metrics": {
            "end_to_end_accuracy": safe_rate(len(matched_rows), total_rows),
            "strict_match_rate": safe_rate(len(strict_matched_rows), total_rows),
            "readable_rate": safe_rate(len(readable_rows), len(found_rows)),
            "accuracy_when_readable": safe_rate(
                sum(row["business_match"] for row in readable_rows),
                len(readable_rows),
            ),
            "auto_validation_rate": safe_rate(len(valid_rows), len(found_rows)),
            "auto_validation_precision": safe_rate(len(valid_correct_rows), len(valid_rows)),
            "auto_validation_error_rate": safe_rate(len(valid_incorrect_rows), len(valid_rows)),
            "review_rate": safe_rate(len(review_rows), len(found_rows)),
            "no_detection_rate": safe_rate(len(no_detection_rows), len(found_rows)),
            "correct_but_reviewed_rate": safe_rate(len(correct_but_review_rows), len(found_rows)),
        },
        "counts": {
            "status": dict(status_counts),
            "outcome": dict(outcome_counts),
            "valid_correct": len(valid_correct_rows),
            "valid_incorrect": len(valid_incorrect_rows),
            "correct_but_reviewed": len(correct_but_review_rows),
        },
        "latency_ms": {
            "total": {
                "mean": safe_mean(timings_total),
                "median": safe_median(timings_total),
                "p95": safe_percentile(timings_total, 95),
            },
            "detection": {
                "mean": safe_mean(timings_detection),
                "median": safe_median(timings_detection),
                "p95": safe_percentile(timings_detection, 95),
            },
            "preprocessing": {
                "mean": safe_mean(timings_pre),
                "median": safe_median(timings_pre),
                "p95": safe_percentile(timings_pre, 95),
            },
            "ocr": {
                "mean": safe_mean(timings_ocr),
                "median": safe_median(timings_ocr),
                "p95": safe_percentile(timings_ocr, 95),
            },
        },
        "subgroups": {
            "background": subgroup_stats(found_rows, "background"),
            "has_decimal": subgroup_stats(found_rows, "has_decimal"),
            "n_digits": subgroup_stats(found_rows, "n_digits"),
        },
    }
    return summary


def render_summary_md(summary: dict) -> str:
    metrics = summary["metrics"]
    counts = summary["counts"]
    latency = summary["latency_ms"]

    def pct(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value:.2%}"

    lines = [
        "# Evaluacion End-to-End - Resumen",
        "",
        f"- Fecha: {summary['generated_at']}",
        f"- Split: `{summary['split']}`",
        f"- Threshold de validacion: `{summary['threshold']}`",
        f"- Casos totales: **{summary['total_rows']}**",
        f"- Imagenes encontradas: **{summary['images_found']}**",
        f"- Imagenes faltantes: **{summary['images_missing']}**",
        "",
        "## Metricas principales",
        "",
        "| Metrica | Valor |",
        "| --- | ---: |",
        f"| End-to-end accuracy | {pct(metrics['end_to_end_accuracy'])} |",
        f"| Strict match rate | {pct(metrics['strict_match_rate'])} |",
        f"| Readable rate | {pct(metrics['readable_rate'])} |",
        f"| Accuracy cuando el pipeline devuelve lectura | {pct(metrics['accuracy_when_readable'])} |",
        f"| Auto-validation rate (`status=valid`) | {pct(metrics['auto_validation_rate'])} |",
        f"| Precision de auto-validacion | {pct(metrics['auto_validation_precision'])} |",
        f"| Error rate de auto-validacion | {pct(metrics['auto_validation_error_rate'])} |",
        f"| Review rate (`status=needs_review`) | {pct(metrics['review_rate'])} |",
        f"| No-detection rate | {pct(metrics['no_detection_rate'])} |",
        f"| Casos correctos enviados a revision | {pct(metrics['correct_but_reviewed_rate'])} |",
        "",
        "## Conteos",
        "",
        f"- Status: `{json.dumps(counts['status'], ensure_ascii=False, sort_keys=True)}`",
        f"- Outcome: `{json.dumps(counts['outcome'], ensure_ascii=False, sort_keys=True)}`",
        f"- Auto-validaciones correctas: **{counts['valid_correct']}**",
        f"- Auto-validaciones incorrectas: **{counts['valid_incorrect']}**",
        f"- Correctos pero enviados a revision: **{counts['correct_but_reviewed']}**",
        "",
        "## Latencia (ms)",
        "",
        "| Etapa | Mean | Median | P95 |",
        "| --- | ---: | ---: | ---: |",
        f"| Total | {latency['total']['mean'] or 'N/A'} | {latency['total']['median'] or 'N/A'} | {latency['total']['p95'] or 'N/A'} |",
        f"| Detection | {latency['detection']['mean'] or 'N/A'} | {latency['detection']['median'] or 'N/A'} | {latency['detection']['p95'] or 'N/A'} |",
        f"| Preprocessing | {latency['preprocessing']['mean'] or 'N/A'} | {latency['preprocessing']['median'] or 'N/A'} | {latency['preprocessing']['p95'] or 'N/A'} |",
        f"| OCR | {latency['ocr']['mean'] or 'N/A'} | {latency['ocr']['median'] or 'N/A'} | {latency['ocr']['p95'] or 'N/A'} |",
        "",
        "## Subgrupos",
        "",
    ]

    for subgroup_name, subgroup_rows in summary["subgroups"].items():
        lines.extend([
            f"### {subgroup_name}",
            "",
            "| Grupo | Total | Match rate | Readable rate |",
            "| --- | ---: | ---: | ---: |",
        ])
        for group_value, stats in subgroup_rows.items():
            lines.append(
                f"| {group_value} | {stats['total']} | {pct(stats['match_rate'])} | {pct(stats['readable_rate'])} |"
            )
        lines.append("")

    lines.extend([
        "## Nota metodologica",
        "",
        "La metrica principal compara la lectura entera normalizada,",
        "ignorando ceros a la izquierda. Esto refleja mejor la validacion",
        "operativa del sistema. El strict match se reporta aparte para",
        "cuantificar diferencias de padding.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images_dir = args.data_root / "obb" / "images" / args.split
    labels_file = args.data_root / "ocr" / f"{args.split}.txt"
    metadata_file = args.data_root / "annotations" / "metadata.csv"

    if not images_dir.exists():
        raise FileNotFoundError(f"Split images dir not found: {images_dir}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Split labels file not found: {labels_file}")
    if not metadata_file.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    selected_images = None
    if args.images:
        selected_images = {item.strip() for item in args.images.split(",") if item.strip()}

    rows = load_split_labels(labels_file, selected_images)
    metadata = load_metadata(metadata_file)
    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        raise RuntimeError("No rows selected for evaluation.")

    print(f"Selected rows: {len(rows)}", flush=True)
    print("Loading full pipeline...", flush=True)
    pipeline = WaterMeterPipeline()
    print("Pipeline loaded. Starting evaluation...", flush=True)

    predictions: list[dict] = []
    failures: list[dict] = []

    for index, row in enumerate(rows, start=1):
        image_name = row["image"]
        image_path = images_dir / image_name
        info = metadata.get(image_name, {})
        expected_digits = digits_only(row["expected_raw"])
        expected_norm = normalize_business_reading(row["expected_raw"])
        expected_digit_len = len(expected_norm) if expected_norm is not None else 0

        print(f"[{index}/{len(rows)}] IMAGE={image_name}", flush=True)

        base_record = {
            "image": image_name,
            "split": args.split,
            "background": info.get("background", ""),
            "n_digits": info.get("n_digits", ""),
            "has_decimal": info.get("has_decimal", ""),
            "group": info.get("group", ""),
            "expected_raw": row["expected_raw"],
            "expected_digits": expected_digits,
            "expected_norm": expected_norm,
            "expected_digit_len": expected_digit_len,
            "image_path": str(image_path),
            "image_found": image_path.exists(),
        }

        if not image_path.exists():
            record = {
                **base_record,
                "reading": None,
                "predicted_digits": None,
                "predicted_norm": None,
                "strict_match": False,
                "business_match": False,
                "status": "missing_image",
                "outcome": "missing_image",
                "global_confidence": None,
                "detection_confidence": None,
                "recognition_confidence": None,
                "orientation_confidence": None,
                "orientation_method": None,
                "total_ms": None,
                "detection_ms": None,
                "preprocessing_ms": None,
                "ocr_ms": None,
                "warnings": "image_not_found",
            }
            predictions.append(record)
            failures.append(record)
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            record = {
                **base_record,
                "reading": None,
                "predicted_digits": None,
                "predicted_norm": None,
                "strict_match": False,
                "business_match": False,
                "status": "image_decode_error",
                "outcome": "missing_image",
                "global_confidence": None,
                "detection_confidence": None,
                "recognition_confidence": None,
                "orientation_confidence": None,
                "orientation_method": None,
                "total_ms": None,
                "detection_ms": None,
                "preprocessing_ms": None,
                "ocr_ms": None,
                "warnings": "image_decode_error",
            }
            predictions.append(record)
            failures.append(record)
            continue

        result = pipeline.predict(image, validation_threshold=args.threshold, collect_intermediates=False)
        predicted_digits = digits_only(result.reading)
        predicted_norm = normalize_business_reading(result.reading)
        strict_match = predicted_digits == expected_digits if predicted_digits is not None else False
        business_match = predicted_norm == expected_norm if predicted_norm is not None else False
        outcome = classify_outcome(
            image_found=True,
            status=result.status,
            predicted_norm=predicted_norm,
            business_match=business_match,
        )

        record = {
            **base_record,
            "reading": result.reading,
            "predicted_digits": predicted_digits,
            "predicted_norm": predicted_norm,
            "strict_match": strict_match,
            "business_match": business_match,
            "status": result.status,
            "outcome": outcome,
            "global_confidence": result.global_confidence,
            "detection_confidence": result.detection_confidence,
            "recognition_confidence": result.recognition_confidence,
            "orientation_confidence": result.orientation_confidence,
            "orientation_method": result.orientation_method,
            "total_ms": result.timing_ms.get("total"),
            "detection_ms": result.timing_ms.get("detection"),
            "preprocessing_ms": result.timing_ms.get("preprocessing"),
            "ocr_ms": result.timing_ms.get("ocr"),
            "warnings": " | ".join(result.warnings),
        }
        predictions.append(record)

        if outcome != "match":
            failures.append(record)

    summary = build_summary(predictions, args)

    prediction_fields = [
        "image",
        "split",
        "background",
        "n_digits",
        "has_decimal",
        "group",
        "expected_raw",
        "expected_digits",
        "expected_norm",
        "expected_digit_len",
        "image_path",
        "image_found",
        "reading",
        "predicted_digits",
        "predicted_norm",
        "strict_match",
        "business_match",
        "status",
        "outcome",
        "global_confidence",
        "detection_confidence",
        "recognition_confidence",
        "orientation_confidence",
        "orientation_method",
        "total_ms",
        "detection_ms",
        "preprocessing_ms",
        "ocr_ms",
        "warnings",
    ]

    write_csv(output_dir / "predictions.csv", predictions, prediction_fields)
    write_csv(output_dir / "failures.csv", failures, prediction_fields)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    with (output_dir / "summary.md").open("w", encoding="utf-8") as fh:
        fh.write(render_summary_md(summary))

    print("\nEvaluation finished.", flush=True)
    print(f"- predictions.csv: {output_dir / 'predictions.csv'}", flush=True)
    print(f"- failures.csv:    {output_dir / 'failures.csv'}", flush=True)
    print(f"- summary.json:    {output_dir / 'summary.json'}", flush=True)
    print(f"- summary.md:      {output_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
