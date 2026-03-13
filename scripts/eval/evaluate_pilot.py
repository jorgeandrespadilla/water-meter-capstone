"""Evaluate the full water meter pipeline on the pilot dataset.

Runs the end-to-end pipeline over `data/pilot/images`, compares predictions
against `data/pilot/readings.csv`, and exports:

- `predictions.csv`: per-image results
- `failures.csv`: subset of non-matching or unreadable cases
- `summary.json`: machine-readable aggregate metrics
- `summary.md`: human-readable report for documentation

The pilot readings are operational values, so evaluation compares predictions
after normalizing leading zeros. This reflects the business comparison between
the OCR reading and the manually entered value.

Examples:
    python scripts/eval/evaluate_pilot.py

    python scripts/eval/evaluate_pilot.py --threshold 0.7 --limit 20

    python scripts/eval/evaluate_pilot.py ^
        --pilot-root data/pilot ^
        --output-dir logs/pilot_eval
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
from statistics import mean, median

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Avoid online source checks when loading local Paddle models.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

from app.pipeline import WaterMeterPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the full pipeline on the pilot dataset.",
    )
    parser.add_argument(
        "--pilot-root",
        default=str(PROJECT_ROOT / "data" / "pilot"),
        help="Pilot dataset root containing `images/` and `readings.csv`.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "logs" / "pilot_eval"),
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
        help="Evaluate only the first N pilot rows (for smoke tests).",
    )
    parser.add_argument(
        "--serials",
        default=None,
        help="Comma-separated list of SERIAL values to evaluate.",
    )
    return parser.parse_args()


def digits_only(value: str | None) -> str | None:
    if value is None:
        return None
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits or None


def normalize_business_reading(value: str | None) -> str | None:
    digits = digits_only(value)
    if digits is None:
        return None
    normalized = digits.lstrip("0")
    return normalized if normalized else "0"


def safe_percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return round(float(np.percentile(values, q)), 1)


def safe_mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(mean(values)), 1)


def safe_median(values: list[float]) -> float | None:
    if not values:
        return None
    return round(float(median(values)), 1)


def safe_rate(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return round(num / den, 4)


def load_pilot_rows(csv_path: Path, selected_serials: set[str] | None) -> list[dict]:
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            serial = str(row["SERIAL"]).strip()
            if selected_serials and serial not in selected_serials:
                continue
            rows.append({
                "serial": serial,
                "cuenta": str(row.get("CUENTA", "")).strip(),
                "medidor": str(row.get("MEDIDOR", "")).strip(),
                "expected_raw": str(row.get("LECTURA", "")).strip(),
            })
    return rows


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


def build_summary(predictions: list[dict], args: argparse.Namespace) -> dict:
    total_rows = len(predictions)
    found_rows = [row for row in predictions if row["image_found"]]
    matched_rows = [row for row in found_rows if row["business_match"]]
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
    digit_len_counts = defaultdict(lambda: {"total": 0, "matches": 0, "readable": 0})
    for row in found_rows:
        key = row["expected_digit_len"]
        digit_len_counts[key]["total"] += 1
        digit_len_counts[key]["matches"] += int(row["business_match"])
        digit_len_counts[key]["readable"] += int(row["predicted_norm"] is not None)

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "pilot_root": str(args.pilot_root),
        "threshold": args.threshold,
        "total_rows": total_rows,
        "images_found": len(found_rows),
        "images_missing": total_rows - len(found_rows),
        "metrics": {
            "pilot_match_rate": safe_rate(len(matched_rows), total_rows),
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
            str(key): {
                "total": stats["total"],
                "match_rate": safe_rate(stats["matches"], stats["total"]),
                "readable_rate": safe_rate(stats["readable"], stats["total"]),
            }
            for key, stats in sorted(digit_len_counts.items(), key=lambda item: int(item[0]))
        },
    }
    return summary


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_summary_md(summary: dict) -> str:
    metrics = summary["metrics"]
    counts = summary["counts"]
    latency = summary["latency_ms"]

    def pct(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value:.2%}"

    lines = [
        "# Validacion Piloto - Resumen",
        "",
        f"- Fecha: {summary['generated_at']}",
        f"- Threshold de validacion: `{summary['threshold']}`",
        f"- Casos totales: **{summary['total_rows']}**",
        f"- Imagenes encontradas: **{summary['images_found']}**",
        f"- Imagenes faltantes: **{summary['images_missing']}**",
        "",
        "## Metricas principales",
        "",
        "| Metrica | Valor |",
        "| --- | ---: |",
        f"| Pilot match rate | {pct(metrics['pilot_match_rate'])} |",
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
        "## Subgrupos por longitud esperada",
        "",
        "| Digitos esperados | Total | Match rate | Readable rate |",
        "| --- | ---: | ---: | ---: |",
    ]

    for digit_len, stats in summary["subgroups"].items():
        lines.append(
            f"| {digit_len} | {stats['total']} | {pct(stats['match_rate'])} | {pct(stats['readable_rate'])} |"
        )

    lines.extend([
        "",
        "## Nota metodologica",
        "",
        "La comparacion principal del piloto se realiza sobre la lectura normalizada,",
        "ignorando ceros a la izquierda. Esto refleja mejor la validacion operativa,",
        "ya que `readings.csv` almacena la lectura manual como valor entero sin padding.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.pilot_root = Path(args.pilot_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pilot_csv = args.pilot_root / "readings.csv"
    pilot_images = args.pilot_root / "images"

    if not pilot_csv.exists():
        raise FileNotFoundError(f"Pilot CSV not found: {pilot_csv}")
    if not pilot_images.exists():
        raise FileNotFoundError(f"Pilot images dir not found: {pilot_images}")

    selected_serials = None
    if args.serials:
        selected_serials = {item.strip() for item in args.serials.split(",") if item.strip()}

    rows = load_pilot_rows(pilot_csv, selected_serials)
    if args.limit is not None:
        rows = rows[: args.limit]

    if not rows:
        raise RuntimeError("No pilot rows selected for evaluation.")

    print(f"Selected pilot rows: {len(rows)}", flush=True)
    print("Loading full pipeline...", flush=True)
    pipeline = WaterMeterPipeline()
    print("Pipeline loaded. Starting evaluation...", flush=True)

    predictions: list[dict] = []
    failures: list[dict] = []

    for index, row in enumerate(rows, start=1):
        serial = row["serial"]
        image_path = pilot_images / f"{serial}.jpg"
        expected_digits = digits_only(row["expected_raw"])
        expected_norm = normalize_business_reading(row["expected_raw"])
        expected_digit_len = len(expected_norm) if expected_norm is not None else 0

        print(f"[{index}/{len(rows)}] SERIAL={serial}", flush=True)

        base_record = {
            "serial": serial,
            "cuenta": row["cuenta"],
            "medidor": row["medidor"],
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
        "serial",
        "cuenta",
        "medidor",
        "expected_raw",
        "expected_digits",
        "expected_norm",
        "expected_digit_len",
        "image_path",
        "image_found",
        "reading",
        "predicted_digits",
        "predicted_norm",
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
