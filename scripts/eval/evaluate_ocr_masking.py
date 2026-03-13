"""Evaluate OCR crops and diagnose decimal-masking leakage.

Runs the fine-tuned PP-OCR recognizer in rec-only mode on OCR crops from
`data/ocr/images/<split>`, compares predictions against integer labels from
`data/ocr/<split>.txt`, and uses `data/annotations/ocr-labels.csv` to
distinguish pure OCR errors from masking leakage on decimal meters.

Exports:
- `results.csv`: per-image predictions and masking diagnostics
- `summary.json`: machine-readable aggregate metrics
- `eval_config.yaml`: run configuration and key metrics
- `summary.md`: human-readable report for documentation

Examples:
    python scripts/eval/evaluate_ocr_masking.py

    python scripts/eval/evaluate_ocr_masking.py --split test

    python scripts/eval/evaluate_ocr_masking.py --split val --limit 20 --force-rerun
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

# Avoid online source checks when loading local Paddle models.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_OCR_MODEL_DIR = (
    PROJECT_ROOT / "models" / "ocr-reader" / "model"
)
DEFAULT_OCR_MODEL_NAME = "en_PP-OCRv4_mobile_rec"
_DIGITS_ONLY = re.compile(r"\D")

RESULT_COLUMNS = [
    "image",
    "rel_path",
    "integer_label",
    "full_reading",
    "full_digits",
    "raw_text",
    "digits_text",
    "prediction",
    "confidence",
    "time_ms",
    "has_decimal",
    "background",
    "n_digits",
    "group",
    "split",
    "is_correct",
    "error_class",
    "decimal_prefix_match_len",
    "matches_full_digits",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OCR crops and diagnose masking leakage.",
    )
    parser.add_argument(
        "--data-root",
        default=str(DEFAULT_DATA_ROOT),
        help="Dataset root containing `ocr/` and `annotations/`.",
    )
    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val", "test"],
        help="Split to evaluate (default: val).",
    )
    parser.add_argument(
        "--model-dir",
        default=str(DEFAULT_OCR_MODEL_DIR),
        help="Directory containing the exported OCR inference model.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_OCR_MODEL_NAME,
        help=f"OCR model name for paddlex.create_model (default: {DEFAULT_OCR_MODEL_NAME}).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where artifacts will be written. Default: logs/ocr_masking_eval/<run-name>.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name. Default: <model-parent>-maskdiag-<split>.",
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
        help="Comma-separated list of image names to evaluate (e.g. 00002.jpg,00007.jpg).",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore cached results and rerun OCR inference.",
    )
    parser.add_argument(
        "--no-sanitize",
        action="store_true",
        help="Keep raw OCR text instead of stripping non-digit characters.",
    )
    return parser.parse_args()


def digits_only(text: str | None) -> str:
    return _DIGITS_ONLY.sub("", text or "")


def normalize_bool(value) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def safe_rate(num: int, den: int) -> float | None:
    if den == 0:
        return None
    return round(num / den, 4)


def levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    if len(a) < len(b):
        a, b = b, a

    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (char_a != char_b)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def compute_metrics(predictions: list[str], ground_truths: list[str]) -> dict:
    correct = sum(1 for pred, gt in zip(predictions, ground_truths) if pred == gt)
    total_chars = sum(len(gt) for gt in ground_truths)
    total_distance = sum(
        levenshtein_distance(pred or "", gt)
        for pred, gt in zip(predictions, ground_truths)
    )
    cer = (total_distance / total_chars) if total_chars else 0.0
    exact_match = (correct / len(ground_truths)) if ground_truths else 0.0
    return {
        "exact_match": round(exact_match, 4),
        "cer": round(cer, 4),
        "crr": round(1.0 - cer, 4),
        "total": len(ground_truths),
        "correct": correct,
    }


def decimal_digits_from_reading(full_reading: str, integer_label: str) -> str:
    full_digits = digits_only(full_reading)
    if full_digits.startswith(integer_label):
        return full_digits[len(integer_label):]
    if "." not in full_reading:
        return ""
    return digits_only(full_reading.split(".", 1)[1])


def decimal_prefix_match_len(prediction: str, integer_label: str, full_reading: str) -> int:
    decimal_digits = decimal_digits_from_reading(full_reading, integer_label)
    if not prediction.startswith(integer_label):
        return 0

    extra = prediction[len(integer_label):]
    if not extra:
        return 0

    match_len = 0
    for pred_char, dec_char in zip(extra, decimal_digits):
        if pred_char != dec_char:
            break
        match_len += 1
    return match_len


def classify_masking_error(prediction: str, integer_label: str, full_reading: str) -> str:
    full_digits = digits_only(full_reading)
    if prediction == integer_label:
        return "correct"
    if prediction == full_digits:
        return "mask_leak_full"
    if decimal_prefix_match_len(prediction, integer_label, full_reading) > 0:
        return "mask_leak_partial"
    return "ocr_error"


def load_ocr_labels(csv_path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            labels[str(row["image"]).strip()] = str(row["reading"]).strip()
    return labels


def load_metadata(csv_path: Path) -> dict[str, dict]:
    metadata: dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image = str(row["image"]).strip()
            metadata[image] = {
                "split": str(row.get("split", "")).strip(),
                "background": str(row.get("background", "")).strip() or "unknown",
                "n_digits": int(str(row.get("n_digits", "")).strip() or 0),
                "has_decimal": normalize_bool(row.get("has_decimal", False)),
                "group": str(row.get("group", "")).strip(),
            }
    return metadata


def load_split_entries(label_file: Path, selected_images: set[str] | None) -> list[dict]:
    entries: list[dict] = []
    with label_file.open("r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            rel_path, integer_label = line.split("\t", 1)
            image_name = Path(rel_path).name
            if selected_images and image_name not in selected_images:
                continue
            entries.append({
                "image": image_name,
                "rel_path": rel_path,
                "integer_label": integer_label.strip(),
            })
    return entries


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


def run_rec(engine, image_bgr, *, sanitize_output: bool) -> tuple[str, str, str, float, float]:
    start = time.perf_counter()
    output = engine.predict(input=image_bgr, batch_size=1)

    raw_text = ""
    confidence = 0.0
    for res in output:
        raw_text, confidence = extract_rec_output(res)
        break

    time_ms = (time.perf_counter() - start) * 1000
    digits_text = digits_only(raw_text)
    prediction = digits_text if sanitize_output else raw_text
    return raw_text, digits_text, prediction, time_ms, confidence


def load_cached_results(csv_path: Path) -> list[dict] | None:
    if not csv_path.exists():
        return None

    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames or []
        if not set(RESULT_COLUMNS).issubset(fieldnames):
            return None

        rows: list[dict] = []
        for row in reader:
            rows.append({
                "image": str(row["image"]).strip(),
                "rel_path": str(row["rel_path"]).strip(),
                "integer_label": str(row["integer_label"]).strip(),
                "full_reading": str(row["full_reading"]).strip(),
                "full_digits": str(row["full_digits"]).strip(),
                "raw_text": str(row["raw_text"]),
                "digits_text": str(row["digits_text"]),
                "prediction": str(row["prediction"]),
                "confidence": float(str(row.get("confidence", "")).strip() or 0.0),
                "time_ms": float(str(row.get("time_ms", "")).strip() or 0.0),
                "has_decimal": normalize_bool(row.get("has_decimal", False)),
                "background": str(row.get("background", "")).strip() or "unknown",
                "n_digits": int(str(row.get("n_digits", "")).strip() or 0),
                "group": str(row.get("group", "")).strip(),
                "split": str(row.get("split", "")).strip(),
                "is_correct": normalize_bool(row.get("is_correct", False)),
                "error_class": str(row.get("error_class", "")).strip() or "ocr_error",
                "decimal_prefix_match_len": int(
                    str(row.get("decimal_prefix_match_len", "")).strip() or 0
                ),
                "matches_full_digits": normalize_bool(row.get("matches_full_digits", False)),
            })
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_yaml(path: Path, payload: dict) -> None:
    try:
        import yaml  # type: ignore

        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(payload, fh, sort_keys=False, allow_unicode=False)
    except Exception:
        # JSON is valid YAML 1.2 and avoids adding a hard dependency.
        with path.open("w", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, indent=2, ensure_ascii=False))
            fh.write("\n")


def subgroup_metrics(rows: list[dict], key: str) -> dict[str, dict]:
    grouped = defaultdict(lambda: {"predictions": [], "labels": []})
    for row in rows:
        group_value = row.get(key)
        if group_value in (None, ""):
            group_value = "unknown"
        grouped[str(group_value)]["predictions"].append(row["prediction"])
        grouped[str(group_value)]["labels"].append(row["integer_label"])

    return {
        group_name: compute_metrics(stats["predictions"], stats["labels"])
        for group_name, stats in sorted(grouped.items(), key=lambda item: item[0])
    }


def build_summary(rows: list[dict], args: argparse.Namespace, run_name: str) -> dict:
    dec_rows = [row for row in rows if row["has_decimal"]]
    nodec_rows = [row for row in rows if not row["has_decimal"]]

    metrics = {
        "global": compute_metrics(
            [row["prediction"] for row in rows],
            [row["integer_label"] for row in rows],
        ),
        "with_decimal": (
            compute_metrics(
                [row["prediction"] for row in dec_rows],
                [row["integer_label"] for row in dec_rows],
            ) if dec_rows else None
        ),
        "without_decimal": (
            compute_metrics(
                [row["prediction"] for row in nodec_rows],
                [row["integer_label"] for row in nodec_rows],
            ) if nodec_rows else None
        ),
    }

    mask_counts = Counter(row["error_class"] for row in dec_rows)
    mask_correct = mask_counts.get("correct", 0)
    mask_leak_full = mask_counts.get("mask_leak_full", 0)
    mask_leak_partial = mask_counts.get("mask_leak_partial", 0)
    ocr_error = mask_counts.get("ocr_error", 0)
    total_decimal_errors = len(dec_rows) - mask_correct
    total_mask_errors = mask_leak_full + mask_leak_partial

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "run_name": run_name,
        "data_root": str(Path(args.data_root)),
        "split": args.split,
        "model": {
            "ocr_model_name": args.model_name,
            "model_dir": str(Path(args.model_dir)),
            "mode": "rec-only",
            "sanitized_prediction": not args.no_sanitize,
        },
        "evaluation": {
            "total_images": len(rows),
            "avg_time_ms": round(sum(row["time_ms"] for row in rows) / len(rows), 1) if rows else 0.0,
            "force_rerun": args.force_rerun,
            "limit": args.limit,
        },
        "metrics": metrics,
        "masking": {
            "decimal_images": len(dec_rows),
            "correct": mask_correct,
            "mask_leak_full": mask_leak_full,
            "mask_leak_partial": mask_leak_partial,
            "ocr_error": ocr_error,
            "total_decimal_errors": total_decimal_errors,
            "total_mask_errors": total_mask_errors,
            "mask_error_share": safe_rate(total_mask_errors, total_decimal_errors),
        },
        "subgroups": {
            "background": subgroup_metrics(rows, "background"),
            "n_digits": subgroup_metrics(rows, "n_digits"),
        },
    }
    return summary


def render_summary_md(summary: dict) -> str:
    metrics = summary["metrics"]
    masking = summary["masking"]
    background = summary["subgroups"]["background"]
    digits = summary["subgroups"]["n_digits"]

    def pct(value: float | None) -> str:
        if value is None:
            return "N/A"
        return f"{value:.2%}"

    lines = [
        f"# {summary['run_name']} - Evaluacion OCR con diagnostico de masking",
        "",
        f"- Fecha: {summary['generated_at']}",
        f"- Split: `{summary['split']}`",
        f"- Imagenes evaluadas: **{summary['evaluation']['total_images']}**",
        f"- Tiempo promedio OCR: **{summary['evaluation']['avg_time_ms']:.1f} ms/imagen**",
        f"- Modelo: `{summary['model']['ocr_model_name']}` (`rec-only`)",
        f"- Prediccion saneada a digitos: **{'si' if summary['model']['sanitized_prediction'] else 'no'}**",
        "",
        "## Metricas principales",
        "",
        "| Grupo | Exact Match | CER | CRR | Correctas | Total |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]

    for key, label in [
        ("global", "Global"),
        ("with_decimal", "Con decimales"),
        ("without_decimal", "Sin decimales"),
    ]:
        group_metrics = metrics.get(key)
        if not group_metrics:
            continue
        lines.append(
            f"| {label} | {pct(group_metrics['exact_match'])} | "
            f"{group_metrics['cer']:.4f} | {group_metrics['crr']:.4f} | "
            f"{group_metrics['correct']} | {group_metrics['total']} |"
        )

    lines.extend([
        "",
        "## Diagnostico de masking",
        "",
        f"- Imagenes con decimales: **{masking['decimal_images']}**",
        f"- Correctas: **{masking['correct']}**",
        f"- `mask_leak_full`: **{masking['mask_leak_full']}**",
        f"- `mask_leak_partial`: **{masking['mask_leak_partial']}**",
        f"- `ocr_error`: **{masking['ocr_error']}**",
        f"- Errores atribuibles a masking: **{masking['total_mask_errors']} / {masking['total_decimal_errors']}**",
        f"- Proporcion de errores atribuibles a masking: **{pct(masking['mask_error_share'])}**",
        "",
        "## Metricas por background",
        "",
        "| Background | Exact Match | CER | CRR | Correctas | Total |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])

    for group_name, group_metrics in background.items():
        lines.append(
            f"| {group_name} | {pct(group_metrics['exact_match'])} | "
            f"{group_metrics['cer']:.4f} | {group_metrics['crr']:.4f} | "
            f"{group_metrics['correct']} | {group_metrics['total']} |"
        )

    lines.extend([
        "",
        "## Metricas por n_digits",
        "",
        "| n_digits | Exact Match | CER | CRR | Correctas | Total |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])

    for group_name, group_metrics in digits.items():
        lines.append(
            f"| {group_name} | {pct(group_metrics['exact_match'])} | "
            f"{group_metrics['cer']:.4f} | {group_metrics['crr']:.4f} | "
            f"{group_metrics['correct']} | {group_metrics['total']} |"
        )

    lines.append("")
    return "\n".join(lines)


def print_console_summary(summary: dict) -> None:
    metrics = summary["metrics"]
    masking = summary["masking"]

    print("\n=== Resultados OCR ===")
    for key, label in [
        ("global", "Global"),
        ("with_decimal", "Con decimales"),
        ("without_decimal", "Sin decimales"),
    ]:
        group_metrics = metrics.get(key)
        if not group_metrics:
            continue
        print(f"\n{label} (n={group_metrics['total']})")
        print(
            f"  Exact Match: {group_metrics['exact_match']:.4f} "
            f"({group_metrics['correct']}/{group_metrics['total']})"
        )
        print(f"  CER:         {group_metrics['cer']:.4f}")
        print(f"  CRR:         {group_metrics['crr']:.4f}")

    print("\n=== Diagnostico de masking ===")
    print(f"  Imagenes con decimales: {masking['decimal_images']}")
    print(f"  Correctas:              {masking['correct']}")
    print(f"  mask_leak_full:         {masking['mask_leak_full']}")
    print(f"  mask_leak_partial:      {masking['mask_leak_partial']}")
    print(f"  ocr_error:              {masking['ocr_error']}")
    print(
        "  Errores por masking:    "
        f"{masking['total_mask_errors']}/{masking['total_decimal_errors']}"
    )
    if masking["mask_error_share"] is not None:
        print(f"  Proporcion:             {masking['mask_error_share']:.1%}")


def resolve_output_dir(args: argparse.Namespace) -> tuple[str, Path]:
    run_name = args.run_name or f"{Path(args.model_dir).parent.name}-maskdiag-{args.split}"
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = PROJECT_ROOT / "logs" / "ocr_masking_eval" / run_name
    return run_name, output_dir


def main() -> None:
    args = parse_args()
    global cv2, np

    import cv2  # type: ignore
    import numpy as np

    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be > 0")

    data_root = Path(args.data_root)
    ocr_root = data_root / "ocr"
    annotations_root = data_root / "annotations"
    model_dir = Path(args.model_dir)
    run_name, output_dir = resolve_output_dir(args)

    label_file = ocr_root / f"{args.split}.txt"
    results_csv = output_dir / "results.csv"
    summary_json = output_dir / "summary.json"
    summary_md = output_dir / "summary.md"
    eval_config_yaml = output_dir / "eval_config.yaml"

    for path, label in [
        (ocr_root, "Dataset OCR"),
        (annotations_root, "Annotations"),
        (label_file, "Archivo del split"),
        (model_dir, "Modelo OCR exportado"),
    ]:
        if not path.exists():
            raise FileNotFoundError(f"{label} no encontrado: {path}")

    selected_images = None
    if args.images:
        selected_images = {
            item.strip() for item in args.images.split(",")
            if item.strip()
        }

    ocr_labels = load_ocr_labels(annotations_root / "ocr-labels.csv")
    metadata = load_metadata(annotations_root / "metadata.csv")
    split_entries = load_split_entries(label_file, selected_images)
    if args.limit is not None:
        split_entries = split_entries[:args.limit]

    if not split_entries:
        raise ValueError("No hay entradas para evaluar con los filtros seleccionados.")

    output_dir.mkdir(parents=True, exist_ok=True)

    cached_rows = None if args.force_rerun else load_cached_results(results_csv)
    expected_images = [entry["image"] for entry in split_entries]
    if cached_rows is not None:
        cached_images = [row["image"] for row in cached_rows]
        if cached_images != expected_images:
            cached_rows = None

    if cached_rows is not None:
        rows = cached_rows
        print(f"Se reutilizan resultados existentes: {results_csv}")
    else:
        from paddlex import create_model

        print("Inicializando OCR fine-tuned...")
        engine = create_model(model_name=args.model_name, model_dir=str(model_dir))
        _ = run_rec(engine, np.zeros((64, 256, 3), dtype=np.uint8), sanitize_output=not args.no_sanitize)
        print("Warmup completado.")

        rows: list[dict] = []
        start_total = time.perf_counter()

        for idx, entry in enumerate(split_entries, start=1):
            image_name = entry["image"]
            rel_path = entry["rel_path"]
            integer_label = entry["integer_label"]
            image_path = ocr_root / rel_path
            image_bgr = cv2.imread(str(image_path))

            if image_bgr is None:
                print(f"Advertencia: no se pudo leer la imagen {image_path}")
                raw_text, digits_text, prediction, time_ms, confidence = "", "", "", 0.0, 0.0
            else:
                raw_text, digits_text, prediction, time_ms, confidence = run_rec(
                    engine,
                    image_bgr,
                    sanitize_output=not args.no_sanitize,
                )

            meta = metadata.get(image_name, {})
            full_reading = ocr_labels.get(image_name, integer_label)
            full_digits = digits_only(full_reading)
            has_decimal = bool(meta.get("has_decimal", "." in full_reading))
            match_len = (
                decimal_prefix_match_len(prediction, integer_label, full_reading)
                if has_decimal else 0
            )
            if has_decimal:
                error_class = classify_masking_error(prediction, integer_label, full_reading)
            else:
                error_class = "correct" if prediction == integer_label else "ocr_error"

            rows.append({
                "image": image_name,
                "rel_path": rel_path,
                "integer_label": integer_label,
                "full_reading": full_reading,
                "full_digits": full_digits,
                "raw_text": raw_text,
                "digits_text": digits_text,
                "prediction": prediction,
                "confidence": round(confidence, 6),
                "time_ms": round(time_ms, 3),
                "has_decimal": has_decimal,
                "background": meta.get("background", "unknown"),
                "n_digits": int(meta.get("n_digits", len(integer_label)) or len(integer_label)),
                "group": meta.get("group", ""),
                "split": meta.get("split", args.split),
                "is_correct": prediction == integer_label,
                "error_class": error_class,
                "decimal_prefix_match_len": match_len,
                "matches_full_digits": prediction == full_digits,
            })

            if idx % 30 == 0 or idx == len(split_entries):
                print(f"  Procesadas {idx}/{len(split_entries)} imagenes...")

        elapsed_total = time.perf_counter() - start_total
        avg_ms = (sum(row["time_ms"] for row in rows) / len(rows)) if rows else 0.0
        print(f"Inferencia completada en {elapsed_total:.1f}s. Tiempo promedio: {avg_ms:.1f} ms/imagen")
        write_csv(results_csv, rows, RESULT_COLUMNS)

    summary = build_summary(rows, args, run_name)
    write_csv(results_csv, rows, RESULT_COLUMNS)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    summary_md.write_text(render_summary_md(summary) + "\n", encoding="utf-8")
    write_yaml(eval_config_yaml, summary)

    print_console_summary(summary)
    print("\n=== Exportado ===")
    print(results_csv)
    print(summary_json)
    print(eval_config_yaml)
    print(summary_md)


if __name__ == "__main__":
    main()
