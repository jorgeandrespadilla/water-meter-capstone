# Scripts

## Pipeline de Datos (`data/`)

Scripts para procesar las imágenes crudas y construir los datasets de entrenamiento.

| Script | Entrada | Salida | Descripción |
|--------|---------|--------|-------------|
| `00_clean_raw_data.py` | `data/raw/batch_XX/` | `data/cleaned/` | Consolida lotes, genera IDs secuenciales, comprime JPEG |
| `01_organize_classification.py` | `data/classification/labels.csv` | `data/classification/{valid,invalid,ambiguous}/` | Organiza imágenes por clase |
| `02_crop_odometers.py` | `data/annotations/` | `data/annotations/ocr-crops/` | Recorta odómetros usando labels OBB |
| `03_generate_metadata.py` | `data/annotations/ocr-labels.csv` | `data/annotations/metadata.csv` | Genera metadata con splits estratificados train/val/test |
| `build_obb.py` | `data/annotations/` | `data/obb/` | Construye dataset YOLO OBB con splits 80/10/10 |
| `build_ocr.py` | `data/annotations/ocr-crops/` | `data/ocr/` | Construye dataset OCR con rotación y máscara decimal |

## Evaluación (`eval/`)

Scripts para evaluar el pipeline completo y sus componentes.

| Script | Descripción |
|--------|-------------|
| `evaluate_end_to_end.py` | Evalúa el pipeline completo sobre cualquier split |
| `evaluate_pilot.py` | Validación piloto sobre datos no vistos (`data/pilot/`) |
| `evaluate_ocr_masking.py` | Evalúa OCR y diagnostica errores del masking decimal |
| `sweep_validation_threshold.py` | Recalcula punto de operación para varios thresholds |
| `benchmark_ocr_latency.py` | Mide latencia del OCR sobre crops de odómetro |
