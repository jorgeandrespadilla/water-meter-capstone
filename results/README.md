# Resultados de Evaluación

Resultados pre-computados de las evaluaciones del pipeline. Cada carpeta contiene
los archivos generados por los scripts de `scripts/eval/`.

## Evaluaciones Disponibles

| Carpeta | Script | Descripción |
|---------|--------|-------------|
| `end_to_end/` | `evaluate_end_to_end.py` | Pipeline completo sobre split test (120 imágenes) |
| `end_to_end_threshold_sweep/` | `sweep_validation_threshold.py` | Búsqueda del threshold óptimo de validación |
| `ocr_masking/` | `evaluate_ocr_masking.py` | Evaluación OCR con diagnóstico de masking decimal |
| `pilot/` | `evaluate_pilot.py` | Validación piloto sobre 300 imágenes de campo no vistas |
| `pilot_threshold_sweep/` | `sweep_validation_threshold.py` | Threshold sweep sobre datos piloto |

## Resumen de Resultados

### End-to-end (test split)

| Métrica | Valor |
|---------|-------|
| End-to-end accuracy | 95.83% |
| Strict match rate | 94.17% |
| Readable rate | 100.00% |
| Auto-validation rate | 93.33% |
| No-detection rate | 0.00% |
| Latencia P95 | 596 ms |

### Piloto (datos de campo no vistos)

| Métrica | Valor |
|---------|-------|
| Pilot match rate | 82.67% |
| Readable rate | 92.67% |
| Auto-validation rate | 83.33% |
| No-detection rate | 6.67% |
| Latencia P95 | 1,509 ms |

## Archivos por Evaluación

Cada evaluación genera:
- `summary.md` / `summary.json` - Resumen de métricas
- `predictions.csv` - Predicciones individuales por imagen
- `failures.csv` - Detalle de casos fallidos
- `threshold_sweep.csv` - Métricas por umbral (en sweeps)
