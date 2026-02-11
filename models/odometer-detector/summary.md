# baseline — Detector de Odómetro
> Configuración base: yolov8n-obb, 640px, 100 epochs

- **Fecha**: 2026-02-25
- **Modelo**: yolov8n-obb.pt
- **Epochs**: 100 (entrenados: 100)
- **Img size**: 640
- **Batch**: 16
- **Patience**: 50

## Métricas (val)

- **mAP50-95: 0.9609** (principal)
- mAP50: 0.9950
- Precision: 0.9995
- Recall: 1.0000

## Archivos

- `best.pt` — Mejor checkpoint
- `training_config.yaml` — Configuración completa
- `results.csv` — Métricas por epoch
- `metrics/` — Gráficas de entrenamiento
