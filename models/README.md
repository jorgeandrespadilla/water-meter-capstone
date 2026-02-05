# Modelos

Modelos entrenados para lectura automatica de odometros en medidores de agua.

## Modelos Disponibles

### odometer-detector

Modelo de produccion para deteccion de odometros mediante Oriented Bounding Boxes (OBB).
Arquitectura YOLOv8n-OBB, entrenado sobre 1,199 imagenes anotadas.

| Metrica | Valor |
|---------|-------|
| mAP50 | 0.995 |
| mAP50-95 | 0.961 |

### ocr-reader

Lector OCR de digitos en crops de odometro. Fine-tuning de PP-OCRv4 mobile con diccionario
de solo digitos (0-9), entrenado sobre crops con mascara LAB de digitos decimales.

| Metrica | Valor |
|---------|-------|
| Exact Match | 0.525 |

### auto-annotator

Modelo auxiliar para generar pre-anotaciones OBB y acelerar el etiquetado manual en CVAT.
No esta optimizado para produccion.

| Metrica | Valor |
|---------|-------|
| mAP50 | 0.893 |

## Uso

```python
from ultralytics import YOLO

model = YOLO("models/odometer-detector/best.pt")
results = model.predict("image.jpg", conf=0.5)

for result in results:
    boxes = result.obb.xyxyxyxy  # Coordenadas OBB
    confs = result.obb.conf      # Confianzas
```

## Estructura

```
models/
├── README.md
├── odometer-detector/
│   ├── best.pt                  # Checkpoint de produccion
│   ├── summary.md               # Resumen del experimento
│   ├── training_config.yaml     # Configuracion de entrenamiento
│   ├── results.csv              # Metricas por epoca
│   └── metrics/                 # Graficas de entrenamiento
├── ocr-reader/
│   ├── model/                   # Modelo PaddleOCR fine-tuned
│   │   ├── best_accuracy.pdparams
│   │   ├── inference.json
│   │   ├── inference.pdiparams
│   │   └── inference.yml
│   ├── summary.md
│   ├── training_config.yaml
│   ├── eval_config.yaml
│   └── results.csv
└── auto-annotator/
    ├── best.pt
    ├── training_config.yaml
    └── metrics/
```

## Entrenamiento

| Modelo | Notebook | Dataset |
|--------|----------|---------|
| `odometer-detector` | `notebooks/03_train_odometer_detector.ipynb` | `data/obb/` |
| `auto-annotator` | `notebooks/02_train_auto_annotator.ipynb` | 300 imagenes semilla |
| `ocr-reader` | `notebooks/05_finetune_ocr.ipynb` | `data/ocr/` |
