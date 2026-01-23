# Water Meter ML Pipeline

Pipeline de ML para lectura automatica de odometros en medidores de agua.
Combina deteccion de odometros (YOLO OBB) y lectura de digitos (PaddleOCR fine-tuned).

## Descripcion del Problema

Las empresas de agua potable realizan miles de lecturas mensuales de medidores mediante fotografias tomadas en campo. La digitacion manual de estas lecturas presenta tasas de error del 3-5%, generando reclamos y perdida de ingresos. Este proyecto implementa un pipeline de ML que automatiza la extraccion y validacion de lecturas a partir de las fotografias.

## Arquitectura del Pipeline

```
Imagen --> Detector YOLO OBB --> Recorte OBB --> Resolucion de Orientacion --> Mascara Decimal --> PaddleOCR --> Lectura
```

Etapas:
1. **Deteccion**: Localiza el odometro en la imagen usando YOLO con Oriented Bounding Boxes (OBB)
2. **Recorte**: Extrae el area del odometro con rotacion y alineacion
3. **Orientacion**: Resuelve si el recorte esta invertido 180 grados (cascada: landscape, posicion del rojo, dual-OCR)
4. **Mascara decimal**: Enmascara digitos rojos decimales mediante segmentacion LAB
5. **OCR**: Lee los digitos con PaddleOCR fine-tuned
6. **Validacion**: Clasifica la lectura como `valid`, `needs_review` o `no_detection` segun confianza global

## Estructura del Proyecto

```
water-meter-capstone/
├── app/                         # Aplicacion (demo + API)
│   ├── pipeline.py              # Pipeline de inferencia principal
│   ├── api.py                   # REST API (FastAPI)
│   └── demo.py                  # Demo interactivo (Gradio)
├── utils/                       # Modulos de utilidad
│   ├── cropping.py              # Recorte OBB
│   ├── orientation.py           # Resolucion de orientacion
│   ├── masking.py               # Mascara de digitos rojos
│   └── logging.py               # Configuracion de logging
├── scripts/
│   ├── data/                    # Pipeline de datos (00-03 + builders)
│   └── eval/                    # Scripts de evaluacion
├── notebooks/                   # Notebooks de entrenamiento y analisis
├── models/                      # Modelos entrenados
│   ├── odometer-detector/       # YOLO OBB (produccion)
│   ├── ocr-reader/              # PaddleOCR fine-tuned (produccion)
│   └── auto-annotator/          # Modelo auxiliar de pre-anotacion
├── data/                        # Datasets
│   ├── annotations/             # Anotaciones CVAT + artefactos
│   ├── obb/                     # Dataset YOLO OBB (train/val/test)
│   └── ocr/                     # Dataset OCR (train/val/test)
└── samples/                     # Imagenes de ejemplo para pruebas
```

## Requisitos Tecnicos

- Python 3.12
- ~800 MB de espacio en disco (modelos + datos)
- CPU suficiente para inferencia (GPU opcional)

Dependencias principales: PyTorch, Ultralytics (YOLO), PaddleOCR, PaddlePaddle, Gradio, FastAPI.

## Instrucciones de Ejecucion

```bash
# 1. Clonar el repositorio
git clone <repo-url>
cd water-meter-capstone

# 2. Crear entorno virtual
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar demo interactivo
python app/demo.py
# Abrir http://localhost:7860 en el navegador

# 5. Ejecutar API REST (alternativa)
python -m app.api
# Documentacion en http://localhost:8000/docs
```

## Uso Programatico

```python
import cv2
from app.pipeline import WaterMeterPipeline

pipeline = WaterMeterPipeline()
image = cv2.imread("samples/00004.jpg")
result = pipeline.predict(image)

print(f"Lectura: {result.reading}")
print(f"Confianza: {result.global_confidence:.2f}")
print(f"Estado: {result.status}")
```

## Modelos

| Modelo | Arquitectura | Metrica Principal | Valor |
|--------|-------------|-------------------|-------|
| Detector de odometro | YOLOv8n-OBB | mAP50 | 0.995 |
| Lector OCR | PP-OCRv4 mobile (fine-tuned) | Exact Match | 0.525 |
| Auto-anotador (auxiliar) | YOLOv8s-OBB | mAP50 | 0.893 |

Ver `models/README.md` para detalles de cada modelo.

## Resultados de Evaluacion

| Evaluacion | Exact Match | Valid Rate |
|-----------|-------------|------------|
| End-to-end (test split) | 0.44 | 0.824 |
| Piloto (datos no vistos) | 0.45 | 0.82 |

Resultados detallados en los notebooks 04, 05 y 07.

## Notebooks

| Notebook | Descripcion |
|----------|-------------|
| `01_eda.ipynb` | Analisis exploratorio de datos |
| `02_train_auto_annotator.ipynb` | Entrenamiento del modelo auxiliar de pre-anotacion |
| `03_train_odometer_detector.ipynb` | Entrenamiento del detector YOLO OBB (iterativo) |
| `04_eval_ocr.ipynb` | Evaluacion de modelos OCR |
| `05_finetune_ocr.ipynb` | Fine-tuning del reconocedor OCR |
| `06_explainability.ipynb` | Analisis de explicabilidad (Grad-CAM, SHAP, LIME) |
| `07_visualize_pipeline_evaluations.ipynb` | Visualizacion comparativa de resultados |

## Pipeline de Datos

| Script | Descripcion |
|--------|-------------|
| `00_clean_raw_data.py` | Consolida lotes, normaliza JPEG |
| `01_organize_classification.py` | Organiza imagenes por clase (valid/invalid/ambiguous) |
| `02_crop_odometers.py` | Recorta odometros usando labels OBB |
| `03_generate_metadata.py` | Genera metadata con splits estratificados |
| `build_obb.py` | Construye dataset YOLO OBB con splits |
| `build_ocr.py` | Construye dataset OCR con rotacion y mascara decimal |

## Autores

- Jorge Andres Padilla Salgado
- Alain Mateo Ruales Quezada
