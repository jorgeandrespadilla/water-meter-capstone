# Lectura Automática de Medidores de Agua

Proyecto Capstone de Maestría en Inteligencia Artificial - UDLA.

Pipeline de ML para lectura automática de odómetros en medidores de agua.
Combina detección de odómetros (YOLO OBB) y lectura de dígitos (PaddleOCR fine-tuned).

## Descripción del Problema

Las empresas de agua potable realizan miles de lecturas mensuales de medidores mediante fotografías tomadas en campo. La digitación manual de estas lecturas presenta tasas de error del 3-5%, generando reclamos y pérdida de ingresos. Este proyecto implementa un pipeline de ML que automatiza la extracción y validación de lecturas a partir de las fotografías.

## Arquitectura del Pipeline

```
Imagen --> Detector YOLO OBB --> Recorte OBB --> Resolución de Orientación --> Máscara Decimal --> PaddleOCR --> Lectura
```

Etapas:
1. **Detección**: Localiza el odómetro en la imagen usando YOLO con Oriented Bounding Boxes (OBB)
2. **Recorte**: Extrae el área del odómetro con rotación y alineación
3. **Orientación**: Resuelve si el recorte está invertido 180 grados (cascada: landscape, posición del rojo, dual-OCR)
4. **Máscara decimal**: Enmascara dígitos rojos decimales mediante segmentación LAB
5. **OCR**: Lee los dígitos con PaddleOCR fine-tuned
6. **Validación**: Clasifica la lectura como `valid`, `needs_review` o `no_detection` según confianza global

## Estructura del Proyecto

```
water-meter-capstone/
├── app/                         # Aplicación (demo + API)
│   ├── pipeline.py              # Pipeline de inferencia principal
│   ├── api.py                   # REST API (FastAPI)
│   └── demo.py                  # Demo interactivo (Gradio)
├── utils/                       # Módulos de utilidad
│   ├── cropping.py              # Recorte OBB
│   ├── orientation.py           # Resolución de orientación
│   ├── masking.py               # Máscara de dígitos rojos
│   └── logging.py               # Configuración de logging
├── scripts/
│   ├── data/                    # Pipeline de datos (00-03 + builders)
│   └── eval/                    # Scripts de evaluación
├── notebooks/                   # Notebooks de entrenamiento y análisis
├── models/                      # Modelos entrenados
│   ├── odometer-detector/       # YOLO OBB (producción)
│   ├── ocr-reader/              # PaddleOCR fine-tuned (producción)
│   └── auto-annotator/          # Modelo auxiliar de pre-anotación
├── data/                        # Datasets
│   ├── annotations/             # Anotaciones CVAT + artefactos
│   ├── obb/                     # Dataset YOLO OBB (train/val/test)
│   └── ocr/                     # Dataset OCR (train/val/test)
└── samples/                     # Imágenes de ejemplo para pruebas
```

## Requisitos Técnicos

- Python 3.12
- ~800 MB de espacio en disco (modelos + datos)
- CPU suficiente para inferencia (GPU opcional)

Dependencias principales: PyTorch, Ultralytics (YOLO), PaddleOCR, PaddlePaddle, Gradio, FastAPI.

## Instrucciones de Ejecución

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
# Documentación en http://localhost:8000/docs
```

## Uso Programático

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

| Modelo | Arquitectura | Métrica Principal | Valor |
|--------|-------------|-------------------|-------|
| Detector de odómetro | YOLOv8n-OBB | mAP50 | 0.995 |
| Lector OCR | PP-OCRv4 mobile (fine-tuned) | Exact Match | 0.525 |
| Auto-anotador (auxiliar) | YOLOv8s-OBB | mAP50 | 0.893 |

Ver `models/README.md` para detalles de cada modelo.

## Resultados de Evaluación

| Evaluación | Exact Match | Valid Rate | Descripción |
|-----------|-------------|------------|-------------|
| End-to-end (test split) | 0.44 | 0.824 | Pipeline completo sobre split de prueba |
| Piloto (datos no vistos) | 0.45 | 0.82 | Validación sobre datos de campo no vistos |

Resultados detallados en los notebooks 04, 05 y 07.

## Notebooks

| Notebook | Descripción |
|----------|-------------|
| `01_eda.ipynb` | Análisis exploratorio de datos |
| `02_train_auto_annotator.ipynb` | Entrenamiento del modelo auxiliar de pre-anotación |
| `03_train_odometer_detector.ipynb` | Entrenamiento del detector YOLO OBB (iterativo) |
| `04_eval_ocr.ipynb` | Evaluación de modelos OCR |
| `05_finetune_ocr.ipynb` | Fine-tuning del reconocedor OCR |
| `06_explainability.ipynb` | Análisis de explicabilidad (Grad-CAM, SHAP, LIME) |
| `07_visualize_pipeline_evaluations.ipynb` | Visualización comparativa de resultados |

## Pipeline de Datos

| Script | Descripción |
|--------|-------------|
| `00_clean_raw_data.py` | Consolida lotes, normaliza JPEG |
| `01_organize_classification.py` | Organiza imágenes por clase (valid/invalid/ambiguous) |
| `02_crop_odometers.py` | Recorta odómetros usando labels OBB |
| `03_generate_metadata.py` | Genera metadata con splits estratificados |
| `build_obb.py` | Construye dataset YOLO OBB con splits |
| `build_ocr.py` | Construye dataset OCR con rotación y máscara decimal |

## Autores

- Jorge Andrés Padilla Salgado
- Alain Mateo Ruales Quezada
