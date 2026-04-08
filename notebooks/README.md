# Notebooks

Notebooks de Jupyter para entrenamiento, evaluación y análisis del pipeline.

| Notebook | Descripción |
|----------|-------------|
| `01_eda.ipynb` | Análisis exploratorio de datos: distribución de lotes, balance de clases, dimensiones |
| `02_train_auto_annotator.ipynb` | Entrenamiento del modelo auxiliar YOLO OBB para pre-anotación (300 imágenes semilla) |
| `03_train_odometer_detector.ipynb` | Entrenamiento iterativo del detector de odómetros YOLO OBB (1,199 imágenes) |
| `04_eval_ocr.ipynb` | Evaluación de modelos OCR: comparación de preprocessing y configuraciones |
| `05_finetune_ocr.ipynb` | Fine-tuning del reconocedor PaddleOCR con diccionario de solo dígitos |
| `06_explainability.ipynb` | Análisis de explicabilidad: Eigen-CAM, Grad-CAM, oclusión |
| `07_visualize_pipeline_evaluations.ipynb` | Visualización comparativa de métricas: evolución por iteración, threshold sweep |

## Ejecución

Los notebooks están diseñados para ejecutarse en orden secuencial. Cada notebook incluye
las salidas pre-computadas, por lo que pueden revisarse sin necesidad de re-ejecutar.

Para re-ejecutar, asegúrese de tener los datos y modelos disponibles en las rutas esperadas
(ver `data/README.md` y `models/README.md`).
