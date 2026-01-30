# Notebooks

Notebooks de Jupyter para entrenamiento, evaluacion y analisis del pipeline.

| Notebook | Descripcion |
|----------|-------------|
| `01_eda.ipynb` | Analisis exploratorio de datos: distribucion de lotes, balance de clases, dimensiones |
| `02_train_auto_annotator.ipynb` | Entrenamiento del modelo auxiliar YOLO OBB para pre-anotacion (300 imagenes semilla) |
| `03_train_odometer_detector.ipynb` | Entrenamiento iterativo del detector de odometros YOLO OBB (1,199 imagenes) |
| `04_eval_ocr.ipynb` | Evaluacion de modelos OCR: comparacion de preprocessing y configuraciones |
| `05_finetune_ocr.ipynb` | Fine-tuning del reconocedor PaddleOCR con diccionario de solo digitos |
| `06_explainability.ipynb` | Analisis de explicabilidad: Grad-CAM, EigenCAM, SHAP, LIME, oclusion |
| `07_visualize_pipeline_evaluations.ipynb` | Visualizacion comparativa de metricas: evolucion por iteracion, threshold sweep |

## Ejecucion

Los notebooks estan diseñados para ejecutarse en orden secuencial. Cada notebook incluye
las salidas pre-computadas, por lo que pueden revisarse sin necesidad de re-ejecutar.

Para re-ejecutar, asegurese de tener los datos y modelos disponibles en las rutas esperadas
(ver `data/README.md` y `models/README.md`).
