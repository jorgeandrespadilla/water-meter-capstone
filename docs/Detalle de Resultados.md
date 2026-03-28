# Detalle de Resultados

Registro consolidado de los experimentos realizados y sus resultados.

---

## Modelo YOLO - Detector de Odómetro

### Configuración Base (Baseline)

- Modelo: yolov8n-obb.pt (3.2M parámetros)
- Resolución: 640px
- Epochs: 100, batch: 16, patience: 50
- GPU: Tesla T4 (Google Colab)
- Tiempo de entrenamiento: ~45 min

**Métricas (val):**

| Métrica | Valor |
|---------|-------|
| mAP50-95 | **0.9609** |
| mAP50 | 0.9950 |
| Precision | 0.9995 |
| Recall | 1.0000 |

El baseline alcanzó métricas muy altas desde el inicio. Precision y Recall prácticamente perfectos. El gap principal está en mAP50-95 (localización fina a IoU altos).

---

### Iteraciones de Mejora

| Iteración | Cambio | mAP50-95 (val) | Delta | Resultado |
|-----------|--------|---------------:|------:|-----------|
| Baseline | YOLOv8n, 640px, 100 ep | 0.9609 | - | Adoptado |
| Iter 1 | Resolución 640 -> 1280px | 0.9556 | -0.53 pp | Revertido |
| Iter 2 | YOLOv8n -> YOLOv8s (11.4M params) | 0.9574 | -0.35 pp | Revertido |
| Iter 3 | YOLOv8n -> YOLO11n (C3k2/SPPF/C2PSA) | 0.9621 | +0.12 pp | Revertido (bajo umbral) |
| Iter 4 | copy_paste=0.3 | 0.9519 | -0.90 pp | Revertido |
| Iter 5 | 200 epochs (early stop en 160) | 0.9565 | -0.44 pp | Revertido |

Ninguna de las 5 iteraciones mejoró sobre el baseline. La configuración mínima ya captura toda la variabilidad del dataset para esta tarea de una sola clase.

---

### Resultado Final - Mejor Modelo YOLO

Configuración ganadora: **Baseline** (yolov8n-obb, 640px, 100 epochs, batch 16).

**Métricas finales (test set):**

| Métrica | Valor |
|---------|-------|
| mAP50-95 | **0.9522** |
| mAP50 | 0.9950 |
| Precision | 0.9920 |
| Recall | 1.0000 |

El gap val->test en mAP50-95 es de -0.87 pp (0.9609 -> 0.9522), dentro de la variabilidad esperada para splits de 120 imágenes. mAP50 y Recall son idénticos en ambos splits, lo que indica buena generalización sin sobreajuste.

---
---

## Modelo OCR - Lector de Dígitos

Los experimentos OCR se dividen en dos fases: primero se afinan los datos y algoritmos de preprocesamiento, luego se optimiza el modelo.

### Fase 1 - Datos y Preprocessing (modelo fijo)

Se evalúan diferentes estrategias de enmascaramiento de dígitos decimales rojos, manteniendo fijo el modelo PP-OCRv5 mobile (det + rec) como instrumento de medición.

| # | Configuración | EM Accuracy (val) | Delta | Decisión |
|---|--------------|------------------:|------:|----------|
| P0 | Sin máscara (raw crops) | 0.83% (1/120) | - | Baseline |
| P1 | Máscara HSV (umbrales fijos) | 5.00% (6/120) | +4.17 pp | Mejora |
| P2 | Máscara LAB (canal a* relativo) | 4.17% (5/120) | +3.34 pp | Adoptado por robustez |

Se adoptó P2 (LAB) a pesar de la diferencia marginal con P1, por las siguientes razones:
1. La diferencia de 1 imagen en 120 no es estadísticamente significativa
2. El algoritmo LAB detecta rojo relativo a cada imagen (vs umbrales absolutos en HSV)
3. El borrado por TELEA inpainting + supresión de a* produce relleno sin artefactos
4. El cuello de botella real es el modelo OCR (59% sin detección), no el masking

---

### Fase 2 - Modelo (datos limpios)

Con los datos optimizados de Fase 1, se optimiza el modelo OCR.

| Iteración | Cambio | EM Accuracy (val) | Delta | Decisión |
|-----------|--------|------------------:|------:|----------|
| Baseline | PP-OCRv5 mobile (det + rec) | 4.17% (5/120) | - | Punto de partida |
| Iter 1 | Bypass del detector de texto (rec-only) | 15.83% (19/120) | +11.66 pp | Adoptado |
| Iter 2 | Mobile -> Server (multilingüe) | 13.33% (16/120) | -2.50 pp | Revertido |
| Iter 3 | PP-OCRv5 -> PP-OCRv4 english-only | 30.00% (36/120) | +14.17 pp | Adoptado |
| Iter 4 | Fine-tuning con diccionario digit-only | 90.00% (108/120) | +60.00 pp | Adoptado |
| Iter 5 | OBB crop padding +5px | 85.00% (102/120) | -5.00 pp | Revertido |
| Iter 6 | Máscara LAB refinada + reentrenamiento | 89.17% (107/120) | -0.83 pp val / +5.00 pp test | Adoptado |

Cambios principales que produjeron mejora:
1. **Bypass del detector** (+11.7 pp): los crops ya están alineados al odómetro, hacer rec-only elimina el cuello de botella del detector de texto
2. **PP-OCRv4 english-only** (+14.2 pp): vocabulario reducido a caracteres latinos, menos confusión con dígitos mecánicos
3. **Fine-tuning** (+60.0 pp): la mejora más grande del proyecto, adaptación directa al dominio de odómetros
4. **Máscara LAB refinada** (+5.0 pp en test): extensión de la compuerta de color a tonos magenta/púrpura

---

### Resultado Final - Mejor Modelo OCR

Configuración ganadora: `en_PP-OCRv4_mobile_rec` fine-tuned, rec-only, diccionario digit-only, dataset OCR con máscara LAB refinada.

**Métricas finales (test set):**

| Métrica | Valor |
|---------|-------|
| Exact Match Accuracy | **92.50% (111/120)** |
| CER | 0.0186 |
| CRR | 0.9814 |
| EM con decimales (n=84) | 94.05% (79/84) |
| EM sin decimales (n=36) | 88.89% (32/36) |
| Tiempo promedio | 9 ms/imagen |

**Análisis de errores en test (9 errores):**

| Tipo de error | Cantidad | % |
|---------------|----------|---|
| missing_digits | 6 | 66.7% |
| extra_digits | 2 | 22.2% |
| wrong_digits | 1 | 11.1% |

El error residual dominante es `missing_digits` (omisión de dígitos). El análisis de explicabilidad (Grad-CAM + oclusión) identificó un sesgo de posición inducido por el desbalance del dataset (70% con decimales, 30% solo enteros), donde el modelo aprendió a deprioritizar las posiciones finales del odómetro.

---
---

## Pipeline End-to-End

### Configuración

- Modelo YOLO: `yolov8n-obb` baseline (`models/odometer-detector/best.pt`)
- Modelo OCR: `en_PP-OCRv4_mobile_rec` fine-tuned (`models/ocr-reader/model/`)
- Preprocesamiento: máscara LAB refinada, orientación por cascada (landscape -> red-position -> dual-OCR)
- Threshold de validación: 0.7
- Hardware: CPU (Intel i7-1365U)
- Split evaluado: test (120 imágenes)

Reportes detallados en `results/end_to_end/`.

### Métricas (test set)

| Métrica | Valor |
|---------|------:|
| End-to-end accuracy | 95.83% (115/120) |
| Strict match rate | 94.17% (113/120) |
| Readable rate | 100.00% (120/120) |
| No-detection rate | 0.00% |
| Auto-validation rate | 93.33% (112/120) |
| Precisión de auto-validación | 96.43% (108/112) |
| Error rate de auto-validación | 3.57% (4/112) |
| Review rate | 6.67% (8/120) |

### Latencia

| Etapa | Media (ms) | Mediana (ms) | P95 (ms) |
|-------|----------:|------------:|--------:|
| Total | 434.9 | 409.9 | 596.2 |
| Detección | 181.9 | 163.5 | 264.1 |
| Preprocesamiento | 41.8 | 37.1 | 69.1 |
| OCR | 210.6 | 162.8 | 330.8 |

### Rendimiento por Subgrupo

| Subgrupo | Total | Match rate |
|----------|------:|-----------:|
| Con decimales | 84 | 97.62% |
| Sin decimales | 36 | 91.67% |
| 4 dígitos | 63 | 93.65% |
| 5 dígitos | 57 | 98.25% |
| Fondo blanco | 115 | 95.65% |
| Fondo negro | 5 | 100.00% |

### Análisis de Errores

Los 5 casos de fallo:

| Imagen | Referencia | Predicción | Causa principal |
|--------|-----------|-----------|----------------|
| 00138 | 3321 | 332 | Omisión de último dígito |
| 01106 | 6870 | 89 | Orientación ambigua + omisión |
| 01778 | 00713 | 007132 | Decimal parcialmente visible tras máscara |
| 02631 | 1174 | 1748 | Sustitución de dígito |
| 02818 | 9486 | 986 | Orientación ambigua + omisión |

El patrón de `missing_digits` se mantiene como error dominante, consistente con el sesgo de posición identificado en el análisis de explicabilidad.

---
---

## Validación Piloto

### Configuración

- Fuente: 300 fotografías operativas reales, capturadas por lectores en campo
- Referencia: lectura digitada manualmente por el operario
- Pipeline: misma configuración que end-to-end
- Threshold de validación: 0.7
- Hardware: CPU (Intel i7-1365U)

Reportes detallados en `results/pilot/`.

### Métricas

| Métrica | Valor |
|---------|------:|
| Pilot match rate | 82.67% (248/300) |
| Readable rate | 92.67% (278/300) |
| Accuracy sobre imágenes procesables | 89.21% (248/278) |
| Auto-validation rate | 83.33% (250/300) |
| Precisión de auto-validación | 90.40% (226/250) |
| Error rate de auto-validación | 9.60% (24/250) |
| Review rate | 10.00% (30/300) |
| No-detection rate | 6.67% (20/300) |

### Latencia

| Etapa | Media (ms) | Mediana (ms) | P95 (ms) |
|-------|----------:|------------:|--------:|
| Total | 1152.7 | 764.3 | 1508.9 |
| Detección | 293.1 | 199.1 | 334.8 |
| Preprocesamiento | 447.2 | 304.6 | 710.2 |
| OCR | 473.2 | 194.7 | 560.9 |

### Rendimiento por Longitud de Lectura

| Dígitos | Total | Match rate | Readable rate |
|---------|------:|-----------:|-------------:|
| 1 | 7 | 71.43% | 85.71% |
| 2 | 11 | 100.00% | 100.00% |
| 3 | 51 | 84.31% | 86.27% |
| 4 | 212 | 83.49% | 94.34% |
| 5 | 19 | 63.16% | 89.47% |

### Clasificación de Casos sin Detección

De las 20 imágenes sin detección:

| Categoría | Cantidad | Descripción |
|-----------|----------|-------------|
| Imagen sin medidor | 12 | Fachadas, portones, tuberías sin medidor |
| Display no legible | 2 | Obstrucción o empañamiento extremo |
| Fallo por condiciones adversas | 4 | Suciedad, inclinación, odómetro poco claro |
| Fallo por modelo no representado | 2 | Tipografía o diseño no visto en entrenamiento |

Las 14 imágenes sin medidor o display no legible no son errores del sistema. Si se consideran solo las 286 imágenes procesables, la tasa de no-detección real es **2.1%** y el match rate ajustado sube a **86.71%**.

### Causas Raíz de Errores de Lectura

Los 32 errores de lectura (30 mismatch + 2 sin lectura):

| Causa raíz | Cantidad | Descripción |
|-----------|----------|-------------|
| Sesgo de posición (missing_digits) | 14 | Omisión del último dígito |
| Calidad visual degradada | 7 | Suciedad, empañamiento, obstrucción |
| Orientación incorrecta | 4 | Inversión 180 grados |
| Imagen rotada/inclinada | 3 | Captura en posición incómoda |
| Posible error del operario | 2 | Imagen sin medidor o lectura inconsistente |
| Otros | 2 | Vegetación, off-by-one |

### Diferencias Respecto al Test Set Controlado

- La latencia media (1153 ms) es 2.6x mayor que en test (435 ms), principalmente por imágenes de mayor resolución.
- El error rate de auto-validación (9.60%) es significativamente mayor que en test (3.57%).
- El subgrupo de 5 dígitos presenta el peor match rate (63.16%).

### Recomendaciones

- No habilitar validación automática total: la tasa de auto-validación incorrecta (9.60%) requiere revisión humana.
- Evaluar un threshold más alto (0.75-0.80) para reducir errores auto-validados (ver `results/pilot_threshold_sweep/`).
- Incorporar imágenes del piloto al dataset de entrenamiento para cerrar la brecha controlado/operativo.

---
---

## Consideraciones Técnicas sobre el Dataset

### Detector de Odómetro (YOLO)

El dataset contiene 1,199 imágenes con una sola clase (odómetro). El baseline (yolov8n, 3.2M parámetros) ya alcanza 0.995 mAP50 y 1.000 Recall, lo que indica que la tarea de detección está esencialmente resuelta con la configuración mínima. Con una sola clase y objetos de tamaño relativamente uniforme (~5.8% del área de la imagen), el dataset no presenta la complejidad que justificaría modelos más grandes o resoluciones mayores.

### Lector de Dígitos (OCR)

El split de validación contiene 120 imágenes, lo que limita la representatividad de las métricas (cada acierto o error individual representa ~0.8 pp). Los modelos de OCR general presentan un desajuste de dominio significativo con la tipografía mecánica de los odómetros. El fine-tuning fue la intervención de mayor impacto (+60 pp), al abordar directamente este desajuste.

### Implicaciones

1. **YOLO - rendimiento saturado:** Ninguna de las 5 iteraciones mejoró sobre el baseline. La configuración mínima ya captura toda la variabilidad del dataset.
2. **OCR - fine-tuning como factor decisivo:** Los modelos pretrained alcanzan máximo 30% EM sin fine-tuning. La adaptación al dominio es clave.
3. **Tamaño del dataset:** 1,199 imágenes es modesto para deep learning. Suficiente para diferencias grandes (>10 pp) pero limitado para diferencias sutiles.

### Hallazgos en las Muestras

Durante la preparación de datos se identificaron:

1. **OBB mal recortado:** Casos aislados donde la anotación no sigue la forma del odómetro.
2. **Muestras con baja calidad visual:** Dígitos poco claros, obstrucciones parciales, iluminación deficiente.
3. **Máscara decimal incompleta:** Muestras donde la máscara LAB inicial no cubría tonos magenta/púrpura, corregidos en la versión refinada.

Estas limitaciones deben considerarse al interpretar los resultados, ya que el rendimiento está condicionado por la calidad de las muestras recopiladas.
