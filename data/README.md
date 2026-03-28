# Datasets

Datasets para entrenamiento y evaluación del pipeline de lectura de medidores de agua.

## Datasets Incluidos

### annotations/

Artefactos de anotación generados desde CVAT. Contiene las imágenes fuente y sus labels.

| Carpeta/Archivo | Descripción |
|----------------|-------------|
| `images/` | 1,199 imágenes clasificadas como válidas |
| `obb/` | Labels en formato YOLO OBB (clase + 4 puntos normalizados) |
| `ocr-crops/` | Recortes de odómetros alineados y orientados |
| `metadata.csv` | Metadata con background, n_digits, has_decimal, grupo y split |
| `ocr-labels.csv` | Lecturas OCR con rotación aplicada |
| `bgblack.txt` | Lista de imágenes con fondo negro |

### obb/

Dataset YOLO OBB listo para entrenamiento, particionado 80/10/10.

```
obb/
├── data.yaml                  # Configuración YOLO
├── images/{train,val,test}/   # Imágenes por split
└── labels/{train,val,test}/   # Labels OBB por split
```

- Train: 959 imágenes
- Val: 120 imágenes
- Test: 120 imágenes

### ocr/

Dataset OCR para fine-tuning de PaddleOCR.

```
ocr/
├── images/{train,val,test}/   # Recortes de odómetros
├── train.txt, val.txt, test.txt  # Manifiestos (imagen + label)
└── mask_decisions.csv         # Decisiones de masking por imagen
```

## Pipeline de Datos

Los datasets se generaron mediante los scripts en `scripts/data/`:

```
raw/ --> [00_clean] --> cleaned/ --> classification/
                                          |
                                   [CVAT Annotation]
                                          |
                                    annotations/
                                          |
                          +---------------+---------------+
                          |               |               |
                    [build_obb]     [02_crop]     [03_metadata]
                          |               |
                        obb/        ocr-crops/
                                          |
                                    [build_ocr]
                                          |
                                        ocr/
```

Los datos intermedios (`raw/`, `cleaned/`, `classification/`) no están incluidos en este
repositorio. Solo se incluyen los datasets finales necesarios para entrenamiento y evaluación.
