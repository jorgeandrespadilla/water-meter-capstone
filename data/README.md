# Datasets

Datasets para entrenamiento y evaluacion del pipeline de lectura de medidores de agua.

## Datasets Incluidos

### annotations/

Artefactos de anotacion generados desde CVAT. Contiene las imagenes fuente y sus labels.

| Carpeta/Archivo | Descripcion |
|----------------|-------------|
| `images/` | 1,199 imagenes clasificadas como validas |
| `obb/` | Labels en formato YOLO OBB (clase + 4 puntos normalizados) |
| `ocr-crops/` | Recortes de odometros alineados y orientados |
| `metadata.csv` | Metadata con background, n_digits, has_decimal, grupo y split |
| `ocr-labels.csv` | Lecturas OCR con rotacion aplicada |
| `bgblack.txt` | Lista de imagenes con fondo negro |

### obb/

Dataset YOLO OBB listo para entrenamiento, particionado 80/10/10.

```
obb/
├── data.yaml                  # Configuracion YOLO
├── images/{train,val,test}/   # Imagenes por split
└── labels/{train,val,test}/   # Labels OBB por split
```

- Train: 959 imagenes
- Val: 120 imagenes
- Test: 120 imagenes

### ocr/

Dataset OCR para fine-tuning de PaddleOCR.

```
ocr/
├── images/{train,val,test}/   # Recortes de odometros
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

Los datos intermedios (`raw/`, `cleaned/`, `classification/`) no estan incluidos en este
repositorio. Solo se incluyen los datasets finales necesarios para entrenamiento y evaluacion.
