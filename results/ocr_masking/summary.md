# iter6-maskdiag-test - Evaluacion OCR con diagnostico de masking

- Fecha: 2026-03-19T18:09:56
- Split: `test`
- Imagenes evaluadas: **120**
- Tiempo promedio OCR: **77.2 ms/imagen**
- Modelo: `en_PP-OCRv4_mobile_rec` (`rec-only`)
- Prediccion saneada a digitos: **si**

## Metricas principales

| Grupo | Exact Match | CER | CRR | Correctas | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| Global | 92.50% | 0.0186 | 0.9814 | 111 | 120 |
| Con decimales | 94.05% | 0.0166 | 0.9834 | 79 | 84 |
| Sin decimales | 88.89% | 0.0229 | 0.9771 | 32 | 36 |

## Diagnostico de masking

- Imagenes con decimales: **84**
- Correctas: **79**
- `mask_leak_full`: **0**
- `mask_leak_partial`: **0**
- `ocr_error`: **5**
- Errores atribuibles a masking: **0 / 5**
- Proporcion de errores atribuibles a masking: **0.00%**

## Metricas por background

| Background | Exact Match | CER | CRR | Correctas | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| bgblack | 100.00% | 0.0000 | 1.0000 | 5 | 5 |
| bgwhite | 92.17% | 0.0195 | 0.9805 | 106 | 115 |

## Metricas por n_digits

| n_digits | Exact Match | CER | CRR | Correctas | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 92.06% | 0.0238 | 0.9762 | 58 | 63 |
| 5 | 92.98% | 0.0140 | 0.9860 | 53 | 57 |

