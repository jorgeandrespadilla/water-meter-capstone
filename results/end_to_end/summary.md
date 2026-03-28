# Evaluacion End-to-End - Resumen

- Fecha: 2026-03-24T02:45:53
- Split: `test`
- Threshold de validacion: `0.7`
- Casos totales: **120**
- Imagenes encontradas: **120**
- Imagenes faltantes: **0**

## Metricas principales

| Metrica | Valor |
| --- | ---: |
| End-to-end accuracy | 95.83% |
| Strict match rate | 94.17% |
| Readable rate | 100.00% |
| Accuracy cuando el pipeline devuelve lectura | 95.83% |
| Auto-validation rate (`status=valid`) | 93.33% |
| Precision de auto-validacion | 96.43% |
| Error rate de auto-validacion | 3.57% |
| Review rate (`status=needs_review`) | 6.67% |
| No-detection rate | 0.00% |
| Casos correctos enviados a revision | 5.83% |

## Conteos

- Status: `{"needs_review": 8, "valid": 112}`
- Outcome: `{"match": 115, "mismatch": 5}`
- Auto-validaciones correctas: **108**
- Auto-validaciones incorrectas: **4**
- Correctos pero enviados a revision: **7**

## Latencia (ms)

| Etapa | Mean | Median | P95 |
| --- | ---: | ---: | ---: |
| Total | 434.9 | 409.9 | 596.2 |
| Detection | 181.9 | 163.5 | 264.1 |
| Preprocessing | 41.8 | 37.1 | 69.1 |
| OCR | 210.6 | 162.8 | 330.8 |

## Subgrupos

### background

| Grupo | Total | Match rate | Readable rate |
| --- | ---: | ---: | ---: |
| bgblack | 5 | 100.00% | 100.00% |
| bgwhite | 115 | 95.65% | 100.00% |

### has_decimal

| Grupo | Total | Match rate | Readable rate |
| --- | ---: | ---: | ---: |
| False | 36 | 91.67% | 100.00% |
| True | 84 | 97.62% | 100.00% |

### n_digits

| Grupo | Total | Match rate | Readable rate |
| --- | ---: | ---: | ---: |
| 4 | 63 | 93.65% | 100.00% |
| 5 | 57 | 98.25% | 100.00% |

## Nota metodologica

La metrica principal compara la lectura entera normalizada,
ignorando ceros a la izquierda. Esto refleja mejor la validacion
operativa del sistema. El strict match se reporta aparte para
cuantificar diferencias de padding.
