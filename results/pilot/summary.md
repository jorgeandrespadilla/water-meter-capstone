# Validacion Piloto - Resumen

- Fecha: 2026-03-24T02:52:01
- Threshold de validacion: `0.7`
- Casos totales: **300**
- Imagenes encontradas: **300**
- Imagenes faltantes: **0**

## Metricas principales

| Metrica | Valor |
| --- | ---: |
| Pilot match rate | 82.67% |
| Readable rate | 92.67% |
| Accuracy cuando el pipeline devuelve lectura | 89.21% |
| Auto-validation rate (`status=valid`) | 83.33% |
| Precision de auto-validacion | 90.40% |
| Error rate de auto-validacion | 9.60% |
| Review rate (`status=needs_review`) | 10.00% |
| No-detection rate | 6.67% |
| Casos correctos enviados a revision | 7.33% |

## Conteos

- Status: `{"needs_review": 30, "no_detection": 20, "valid": 250}`
- Outcome: `{"match": 248, "mismatch": 30, "no_detection": 20, "no_reading": 2}`
- Auto-validaciones correctas: **226**
- Auto-validaciones incorrectas: **24**
- Correctos pero enviados a revision: **22**

## Latencia (ms)

| Etapa | Mean | Median | P95 |
| --- | ---: | ---: | ---: |
| Total | 1152.7 | 764.3 | 1508.9 |
| Detection | 293.1 | 199.1 | 334.8 |
| Preprocessing | 447.2 | 304.6 | 710.2 |
| OCR | 473.2 | 194.7 | 560.9 |

## Subgrupos por longitud esperada

| Digitos esperados | Total | Match rate | Readable rate |
| --- | ---: | ---: | ---: |
| 1 | 7 | 71.43% | 85.71% |
| 2 | 11 | 100.00% | 100.00% |
| 3 | 51 | 84.31% | 86.27% |
| 4 | 212 | 83.49% | 94.34% |
| 5 | 19 | 63.16% | 89.47% |

## Nota metodologica

La comparacion principal del piloto se realiza sobre la lectura normalizada,
ignorando ceros a la izquierda. Esto refleja mejor la validacion operativa,
ya que `readings.csv` almacena la lectura manual como valor entero sin padding.
