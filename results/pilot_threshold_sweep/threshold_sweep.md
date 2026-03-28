# Barrida de Threshold de Validacion

- Fecha: 2026-03-24T09:37:26
- Fuente: `logs\pilot_eval\predictions.csv`

| Threshold | Auto-validation rate | Precision | Error rate | Review rate | No-detection rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 91.67% | 89.45% | 10.55% | 1.67% | 6.67% |
| 0.55 | 91.33% | 89.78% | 10.22% | 2.00% | 6.67% |
| 0.60 | 89.67% | 89.96% | 10.04% | 3.67% | 6.67% |
| 0.65 | 88.00% | 90.15% | 9.85% | 5.33% | 6.67% |
| 0.70 | 83.33% | 90.40% | 9.60% | 10.00% | 6.67% |
| 0.75 | 70.33% | 91.47% | 8.53% | 23.00% | 6.67% |
| 0.80 | 53.33% | 90.62% | 9.38% | 40.00% | 6.67% |
| 0.85 | 31.00% | 90.32% | 9.68% | 62.33% | 6.67% |
| 0.90 | 8.67% | 88.46% | 11.54% | 84.67% | 6.67% |
| 0.95 | 0.00% | N/A | N/A | 93.33% | 6.67% |

La barrida recalcula el estado `valid` vs `needs_review` a partir de
`global_confidence`, sin volver a ejecutar el pipeline.
