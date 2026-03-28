# Barrida de Threshold de Validacion

- Fecha: 2026-03-24T09:37:24
- Fuente: `logs\end_to_end_eval\predictions.csv`

| Threshold | Auto-validation rate | Precision | Error rate | Review rate | No-detection rate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.50 | 100.00% | 95.83% | 4.17% | 0.00% | 0.00% |
| 0.55 | 100.00% | 95.83% | 4.17% | 0.00% | 0.00% |
| 0.60 | 100.00% | 95.83% | 4.17% | 0.00% | 0.00% |
| 0.65 | 100.00% | 95.83% | 4.17% | 0.00% | 0.00% |
| 0.70 | 93.33% | 96.43% | 3.57% | 6.67% | 0.00% |
| 0.75 | 85.00% | 98.04% | 1.96% | 15.00% | 0.00% |
| 0.80 | 60.00% | 100.00% | 0.00% | 40.00% | 0.00% |
| 0.85 | 35.00% | 100.00% | 0.00% | 65.00% | 0.00% |
| 0.90 | 10.00% | 100.00% | 0.00% | 90.00% | 0.00% |
| 0.95 | 0.00% | N/A | N/A | 100.00% | 0.00% |

La barrida recalcula el estado `valid` vs `needs_review` a partir de
`global_confidence`, sin volver a ejecutar el pipeline.
