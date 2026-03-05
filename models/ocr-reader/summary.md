# f2/iter6-mask-lab — Lector OCR (Fine-Tuned)
> Iter 6: Fine-tuning en_PP-OCRv4_mobile_rec con crops regenerados con la máscara LAB refinada

- **Fecha**: 2026-03-19
- **Modelo base**: en_PP-OCRv4_mobile_rec
- **Fine-tuned**: Si (diccionario digit-only)
- **Epochs**: 20
- **Learning rate**: 0.0001
- **Batch size**: 8
- **Sanitization**: True
- **Split**: val (120 imagenes)
- **Tiempo promedio**: 17 ms/imagen

## Metricas

- **Exact Match Accuracy: 0.8917** (principal)
- CER: 0.0262
- CRR: 0.9738

### Con decimales (n=84)
- EM Accuracy: 0.8929
- CER: 0.0249

### Sin decimales (n=36)
- EM Accuracy: 0.8889
- CER: 0.0287
