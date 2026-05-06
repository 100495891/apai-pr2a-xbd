# APAI · Proyecto P2A — Segmentación de Daños con DeepLab-V3 sobre xBD
**Grupo 08 · Curso 2025/2026**
María Montanet Estebaranz · Guillermo Sánchez Sanlucas · Lara Gómez Carrera

---

## Descripción general

Este proyecto aborda la segmentación semántica multiclase de daños post-desastre sobre el dataset **xBD**, utilizando DeepLab-V3 con backbone ResNet-101. El problema central es el desbalanceo extremo de clases: `no-damage` representa ~84% de las muestras, mientras que `major-damage` y `destroyed` apenas alcanzan el 5% y el 2%.

El trabajo se organiza en tres bloques de extensión independientes (M1, M2, M3), cada uno con su propio cuaderno de evaluación, y un cuaderno final integrado que combina las mejores decisiones de cada bloque.

---

## Estructura del repositorio

```
.
├── APAI_Pr2A_ImageSegmentation_2025_2026_M1.ipynb   ← Cuaderno de evaluación M1
├── APAI_Pr2A_ImageSegmentation_2025_2026_M2.ipynb   ← Cuaderno de evaluación M2
├── APAI_Pr2A_ImageSegmentation_2025_2026_M3.ipynb   ← Cuaderno de evaluación M3
├── APAI_Pr2A_ImageSegmentation_FINAL.ipynb          ← Cuaderno final integrado
│
├── xbd_dataset.py       ← Clase xBDDataset + baseline compartido
├── augment.py           ← M1: JointTransform y ABLATION_CONFIGS
├── losses.py            ← M2: funciones de pérdida y selector get_loss()
├── arch.py              ← M3: get_deeplabv3_xbd() con variantes arquitectónicas
├── train_utils.py       ← Bucle de entrenamiento y evaluación (común)
├── external.py          ← Utilidades auxiliares de visualización
│
├── APAI_Proyecto2_2025_2026_Informe_Grupo08   ← Informe del proyecto
├── mejores_modelos.txt  ← Registro de los mejores runs por bloque
├── requirements.txt     ← Dependencias del proyecto
└── README.md            ← Este fichero
```

---

## Cuadernos de evaluación

### M1 — Data Augmentation (`APAI_Pr2A_ImageSegmentation_2025_2026_M1.ipynb`)
**Responsable: Lara Gómez**

Explora el aumento de datos sincronizado imagen–máscara como estrategia complementaria al balanceo por muestreo. Contiene:

- Implementación del **baseline compartido**: re-split estratificado 85/15 y `WeightedRandomSampler`.
- Ablación de 5 configuraciones de aumento (`aug_none`, `aug_flips`, `aug_geo`, `aug_full`, `aug_crop`), evaluadas durante 10 épocas con `img_size=128`.
- Análisis del efecto combinado de `aug_crop` con y sin WRS sobre las clases minoritarias.

**Conclusión principal:** con pocas épocas, `aug_none` obtiene el mejor Val mIoU (0,6226); el aumento agresivo necesita un horizonte de entrenamiento mayor para ser beneficioso. La combinación WRS + `aug_crop` mejora major-damage en +72,7% y minor-damage en +37,3%.

---

### M2 — Función de Pérdida (`APAI_Pr2A_ImageSegmentation_2025_2026_M2.ipynb`)
**Responsable: María Montanet**

Analiza cómo distintas funciones de pérdida interactúan con la estrategia de muestreo. Contiene:

- Ablación 2D: 6 funciones de pérdida × 2 estrategias de muestreo (con/sin WRS), 8 épocas por configuración con `img_size=128`, `batch=4`, AdamW + CosineAnnealing.
- Identificación del **colapso de pérdidas regionales** (Dice, Focal-Tversky) al combinarse con WRS activo (mIoU ≈ 0,003).
- Selección de **ComboLoss** (α·CE + (1-α)·Dice, α=0,5) como pérdida final por su robustez ante ambos regímenes de muestreo.

**Conclusión principal:** el balanceo en el sampler y el balanceo en la pérdida son alternativos, no complementarios. Combo es la única pérdida que mantiene su rendimiento independientemente de si el WRS está activo.

---

### M3 — Arquitectura y Optimizador (`APAI_Pr2A_ImageSegmentation_2025_2026_M3.ipynb`)
**Responsable: Guillermo Sánchez**

Explora mejoras en la arquitectura de red y la estrategia de optimización. Contiene:

- Comparación de optimizadores (**SGD vs. AdamW**) sobre ResNet-18 para selección rápida del motor de entrenamiento. SGD colapsa a mIoU ≈ 0,0; AdamW + PolynomialLR converge establemente.
- Evaluación de cuatro variantes arquitectónicas sobre ResNet-101: baseline, clasificador auxiliar (`aux_classifier`), congelación de capas tempranas (`freeze_backbone`) y tasas ASPP reducidas.

**Conclusión principal:** AdamW + PolynomialLR se fija como motor definitivo. Las variantes arquitectónicas aportan mejoras marginales en clases escasas, pero no son suficientes por sí solas para contrarrestar el desbalanceo extremo.

---

### Cuaderno Final (`APAI_Pr2A_ImageSegmentation_FINAL.ipynb`)

Integra las mejores decisiones de los tres bloques en un único pipeline de entrenamiento y evaluación completo:

- **Datos:** re-split estratificado 85/15, sin WRS (incompatible con ComboLoss en este contexto).
- **Pérdida:** ComboLoss (CE + Dice, α=0,5).
- **Arquitectura:** DeepLab-V3 + ResNet-101, tasas ASPP ajustadas.
- **Optimizador:** AdamW + PolynomialLR.
- **Entrenamiento:** 15 épocas sobre el conjunto completo, evaluación sobre `test_with_labels` (5 desastres no vistos).

**Resultados finales sobre test:**

| Clase | IoU |
|---|---|
| no-damage | 48,41% |
| minor-damage | 0,98% |
| major-damage | 0,01% |
| destroyed | 0,54% |
| **mIoU (4 clases)** | **12,49%** |

---

## Módulos Python factorizados

El código reutilizable entre cuadernos se ha extraído a los siguientes módulos:

### `xbd_dataset.py`
Clase `xBDDataset` que gestiona la carga, normalización y construcción de máscaras de segmentación a partir de los ficheros JSON de xBD. Incluye el baseline compartido (re-split estratificado + WRS). **Módulo común — no modificar sin consenso del equipo.**

### `augment.py`
Define `JointTransform`, que aplica transformaciones espaciales (flip, rotación, RandomResizedCrop) de forma sincronizada sobre la imagen pre-desastre, la imagen post-desastre y la máscara, usando interpolación BILINEAR para imágenes y NEAREST para máscaras. También exporta `ABLATION_CONFIGS` con las 5 configuraciones predefinidas del estudio de ablación de M1.

### `losses.py`
Implementa las funciones de pérdida para segmentación multiclase desbalanceada: `DiceLoss`, `FocalLoss`, `TverskyLoss`, `FocalTverskyLoss` y `ComboLoss`. Expone el selector unificado `get_loss(name, weight=None, **kwargs)` para instanciar cualquier pérdida por nombre.

### `arch.py`
Expone `get_deeplabv3_xbd(num_classes, ...)`, que construye DeepLab-V3 con ResNet-101 preentrenado y admite las variantes exploradas en M3: `aux_classifier=True` (cabeza auxiliar FCNHead), `freeze_backbone="partial"` (congelación de capas tempranas) y `aspp_rates=(...)` (tasas de dilatación del ASPP personalizables). También incluye `get_deeplabv3_xbd_resnet18()` como variante ligera para experimentación rápida.

### `train_utils.py`
Contiene `train_model_xbd()`, el bucle de entrenamiento con checkpointing por época y soporte al clasificador auxiliar de M3 (`use_aux=True, aux_weight=0.4`), y `test_segmentation_model_xbd()`, que evalúa el modelo calculando IoU por clase, genera la matriz de confusión normalizada por fila y guarda las predicciones visuales. **Módulo común — no modificar sin consenso del equipo.**

### `external.py`
Utilidades auxiliares de visualización y extracción de activaciones intermedias de la red, usadas principalmente en los experimentos de análisis del ASPP.

---

## Configuración y ejecución

**Dependencias:**
```bash
pip install -r requirements.txt
```

**Ejecución en Kaggle / Colab:** abrir el cuaderno correspondiente, configurar `data_dir` con la ruta al dataset xBD y activar aceleración GPU. Para el cuaderno final, asegurarse de que los módulos `.py` están en el mismo directorio de trabajo.

**Semilla de reproducibilidad:** todos los experimentos usan `random_state=42` para el split y `manualSeed=999` para PyTorch.