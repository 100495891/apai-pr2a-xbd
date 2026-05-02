# Baseline Compartido — Guía del Equipo
## Proyecto APAI-PR2A | Segmentación Semántica xBD | Grupo 08

---

## ¿Qué es este documento?

Este documento explica los cambios implementados en la rama `main` del proyecto que forman el **baseline compartido** del que parten los tres miembros del equipo (M1, M2, M3). Incluye qué se ha hecho, cómo verificarlo y qué debe añadir cada uno en su rama propia para resolver el problema de desbalanceo de clases.

---

## El problema: desbalanceo severo de clases

El dataset xBD_UC3M tiene una distribución de clases **muy desigual**. Aproximadamente:

| Clase | Descripción | % aprox. muestras |
|-------|-------------|-------------------|
| 0 | background | — (píxeles vacíos) |
| 1 | no-damage | ~84 % |
| 2 | minor-damage | ~9 % |
| 3 | major-damage | ~5 % |
| 4 | destroyed | ~2 % |

Sin ninguna corrección, el modelo aprende a predecir casi siempre "no-damage" y obtiene métricas de IoU cercanas a 0 en las clases de daño reales (minor, major, destroyed). Esto hace que la métrica principal del proyecto (mIoU sobre las 4 clases de daño, excluyendo background) sea inutilizable.

---

## Qué hay implementado en `main` (baseline compartido)

### Celda "BASELINE COMPARTIDO" del notebook

Es la celda cuya primera línea de código es `from collections import Counter`. Reemplaza la creación simple de datasets y DataLoaders que había antes. Implementa exactamente la misma estrategia que usamos en el Proyecto 1 (células 37 y 41 de APAI_Proyecto1_Grupo08):

#### Paso 1 — TEST separado (no se toca)
```python
dataset_test = xBDDataset(split=["test"], ...)
```
El split de test se carga aparte y nunca participa en el balanceo.

#### Paso 2 — Pool combinado train+val
```python
dataset_all = xBDDataset(split=["train", "val"], transform=None, ...)
```
Se combinan train y val en un único pool de muestras para poder re-dividir de forma controlada.

#### Paso 3 — Re-split estratificado 85% / 15%
```python
train_idx, val_idx = train_test_split(
    all_indices,
    test_size=0.15, stratify=dominant_classes, random_state=42
)
```
Usando `sklearn.train_test_split` con `stratify`, garantizamos que ambos splits tengan la **misma distribución proporcional** de clases. `random_state=42` asegura reproducibilidad.

La clase de estratificación de cada muestra es la **clase de daño más grave** presente en el patch (sin background), igual que en P1.

#### Paso 4 — WeightedRandomSampler en el DataLoader de train
```python
# Fórmula exacta del Proyecto 1 (célula 41):
cls_w = { cls: total_train / (num_cls * count) for cls, count in counts.items() }
sample_weights = torch.DoubleTensor([cls_w[d] for d in train_dominant])

wrs_sampler = WeightedRandomSampler(
    weights=sample_weights, num_samples=len(sample_weights), replacement=True
)

DataLoader(dataset_train, sampler=wrs_sampler, ...)  # NO shuffle, WRS lo sustituye
```
Los pesos hacen que muestras de clases minoritarias (major-damage, destroyed) se seleccionen con más frecuencia, compensando el desbalanceo a nivel de batch.

#### Resultado final
```python
dataloaders = {
    "Train": ...,   # con WRS activo
    "Val":   ...,   # sin WRS (distribución real)
    "Test":  ...,   # sin WRS (distribución real)
}
```
Estas variables `dataloaders` y `dataset_train/val/test` son las que usa el resto del notebook.

---

## Cómo verificar que funciona (antes de vuestros experimentos)

Ejecutar la celda `from collections import Counter`. Debéis ver un output como este (los números exactos varían según el dataset completo):

```
Cargando splits train+val combinados ...
  Total muestras combinadas: 1015

Distribución ANTES del re-split:
  no-damage      :   643  (63.3%)
  minor-damage   :   171  (16.8%)
  major-damage   :    86  (8.5%)
  destroyed      :   115  (11.3%)

Re-split 85/15 → Train: 862  Val: 153

Pesos WeightedRandomSampler:
  no-damage      : 0.316×    ← submuestreado (clase dominante)
  minor-damage   : 1.189×
  major-damage   : 2.362×    ← sobremuestreado (clase escasa)
  destroyed      : 1.759×

✅ DataLoaders listos:
  Train :   862 muestras  (WRS activo)
  Val   :   153 muestras
  Test  :   287 muestras
```

Lo importante es que los pesos de no-damage sean < 1× y los de major-damage / destroyed sean > 1×. Si el IoU de major-damage y destroyed ya no es 0.000 tras entrenar, está funcionando.

---

## Qué debe añadir cada miembro en su rama

El baseline corrige el desbalanceo a **nivel de DataLoader** (qué muestras se seleccionan por batch). Cada miembro puede complementarlo desde su ángulo específico:

---

### M1 — Data Augmentation (Lara — rama `feat/augment`)

**Lo que ya tienes:** JointTransform, 5 configs de ablación (aug_none, aug_flips, aug_geo, aug_full, aug_crop), WeightedRandomSampler integrado.

**Sugerencia complementaria al baseline:** El augmentation actúa como segunda capa de balanceo. Cuando el WRS sobremuestra una imagen de "destroyed", JointTransform genera variantes diferentes de esa misma imagen (flips, rotaciones, crops) → más diversidad sin perder el efecto de balanceo.

**Experimento recomendado para el informe:** Compara en tabla:
- Baseline sin aug + sin WRS → (referencia)
- Baseline con WRS + sin aug
- Baseline con WRS + aug_flips
- Baseline con WRS + aug_crop ← tu mejor resultado esperado

Esto muestra que WRS y augmentation son **complementarios**, no redundantes.

**Cómo inyectar tu transform en la celda de baseline:**
```python
# En celda 60, línea "baseline_aug =":
baseline_aug = JointTransform(hflip_p=0.5, vflip_p=0.5)  # baseline mínimo

# En tus experimentos M1, sustituye por:
# dataset_train = TransformSubset(dataset_all, train_idx, transform=cfg['transform'])
```
`TransformSubset` ya está definida en la celda 60 y acepta cualquier `JointTransform`.

---

### M2 — Funciones de Pérdida (rama `feat/losses`)

**Tu archivo:** `losses.py` con `get_loss(name, ...)`.

**El problema que resuelves:** CrossEntropy estándar trata todas las clases igual → ignora que "destroyed" aparece en el 2% de los batches. Incluso con WRS, la función de pérdida puede seguir favoreciendo la clase mayoritaria.

**Sugerencias concretas:**

**Opción A — CrossEntropy ponderada (punto de partida fácil):**
```python
# Los pesos son inversos a la frecuencia (igual que WRS pero en la loss)
class_weights = torch.tensor([
    total / (num_cls * counts[cls]) for cls in range(num_cls)
], dtype=torch.float32).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=255)
```
Combinar WRS + CrossEntropy ponderada suele ser la combinación más estable.

**Opción B — Focal Loss:**
```python
# Focal Loss: reduce el peso de ejemplos "fáciles" (no-damage bien clasificado)
# y amplifica la señal de ejemplos "difíciles" (minor/major/destroyed)
# gamma=2 es el valor canónico del paper original (Lin et al. 2017)
loss = FocalLoss(gamma=2, alpha=class_weights, ignore_index=255)
```
Focal Loss es especialmente efectiva en conjuntos desbalanceados porque suprime la señal de los ejemplos donde el modelo ya tiene alta confianza.

**Opción C — Dice Loss:**
```python
# Dice Loss: directamente optimiza el IoU (lo que medimos al final)
# No sufre del desbalanceo porque normaliza por la unión de predicción y GT
loss = DiceLoss(smooth=1.0, ignore_background=True)
```
Dice Loss es invariante al desbalanceo por construcción, pero puede ser inestable al principio del entrenamiento.

**Experimento recomendado:**
| Loss | mIoU val | IoU major | IoU destroyed |
|------|----------|-----------|---------------|
| CrossEntropy (baseline) | | | |
| CrossEntropy + pesos | | | |
| Focal (γ=2) | | | |
| Dice | | | |
| Focal + Dice (combinada) | | | |

**Cómo usar `get_loss` en el training loop:** En `train_utils.py` ya hay un hook. Pasas `loss_fn=get_loss('focal')` al llamar a `train_model_xbd`.

---

### M3 — Arquitectura y Entrenamiento (rama `feat/arch`)

**Tu archivo:** `arch.py` con `get_deeplabv3_xbd(...)`.

**El problema que resuelves:** La arquitectura puede reforzar el sesgo hacia clases mayoritarias si el clasificador final no está bien calibrado para clases raras.

**Sugerencias concretas:**

**Opción A — Clasificador auxiliar (aux_classifier):**
```python
model = get_deeplabv3_xbd(num_classes=5, aux_classifier=True)
# Añade FCNHead sobre layer3 del backbone. Su pérdida se pondera:
# loss_total = loss_main + aux_weight * loss_aux   (aux_weight=0.4 sugerido)
```
La cabeza auxiliar actúa como regularización: obliga al backbone a aprender representaciones útiles para clases difíciles en una capa más temprana.

**Opción B — Congelar backbone (fine-tuning más controlado):**
```python
model = get_deeplabv3_xbd(num_classes=5, freeze_backbone="partial")
# Solo layer4 y el clasificador se entrenan. El backbone mantiene features
# preentrenados en ImageNet/COCO → más estable con pocas muestras de clase rara.
```
Con el backbone parcialmente congelado, el modelo no "olvida" features genéricos al sobreajustarse a "no-damage".

**Opción C — Tasas ASPP ajustadas:**
```python
model = get_deeplabv3_xbd(num_classes=5, aspp_rates=(6, 12, 18))
# vs. baseline (12, 24, 36). Tasas menores → receptivo más local.
# Para xBD con parches 128×128, tasas altas pueden captar contexto fuera del parche.
```

**Opción D — Scheduler de LR con warm-up:**
```python
# ReduceLROnPlateau ya está en el baseline. Prueba añadir warm-up lineal:
# epochs 1-3: LR crece de lr/10 a lr, luego decae normalmente
# Esto ayuda con WRS porque los primeros batches pueden ser "raros" (todo destroyed)
```

**Experimento recomendado:**
| Config | mIoU | IoU minor | IoU major | IoU dest |
|--------|------|-----------|-----------|----------|
| Baseline (ResNet101, sin aux) | | | | |
| + aux_classifier (aux_w=0.4) | | | | |
| + freeze_backbone="partial" | | | | |
| + aspp_rates=(6,12,18) | | | | |
| Mejor combinación | | | | |

---

## Estado del repositorio

```
main
├── APAI_Pr2A_ImageSegmentation_2025_2026.ipynb  ← celda "from collections import Counter" = baseline balanceado
├── xbd_dataset.py    ← DataLoader base (común, no tocar)
├── train_utils.py    ← train_model_xbd / test_segmentation_model_xbd (común)
├── augment.py        ← JointTransform, ABLATION_CONFIGS (M1)
├── losses.py         ← get_loss (M2)
├── arch.py           ← get_deeplabv3_xbd (M3)
└── README.md

feat/augment  ← rama de Lara (M1)
feat/losses   ← rama de M2
feat/arch     ← rama de M3
```

**Flujo de trabajo para incorporar el baseline (ejecutar una sola vez):**
```bash
git checkout feat/losses    # o feat/arch según quién seas
git merge main
git push origin feat/losses
```
Si hay conflictos en el notebook, quedarse siempre con la versión que tiene `from collections import Counter` en la celda del baseline.

---

## Resumen rápido

| Qué | Dónde | Quién lo usa |
|-----|-------|--------------|
| Re-split estratificado 85/15 | Celda `from collections import Counter` | Todos |
| WeightedRandomSampler en train | Celda `from collections import Counter` | Todos |
| JointTransform sincronizado | `augment.py` | M1 (Lara) |
| Focal Loss / Dice Loss | `losses.py` | M2 |
| aux_classifier / freeze_backbone | `arch.py` | M3 |

El baseline de la celda 60 es el punto de partida **igual para todos**. Las mejoras de M1, M2 y M3 son **complementarias** entre sí y cada una aborda el desbalanceo desde un ángulo diferente.
