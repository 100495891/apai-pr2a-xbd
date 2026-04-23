# APAI · Proyecto P2A — xBD + DeepLab-V3

## Estructura

```
.
├── notebook.ipynb     ← driver para Colab
├── xbd_dataset.py     ← clase xBDDataset
├── augment.py         ← M1: JointTransform + ABLATION_CONFIGS
├── losses.py          ← M2: get_loss(name)  (stub, M2 amplía)
├── arch.py            ← M3: get_deeplabv3_xbd con hooks aux/freeze/aspp
└── train_utils.py     ← train_model_xbd (con hook aux), test_segmentation_model_xbd
```

## Setup inicial (una vez por persona/máquina)

```bash
git clone https://github.com/<usuario>/<repo>.git
cd <repo>
pipx install nbstripout      # si pipx no, usar: pip install --user nbstripout
nbstripout --install
```

## Reparto de ramas

| Rama          | Persona | Edita                          |
|---------------|---------|--------------------------------|
| `feat/augment`| M1      | `augment.py`                   |
| `feat/losses` | M2      | `losses.py`                    |
| `feat/arch`   | M3      | `arch.py` (+ optimizer/scheduler en notebook) |

`train_utils.py` y `xbd_dataset.py` son comunes — **no se tocan** después de
la Fase 0 sin avisar a las otras dos.

## Protocolo de comparabilidad (NO desviarse)

Para que los runs de las tres ramas sean comparables:

- **Semilla**: `999`
- **Patch size**: `128` para ablaciones; `256` para el run final
- **Epochs**: `3` para ablaciones
- **Batch train**: `4`
- **Métrica principal**: mIoU **excluyendo background** (índice 0)
- Reportar siempre **IoU por clase** además del mIoU
- **Dataset** en Drive en `MyDrive/VA_P2/xBD_UC3M/`

## Workflow en Colab

1. Subir `notebook.ipynb` desde GitHub (Colab → Open notebook → GitHub).
2. Editar `BRANCH = "..."` en la celda 1 con tu rama.
3. Editar `REPO_URL = "..."` con la URL de este repo.
4. Activar GPU: *Entorno de ejecución → Cambiar tipo de entorno → GPU T4*.
5. Ejecutar todo.

Para hacer push de cambios desde Colab:

```python
!cd /content/repo && git add <archivo>.py && git commit -m "..." && git push
```

## Smoke test

Para verificar que todo funciona end-to-end sin esperar horas:
en el notebook, dejar `SMOKE_TEST = True` (entrena con 30 train / 10 val).
Debe completarse en pocos minutos en GPU.
