"""
losses.py — Funciones de pérdida para xBD (Bloque M2).

Estado actual: STUB con CrossEntropyLoss únicamente.
M2 amplía este módulo con WeightedCE, FocalLoss, DiceLoss, TverskyLoss, ComboLoss.

Interfaz contractual
--------------------
Todas las pérdidas deben aceptar la llamada:

    loss = criterion(logits, target)

donde:
    logits : Tensor (N, C, H, W) — salida cruda del modelo (antes de softmax)
    target : Tensor (N, H, W)    — etiquetas enteras 0..C-1

Y devolver un escalar Tensor para hacer .backward().

Uso
---
    from losses import get_loss

    criterion = get_loss("cross_entropy")
    # Cuando M2 termine, también funcionará:
    # criterion = get_loss("focal", gamma=2.0)
    # criterion = get_loss("weighted_ce", class_weights=weights)
"""

import torch
import torch.nn as nn


# ─── Factory ──────────────────────────────────────────────────────────────────

def get_loss(name: str = "cross_entropy", **kwargs) -> nn.Module:
    """
    Factory de pérdidas. M2 amplía este registro.

    Parámetros
    ----------
    name : str — identificador de la pérdida
    kwargs     : argumentos específicos de cada pérdida

    Pérdidas disponibles (M2 ampliará):
      - 'cross_entropy' : torch.nn.CrossEntropyLoss
    """
    name = name.lower()

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)

    # ─────────────────────────────────────────────────────────────────────────
    # ↓↓↓ A IMPLEMENTAR POR M2 ↓↓↓
    # if name == "weighted_ce":
    #     return WeightedCrossEntropy(**kwargs)
    # if name == "focal":
    #     return FocalLoss(**kwargs)
    # if name == "dice":
    #     return DiceLoss(**kwargs)
    # if name == "tversky":
    #     return TverskyLoss(**kwargs)
    # if name == "combo":
    #     return ComboLoss(**kwargs)
    # ↑↑↑ A IMPLEMENTAR POR M2 ↑↑↑
    # ─────────────────────────────────────────────────────────────────────────

    raise ValueError(
        f"Pérdida desconocida: '{name}'. Disponibles: ['cross_entropy']. "
        f"M2: añade tu pérdida en losses.py:get_loss()."
    )


# ─── Utilidad para M2: distribución de clases del train set ───────────────────

def compute_class_pixel_distribution(dataloader, num_classes: int = 5):
    """
    Cuenta los píxeles de cada clase en todo el dataloader.
    Útil para calcular pesos inversamente proporcionales a la frecuencia
    (WeightedCrossEntropy).

    Devuelve un dict {class_idx: pixel_count}.

    Ejemplo de uso por M2:
        counts = compute_class_pixel_distribution(dataloaders["Train"])
        total  = sum(counts.values())
        weights = torch.tensor(
            [total / (num_classes * counts[i]) if counts[i] > 0 else 0.0
             for i in range(num_classes)],
            dtype=torch.float32,
        )
        criterion = get_loss("weighted_ce", class_weights=weights)
    """
    import numpy as np
    from collections import Counter
    from tqdm import tqdm

    counts = Counter()
    for sample in tqdm(dataloader, desc="Contando píxeles por clase"):
        masks = sample["mask_patch"].numpy().ravel()
        c = Counter(masks.tolist())
        counts.update(c)

    return {i: counts.get(i, 0) for i in range(num_classes)}
