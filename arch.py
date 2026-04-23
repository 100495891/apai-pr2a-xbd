"""
arch.py — Arquitectura DeepLab-V3 adaptada a xBD (Bloque M3).

Por defecto reproduce el baseline (cell 64 del notebook de M1):
    DeepLabV3-ResNet101 pretrained, classifier = DeepLabHead(2048, 5).

M3 amplía este módulo con:
  - aux_classifier (FCNHead sobre layer3) y su pérdida ponderada
  - congelado/descongelado del backbone
  - tasas de dilatación del ASPP personalizables
  - opcionalmente, variante ligera con ResNet-18

Uso
---
    from arch import get_deeplabv3_xbd

    # Baseline (idéntico a M1):
    model = get_deeplabv3_xbd(num_classes=5)

    # M3 — con aux_classifier:
    model = get_deeplabv3_xbd(num_classes=5, aux_classifier=True)

    # M3 — con backbone parcialmente congelado:
    model = get_deeplabv3_xbd(num_classes=5, freeze_backbone="partial")

    # M3 — con tasas ASPP del enunciado:
    model = get_deeplabv3_xbd(num_classes=5, aspp_rates=(6, 12, 18))
"""

import warnings
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision.models.segmentation.fcn import FCNHead


def get_deeplabv3_xbd(
    num_classes: int = 5,
    aux_classifier: bool = False,
    freeze_backbone: str = "none",
    aspp_rates=None,
):
    """
    Construye DeepLabV3 con ResNet-101 preentrenado y cabezas adaptadas a xBD.

    Parámetros
    ----------
    num_classes     : número de clases de salida (5 para xBD).
    aux_classifier  : si True, añade FCNHead sobre layer3 del backbone.
                      Su salida queda en outputs["aux"] y se usa en la pérdida
                      auxiliar del bucle de entrenamiento (ver train_utils.py).
    freeze_backbone : 'none'    → todo el backbone se entrena (default, baseline).
                      'all'     → todo el backbone congelado.
                      'partial' → solo layer4 descongelado (resto congelado).
    aspp_rates      : tupla con tasas de dilatación del ASPP. None usa
                      las del baseline (12, 24, 36). El enunciado sugiere
                      probar (6, 12, 18).

    Devuelve
    --------
    nn.Module — el modelo, todavía en CPU (mover con .to(device) fuera).
    """
    # NOTA: usamos pretrained=True (compat con torchvision usado por M1).
    # En torchvision >=0.13 emite DeprecationWarning; suprimimos para limpiar logs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = deeplabv3_resnet101(
            pretrained=True,
            progress=True,
            aux_loss=aux_classifier,
        )

    # ── Cabeza principal (ASPP + classifier) ──────────────────────────────────
    if aspp_rates is None:
        model.classifier = DeepLabHead(2048, num_classes)
    else:
        model.classifier = DeepLabHead(2048, num_classes, atrous_rates=aspp_rates)

    # ── Cabeza auxiliar (FCNHead sobre layer3) ────────────────────────────────
    if aux_classifier:
        # ResNet-101 layer3 tiene 1024 canales de salida
        model.aux_classifier = FCNHead(1024, num_classes)

    # ── Congelado del backbone (M3) ───────────────────────────────────────────
    if freeze_backbone == "none":
        pass
    elif freeze_backbone == "all":
        for p in model.backbone.parameters():
            p.requires_grad = False
    elif freeze_backbone == "partial":
        # Descongelar SOLO layer4; congelar el resto
        for name, p in model.backbone.named_parameters():
            p.requires_grad = "layer4" in name
    else:
        raise ValueError(
            f"freeze_backbone debe ser 'none' | 'all' | 'partial', "
            f"recibido '{freeze_backbone}'"
        )

    return model


# ─── Helper opcional para M3: variante ligera con ResNet-18 ───────────────────

def get_deeplabv3_xbd_resnet18(num_classes: int = 5):
    """
    Variante ligera para entrenamientos rápidos de prueba (cell 75 del baseline).
    M3 puede usarla para iterar más rápido en el barrido de hiperparámetros.

    Pendiente de implementación detallada por M3 (requiere ensamblar manualmente
    DeepLabV3 sobre torchvision.models.resnet18).
    """
    raise NotImplementedError(
        "Variante ResNet-18 pendiente. M3 implementa esto si decide usarla."
    )
