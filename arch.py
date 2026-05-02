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
    Soluciona el error de carga de pesos preentrenados forzando aux_loss=True.
    """
    # ── 1. Carga del modelo base ──────────────────────────────────────────────
    # NOTA IMPORTANTE: Para cargar los pesos preentrenados de torchvision, 
    # DEBEMOS poner aux_loss=True, ya que los pesos originales incluyen la rama auxiliar.
    # Si ponemos False, torchvision lanza un ValueError.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = deeplabv3_resnet101(
            pretrained=True,
            progress=True,
            aux_loss=True,  # <--- Siempre True para evitar el ValueError
        )

    # ── 2. Cabeza principal (ASPP + classifier) ──────────────────────────────
    # Sustituimos la cabeza de 21 clases (COCO) por la de 5 clases (xBD)
    if aspp_rates is None:
        model.classifier = DeepLabHead(2048, num_classes)
    else:
        # Permite probar las tasas (6, 12, 18) sugeridas en el enunciado
        model.classifier = DeepLabHead(2048, num_classes, atrous_rates=aspp_rates)

    # ── 3. Gestión de la cabeza auxiliar (M3) ─────────────────────────────────
    if aux_classifier:
        # Si el experimento pide auxiliar, adaptamos la cabeza a nuestras 5 clases
        # ResNet-101 layer3 tiene 1024 canales de salida
        model.aux_classifier = FCNHead(1024, num_classes)
    else:
        # Si el experimento NO quiere auxiliar (Baseline), la eliminamos
        # después de haber cargado los pesos preentrenados con éxito.
        model.aux_classifier = None

    # ── 4. Congelado del backbone (M3) ───────────────────────────────────────────
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

from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models import resnet18

def get_deeplabv3_xbd_resnet18(num_classes: int = 5, aux_classifier: bool = False):
    """
    Variante ligera con ResNet-18 para entrenamientos rápidos de prueba.
    """
    # 1. Cargamos el esqueleto básico y el backbone ligero
    model = deeplabv3_resnet101(pretrained=True, progress=True)
    backbone = resnet18(pretrained=True)
    
    # 2. Conectamos las capas del backbone
    return_layers = {'layer4': 'out'}
    if aux_classifier:
        return_layers['layer3'] = 'aux'
        
    model.backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    
    # 3. Mantenemos la resolución espacial (cambiando strides por dilations)
    model.backbone.layer3[0].conv1.stride = (1, 1)
    model.backbone.layer4[0].conv1.stride = (1, 1)
    model.backbone.layer3[0].conv1.dilation = (2, 2)
    model.backbone.layer3[0].conv1.padding = (2, 2)
    model.backbone.layer4[0].conv1.dilation = (4, 4)
    model.backbone.layer4[0].conv1.padding = (4, 4)
    model.backbone.layer3[0].downsample[0].stride = (1, 1)
    model.backbone.layer4[0].downsample[0].stride = (1, 1)

    # 4. Adaptamos las cabezas a los canales de ResNet-18 (512 en vez de 2048)
    model.classifier = DeepLabHead(512, num_classes)
    
    if aux_classifier:
        model.aux_classifier = FCNHead(256, num_classes)
    else:
        model.aux_classifier = None
        
    return model