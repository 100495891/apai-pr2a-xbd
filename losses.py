"""
losses.py — Funciones de pérdida para xBD (Bloque M2).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ─── Clases de Pérdida Personalizadas (M2) ────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(
            logits, targets, weight=self.alpha, 
            ignore_index=self.ignore_index, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_background=False):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits, targets):
        # Aplicamos softmax para obtener probabilidades (0 a 1)
        probs = F.softmax(logits, dim=1)
        num_classes = probs.size(1)
        
        # Convertimos el target a one-hot encoding (B, C, H, W)
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # Intersección y Unión por clase
        dims = (0, 2, 3) # Reducimos por batch, altura y anchura (solo queda la clase)
        intersection = torch.sum(probs * targets_one_hot, dim=dims)
        cardinality = torch.sum(probs + targets_one_hot, dim=dims)
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        
        if self.ignore_background:
            # Ignoramos la clase 0 (background) en el cálculo de la media
            dice_loss = 1.0 - dice_score[1:].mean()
        else:
            dice_loss = 1.0 - dice_score.mean()
            
        return dice_loss

# ─── Factory ──────────────────────────────────────────────────────────────────

def get_loss(name: str = "cross_entropy", **kwargs) -> nn.Module:
    name = name.lower()

    if name == "cross_entropy":
        return nn.CrossEntropyLoss(**kwargs)

    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=kwargs.get("class_weights"))
        
    if name == "focal":
        return FocalLoss(
            alpha=kwargs.get("class_weights"), 
            gamma=kwargs.get("gamma", 2.0)
        )
        
    if name == "dice":
        return DiceLoss(ignore_background=True)

    raise ValueError(f"Pérdida desconocida: '{name}'.")

# ─── Utilidad para M2: distribución de clases del train set ───────────────────

def compute_class_pixel_distribution(dataloader, num_classes: int = 5):
    import numpy as np
    from collections import Counter
    from tqdm import tqdm

    counts = Counter()
    for sample in tqdm(dataloader, desc="Contando píxeles por clase"):
        masks = sample["mask_patch"].numpy().ravel()
        c = Counter(masks.tolist())
        counts.update(c)

    return {i: counts.get(i, 0) for i in range(num_classes)}