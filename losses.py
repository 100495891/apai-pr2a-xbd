"""
losses.py — Funciones de pérdida para segmentación multi-clase desbalanceada (xBD).

Provee:
  - DiceLoss          : Dice multi-clase, con opción de ignorar background y pesos por clase
  - FocalLoss         : Focal multi-clase con alpha por clase (vector)
  - TverskyLoss       : generaliza Dice con control FP vs FN  (β alto → penaliza FN)
  - FocalTverskyLoss  : Tversky con factor focal γ (clases difíciles)
  - ComboLoss         : α·CE_weighted + (1-α)·Dice  (recomendado por defecto)
  - LovaszSoftmax     : optimiza directamente IoU (opcional, se importa si está)

Selector
--------
    criterion = get_loss("combo", weight=class_weights)
    criterion = get_loss("focal", alpha=class_weights, gamma=2.0)
    criterion = get_loss("focal_tversky", alpha=0.7, beta=0.3, gamma=0.75)

Notas para xBD
--------------
- En xBD el background ocupa ~70% de los píxeles. Casi todas las pérdidas
  multi-clase incluyen background salvo que pases ignore_index=0 o
  ignore_background=True (donde aplique).
- Si ya usas WeightedRandomSampler, NO uses pesos extremos en la loss.
  Recomendación: pesos suaves (sqrt o log-inverse) o directamente sin pesos.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── DICE LOSS ────────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Dice multi-clase.

    Parámetros
    ----------
    smooth             : laplace smoothing
    ignore_background  : si True, excluye la clase 0 del promedio (recomendado en xBD)
    weight             : tensor (C,) con pesos por clase para promedio ponderado
    """
    def __init__(self, smooth=1e-6, ignore_background=False, weight=None):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background
        self.weight = weight  # tensor (C,) o None

    def forward(self, outputs, targets):
        C = outputs.shape[1]
        probs = F.softmax(outputs, dim=1)
        targets_oh = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        intersection = (probs * targets_oh).sum(dim=dims)
        union = probs.sum(dim=dims) + targets_oh.sum(dim=dims)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)  # (C,)

        if self.ignore_background:
            dice = dice[1:]
            w = self.weight[1:] if self.weight is not None else None
        else:
            w = self.weight

        if w is not None:
            w = w.to(dice.device)
            dice = (dice * w).sum() / w.sum()
        else:
            dice = dice.mean()

        return 1.0 - dice


# ─── FOCAL LOSS multi-clase ───────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss multi-clase: FL = -α·(1-p_t)^γ · log(p_t).

    Parámetros
    ----------
    alpha : escalar o tensor (C,) con pesos por clase
    gamma : factor focal (γ=2 estándar; subir si las clases minoritarias tardan en aprender)
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, outputs, targets):
        # Pesos por clase (alpha vector) → se aplican via weight de CE
        if isinstance(self.alpha, torch.Tensor):
            ce = F.cross_entropy(outputs, targets,
                                 weight=self.alpha.to(outputs.device),
                                 reduction="none",
                                 ignore_index=self.ignore_index)
        else:
            ce = F.cross_entropy(outputs, targets,
                                 reduction="none",
                                 ignore_index=self.ignore_index)
            if self.alpha is not None:
                ce = self.alpha * ce
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ─── TVERSKY / FOCAL-TVERSKY ──────────────────────────────────────────────────

class TverskyLoss(nn.Module):
    """
    Tversky Loss = 1 - TP / (TP + α·FP + β·FN).
    Con α=β=0.5 → Dice.   Con β > α → penaliza más los FN (útil cuando importa
    no perderse píxeles de daño, como aquí).

    Recomendado para xBD: α=0.3, β=0.7.
    """
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6, ignore_background=True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, outputs, targets):
        C = outputs.shape[1]
        probs = F.softmax(outputs, dim=1)
        targets_oh = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

        dims = (0, 2, 3)
        TP = (probs * targets_oh).sum(dim=dims)
        FP = (probs * (1 - targets_oh)).sum(dim=dims)
        FN = ((1 - probs) * targets_oh).sum(dim=dims)
        tversky = (TP + self.smooth) / (TP + self.alpha*FP + self.beta*FN + self.smooth)

        if self.ignore_background:
            tversky = tversky[1:]
        return 1.0 - tversky.mean()


class FocalTverskyLoss(TverskyLoss):
    """
    Focal Tversky = (1 - Tversky)^γ. Con γ < 1 amplifica errores de clases difíciles.
    Recomendado: α=0.3, β=0.7, γ=0.75.
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6,
                 ignore_background=True):
        super().__init__(alpha, beta, smooth, ignore_background)
        self.gamma = gamma

    def forward(self, outputs, targets):
        loss = super().forward(outputs, targets)
        return loss ** self.gamma


# ─── COMBO LOSS (CE + Dice) — recomendada por defecto ─────────────────────────

class ComboLoss(nn.Module):
    """
    Combo = α·CE_weighted + (1 - α)·Dice.

    Es la receta estándar para segmentación multi-clase desbalanceada:
      - CE estabiliza el aprendizaje píxel a píxel
      - Dice empuja directamente al IoU (la métrica que evalúas)
    """
    def __init__(self, alpha=0.5, ce_weight=None, dice_ignore_background=True,
                 dice_weight=None):
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss(weight=ce_weight)
        self.dice = DiceLoss(ignore_background=dice_ignore_background,
                             weight=dice_weight)

    def forward(self, outputs, targets):
        return self.alpha * self.ce(outputs, targets) + \
               (1 - self.alpha) * self.dice(outputs, targets)


# ─── (opcional) Lovasz-Softmax ────────────────────────────────────────────────
# Optimiza IoU directamente. Si quieres usarla, instala:
#   !pip install -q git+https://github.com/bermanmaxim/LovaszSoftmax.git
try:
    from lovasz_losses import lovasz_softmax  # type: ignore

    class LovaszSoftmax(nn.Module):
        def __init__(self, ignore=0):
            super().__init__()
            self.ignore = ignore

        def forward(self, outputs, targets):
            probs = F.softmax(outputs, dim=1)
            return lovasz_softmax(probs, targets, ignore=self.ignore)
except ImportError:
    LovaszSoftmax = None


# ─── Selector ─────────────────────────────────────────────────────────────────

def get_loss(loss_name, weight=None, **kwargs):
    """
    weight : tensor de pesos por clase (para CE y variantes).
    kwargs : se pasan al constructor de la loss correspondiente.

    Disponibles:
      "cross_entropy" / "ce"
      "weighted_ce"          (igual que CE, pero pasa weight)
      "dice"                  (kwargs: ignore_background, weight)
      "focal"                 (kwargs: alpha, gamma)
      "tversky"               (kwargs: alpha, beta, ignore_background)
      "focal_tversky"         (kwargs: alpha, beta, gamma, ignore_background)
      "combo"                 (kwargs: alpha, ce_weight, dice_ignore_background)
      "lovasz"                (si está instalado)
    """
    name = loss_name.lower()

    if name in ("cross_entropy", "ce"):
        return nn.CrossEntropyLoss(weight=weight)

    if name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weight)

    if name == "dice":
        # Por defecto en xBD: ignorar background
        kwargs.setdefault("ignore_background", True)
        if weight is not None and "weight" not in kwargs:
            kwargs["weight"] = weight
        return DiceLoss(**kwargs)

    if name == "focal":
        # Si pasan `weight`, lo usamos como alpha vector
        if weight is not None and "alpha" not in kwargs:
            kwargs["alpha"] = weight
        kwargs.setdefault("gamma", 2.0)
        return FocalLoss(**kwargs)

    if name == "tversky":
        return TverskyLoss(**kwargs)

    if name == "focal_tversky":
        return FocalTverskyLoss(**kwargs)

    if name == "combo":
        kwargs.setdefault("alpha", 0.5)
        if weight is not None and "ce_weight" not in kwargs:
            kwargs["ce_weight"] = weight
        return ComboLoss(**kwargs)

    if name == "lovasz":
        if LovaszSoftmax is None:
            raise ImportError("Instala lovasz: pip install git+https://github.com/bermanmaxim/LovaszSoftmax.git")
        return LovaszSoftmax(**kwargs)

    raise ValueError(
        f"Loss '{loss_name}' no reconocida. "
        f"Disponibles: ce, weighted_ce, dice, focal, tversky, focal_tversky, combo, lovasz"
    )


# ─── Helper: pesos de clase desde el dataset ──────────────────────────────────

def compute_class_weights(class_counts, scheme="inv_freq", num_classes=5,
                          ignore_background=False, device=None):
    """
    class_counts : tensor o lista (C,) con # píxeles por clase en el train set.
    scheme       : 'inv_freq' (clásico), 'sqrt_inv_freq' (suave), 'effective'
                   (Cui et al. 2019, β=0.9999), 'median_freq'.

    Devuelve tensor (C,) de pesos.

    Recomendación si YA usas WeightedRandomSampler:
      - usa 'sqrt_inv_freq' (suave) o pesos plano (None) — evita doble balanceo.
    Recomendación SIN sampler:
      - 'inv_freq' o 'effective'.
    """
    counts = torch.as_tensor(class_counts, dtype=torch.float32)
    counts = torch.clamp(counts, min=1.0)
    total = counts.sum()

    if scheme == "inv_freq":
        w = total / (num_classes * counts)
    elif scheme == "sqrt_inv_freq":
        w = torch.sqrt(total / (num_classes * counts))
    elif scheme == "effective":
        beta = 0.9999
        eff = 1.0 - torch.pow(beta, counts)
        w = (1.0 - beta) / eff
    elif scheme == "median_freq":
        freq = counts / total
        w = torch.median(freq) / freq
    else:
        raise ValueError(f"scheme '{scheme}' desconocido")

    # Normaliza para que el peso medio sea 1
    w = w / w.mean()

    if ignore_background:
        w[0] = 0.0

    if device is not None:
        w = w.to(device)
    return w