import torch
import torch.nn as nn
import torch.nn.functional as F

# --- DICE LOSS ---
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, outputs, targets):
        probs = F.softmax(outputs, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=outputs.shape[1]).permute(0, 3, 1, 2).float()
        intersection = (probs * targets_one_hot).sum(dim=(0, 2, 3))
        union = probs.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

# --- FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, outputs, targets):
        ce_loss = F.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return focal_loss.mean()

# --- SELECTOR DE PÉRDIDAS ---
def get_loss(loss_name, weight=None):
    loss_name = loss_name.lower()
    if loss_name == "cross_entropy":
        return nn.CrossEntropyLoss(weight=weight)
    elif loss_name == "dice":
        return DiceLoss()
    elif loss_name == "focal":
        return FocalLoss()
    elif loss_name == "weighted_ce":
        return nn.CrossEntropyLoss(weight=weight)
    else:
        raise ValueError(f"Loss '{loss_name}' no reconocida. Prueba con: dice, focal, cross_entropy.")