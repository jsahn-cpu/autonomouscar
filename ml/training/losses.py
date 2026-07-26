"""BCE + Dice loss for binary lane-marking segmentation.

Lane-marking pixels are a small minority of each frame, so plain BCE alone
tends to collapse toward predicting all-background; Dice directly rewards
foreground overlap and is a standard counterweight for this kind of
class imbalance.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets)
        dice = dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice


@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds = (torch.sigmoid(logits) > threshold).float()
    intersection = (preds * targets).sum().item()
    union = ((preds + targets) > 0).float().sum().item()
    return intersection / union if union else 1.0
