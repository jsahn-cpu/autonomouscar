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


def extract_boundary(label: torch.Tensor, width: int = 2) -> torch.Tensor:
    """Binary boundary band (foreground/background edge) of a {0,1} label,
    via dilation-minus-erosion -- both implemented as max_pool2d (dilation
    directly, erosion as max_pool2d on the inverted mask), so this stays a
    cheap tensor op computed fresh each batch rather than a per-sample cv2
    round-trip. Used to supervise PIDNetLite's boundary branch."""
    k = 2 * width + 1
    dilated = F.max_pool2d(label, kernel_size=k, stride=1, padding=width)
    eroded = 1.0 - F.max_pool2d(1.0 - label, kernel_size=k, stride=1, padding=width)
    return (dilated - eroded).clamp(0.0, 1.0)


class PIDNetLoss(nn.Module):
    """Main segmentation loss (BCE+Dice, same as TinyUNet's) + auxiliary
    semantic loss (deep supervision on the I/context branch, lower weight)
    + boundary BCE (supervises the D branch against extract_boundary's
    target). boundary_weight is much larger than the others because
    boundary pixels are a tiny minority even within an already-sparse
    foreground mask -- matches PIDNet's own training recipe of weighting
    the boundary term heavily."""

    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        aux_weight: float = 0.4,
        boundary_weight: float = 20.0,
        boundary_width: int = 2,
    ) -> None:
        super().__init__()
        self.seg_loss = BCEDiceLoss(bce_weight, dice_weight)
        self.aux_weight = aux_weight
        self.boundary_weight = boundary_weight
        self.boundary_width = boundary_width

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        main_logits, aux_logits, boundary_logits = outputs
        boundary_targets = extract_boundary(targets, self.boundary_width)
        loss = self.seg_loss(main_logits, targets)
        loss = loss + self.aux_weight * self.seg_loss(aux_logits, targets)
        loss = loss + self.boundary_weight * F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
        return loss
