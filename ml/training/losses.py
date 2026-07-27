"""Cross-entropy + Dice loss for multi-class lane segmentation, plus
PIDNetLite's auxiliary/boundary terms.

Background dominates the pixel count (lane markings/areas are a small
fraction of each frame), so plain cross-entropy alone tends to underweight
the rare foreground classes; Dice directly rewards per-class overlap and is
a standard counterweight for this kind of imbalance. `class_weights` (per
ml/configs/train.yaml) lets CE additionally upweight the rarest classes
(e.g. center_dashed, which covers far fewer pixels than a solid line).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """logits: (B,C,H,W). targets: (B,H,W) class indices. Averaged over all
    C classes (including background) so every class gets an overlap
    gradient, not just the ones CE already weights heavily."""
    num_classes = logits.shape[1]
    probs = F.softmax(logits, dim=1).flatten(2)  # (B,C,HW)
    targets_onehot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float().flatten(2)  # (B,C,HW)
    intersection = (probs * targets_onehot).sum(dim=2)  # (B,C)
    union = probs.sum(dim=2) + targets_onehot.sum(dim=2)
    dice = (2 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5, class_weights=None) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        weight_t = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.register_buffer("class_weight_t", weight_t, persistent=False)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.class_weight_t)
        dice = dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice


@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Mean IoU over the FOREGROUND classes only (excludes background,
    class 0) -- background is the vast majority of every frame and
    trivially easy, so including it would drown out the metric that
    actually matters: how well the 5 lane classes are segmented."""
    num_classes = logits.shape[1]
    preds = logits.argmax(dim=1)
    ious = []
    for c in range(1, num_classes):
        pred_c = preds == c
        target_c = targets == c
        intersection = (pred_c & target_c).sum().item()
        union = (pred_c | target_c).sum().item()
        if union > 0:
            ious.append(intersection / union)
    return sum(ious) / len(ious) if ious else 1.0


@torch.no_grad()
def update_iou_stats(
    logits: torch.Tensor, targets: torch.Tensor,
    intersection: torch.Tensor, union: torch.Tensor,
) -> None:
    """Accumulate per-class intersection and union IN PLACE across a whole
    val set, for a dataset-global per-class IoU -- more stable and more
    informative than iou_score's per-image averaging: a frame with only a
    few lane pixels no longer counts the same as one full of them, and
    keeping the classes separate (rather than pre-averaging) is what lets
    the caller print which specific class is lagging (e.g. right_solid or
    lane_2, the ones that kept coming out weak). intersection/union are
    length-num_classes tensors on the same device as logits.

    Global IoU per class c is then intersection[c] / union[c]; mean
    foreground IoU is the mean of that over classes 1..num_classes-1 with
    nonzero union."""
    num_classes = intersection.shape[0]
    preds = logits.argmax(dim=1)
    for c in range(num_classes):
        pred_c = preds == c
        target_c = targets == c
        intersection[c] += (pred_c & target_c).sum()
        union[c] += (pred_c | target_c).sum()


def extract_boundary(targets: torch.Tensor, num_classes: int, width: int = 2) -> torch.Tensor:
    """Binary boundary band (any class-to-class edge, not just fg/bg) of a
    (B,H,W) class-index label, via one-hot + per-class dilation-minus-
    erosion (both as max_pool2d), taking the max over classes -- a pixel on
    the border between e.g. lane_1 and lane_2 shows up as an edge in both
    channels. Used to supervise PIDNetLite's boundary branch."""
    one_hot = F.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()  # (B,C,H,W)
    k = 2 * width + 1
    dilated = F.max_pool2d(one_hot, kernel_size=k, stride=1, padding=width)
    eroded = 1.0 - F.max_pool2d(1.0 - one_hot, kernel_size=k, stride=1, padding=width)
    boundary_per_class = (dilated - eroded).clamp(0.0, 1.0)
    return boundary_per_class.amax(dim=1, keepdim=True)  # (B,1,H,W)


class PIDNetLoss(nn.Module):
    """Main segmentation loss (CE+Dice) + auxiliary semantic loss (deep
    supervision on the I/context branch, lower weight) + boundary BCE
    (supervises the D branch against extract_boundary's target).
    boundary_weight is much larger than the others because boundary pixels
    are a tiny minority even within an already-sparse foreground mask --
    matches PIDNet's own training recipe of weighting the boundary term
    heavily."""

    def __init__(
        self,
        num_classes: int,
        ce_weight: float = 0.5,
        dice_weight: float = 0.5,
        aux_weight: float = 0.4,
        boundary_weight: float = 20.0,
        boundary_width: int = 2,
        class_weights=None,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.seg_loss = CEDiceLoss(ce_weight, dice_weight, class_weights=class_weights)
        self.aux_weight = aux_weight
        self.boundary_weight = boundary_weight
        self.boundary_width = boundary_width

    def forward(self, outputs, targets: torch.Tensor) -> torch.Tensor:
        main_logits, aux_logits, boundary_logits = outputs
        boundary_targets = extract_boundary(targets, self.num_classes, self.boundary_width)
        loss = self.seg_loss(main_logits, targets)
        loss = loss + self.aux_weight * self.seg_loss(aux_logits, targets)
        loss = loss + self.boundary_weight * F.binary_cross_entropy_with_logits(boundary_logits, boundary_targets)
        return loss
