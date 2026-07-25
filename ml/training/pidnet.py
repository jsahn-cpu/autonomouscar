"""PIDNet-lite for binary lane-marking segmentation (ml/README.md step 4,
alternative candidate to TinyUNet).

This is a from-scratch, deliberately simplified reimplementation of the
core ideas from "PIDNet: A Real-time Semantic Segmentation Network Inspired
by PID Controllers" (Xu et al., CVPR 2023) -- NOT a copy of, or
weight-compatible with, the official repo/Cityscapes checkpoints. Written
this way (like TinyUNet) to stay self-contained in this codebase and avoid
pulling in an external framework/repo whose training conventions and
19-class Cityscapes head don't match our single-channel-grayscale,
single-class-binary setup anyway.

Why PIDNet's design is a good fit here: lane markings are thin,
boundary-dominated structures, which is exactly what PIDNet's three-branch
design targets --
  - P (detail) branch: shallow, stays at higher resolution, preserves fine
    spatial detail (thin lines survive here, unlike in a deep encoder).
  - I (context) branch: a normal downsampling encoder + pyramid pooling,
    captures global scene context (where's the floor, where's the wall).
  - D (boundary) branch: shallow, predicts an auxiliary boundary map, and
    that boundary map gates how much the final prediction trusts the
    detail branch vs. the context branch at each pixel (the "Bag" fusion
    below) -- exactly where a thin line's edge is, trust detail more.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class LitePAPPM(nn.Module):
    """Simplified pyramid pooling (stand-in for PIDNet's PAPPM): pools the
    I branch's deepest features at a few scales, upsamples back, and fuses
    -- cheap way to inject global context before handing off to the detail
    branch."""

    def __init__(self, in_ch: int, out_ch: int, pool_sizes=(1, 2, 4, 8)) -> None:
        super().__init__()
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(p) for p in pool_sizes])
        self.convs = nn.ModuleList([nn.Conv2d(in_ch, out_ch, 1, bias=False) for _ in pool_sizes])
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * len(pool_sizes) + in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        feats = [x]
        for pool, conv in zip(self.pools, self.convs):
            y = conv(pool(x))
            y = F.interpolate(y, size=(h, w), mode="bilinear", align_corners=False)
            feats.append(y)
        return self.fuse(torch.cat(feats, dim=1))


class PIDNetLite(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels

        # Shared stem -> 1/4 resolution
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )

        # I (context) branch: keeps downsampling from the stem
        self.i_layer1 = BasicBlock(c, c)                 # 1/4
        self.i_layer2 = BasicBlock(c, c * 2, stride=2)    # 1/8
        self.i_layer3 = BasicBlock(c * 2, c * 4, stride=2)  # 1/16 -- aux head taps here
        self.i_layer4 = BasicBlock(c * 4, c * 8, stride=2)  # 1/32
        self.pappm = LitePAPPM(c * 8, c * 2)
        self.i_project = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 1, bias=False), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
        )

        # P (detail) branch: shallow, only ever downsampled once (stem is
        # already 1/4, one more stride-2 gets it to 1/8 to match I/D)
        self.p_in = nn.Sequential(
            nn.Conv2d(c, c * 2, 3, 2, 1, bias=False), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
        )
        self.p_block = BasicBlock(c * 2, c * 2)

        # D (boundary) branch: even shallower/thinner than P, also at 1/8
        self.d_in = nn.Sequential(
            nn.Conv2d(c, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
        )
        self.d_block = BasicBlock(c, c)
        self.boundary_head = nn.Conv2d(c, 1, 1)

        # Bag (Boundary-Attention-Guided) fusion: D's features gate how
        # much of the final prediction comes from P (detail) vs. I
        # (context) at each pixel.
        self.bag_attn = nn.Conv2d(c, 1, 1)
        self.bag_fuse = nn.Sequential(
            nn.Conv2d(c * 2, c * 2, 3, 1, 1, bias=False), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True),
        )

        self.main_head = nn.Conv2d(c * 2, 1, 1)
        self.aux_head = nn.Conv2d(c * 4, 1, 1)  # deep supervision on i_layer3

    def forward(self, x: torch.Tensor):
        h, w = x.shape[-2:]
        stem = self.stem(x)

        i1 = self.i_layer1(stem)
        i2 = self.i_layer2(i1)   # 1/8 -- P/D branches match this resolution
        i3 = self.i_layer3(i2)   # 1/16
        i4 = self.i_layer4(i3)   # 1/32
        i4 = self.pappm(i4)
        i_up = F.interpolate(self.i_project(i4), size=i2.shape[-2:], mode="bilinear", align_corners=False)

        p = self.p_block(self.p_in(stem))  # 1/8

        d = self.d_block(self.d_in(stem))  # 1/8
        boundary_logits = self.boundary_head(d)

        attn = torch.sigmoid(self.bag_attn(d))
        fused = self.bag_fuse(attn * p + (1 - attn) * i_up)

        main_logits = F.interpolate(self.main_head(fused), size=(h, w), mode="bilinear", align_corners=False)

        if not self.training:
            return main_logits

        aux_logits = F.interpolate(self.aux_head(i3), size=(h, w), mode="bilinear", align_corners=False)
        boundary_logits = F.interpolate(boundary_logits, size=(h, w), mode="bilinear", align_corners=False)
        return main_logits, aux_logits, boundary_logits


if __name__ == "__main__":
    model = PIDNetLite()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PIDNetLite params: {n_params / 1e6:.2f}M")

    dummy = torch.randn(2, 1, 288, 512)
    model.train()
    main, aux, boundary = model(dummy)
    print(f"train mode -> main={tuple(main.shape)} aux={tuple(aux.shape)} boundary={tuple(boundary.shape)}")
    model.eval()
    with torch.no_grad():
        out = model(dummy)
    print(f"eval mode -> {tuple(out.shape)}")
