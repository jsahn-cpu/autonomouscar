"""Tiny U-Net for binary lane-marking segmentation (ml/README.md step 4).

~0.48M params (base_channels=16, measured via `python model.py`) --
deliberately small: this is a single-class, fixed-camera-
angle, low-resolution problem, so a pretrained encoder isn't needed to
converge, and staying small keeps inference on the 3050 Ti (or eventually
whatever the vehicle's onboard compute turns out to be) essentially free --
the real risk for this model is data diversity, not inference speed. See
ml/README.md for the MobileNetV3-Small backbone alternative if this
underfits on the initial dataset.
"""
import torch
import torch.nn as nn


def conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TinyUNet(nn.Module):
    def __init__(self, in_channels: int = 1, base_channels: int = 16) -> None:
        super().__init__()
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8

        self.enc1 = conv_block(in_channels, c1)
        self.enc2 = conv_block(c1, c2)
        self.enc3 = conv_block(c2, c3)
        self.bottleneck = conv_block(c3, c4)

        self.pool = nn.MaxPool2d(2)

        self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
        self.dec3 = conv_block(c3 * 2, c3)
        self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
        self.dec2 = conv_block(c2 * 2, c2)
        self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
        self.dec1 = conv_block(c1 * 2, c1)

        self.out_conv = nn.Conv2d(c1, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out_conv(d1)  # logits, BCEWithLogitsLoss/dice_loss apply sigmoid themselves


if __name__ == "__main__":
    model = TinyUNet()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TinyUNet params: {n_params / 1e6:.2f}M")
    dummy = torch.randn(2, 1, 288, 512)
    out = model(dummy)
    print(f"output shape: {tuple(out.shape)}")
