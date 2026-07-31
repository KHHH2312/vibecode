"""CELLECT-lite backbone + dense heads (Session 0 skeleton)."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn_silu(ch: int, groups: int = 8) -> nn.Sequential:
    g = min(groups, ch)
    while ch % g != 0 and g > 1:
        g -= 1
    return nn.Sequential(nn.GroupNorm(g, ch), nn.SiLU(inplace=True))


class ConvBlock(nn.Module):
    def __init__(self, cin: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1, bias=False),
            _gn_silu(cout),
            nn.Conv3d(cout, cout, 3, padding=1, bias=False),
            _gn_silu(cout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CellectLite(nn.Module):
    """Anisotropic temporal U-Net-ish stack with dense heads.

    Input: (B, T, 1, Z, Y, X) or (B, 1, Z, Y, X) for T=1.
    Pool strides default (1,2,2), (1,2,2), (2,2,2) per plan.
    """

    def __init__(
        self,
        in_channels: int = 1,
        channels: tuple[int, ...] = (24, 48, 96, 192),
        embedding_dim: int = 64,
        time_frames: int = 2,
    ):
        super().__init__()
        self.time_frames = time_frames
        c0, c1, c2, c3 = channels
        self.stem = ConvBlock(in_channels * time_frames, c0)
        self.down1 = nn.Sequential(nn.Conv3d(c0, c1, 3, stride=(1, 2, 2), padding=1), _gn_silu(c1), ConvBlock(c1, c1))
        self.down2 = nn.Sequential(nn.Conv3d(c1, c2, 3, stride=(1, 2, 2), padding=1), _gn_silu(c2), ConvBlock(c2, c2))
        self.down3 = nn.Sequential(nn.Conv3d(c2, c3, 3, stride=(2, 2, 2), padding=1), _gn_silu(c3), ConvBlock(c3, c3))
        self.up2 = nn.ConvTranspose3d(c3, c2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.dec2 = ConvBlock(c2 + c2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = ConvBlock(c1 + c1, c1)
        self.up0 = nn.ConvTranspose3d(c1, c0, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec0 = ConvBlock(c0 + c0, c0)

        self.head_center = nn.Conv3d(c0, 1, 1)
        self.head_offset = nn.Conv3d(c0, 3, 1)
        self.head_flow = nn.Conv3d(c0, 3, 1)
        self.head_mitosis = nn.Conv3d(c0, 1, 1)
        self.head_embed = nn.Conv3d(c1, embedding_dim, 1)  # at (1,2,2) stride

    def _encode_time(self, x: torch.Tensor) -> torch.Tensor:
        # x: B,T,C,Z,Y,X or B,C,Z,Y,X
        if x.ndim == 6:
            b, t, c, z, y, w = x.shape
            x = x.reshape(b, t * c, z, y, w)
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self._encode_time(x)
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        d2 = self.up2(s3)
        if d2.shape[-3:] != s2.shape[-3:]:
            d2 = F.interpolate(d2, size=s2.shape[-3:], mode="trilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))
        d1 = self.up1(d2)
        if d1.shape[-3:] != s1.shape[-3:]:
            d1 = F.interpolate(d1, size=s1.shape[-3:], mode="trilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))
        d0 = self.up0(d1)
        if d0.shape[-3:] != s0.shape[-3:]:
            d0 = F.interpolate(d0, size=s0.shape[-3:], mode="trilinear", align_corners=False)
        d0 = self.dec0(torch.cat([d0, s0], dim=1))

        return {
            "center_logits": self.head_center(d0),
            "offset_vox": self.head_offset(d0),
            "backward_flow_um": self.head_flow(d0),
            "mitosis_logits": self.head_mitosis(d0),
            "embedding_map": self.head_embed(d1),
        }


@dataclass
class DetectorOutput:
    center_logits: torch.Tensor
    offset_vox: torch.Tensor
    backward_flow_um: torch.Tensor
    embedding_map: torch.Tensor
    mitosis_logits: torch.Tensor
