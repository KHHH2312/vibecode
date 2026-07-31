"""Physical Gaussian centers, subvoxel offsets, backward flow (plan recipe A)."""
from __future__ import annotations

import math
from typing import Sequence

import torch

from .constants import SIGMA_UM, SPACING_ZYX_UM


def make_targets(
    shape_zyx: tuple[int, int, int],
    centers_zyx: Sequence[Sequence[float]] | torch.Tensor,
    parent_index: Sequence[int] | torch.Tensor,
    *,
    spacing_zyx_um: Sequence[float] = SPACING_ZYX_UM,
    sigma_um: float = SIGMA_UM,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build heatmap, offset (3,Z,Y,X), flow_um (3,Z,Y,X), offset_valid, flow_valid.

    centers_zyx: floating voxel coordinates (z,y,x) in the patch.
    parent_index[i] = parent center index or -1 if none / birth.
    """
    device = device or torch.device("cpu")
    Z, Y, X = shape_zyx
    spacing = torch.tensor(list(spacing_zyx_um), dtype=torch.float32, device=device)

    if not isinstance(centers_zyx, torch.Tensor):
        centers = torch.tensor(centers_zyx, dtype=torch.float32, device=device).reshape(-1, 3)
    else:
        centers = centers_zyx.to(device=device, dtype=torch.float32).reshape(-1, 3)

    if not isinstance(parent_index, torch.Tensor):
        parents = torch.tensor(list(parent_index), dtype=torch.long, device=device)
    else:
        parents = parent_index.to(device=device, dtype=torch.long)

    heat = torch.zeros((Z, Y, X), dtype=torch.float32, device=device)
    offset = torch.zeros((3, Z, Y, X), dtype=torch.float32, device=device)
    flow_um = torch.zeros((3, Z, Y, X), dtype=torch.float32, device=device)
    offset_valid = torch.zeros((Z, Y, X), dtype=torch.bool, device=device)
    flow_valid = torch.zeros((Z, Y, X), dtype=torch.bool, device=device)

    if centers.numel() == 0:
        return heat, offset, flow_um, offset_valid, flow_valid

    # radius in voxels per axis for 3σ
    rad_vox = [max(1, int(math.ceil(3.0 * sigma_um / float(spacing[i])))) for i in range(3)]

    for child_i, center in enumerate(centers):
        cz, cy, cx = float(center[0]), float(center[1]), float(center[2])
        az, ay, ax = int(math.floor(cz)), int(math.floor(cy)), int(math.floor(cx))
        if not (0 <= az < Z and 0 <= ay < Y and 0 <= ax < X):
            # still paint if nearby; clamp anchor for offset write only if in bounds
            pass

        z0 = max(0, int(math.floor(cz)) - rad_vox[0])
        z1 = min(Z, int(math.floor(cz)) + rad_vox[0] + 2)
        y0 = max(0, int(math.floor(cy)) - rad_vox[1])
        y1 = min(Y, int(math.floor(cy)) + rad_vox[1] + 2)
        x0 = max(0, int(math.floor(cx)) - rad_vox[2])
        x1 = min(X, int(math.floor(cx)) + rad_vox[2] + 2)
        if z0 >= z1 or y0 >= y1 or x0 >= x1:
            continue

        zz = torch.arange(z0, z1, device=device, dtype=torch.float32)
        yy = torch.arange(y0, y1, device=device, dtype=torch.float32)
        xx = torch.arange(x0, x1, device=device, dtype=torch.float32)
        gz, gy, gx = torch.meshgrid(zz, yy, xx, indexing="ij")
        dz_um = (gz - cz) * spacing[0]
        dy_um = (gy - cy) * spacing[1]
        dx_um = (gx - cx) * spacing[2]
        dist2 = dz_um * dz_um + dy_um * dy_um + dx_um * dx_um
        gaussian = torch.exp(-0.5 * dist2 / (sigma_um**2))
        heat[z0:z1, y0:y1, x0:x1] = torch.maximum(heat[z0:z1, y0:y1, x0:x1], gaussian)

        if 0 <= az < Z and 0 <= ay < Y and 0 <= ax < X:
            offset[0, az, ay, ax] = cz - az
            offset[1, az, ay, ax] = cy - ay
            offset[2, az, ay, ax] = cx - ax
            offset_valid[az, ay, ax] = True

            p = int(parents[child_i].item()) if child_i < len(parents) else -1
            if 0 <= p < centers.shape[0]:
                # backward flow: parent - child in µm
                delta_vox = centers[p] - center
                flow_um[:, az, ay, ax] = delta_vox * spacing
                flow_valid[az, ay, ax] = True

    return heat, offset, flow_um, offset_valid, flow_valid


def decode_peaks(
    heat: torch.Tensor,
    offset: torch.Tensor | None = None,
    *,
    threshold: float = 0.1,
    nms_um: float = 3.0,
    spacing_zyx_um: Sequence[float] = SPACING_ZYX_UM,
    max_peaks: int = 2000,
) -> torch.Tensor:
    """Greedy physical-NMS peak decode → (N,3) float zyx voxels."""
    assert heat.ndim == 3
    spacing = torch.tensor(list(spacing_zyx_um), dtype=torch.float32, device=heat.device)
    # local max 3x3x3
    import torch.nn.functional as F

    x = heat[None, None]
    pooled = F.max_pool3d(x, kernel_size=3, stride=1, padding=1)
    is_peak = (heat >= threshold) & (heat == pooled[0, 0])
    coords = is_peak.nonzero(as_tuple=False).float()  # z,y,x integer
    if coords.numel() == 0:
        return torch.zeros((0, 3), device=heat.device)

    vals = heat[is_peak]
    order = torch.argsort(vals, descending=True)
    coords = coords[order]
    vals = vals[order]

    if offset is not None:
        zi = coords[:, 0].long().clamp(0, heat.shape[0] - 1)
        yi = coords[:, 1].long().clamp(0, heat.shape[1] - 1)
        xi = coords[:, 2].long().clamp(0, heat.shape[2] - 1)
        coords = coords + offset[:, zi, yi, xi].T

    # greedy NMS in µm
    kept: list[int] = []
    for i in range(coords.shape[0]):
        if len(kept) >= max_peaks:
            break
        c = coords[i]
        ok = True
        for j in kept:
            d = (c - coords[j]) * spacing
            if float(d.pow(2).sum().sqrt()) < nms_um:
                ok = False
                break
        if ok:
            kept.append(i)
    if not kept:
        return torch.zeros((0, 3), device=heat.device)
    return coords[kept]
