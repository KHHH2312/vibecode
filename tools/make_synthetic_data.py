"""
Generate a tiny synthetic OME-Zarr 3D+time dataset (+ sample_submission.csv) that
mimics the Kaggle Biohub layout closely enough to exercise the pipeline locally.

This is ONLY for smoke testing the code paths and the output format in this
sandbox -- it is not the competition data and says nothing about the LB score.

Layout produced under `root`:
    root/
      sample_submission.csv          # canonical schema: track_id,t,z,y,x,parent_track_id
      test/seq01.zarr/               # OME-Zarr group, array '0' shape (T,Z,Y,X)
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd


def _render(volume, center_zyx, sigma_zyx, amp):
    """Add an anisotropic Gaussian blob into `volume` in-place."""
    zz, yy, xx = np.ogrid[: volume.shape[0], : volume.shape[1], : volume.shape[2]]
    cz, cy, cx = center_zyx
    sz, sy, sx = sigma_zyx
    g = amp * np.exp(-(((zz - cz) ** 2) / (2 * sz ** 2)
                       + ((yy - cy) ** 2) / (2 * sy ** 2)
                       + ((xx - cx) ** 2) / (2 * sx ** 2)))
    volume += g.astype(volume.dtype)


def make_dataset(root, T=8, shape_zyx=(16, 84, 84), spacing_um=(2.0, 0.5, 0.5),
                 n_cells=12, n_divisions=2, seed=0, extra_column=False):
    """Write the synthetic OME-Zarr + sample_submission.csv. Returns paths."""
    import zarr

    rng = np.random.default_rng(seed)
    Z, Y, X = shape_zyx
    sz, sy, sx = spacing_um
    sigma_vox = (max(1.0, 2.5 / sz), max(1.5, 2.5 / sy), max(1.5, 2.5 / sx))

    # -- lay out cell trajectories on a jittered grid so detections are well
    #    separated (> the pipeline's min-separation), with slow linear drift --- #
    cells = []
    margin = 10
    gy = np.linspace(margin, Y - margin, 4)
    gx = np.linspace(margin, X - margin, 3)
    grid = [(y, x) for y in gy for x in gx]
    rng.shuffle(grid)
    for k in range(min(n_cells, len(grid))):
        gyk, gxk = grid[k]
        start = np.array([rng.uniform(4, Z - 4),
                          gyk + rng.uniform(-2, 2),
                          gxk + rng.uniform(-2, 2)])
        vel = rng.normal(0, [0.15, 0.6, 0.6])   # voxels/frame, mostly in-plane
        cells.append({"start": start, "vel": vel, "t0": 0, "t1": T - 1, "parent": -1})

    divide_idx = list(range(min(n_divisions, len(cells))))
    tracks_gt = []
    next_id = 0

    def emit(cell):
        nonlocal next_id
        tid = next_id
        next_id += 1
        for t in range(cell["t0"], cell["t1"] + 1):
            pos = cell["start"] + cell["vel"] * (t - cell["t0"])
            tracks_gt.append((tid, t, cell["parent"], pos))
        return tid

    # Build the volume stack while emitting ground-truth positions.
    vol = np.zeros((T, Z, Y, X), dtype=np.float32)

    def paint(tid, t, pos):
        pos = np.clip(pos, [1, 2, 2], [Z - 2, Y - 3, X - 3])
        _render(vol[t], pos, sigma_vox, amp=1.0)

    for ci, cell in enumerate(cells):
        if ci in divide_idx:
            t_div = T // 2
            parent = dict(cell, t1=t_div)
            pid = emit_and_paint(parent, paint, emit)
            div_pos = parent["start"] + parent["vel"] * (t_div - parent["t0"])
            for sign in (-1.0, 1.0):
                d = {"start": div_pos + np.array([0, sign * 8, sign * 8]),
                     "vel": cell["vel"] + rng.normal(0, [0.2, 0.8, 0.8]),
                     "t0": t_div + 1, "t1": T - 1, "parent": pid}
                emit_and_paint(d, paint, emit)
        else:
            emit_and_paint(cell, paint, emit)

    # background + shot-ish noise
    vol += rng.normal(0.02, 0.01, size=vol.shape).astype(np.float32).clip(0)
    vol = (vol / vol.max() * 4000).astype(np.uint16)

    # -- write OME-Zarr group with a single pyramid level '0' ------------------ #
    test_dir = os.path.join(root, "test")
    os.makedirs(test_dir, exist_ok=True)
    zpath = os.path.join(test_dir, "seq01.zarr")
    g = zarr.open_group(zpath, mode="w")
    try:
        arr = g.create_array(name="0", shape=vol.shape, dtype=vol.dtype,
                             chunks=(1, Z, Y, X))
    except TypeError:  # older zarr API
        arr = g.create_dataset("0", shape=vol.shape, dtype=vol.dtype,
                               chunks=(1, Z, Y, X))
    arr[:] = vol
    g.attrs["multiscales"] = [{
        "version": "0.4",
        "axes": [
            {"name": "t", "type": "time", "unit": "frame"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ],
        "datasets": [{
            "path": "0",
            "coordinateTransformations": [
                {"type": "scale", "scale": [1.0, sz, sy, sx]}],
        }],
    }]

    # -- sample_submission.csv: canonical schema + a valid tiny example --------- #
    cols = ["track_id", "t", "z", "y", "x", "parent_track_id"]
    example = pd.DataFrame(
        [[0, 0, Z // 2, Y // 2, X // 2, -1],
         [0, 1, Z // 2, Y // 2, X // 2, -1]], columns=cols)
    if extra_column:
        example.insert(0, "dataset", "seq01")
    sample_path = os.path.join(root, "sample_submission.csv")
    example.to_csv(sample_path, index=False)

    gt = pd.DataFrame(
        [(tid, t, par, *pos) for (tid, t, par, pos) in tracks_gt],
        columns=["track_id", "t", "parent_track_id", "z", "y", "x"])
    return {"zarr": zpath, "sample": sample_path, "root": root, "gt": gt,
            "n_divisions": len(divide_idx)}


def emit_and_paint(cell, paint, emit):
    """Emit a cell's ground-truth rows and paint its blobs frame by frame."""
    tid = emit(cell)
    for t in range(cell["t0"], cell["t1"] + 1):
        pos = cell["start"] + cell["vel"] * (t - cell["t0"])
        paint(tid, t, pos)
    return tid


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_input"
    info = make_dataset(out)
    print("wrote", info["zarr"], "and", info["sample"])
