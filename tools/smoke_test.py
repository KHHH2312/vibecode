"""
End-to-end smoke test for the CPU pipeline on synthetic data.

Verifies CODE PATHS and OUTPUT FORMAT only -- it does NOT and cannot validate the
Kaggle leaderboard score (the competition data is not available in this sandbox).

Checks:
  * the pipeline runs detect -> link -> gap-close -> divide -> assemble -> write
  * submission.csv columns exactly match sample_submission.csv (both schemas)
  * output is non-empty, has no NaNs, and integer-typed ids/time
  * at least one division (parent_track_id != -1) is recovered
  * detection recall against ground truth is sane (pipeline finds real cells)
  * runtime for the tiny volume is small
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "kernel"))
sys.path.insert(0, HERE)

import biohub_cell_tracking_cpu as pipe          # noqa: E402
from make_synthetic_data import make_dataset       # noqa: E402


def _detection_recall(sub, gt, tol_vox=4.0):
    """Fraction of GT (t, position) points with a predicted node nearby."""
    from scipy.spatial import cKDTree
    hits = 0
    for t, g in gt.groupby("t"):
        s = sub[sub["t"] == t]
        if s.empty:
            continue
        tree = cKDTree(s[["z", "y", "x"]].values)
        d, _ = tree.query(g[["z", "y", "x"]].values, k=1)
        hits += int((d <= tol_vox).sum())
    return hits / max(1, len(gt))


def run_case(tmp, extra_column):
    label = "extra-grouping-column" if extra_column else "canonical-schema"
    print(f"\n=== case: {label} ===")
    root = os.path.join(tmp, label)
    os.makedirs(root, exist_ok=True)
    info = make_dataset(root, extra_column=extra_column, seed=0)
    sample = pd.read_csv(info["sample"])

    cfg = dict(pipe.CONFIG)
    cfg["min_track_len"] = 2
    cfg["verbose"] = True

    out_path = os.path.join(root, "submission.csv")
    t0 = time.time()
    sub = pipe.run_pipeline(input_root=root, out_path=out_path, cfg=cfg)
    elapsed = time.time() - t0

    # --- format assertions ------------------------------------------------- #
    assert os.path.exists(out_path), "submission.csv was not written"
    on_disk = pd.read_csv(out_path)
    assert list(on_disk.columns) == list(sample.columns), (
        f"columns {list(on_disk.columns)} != sample {list(sample.columns)}")
    assert len(on_disk) > 0, "submission is empty"
    assert not on_disk.isnull().values.any(), "submission has NaNs"
    assert on_disk["track_id"].dtype.kind in "iu", "track_id must be integer"
    assert on_disk["t"].dtype.kind in "iu", "t must be integer"

    # --- behaviour assertions --------------------------------------------- #
    n_tracks = on_disk["track_id"].nunique()
    n_div_rows = int((on_disk["parent_track_id"] != -1).sum())
    recall = _detection_recall(sub, info["gt"], tol_vox=4.0)
    print(f"tracks={n_tracks} nodes={len(on_disk)} div_rows={n_div_rows} "
          f"recall={recall:.2f} elapsed={elapsed:.2f}s")

    assert n_tracks >= 5, f"implausibly few tracks: {n_tracks}"
    assert n_div_rows >= 1, "no divisions recovered (parent_track_id all -1)"
    assert recall >= 0.6, f"detection recall too low: {recall:.2f}"
    assert elapsed < 60, f"tiny volume took too long: {elapsed:.1f}s"
    if extra_column:
        assert "dataset" in on_disk.columns and (on_disk["dataset"] == "seq01").all()
    print(f"PASS: {label}")


def main():
    tmp = tempfile.mkdtemp(prefix="biohub_smoke_")
    try:
        run_case(tmp, extra_column=False)
        run_case(tmp, extra_column=True)
        print("\nALL SMOKE TESTS PASSED")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
