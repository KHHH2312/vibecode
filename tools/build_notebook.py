"""
Build the self-contained Kaggle notebook from the pipeline source of truth.

The notebook embeds kernel/biohub_cell_tracking_cpu.py verbatim (minus its
__main__ guard) in one code cell, then a final cell runs full inference. Keeping
the .py as the single source and generating the .ipynb avoids drift.

Usage:  python tools/build_notebook.py
"""

from __future__ import annotations

import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "kernel", "biohub_cell_tracking_cpu.py")
OUT = os.path.join(ROOT, "kernel", "biohub_cell_tracking_cpu.ipynb")

INTRO = """\
# Biohub - Cell Tracking During Development - CPU full-inference pipeline

**Code-competition kernel.** This notebook *recomputes* the tracking graph from
the image volumes on every run (detect -> link -> gap-recover -> divisions ->
tracks) and writes `submission.csv`. It is **not** a postprocess kernel and never
rewrites a precomputed CSV.

**Runtime settings (right-hand panel):** Accelerator = **None (CPU)**,
Internet = **Off**, and add the competition data as the input source.

**How it works**
1. Discover the test OME-Zarr volumes and `sample_submission.csv` under `/kaggle/input`.
2. Detect nuclei per timepoint (background-relative threshold + local maxima), on a
   downsampled pyramid level chosen to fit the CPU time budget.
3. Link frames with KD-tree-gated, motion-aware, ascending-distance matching.
4. Close short temporal gaps (interpolating skipped frames) to recover missed links.
5. Detect divisions (one parent -> two symmetric, velocity-consistent daughters).
6. Assemble tracks, drop spurious short tracks, and write `submission.csv` using the
   **exact columns/units discovered from `sample_submission.csv` at runtime**.

**First-run sanity checks (read the log the first cell prints):**
- `column mapping` should cover track_id / t / z / y / x / parent_track_id.
- If `sample_submission` coordinates are physical micrometres rather than voxel
  indices, set `CONFIG["coord_mode"] = "um"`.
- Check the chosen pyramid level / timepoint count against the competition time limit;
  raise `CONFIG["target_max_voxels"]` for accuracy or lower it for speed.

**Honest note:** the authors could not run this against the real competition data
when writing it, so the leaderboard score is unproven. It aims to be a robust,
correctly-formatted full-inference baseline in the public classical class - not a
guaranteed jump over a prior score. Tune the `CONFIG` knobs on the training split.
"""

RUN_CELL = """\
# Full inference -> submission.csv (this recomputes the graph from the volumes).
submission = run_pipeline(input_root="/kaggle/input", out_path="submission.csv", cfg=CONFIG)
print("submission shape:", submission.shape)
submission.head()
"""


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        source = f.read()
    marker = '\nif __name__ == "__main__":'
    if marker in source:
        source = source[: source.index(marker)].rstrip() + "\n"

    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(INTRO),
        nbf.v4.new_code_cell(source),
        nbf.v4.new_markdown_cell("## Run full inference"),
        nbf.v4.new_code_cell(RUN_CELL),
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    with open(OUT, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("wrote", OUT, f"({len(source.splitlines())} source lines embedded)")


if __name__ == "__main__":
    main()
