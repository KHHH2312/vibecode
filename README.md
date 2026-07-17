# Biohub – Cell Tracking During Development — CPU full-inference notebook

A self-contained, **CPU-only** Kaggle notebook that runs **full inference** end to
end (detect → link → gap-recover → divisions → tracks) and writes
`submission.csv` for the
[Biohub – Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
code competition. It recomputes the tracking graph from the image volumes on every
run — it is **not** a postprocess-only kernel and never rewrites a precomputed CSV
(the failure mode that gets rejected as *incorrect format*).

---

## ⚠️ Honest scope — read this first

- This notebook was written **without access to the competition data** (Kaggle data
  needs auth + rules acceptance; internet is off at run time). It has been verified
  end-to-end **only on synthetic 3D+time data** for code paths and output format —
  **the leaderboard score is unproven.**
- Field context: the naive nearest-neighbour baseline scores ≈ **0.505**, the public
  classical pack ≈ **0.90**, a strong full-pipeline bank ≈ **0.903**, and top private
  teams ≈ **0.97**. A genuine step-change toward ~0.97 realistically needs the
  **non-public GPU deep-detection + global-ILP** methods, which are out of reach under
  the CPU-only / internet-off / no-GPU constraints here.
- Realistic aim: a **robust, correctly-formatted, full-inference baseline in the
  public classical class** with genuinely stronger association than a single
  hyperparameter tweak. It is **not** a guaranteed jump over 0.903. Treat the numbers
  it produces as a starting point and **tune `CONFIG` on the training split**.

If you can paste the real `sample_submission.csv` header + first row and an
`ls /kaggle/input/...` listing, the column mapping and coordinate units can be
hard-validated instead of auto-detected.

---

## Repository layout

```
kernel/
  biohub_cell_tracking_cpu.py      # pipeline source of truth (readable, reviewable)
  biohub_cell_tracking_cpu.ipynb   # self-contained notebook to fork/upload/submit
  kernel-metadata.json             # Kaggle kernel metadata (CPU, internet off)
tools/
  make_synthetic_data.py           # synthetic OME-Zarr + sample_submission generator
  smoke_test.py                    # end-to-end format/code-path test on synthetic data
  build_notebook.py                # regenerates the .ipynb from the .py
requirements-kaggle.txt            # dependency lower bounds (see offline-zarr note)
```

The `.ipynb` embeds the `.py` verbatim, so the notebook is fully self-contained.
If you edit the `.py`, run `python tools/build_notebook.py` to regenerate the notebook.

---

## How it works (and how each stage maps to the metric)

The metric is roughly **edge Jaccard @ 7 µm + ~0.1 × division Jaccard − node-count
penalty (α ≈ 0.1)**. Design choices target those terms:

1. **I/O discovery** — locate the test OME-Zarr v3 volumes and `sample_submission.csv`
   under `/kaggle/input`; read OME axes + voxel spacing (needed because matching is at
   7 µm); pick the finest pyramid level whose per-frame volume fits `target_max_voxels`.
2. **Detection** — light smoothing (~radius/3) then a **background-relative threshold**
   (`median + k·MAD`, robust when nuclei are a small fraction of voxels) and local-maxima
   peaks with a min-separation. Adaptive thresholding + separation **control the node
   count** (the α penalty).
3. **Linking** — KD-tree-gated, **motion-aware** (running median flow), **ascending-
   distance mutual matching** between consecutive frames. Stronger than the official
   greedy nearest-neighbour because it resolves conflicts globally by distance.
4. **Gap closing** — bridge track ends → starts across up to `max_gap` frames and, by
   default, **interpolate the skipped frames** so missed detections are recovered. This
   lifts **edge recall**, the dominant Jaccard term.
5. **Divisions** — turn a parent into a split when two plausible daughters exist at
   t+1: either a track end with two births, **or** a continuation plus a nearby birth
   that form a symmetric, velocity-consistent sister pair (the common case after
   one-to-one linking). Feeds the **division term** with guards against false splits.
6. **Assembly & filtering** — split the graph at divisions into tracks, assign
   `parent_track_id`, and drop isolated short tracks (`min_track_len`) for precision.
7. **Write** — map to the **exact `sample_submission.csv` columns/units discovered at
   runtime** and validate (non-empty, no NaNs, integer ids/time). This is the primary
   defense against *incorrect format*.

---

## Submitting on Kaggle

1. **Fork/upload the notebook.** Either upload `kernel/biohub_cell_tracking_cpu.ipynb`
   via *Create → New Notebook → File → Import Notebook*, or push with the Kaggle CLI:
   ```bash
   cd kernel && kaggle kernels push        # uses kernel-metadata.json
   ```
   (Edit the `id` in `kernel-metadata.json` if your username differs from `khalid000000`.)
2. **Attach the data.** Add the competition *biohub-cell-tracking-during-development*
   as the notebook input.
3. **Set hardware/runtime.** Accelerator = **None (CPU)**, Internet = **Off**
   (already encoded in `kernel-metadata.json`: `enable_gpu:false`, `enable_internet:false`).
4. **Run all.** The first cell prints the discovered schema, datasets, and per-dataset
   track/division counts; the last cell writes `submission.csv`.
5. **Submit** the notebook output to the competition.

### Offline `zarr` note

The competition volumes are OME-Zarr **v3**, which needs `zarr-python ≥ 3`. `zarr` is
sometimes **not** pre-installed on the Kaggle image, and internet is off at run time.
The notebook's `import_zarr()` first tries a plain `import zarr`, then falls back to
installing offline from an attached wheels dataset. If you hit the "could not import
zarr" error: create/attach a Kaggle dataset containing `zarr` (+ `numcodecs`) wheels
and re-run — the fallback scans `/kaggle/input/*zarr*`, `*wheel*`, `*offline*`, etc.

---

## First-run sanity checks

Read the log the first run prints and confirm:

- **Column mapping** covers `track_id / t / z / y / x / parent_track_id`. Any unmapped
  sample column is treated as a per-dataset grouping id (e.g. `dataset`/`fov`).
- **Coordinate units.** Output defaults to **full-resolution voxel indices**. If the
  sample's `z/y/x` are physical micrometres instead, set `CONFIG["coord_mode"] = "um"`.
- **Runtime budget.** Check the chosen pyramid level and timepoint count against the
  competition time limit. Raise `CONFIG["target_max_voxels"]` for accuracy (finer level)
  or lower it for speed (coarser level); cap frames with `CONFIG["max_timepoints"]`
  while iterating.

---

## Tuning knobs (`CONFIG` at the top of the pipeline)

| Key | Meaning | Metric effect |
|---|---|---|
| `target_max_voxels` | per-frame voxel budget → pyramid level | speed ↔ detection accuracy |
| `cell_radius_um` | nuclear radius → smoothing scale | detection quality |
| `detect_min_distance_um` | min separation between detections | node count (precision) |
| `detect_threshold_mode` / `detect_mad_k` | adaptive foreground threshold | recall ↔ node count |
| `max_link_dist_um` | max frame-to-frame motion | link recall ↔ false links |
| `max_gap` / `gap_dist_um` / `interpolate_gaps` | gap recovery | **edge recall** |
| `sister_radius_um` / `sister_pair_max_um` / `division_symmetry_tol_um` | division gates | division term |
| `min_track_len` | drop short spurious tracks | precision + node count |
| `coord_mode` | `voxel` or `um` output coordinates | format correctness |

Recommended workflow: fork the notebook, read the training split's `geff` ground truth
(`nodes/props/{t,z,y,x}/values`, `edges/ids`) to calibrate cell radius, typical motion,
and division rate, then grid-search the linking/gap/division knobs against the metric.

---

## Local testing (no competition data needed)

```bash
pip install -r requirements-kaggle.txt nbformat
python tools/smoke_test.py          # end-to-end on synthetic data (format + code paths)
python tools/build_notebook.py      # regenerate the .ipynb from the .py
```

The smoke test verifies the pipeline runs, the output columns exactly match
`sample_submission.csv`, there are no NaNs, ids/time are integers, at least one
division is recovered, and detection recall against synthetic ground truth is sane.
It **does not** and **cannot** validate the Kaggle leaderboard score.
