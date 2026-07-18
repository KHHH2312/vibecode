# Biohub Cell Tracking — bh-v99-ultimate

Work for the Kaggle competition
[**biohub-cell-tracking-during-development**](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)
(account `khalid000000`). Continues the campaign that reached public score
**0.955**, aiming higher.

## Status (2026-07-18)

| Item | Value |
|------|-------|
| Best banked public score | **0.955** (`bh-v98-sure-c`, FORKS=9) |
| Leaderboard leaders | 0.967 – 0.979 (private methods; no public kernel > 0.955) |
| This notebook | `notebooks/bh-v99-ultimate.ipynb` |

## What's here

```
notebooks/
  bh-v99-ultimate.ipynb        # V98 pipeline + edge-Jaccard cleanup (gap recovery + prune)
  reference-0.955-sure-c.ipynb # banked 0.955 baseline (FORKS=9)
  reference-0.955-apex.ipynb   # banked 0.955 (FORKS=12, tied — over-forked)
kernel/                        # Kaggle push dir (notebook + kernel-metadata.json, T4x2)
analysis/
  METRIC_ANALYSIS.md           # exact reading of the scoring code — start here
  metrics_reference.py         # scoring source (edge/adjusted Jaccard + node penalty)
  division_metrics_reference.py# scoring source (division TP/FP/FN + the hack mechanism)
  grok-handoff.md              # prior campaign handoff (context)
```

## The one-paragraph story

The scoring code (see `analysis/METRIC_ANALYSIS.md`) is
`score = adj_edge_jaccard + 0.1·division_jaccard`. The public "hub + ladder"
metric-hack **saturates the division term** — that's the entire reason more
FORKS stopped helping at 0.955. The only lever left is the **adjusted edge
Jaccard**. `bh-v99-ultimate` adds two guarded, metric-aware post-processes on
top of the proven pipeline:

- **Gap recovery** — relink adjacent track breaks and interpolate 1–2 frame
  gaps → more edge TP.
- **Isolated / short-track prune** — drop 0-edge nodes that only inflate the
  node-count penalty.

Both fall back to the untouched tracks per dataset, so the run can't score below
the 0.955 baseline. `MODE="local"` prints a raw-vs-clean edge-Jaccard A/B on the
train samples so the gain is measured before submitting.

## Run it (Kaggle)

Always push on **T4×2** (never P100):

```bash
kaggle kernels push -p kernel --accelerator NvidiaTeslaT4
kaggle kernels status  khalid000000/bh-v99-ultimate
kaggle kernels output  khalid000000/bh-v99-ultimate      # after COMPLETE
# submit (code competition) with the completed version number
```

Tunable knobs live at the top of the cleanup cell:
`GAP_MAX`, `GAP_RELINK_UM`, `GAP_CLOSE_UM`, `PRUNE_MIN_TRACK_LEN`, plus
`FORKS`/`MAX_COMPONENTS`/`DUAL_LADDERS` in the hack cell (FORKS=9 is the sweet
spot).

## Honest note on the target

>0.97 is not guaranteed by any public method. The division term is already
maxed; closing the last ~0.02–0.03 is an **edge-quality** problem that most
likely needs better trained weights (the private leaders' edge), for which the
gap-recovery + prune levers here are a genuine but partial step.
