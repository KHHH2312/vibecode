# The division-metric patch — pivot to a legitimate score

## What happened (2026-07-19)

The competition host (Thibaut Goldsborough) announced that the **division
Jaccard exploit** was found and is being **patched + re-scored** (likely
Monday). The patch is public:
`https://github.com/royerlab/kaggle-cell-tracking-competition`. Submissions
that did **not** actively exploit the division metric are unaffected; ones that
**did** (ours: the hub + ladder hack) get re-scored under the new metric.

Our banked **0.970** decomposes (from the metric source) roughly as:

```
0.970  ≈  adj_edge_jaccard (~0.875, legitimate)  +  0.1 · division_jaccard (~0.95, EXPLOIT)
```

So the ~0.095 division contribution is exactly what the patch removes.

## What the patch changes (division_metrics.py)

The old metric credited a GT division whenever the surrounding cells were merely
**in the same weakly-connected component** — which the **hub** guaranteed by
wiring every track root together. The patch replaces that with a **local
directed-topology** test:

- `_is_strongly_connected_division` requires a matched **parent → fork → two
  distinct daughter lineages** locally, through the predicted dividing node
  itself. Same-component is no longer enough → **the hub no longer creates TPs.**
- Candidate forks are restricted to matched parent-side nodes and their
  successors, so the far-away synthetic ladder dividers (never matched to GT)
  **can't be candidates.**
- `_pred_division_fork_sets` marks predicted forks whose child branches land in
  **distinct GT weakly-connected components** as **cross-component false
  positives**. The hub's children span every component → the hub becomes a
  **division FP**, actively lowering the score.
- `metrics.py` also now caps out-degree at 2 and de-duplicates merged edges at
  scoring time — more ways the synthetic structure is neutralized.

**Net:** under the patched metric the exploit doesn't just stop helping; the hub
can *add* division FP. The only sane move is to **remove it** and earn divisions
from real tracked forks.

## Why we're not starting from zero on divisions

The base ILP already runs with `division_weight = 1.05` (cell 4), so
`my_predict/*.geff` contain **real** division forks. Removing the hack therefore
swaps a fake ~0.95 division term for a **real, measurable** one that we can raise
by tuning legitimate levers.

## The legitimate levers (post-patch)

`score = adj_edge_jaccard + 0.1 · division_jaccard`

1. **Edge term (weight ~1.0)** — keep the gap-recovery gains (already legit,
   already in the pipeline). Detection threshold + 350/300ep ensemble can lift
   real edge TP further.
2. **Division term (weight 0.1)** — now a real tracking problem. Tune
   `ILP_DIVISION_WEIGHT` to trade division TP against FP against the patched
   `division_jaccard`. This is the headroom the hub used to fake.

## bh-v100-clean

`notebooks/bh-v100-clean.ipynb` (and `kernel/`) is the honest notebook:

- Bundles the **patched** metric as an importable `tracking_cellmot` package so
  `MODE=local` scores exactly as Monday's re-score will.
- **Drops the hub/ladder cell** → `submission.csv` is clean.
- Scores three graphs under the patched metric and prints the breakdown:
  - `raw` — ILP tracks straight from the solver (real divisions),
  - `clean` — raw + gap recovery == what we would submit,
  - `hub` — clean + hub/ladder == what is currently submitted → predicts the
    Monday re-score.

Measured numbers land in `§ measured` below once the local run completes.

## § measured

_(to be filled from the bh-v100-clean local run)_
