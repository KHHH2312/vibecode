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

## § measured (bh-v100-clean local run, 2026-07-19)

Local validation = the 5 training clips, scored under the **patched** metric
(`tracking_cellmot` bundled into the notebook). Aggregate = micro-average
(pooled TP/FP/FN), which is how the real LB behaves.

| variant | SCORE | adjEdgeJ | edgeJ | divJ | div TP/FP/FN |
|---------|-------|----------|-------|------|--------------|
| raw (ILP tracks) | 0.9124 | 0.9124 | 0.9115 | 0.0000 | 0/0/3 |
| **clean (gap-recovery = SUBMIT)** | **0.9250** | 0.9250 | 0.9262 | 0.0000 | 0/0/3 |
| hub (Monday re-score, dense-only) | 0.9208 | 0.9208 | 0.9220 | 0.0000 | 0/**2**/3 |

Per-movie (clean = what we'd submit):

| movie | edgeJ | adj | eTP/FP/FN | Npred | Ntot | divJ |
|-------|-------|-----|-----------|-------|------|------|
| 44b6_0113de3b | 1.0000 | 1.0009 | 50/0/0 | 25517 | 25755 | nan |
| 44b6_0b24845f | 0.9600 | 0.9463 | 48/1/1 | **37487** | 32795 | nan |
| 6bba_05b6850b | 0.9638 | 0.9658 | 826/12/19 | 6234 | 6362 | nan |
| **6bba_05db0fb1** | **0.8935** | **0.8908** | **1124/75/59** | 71902 | 69800 | **0.0** |
| 44b6_33b596bf | 1.0000 | 0.9928 | 49/0/0 | 25015 | 23330 | nan |

### What the numbers prove

1. **Divisions score ZERO on our real tracks** under the patched metric. Only
   1 of 5 local clips even contains GT divisions (3 of them, in
   `6bba_05db0fb1`), and our ILP forks match **none** of them
   (`div TP/FP/FN = 0/0/3`). Divisions are effectively **unmeasurable
   locally** → they contribute ~0 to the honest score and we have almost no
   local signal to tune them.

2. **Removing the hub is strictly correct.** Post-patch the hub creates
   **cross-component division false positives** (`0/2/3` → divJ still 0) *and*
   inflates `Npred` (extra node penalty). hub (0.9208) < clean (0.9250) even
   dense-only. Keeping it can only lose points Monday. **→ submit clean.**

3. **Honest local score = 0.9250**, entirely from edges. The exploit's ~0.095
   division contribution is gone. Because the LB clips are harder than local
   (the 0.970 decomposition put the honest edge term at ~0.875), the **Monday
   re-score / honest LB is expected around 0.875–0.885**, not 0.970. This drop
   is the true cost of the exploit and is exactly what the user asked to face
   honestly.

4. **Gap recovery is net +0.0126** (raw 0.9124 → clean 0.9250): it helps 4
   movies (recovers real edge TP, cuts FN) but **hurts `44b6_0b24845f`**
   (adj 0.9551 → 0.9463) by over-adding nodes (Npred 34456 → 37487, **+14%**
   over Ntot) → node penalty. Tightening gap recovery on that movie is free
   points.

### The legitimate levers, ranked by measured headroom

1. **`6bba_05db0fb1` edge recall** — adj **0.8908**, the weakest movie and the
   dominant one by edge weight (**1124 of 2097** pooled TP). FN=59 (missed real
   edges → lower `POINT_THRESHOLD` / stronger detection / ensemble) and FP=75
   (spurious edges → pruning). This is where the real edge points are.
2. **Gap-recovery node control on `44b6_0b24845f`** — stop the +14% Npred bloat
   to recover ~+0.009 on that movie (small pooled weight, but free).
3. **Real divisions (LB gamble)** — the test set has many more divisions than
   the 3 local ones. Raising `ILP_DIVISION_WEIGHT` may capture real forks and
   earn `division_jaccard` on the LB, but there is **no local signal** to
   validate it — treat as a measured LB experiment, not a safe knob.

### Honest read on the 0.980 target

With divisions ≈ 0 and edge ≈ 0.925 local (~0.88 LB), a legitimate 0.980 needs
`adj_edge_jaccard` ≈ 0.97+ (near-perfect edge tracking) *or* a substantial real
division contribution. That is a **modeling improvement**, not a knob tweak. The
honest near-term ceiling is edge gains on the dense movies; 0.980-legit is a
stretch goal, not a next-submission expectation. The correct immediate move is
to **lock in the clean submission** so the Monday re-score can't penalize us for
the hub, then push the edge term.
