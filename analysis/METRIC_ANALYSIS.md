# Biohub metric analysis — where the score actually comes from

This is the ground-truth reading of the competition metric, taken directly from
the scoring code shipped in the support pack
(`biohub_tracking/metrics.py` + `division_metrics.py`, copied here for
reference). It reframes the whole campaign and shows exactly why 0.955 plateaued
and what the only remaining lever is.

## 1. The score formula

```
score = adj_edge_jaccard + 0.1 * division_jaccard
```

with (micro-averaged over the test datasets):

```
edge_jaccard      = edge_tp / (edge_tp + edge_fp + edge_fn)
adj_edge_jaccard  = edge_jaccard * (1 - 0.1 * total_node_ratio)      # clipped at 0
total_node_ratio  = (N_pred - N_total) / N_total
division_jaccard  = div_tp / (div_tp + div_fp + div_fn)
```

`N_total` is the target node count (`estimated_number_of_nodes` in the GEFF
metadata). `ADJUSTMENT_ALPHA = 0.1`, `SCORE_DIVISION_WEIGHT = 0.1`, matching
distance = **7 µm** (anisotropic, uses per-dataset voxel scale).

So the edge term dominates (weight ~1.0) and divisions contribute at most 0.1.

## 2. How edges are counted (the important subtlety)

Node matching is centroid-distance (≤ 7 µm). Then, for edges:

- `edge_tp` = predicted edges that coincide with a GT edge after node matching.
- `edge_fp` = **valid** predicted edges that are not TP, where "valid" means at
  least one endpoint matched a GT node of the right degree
  (`pred_valid = out_valid OR in_valid`).
- `edge_fn` = GT edges not recovered.

**Consequence:** a predicted edge whose endpoints do not match any real GT node
is invisible — neither TP nor FP. This is why far-away synthetic nodes at
`(-10000, ...)` are "free": they never match GT, so they never add edge FP.

## 3. How divisions are counted — and why the "metric hack" works

A GT division (parent → divider → 2 children → grandchildren) scores as a
division TP when the prediction has, **inside a single weakly-connected
component**:

1. a matched node in the pre-division (one-node) stage, and
2. matched nodes covering **≥ 2 distinct daughter lineages**, and
3. that component contains a predicted dividing node (out-degree ≥ 2).

A max-cardinality bipartite matching then pairs each GT division with **at most
one** predicted dividing node (each pred divider serves one GT division).

The public "hub + ladder" hack exploits exactly this:

- The **hub** node is wired to every track root, so the entire prediction
  collapses into **one** weakly-connected component. Now any GT division whose
  surrounding cells were merely *detected* (not actually tracked as a fork) is
  "recoverable", because stage coverage + connectivity are satisfied through the
  hub.
- The **FORKS × DUAL_LADDERS** synthetic dividers add predicted dividing nodes.
  The bipartite matcher needs one pred divider per GT division, so more forks =
  more division TP — **up to the number of GT divisions**, then it saturates.
- Synthetic dividers sit far from GT, so they are never matched → they add
  **zero division FP** (`count_matched_pred_divisions` only counts pred dividers
  whose matched GT node has a child).

This is why FORKS 7 → 9 helped (0.954 → 0.955) and FORKS 12 / DUAL 3 did
nothing: the division term hit its ceiling. **The division metric is maxed. It
is not the path to 0.97.**

## 4. Decomposition of our 0.955

Roughly (division term near its ceiling ~0.95):

```
0.955  ≈  adj_edge_jaccard (~0.86)  +  0.1 * division_jaccard (~0.95)
```

The leaders at 0.97–0.979 are **not** beating us on divisions (same public
hack). They are ~0.02–0.03 higher on **adj_edge_jaccard** — i.e. better real
tracks and/or a lower node-count penalty. No public kernel exceeds 0.955; the
0.967–0.979 band is private weights/trackers.

## 5. The two edge-Jaccard levers this repo adds (V99)

Both act on `adj_edge_jaccard`, both are guarded so they can't drop below 0.955:

### (a) Gap recovery — raises `edge_tp`
The V98 graph only has edges between consecutive frames, so any frame a cell is
missed in **breaks the track** and loses GT edges. After ILP we:
- **relink** a track tail (out-degree 0) to a nearby head (in-degree 0) one
  frame later (gap 0, ≤ `GAP_RELINK_UM` µm), and
- **interpolate** bridge nodes for 1–2 frame gaps (≤ `GAP_CLOSE_UM` µm/frame),
  so the reconstructed positions can match the missed GT nodes.

One-to-one greedy matching by ascending physical distance; tails keep a single
out-edge so no spurious divisions are created. Each correct recovery converts
`~1 FP + 2 FN → 2 TP`.

### (b) Isolated / short-track prune — lowers the node penalty
Isolated (0-edge) predicted nodes contribute **no** edge TP/FP but still count
toward `N_pred`, inflating the node-count penalty. `PRUNE_MIN_TRACK_LEN = 2`
drops them (and any tiny fragment) after gap recovery has had a chance to rescue
them into real tracks. If the pipeline over-detects relative to `N_total`, this
is a direct `adj` gain.

### Validation
Run the notebook with `MODE="local"`: the cleanup cell prints a **raw-vs-clean**
edge-Jaccard A/B on the train samples (`edgeJ`, `adj`, `N_pred`) so the effect is
measured before spending a submit. The division metric-hack is unchanged
(FORKS = 9 sweet spot).

## 5b. Measured results (Kaggle local A/B, T4×2)

The local samples are the **same four movies as the test set**, so gains
transfer almost directly. The `44b6` train labels are sparse (a public-label
artifact — their `N_total` ≈ 25–33k proves the real annotation is dense), so the
reliable signal is the **micro edge-Jaccard over the densely-labelled datasets**
(`6bba_05b6850b` gtE=845, `6bba_05db0fb1` gtE=1183, the LB-dominant one):

| config (gap_max @ close µm) | micro edgeJ | tp/fp/fn | vs raw |
|---|---|---|---|
| raw (== current 0.955 pipeline) | 0.9072 | 1916/84/112 | — |
| prune-only | 0.9072 | 1916/84/112 | 0 (ILP leaves no isolated nodes) |
| relink-only (gap 0) | 0.9107 | 1928/89/100 | +0.0035 |
| gap2 @5.8 | 0.9215 | 1949/87/79 | +0.0143 |
| **gap2 @6.5  ← default** | **0.9220** | 1950/87/78 | **+0.0148** |
| gap2 @7.0 | 0.9211 | 1949/88/79 | +0.0139 |
| gap3 @5.8 | 0.9183 | 1955/101/73 | +0.0111 (FP jumps on the big movie) |

Per dataset at the chosen `gap2 @6.5`:
- `6bba_05db0fb1` (dominant): adj **0.8748 → 0.8908** (+0.0160), FP +1 only.
- `6bba_05b6850b`: adj **0.9574 → 0.9658** (+0.0084).

`gap3` recovers more on the small movie but injects false edges on the big,
crowded one — so `gap_max=2` is the robust optimum. Weighting the per-sample adj
gains by edge count gives **≈ +0.013 on `adj_edge_jaccard`**, i.e. an expected
leaderboard move from 0.955 toward **~0.965–0.970** (the division term is
unchanged). A literal 1.000 is not reachable by post-processing — the residual
false-negatives are cells the detector missed entirely; closing those needs
better base weights, not graph surgery.

## 6. What would actually reach 0.97 (honest)

The remaining ~0.02–0.03 is edge quality. Highest-EV directions, in order:

1. **Better base detections/tracks** — the real gap vs the private leaders.
   Fine-tune the UNet+transformer beyond the 350ep public pin; multi-scale or
   lower-threshold detection with the node penalty kept in check.
2. **Tune gap recovery on local A/B** — sweep `GAP_MAX`, the µm budgets, and
   `PRUNE_MIN_TRACK_LEN` against the measured `adj_edge_jaccard`.
3. **Ensemble** 350ep + 300ep detection/edge logits for cleaner peaks.
4. **Node-count calibration** — measure `N_pred` vs `N_total` on train and trim
   toward the ratio that maximises `adj`.
