# Safe-division sweep verdict (bh-divsweep-v1)

_2026-07-22. CPU sweep on the leak-free pre-safe cache (50ep base, 31 val stems),
scored with the bundled PATCHED metric (== Monday re-score) vs local GT._

## Result: the bank safe-division config is already optimal. Division is a dead lever.

| config | ALL score | Δ vs bank | edgeJ | divJ | divTP/FP/FN |
|--------|-----------|-----------|-------|------|-------------|
| **baseline (bank)** | **0.8948** | — | 0.8896 | 0.0229 | 3/58/70 |
| safe_OFF | 0.8926 | −0.0022 | 0.8895 | 0.0000 | 0/0/73 |
| gcap 0.0075…0.10 | 0.8948 | ~0 | 0.8897 | 0.0224 | 3/61/70 |
| maxum 5.5 / 6.5 / 7.5 | 0.8933 / 0.8916 / 0.8883 | −0.0015…−0.0065 | ↓ | ~0.022 | FP explodes |
| combo_A/B (aggressive) | 0.8925 | −0.0023 | 0.8866 | 0.0343 | 11/248/62 |
| combo_C (most aggressive) | 0.8915 | −0.0033 | 0.8852 | 0.0410 | 16/317/57 |

## Why it's dead

- **Division detection precision is ~5% and stays ~5% at every aggression level**
  (bank 4.9%, combo_C 4.8%). Raising caps/geometry scales true and false divisions
  *together* — it never improves precision.
- Only **73 real divisions across 31 movies (~2.4/movie)**; 27/31 movies have ≥1, max 5.
  The geometric safe-division cannot distinguish a true missed sister from a false one.
- The bank's conservative config already banks the entire usable division gain: **+0.0022**
  (safe_OFF 0.8926 → bank 0.8948). Every more-aggressive variant nets negative because the
  false division edges cost more edge-Jaccard than the 0.1-weighted divJ gain returns.

## Strategic consequence

The score is edge-dominated (adj_edge_jaccard weight 1.0 vs division 0.1). With:
- division tuning **exhausted** (this sweep),
- better public weights **regressed** (v110 350ep → 0.901, v112 v34 → 0.901),
- association tweaks **regressed** (v102/v103 → 0.895),
- per-movie threshold **dead** (v101 +0.0002),

the honest ceiling with available assets is confirmed at **~0.902 pre-patch / ~0.872
post-patch**. No post-processing lever bridges the gap to 0.920+. The only remaining
honest path is a materially better edge predictor — but the 350ep-regressed-vs-50ep
result is evidence *against* "just train longer" helping this metric.

**Recommendation:** honest best submission remains the bank (0.902). Do NOT submit any
safe-division variant — none beat the bank.

---

## Ensemble edge-predictor screen (bh-ens-screen-v1) — 2026-07-22

**Result: the 3-model edge-logit ensemble HURTS. Dead lever.**

Averaging `sigmoid(predict_edges)` across {50ep, 350ep, v34} (identical
architecture, detector fixed to 50ep so node sets match), scored raw-ILP
`edge_jaccard` with the patched metric on 6 GT train movies:

| movie | single 50ep | ensemble | Δ | eTP/eFP/eFN single→ens |
|-------|------------:|---------:|------:|------------------------|
| 6bba_57b7cc1e (dominant) | 0.6565 | 0.6387 | −0.0178 | 1225/274/367 → 1188/268/404 |
| 44b6_d5e7d891 | 0.8092 | 0.7932 | −0.0160 | 789/82/104 → 775/84/118 |
| 44b6_12dfb391 | 0.8764 | 0.8707 | −0.0056 | 716/44/57 → 714/47/59 |
| 6bba_337b1b3a | 0.9611 | 0.9579 | −0.0032 | 1186/21/27 → 1182/21/31 |
| 44b6_0c582fdc | 0.9041 | 0.9041 | 0.0000 | 66/3/4 → 66/3/4 |
| 6bba_062c8d37 | 0.9978 | 0.9978 | 0.0000 | 896/0/2 → 896/0/2 |
| **micro-avg** | **0.8320** | **0.8224** | **−0.0096** | |

**Why:** 350ep and v34 are individually *weaker* edge predictors (both
regressed to 0.901 on the LB vs 50ep's 0.902). Prob-averaging them in *dilutes*
the stronger 50ep signal — it drops true edges (eTP down) and adds misses
(eFN up), worst on the dominant movie. There is no diversity benefit because the
extra models are not better, just worse-and-correlated.

**Consequence:** "better edge predictor via public weights" is now fully
exhausted (single-350ep regressed, single-v34 regressed, 3-way ensemble hurts).
Combined with the division sweep, weight swaps, refine/assoc tweaks, and
threshold sweep all landing ≤ bank, the honest ceiling with public assets is
**confirmed at ~0.902 pre-patch**. The gap to the 0.96+ leaders reflects private
training data/weights, not a public post-processing or ensembling lever.
