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
