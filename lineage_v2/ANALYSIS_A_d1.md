# Analysis: mimiiiii0/bh-lineage-reala-d1

**Analyzed:** 2026-07-28 ~13:05 UTC  
**Kernel status:** COMPLETE  
**Decision:** **PUSH** `bh-lineage-realb-d1` (Stage B)

## Summary

| Field | Value |
|-------|-------|
| Session | realA_d1 |
| Steps | **44509** / 70000 target |
| Stop reason | **time_budget** (11h) |
| Best loss | **0.000810** |
| Elapsed | 10.75 h |
| Stems | 195 / 195 |
| Warm from | `mimiiiii0/lineage-v2-warm/last.pt` (missing=0 unexpected=0) |
| last.pt | 53.3 MB |
| best_exact.pt | 53.3 MB |
| nonfinite | **0** |
| ABORT | **no** |
| TRAIN_LOOP | **yes** |

## Notes

- Hit time budget before full 70k; still strong step count and best loss (better than A_c1 ~0.0009).
- Warm loaded cleanly from B_c1 handoff weights.
- Gates for next push: last.pt large, health OK, no invalid-dataset push allowed.

## Next

- Publish warm from A_d1 `last.pt` → `mimiiiii0/lineage-v2-warm`
- Push `mimiiiii0/bh-lineage-realb-d1` (Stage B, 35k, AMP off)
- No submit
