# Biohub campaign — live scores (verified against Kaggle CLI)

**Last poll:** 2026-07-21 (this session). Source: `python -m kaggle competitions submissions -c biohub-cell-tracking-during-development -v`.

## Best honest COMPLETE

**0.903** — ref `54748675` / `54704551` / `54704922`
("det-only weights + praxel902 PP one-axis", 2026-07-15). The handoff called best
0.902; the live CLI shows a 0.903 lineage. `best = 0.903`.

Yesterday's re-floor `54862719` (v100c bank restore) = **0.902** COMPLETE — floor held.

## This week's campaign results (verified)

| ref | date UTC | kernel / desc | status | publicScore | verdict |
|-----|----------|---------------|--------|-------------|---------|
| 54862719 | 2026-07-20 21:28 | v100c bank restore | COMPLETE | **0.902** | floor held |
| 54863143 | 2026-07-20 22:05 | v110 350ep bank PP | COMPLETE | **0.901** | REGRESSED vs bank |
| 54863473 | 2026-07-20 22:35 | v111 gapfn | **PENDING** (~14h) | — | stuck in queue |
| 54840764 | 2026-07-20 00:00 | v102 refine | COMPLETE | 0.895 | regressed |
| 54844836 | 2026-07-20 04:21 | v103 assoc | COMPLETE | 0.895 | regressed |

## Key finding: the 350ep lever is dead

v110 used *more training epochs* (350 vs 50) of the public checkpoint and scored
**below** the 50ep bank (0.901 < 0.902/0.903). More epochs of that public lineage
HURT. So "better-trained public weights" is NOT the lever the handoff hoped.

The remaining honest weight lever is a **different training lineage**, not more
epochs → v34-retrain mirror (2026-07-08), A/B on identical bank PP.

## In flight this session

| kernel | version | dataset weights | status |
|--------|---------|-----------------|--------|
| bh-v112-v34 | 1 | support-pack 50ep + v34-retrain mirror (PREFER_350EP=0) | pushed, RUNNING at T4x2 |

Decision on completion: run guards; submit only if it beats best (0.903), else it's
a negative result and we stay on bank (0.902 already banked — floor safe).

## Exploit-era (FORBIDDEN, do not revive)

0.970 / 0.955 / 0.954 / 0.952 — hub/ladder metric hack. Off-limits.
