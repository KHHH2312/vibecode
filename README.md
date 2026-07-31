# vibecode — Biohub cell tracking

Private work for
[biohub-cell-tracking-during-development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

## Active stack (2026-07-30)

**`lineage_v2/`** is the live training + submit path (CELLECT-lite real train → ultimate infer).

| Doc | Purpose |
|-----|---------|
| [`lineage_v2/CONTINUATION_PLAN.md`](lineage_v2/CONTINUATION_PLAN.md) | Full handoff plan for the next agent |
| [`lineage_v2/HANDOFF_TRAINE_DE_STATUS.md`](lineage_v2/HANDOFF_TRAINE_DE_STATUS.md) | Stage DE live status |
| [`lineage_v2/NO_AUTO_SUBMIT.flag`](lineage_v2/NO_AUTO_SUBMIT.flag) | Gate: submit only if expected **>0.915** |

### Snapshot

- Stage C COMPLETE — step 32913, best_loss 0.521 (dual B4, AMP off)
- Stage DE RUNNING — `jiiiiiiiim/bh-lineage-realde-f1` (kaggle 4 / trainF), no LB submit
- Primary best recent public ~**0.909**; hybrids 0.896/0.908
- Public ref to beat: Yusuke **0.914**
- **No auto-submit ≤0.915**

### Older campaign artifacts

Earlier `kernel/`, `notebooks/`, `analysis/` trees (v99–v112, 0.90x edge-TTA era) may still exist on `main` history. Prefer `lineage_v2/` for continuing work.

## Clone

```bash
gh repo clone KHHH2312/vibecode
cd vibecode
```
