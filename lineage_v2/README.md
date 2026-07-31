# lineage_v2 — Biohub real-train + ultimate submit stack

CELLECT-lite + association + fork heads for
[biohub-cell-tracking-during-development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development).

**Start here for the next agent:** [`CONTINUATION_PLAN.md`](CONTINUATION_PLAN.md)

## Quick status

- Stage C COMPLETE (step 32913, best 0.521, dual B4) on `ikoooooooooooop`
- Stage DE RUNNING on `jiiiiiiiim/bh-lineage-realde-f1` (kaggle 4 / trainF)
- Primary best recent public **0.909**; hybrids 0.896/0.908; **no auto-submit ≤0.915**

## Layout

| Path | Purpose |
|------|---------|
| `lineage_v2/` | Python package |
| `dataset_src_trainF/` | DE train script + package for Kaggle dataset |
| `kernel_realDE_f1/` | Live DE notebook |
| `kernel_realC_e1/`, `kernel_stageC_c1/`, `kernel_train11h/` | C and full stage scripts |
| `kernel_yusuke_pure914/`, `kernel_hybrid_submit/` | Public-PP baselines / hybrids |
| `wait_de_f1.py` | Watch DE → pull; never submit |
| `NO_AUTO_SUBMIT.flag` | Gate file |

Weights (`*.pt`) and `out_*` trees are **not** in git — use Kaggle datasets / `kernels output`.
