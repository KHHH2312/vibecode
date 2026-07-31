# Stage DE — live handoff (kaggle 4 / jiiiiiiiim)

Updated: **2026-07-30** (watcher DNS-timeout; kernel still was RUNNING at last good poll)

| Item | Value |
|------|--------|
| Account | `jiiiiiiiim` (`kaggle (4).json` → `~/.kaggle_trainF`) |
| Kernel | `jiiiiiiiim/bh-lineage-realde-f1` |
| Warm | Stage C step **32913** `best_exact.pt` / `last.pt` (best_loss **0.521**) |
| Datasets | `jiiiiiiiim/lineage-v2-src`, `jiiiiiiiim/lineage-v2-warm` |
| Dual GPU | ON batch=4 |
| Time budget | **11.85 h** (finish ~11h45 under 12h) |
| Step cap | 200000 (time is real limiter) |
| Expected steps @~0.49/s | **~18–20k** in 11h |
| AMP | OFF |
| Competition submit | **NO** |
| Local watcher | `wait_de_f1.py` — pull on COMPLETE, **no auto-submit** |
| Watcher note | Timed out ~20:40 UTC after api.kaggle.com DNS failures; **re-check status** |

## After COMPLETE

1. `python -m kaggle kernels output jiiiiiiiim/bh-lineage-realde-f1 -p out_stageDE_f1 --force`
2. Read `export/train_meta.json` + kernel log — health check
3. Do **not** LB-submit
4. If healthy → publish warm DE weights → build real-weight ultimate infer (see `CONTINUATION_PLAN.md`)
5. Gate: submit primary only if expected **>0.915**
