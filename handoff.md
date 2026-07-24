# BIOHUB CELL-TRACKING KAGGLE CAMPAIGN — HANDOFF
_Last updated: 2026-07-22 (UTC). Workspace: `C:\Users\Khalid\Desktop\New_folder\vibecode`. Supersedes all prior handoffs._

## 0. THE GOAL (verbatim intent)
- Compete in Kaggle comp `biohub-cell-tracking-during-development`, account `khalid000000`.
- Score as high as possible **WITHOUT the exploit** (hub/ladder metric hack is FORBIDDEN — see §1).
- User's cadence: **start at 0.909 at midnight UTC, then add ≥ +0.005 every day** ("0.05" in the goal string = +0.005/day). Next notebooks must clear **0.914**.
- North star 0.985 honest. **HONEST ASSESSMENT: 0.985 almost certainly NOT reachable** with current public assets — clean public field tops ~0.909; every 0.985+ LB score is the exploit. Incremental daily gains are the realistic plan.

## 1. HARD RULES (non-negotiable)
1. **No exploit ever.** Grep and reject sentinels: `hub_id`, `FORKS=`, nodes at `(-10000,-10000,-10000)`, `MAX_COMPONENTS`, `DUAL_LADDERS`, `divider_id`, negative-time nodes.
2. **One submit per notebook, EVER.** (We wasted 3 slots re-submitting the same notebook — never again.)
3. **Max 5 submits/day**, reset 00:00 UTC.
4. **Never submit unless confident it scores well** (measured locally first, or a faithful reproduction of a known-good public score).
5. Only submit notebooks producing **exactly the 4 test stems**: `44b6_0113de3b, 44b6_0b24845f, 6bba_05b6850b, 6bba_05db0fb1`.
6. **Guards every submit**: 4 stems exact, 0 dangling edges, 0 exploit sentinels, 0 nodes at -10000.
7. **Training exploit boundary**: the 4 test stems have GT `.geff` in the train mount. Training on them = memorizing answers = FORBIDDEN. Train only on train stems excluding the 4 test + 8 val stems; validate on held-out train stems (legit CV).
8. Submit path ONLY: `python -m kaggle competitions submit biohub-cell-tracking-during-development -k khalid000000/<slug> -v <version> -f submission.csv -m "<msg>"`
9. T4×2 (`machine_shape: NvidiaTeslaT4`), internet off for submissions. Kernel push+run is FREE; only `competitions submit` costs a slot.

## 2. THE METRIC (drives every decision)
`score = adj_edge_jaccard + 0.1 * division_jaccard`
- `adj_edge_jaccard = max(0, edge_jaccard * (1 - 0.1*(N_pred - N_true)/N_true))`, per dataset.
- division_jaccard pooled across datasets before Jaccard. Node match by centroid dist ≤7µm; scale z=1.625, y=x=0.40625 µm/voxel.
- **Implication:** edge_jaccard already ~0.943 locally (nearly tapped). The untapped term is **0.1 × division_jaccard** — divisions are the honest lever.

## 3. VERIFIED SCORES (live Kaggle CLI)
- **Best honest COMPLETE = 0.903** (refs 54748675 / 54704551 / 54704922).
- Banked floor 0.902 (54862719).
- **bh-clean-909 = faithful repro of yusuketogashi clean public 0.909. NEVER SUBMITTED YET.** Guard-safe: 236185 rows, 4 stems, 0 dangling, 0 sentinels. Expected ~0.909 = first gain above 0.903. **Cron queued to submit at 00:05 UTC** (see §7).
- Exploit-era (FORBIDDEN, never revive): 0.970 / 0.955 / 0.954 / 0.952.

## 4. LEVERS — DEAD vs OPEN
**DEAD (do not revisit):**
- Public weight-swaps: 350ep → 0.901 LB, v34 alt-lineage → 0.901 LB. Both REGRESSED vs 50ep bank. Family exhausted.
- ILP `division_weight`: bank-proxy peaks div_w=0.70 (local 0.9693) but does NOT transfer — on real clean909 pipeline, 1.0→0.60 moved output by ONE division (309 vs 308). clean909's safe-division PP re-adds divisions after ILP, overriding the ILP weight. div060/070/075 ≈ 0.909.
- Swap-gate loosening: tapped out (decimal-place).

**OPEN / PROMISING:**
- **Safe-division PP gating params** = the live division lever. PURELY GEOMETRIC in clean909 (deepcenter veto OFF), gated only by distance caps:
  - `BIOHUB_SAFE_DIV_MAX_UM=4.66`, `BIOHUB_SAFE_DIV_SISTER_MAX_UM=8.5`, `BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM=7.65`
  - `BIOHUB_SAFE_DIV_FRAME_FRAC_CAP=0.0076`, `BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP=0.00375`
  Geometry-only ⇒ can be swept FULLY LOCALLY (no images) on cached candidate graphs + GT. **Most promising path to +0.005.**
- **Fine-tune** (only theoretical 0.985-magnitude lever) — but historically regresses; see §6.

## 5. THE MEASUREMENT UNLOCK (crown jewel)
Compute the **EXACT competition metric locally** on train GT, FREE, no Kaggle:
- `biohub_tracking.metrics` (evaluate/node_recall/per_sample_metrics/summarise) — source at `out_clean909/tracking_repo/src/biohub_tracking/`.
- Local ILP: tracksdata 0.1.0rc6 + ilpy 0.6.0 + pyscipopt (SCIP, ~20-26s/solve; Gurobi absent).
- **Validated EXACT**: local div_w=1.05 → 0.9440 / edgeJ 0.9434 — identical to paid Kaggle run.
- Harness: `sweep_worker.py` (one ILP solve/subprocess, fixes OOM), `sweep_driver.py`.

**Problem last session solved:** original 8 val stems have only 5 GT divisions → division signal too weak to tune safe-div lever.
**Fix (DONE):** `bh-divcensus` kernel counted divisions across 195 train stems (**148 total; 86 stems have >0**) and exported GT for the **24 most division-rich stems** → `out_divcensus/gt_divrich.zip` (NEEDS UNZIP). Top: 6bba_48816121(5), 6bba_09961292(4), 6bba_afb141ff(4), 6bba_cdcfe533(4), 6bba_debd7bfa(4), 6bba_df673a83(4), then several with 3.

## 6. FINE-TUNE (Task #1) — DESIGN NOTES
- Script: `out_clean909/tracking_repo/scripts/train_unet_transformer.py`. Args: `--data-dir --splits --unet-weights --epochs --lr --batch-size --single-gpu/--data-parallel --det-loss-weight --det-neg-weight`.
- **Leak-free split DONE**: `finetune_split.json` — train=187 (199 − 4 test − 8 val), test=8 val. Verified disjoint. Also `all_stems.json`.
- **TWO TRAPS that likely caused prior regressions:**
  1. `--unet-weights X` loads X into ONLY the UNet submodule (strict=False) → **silently drops the transformer edge head** (fresh transformer each run). Full checkpoint `weights/unet_transformer/split_0/edge_predictor_best.pth` is the FULL state_dict. PATCH the script to load the full model state before the loop to truly continue training.
  2. Checkpoint selection uses `test_acc*test_recall` proxy (line ~1165), NOT the comp metric. Fix: save checkpoints periodically, run full inference+ILP+PP on val per checkpoint, local_score, pick by REAL metric.
- LOW odds of beating 0.909; do NOT submit a fine-tune output unless it beats 0.909 locally first.

## 7. QUEUED / IN FLIGHT
- **Cron `808a80c4`** (durable, `.claude/scheduled_tasks.json`): fires 00:05 UTC (01:05 local, UTC+1). Re-verifies slots+guards+that bh-clean-909 was never submitted, then submits `bh-clean-909 v1` ONCE → banks 0.909.
- Kaggle kernels (all COMPLETE): `bh-clean-909` (0.909 repro, submission ready), `bh-clean909-div060` (div lever test, dead), `bh-gtexport`, `bh-val-baseline` (local baseline 0.9440), `bh-stemlist` (199 stems), `bh-divcensus` v2 (census+GT export), `bh-v913-swapgate`.
- Local artifacts: `dsout/cand_graphs/` (8 cached pre-ILP graphs w/ edge_prob), `gt_local/` (8 val GT), `out_divcensus/gt_divrich.zip` (24 div-rich GT), `out_clean909/submission.csv` (the 0.909), `sweep_out/` (full div_w sweep).

## 8. PLAN TO HIT 0.914 (next concrete steps)
1. Let cron bank 0.909 at 00:05 UTC; verify LB score.
2. Unzip `out_divcensus/gt_divrich.zip` → 24 div-rich GT stems.
3. Run a free GPU kernel (like `bh-val-baseline`) to produce candidate graphs for the 24 div-rich stems.
4. Reproduce clean909's `add_safe_divisions_postlink` geometric gating in a standalone local script; sweep `SAFE_DIV_MAX_UM / SISTER_MAX_UM / FRAME_FRAC_CAP / GLOBAL_FRAC_CAP` to maximize division_jaccard WITHOUT hurting edge_jaccard, scored with `biohub_tracking.metrics`.
5. If a param set beats clean909 division recall locally by enough to clear +0.005, bake those env vars into a NEW notebook (e.g. `bh-clean909-safediv-v2`), guard, submit ONCE at next reset.
6. In parallel (free): proper full-checkpoint-warm-start fine-tune with real-metric checkpoint selection. Submit only if it beats 0.909 locally.

## 9. GOTCHAS
- geff edge ids = zarr v3, zstd uint64 `[E,2]` at `<stem>.geff/edges/ids/c/0/0`. Local machine has NO zstd/zarr; decode on Kaggle with `numcodecs.Zstd().decode()` (how divcensus reads them).
- `io.open_dataset` imports torch → bypass locally: load `.geff` directly with `td.graph.IndexedRXGraph.from_geff`, parse scale from `zarr.json` multiscales.
- Solver output (GraphView) must be `.detach()`'d before `evaluate` (`.copy()` unsupported locally).
- `PYTHONIOENCODING=utf-8` when printing notebook content (Windows cp1252).
- Foreground `sleep` polls hit a 2-min Bash cap; use background watchers.
- `kaggle kernels push` sometimes throws transient "Expecting value: line 1 column 1" — just retry.

## 10. FILE INDEX
- `campaign_scores.md` — full verified-scores log + all sweep findings.
- `finetune_split.json`, `all_stems.json` — leak-free training split.
- `sweep_worker.py`, `sweep_driver.py` — local ILP metric harness.
- `kernel_clean909/` — the 0.909 notebook. `kernel_divcensus/` — census + GT export.
- `out_divcensus/divcensus.json` — per-stem division counts (195 stems).
- `out_divcensus/gt_divrich.zip` — GT for 24 div-rich stems (unzip me).
- Memory (persists across sessions): `C:\Users\Khalid\.claude\projects\C--Users-Khalid-Desktop-New-folder\memory\` — `biohub-verified-scores-2026-07-21.md`, `biohub-finetune-plan.md`.

---

## POST-RESCORE UPDATE (2026-07-24)

### 0.911 claim on Downloads notebooks (Claude)
- Notebooks: `Downloads/bhdsc350epcv.ipynb` = kernel `bh-dsc-350ep-cv`; `bhdsem2div.ipynb` = m2div variant E.
- **Fixed-8 CV is REAL:** score **0.88163**, delta_vs_124 **+0.00241**, divJ **0.0** (file `out_dsc/127_lite_cv_status.json`).
- **"0.911 minimum LB" is NOT a guarantee** — CV→LB extrapolation (0.879 CV ↔ 0.908 LB ⇒ +0.029). Expected LB ≈ **0.910–0.911 if transfer holds**.
- Submitted: ref **54939707** (PENDING). Best prior COMPLETE honest: **0.909**.

### Path to ~0.930 (honest)
- Leaders post-rescore ≈ **0.929** — honest board; exploit dead.
- Division term still 0 on our best CV → m2div graft is the large honest lever (`bh-dse-m2div-v2` pushed RUNNING).
- Do not re-enable hub/ladder.

### Ops
```
python -m kaggle competitions submissions -c biohub-cell-tracking-during-development
python -m kaggle kernels status khalid000000/bh-dse-m2div-v2
```
