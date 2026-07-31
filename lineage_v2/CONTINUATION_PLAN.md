# Continuation plan — Biohub lineage_v2 (as of 2026-07-30)

## Hard rules (do not break)

1. **NO auto-submit** to competition LB unless expected **public score > 0.915**.
   - Flag file: `NO_AUTO_SUBMIT.flag`
2. Train workers **never** competition-submit. Only **primary** (`khalid000000`) submits, **≤3/day**.
3. Notebook narrative: training described as **main-account only**. Never mention multi-account / farm.
4. Session budget **≤11–12h** per kernel; prefer finish ~11h45 with `STOP_REMAIN≈0.12h`.
5. AMP **OFF** for CELLECT train (AMP caused NaNs on Stage C).

## Accounts / Kaggle configs (local machine)

| Role | Username | Config dir | Role |
|------|----------|------------|------|
| Primary submit | `khalid000000` | `~/.kaggle` (or `kaggle.json`) | LB only ≤3 |
| Train C done | `ikoooooooooooop` | `~/.kaggle_trainE` (kaggle 3) | Stage C COMPLETE |
| Train DE **active** | `jiiiiiiiim` | `~/.kaggle_trainF` (kaggle 4) | Stage DE RUNNING |
| Older trains | mimiiiii0 / others | trainB–D | A/B history; stop new A/B |

CLI pattern:

```powershell
$env:KAGGLE_CONFIG_DIR = "$env:USERPROFILE\.kaggle_trainF"
python -m kaggle kernels status jiiiiiiiim/bh-lineage-realde-f1
```

## Current training state

| Stage | Kernel | Account | Status | Notes |
|-------|--------|---------|--------|-------|
| A/B real GEFF | various | train C/D | done / stopped | 195 stems real GEFF |
| **Stage C** | `ikoooooooooooop/bh-lineage-realc-e1` | trainE | **COMPLETE** | step **32913**, best_loss **0.521**, dual B4, AMP off, ~11.73h |
| **Stage DE** | `jiiiiiiiim/bh-lineage-realde-f1` | trainF | **RUNNING** (pushed ~16:57 UTC 2026-07-30) | resume C warm; dual B4; TIME 11.85h; STEPS 200000; **no submit** |
| Warm datasets (F) | `jiiiiiiiim/lineage-v2-src`, `jiiiiiiiim/lineage-v2-warm` | trainF | published | C export best/last |

Local watcher `wait_de_f1.py` **timed out** after DNS blips (~20:40 UTC) while kernel still RUNNING. **Re-arm watch**; do not assume COMPLETE.

Expected DE: ~**18–20k steps** in ~11h @ ~0.49 step/s dual B4 (time-limited, not step-capped).

## Primary LB / submissions (watch these)

**Competition:** `biohub-cell-tracking-during-development`

| Public | Kernel / desc | Note |
|--------|---------------|------|
| **0.909** | CLEAN 0.909 + edge-TTA | best banked honest-ish |
| **0.908** | hyb-b ULT1+Yusuke 350ep; dsc-350ep; many cands | plateau |
| **0.896** | hyb-a Yusuke+350ep | worse |
| Public SOTA ref | Yusuke-style **0.914** (public notebook) | pure914 kernel COMPLETE on primary, **NOT submitted** (gate) |

Recent primary kernels:

- `khalid000000/bh-yusuke-pure914` — **COMPLETE**, not submitted (gate >0.915)
- `khalid000000/bh-hyb-a-yusuke350` / `bh-hyb-b-ult1yusuke` — COMPLETE, scored 0.896 / 0.908
- Stage C CELLECT infer — **ERROR** (arch/PP mismatch vs Yusuke UNet)

**Why hybrids failed to beat 0.914:** used public **350ep** weights, not real CELLECT Stage C/DE. Stage C/DE train path used **synthetic** script path earlier; C weights are **not** drop-in for Yusuke UNet edge_predictor.

## Score model (always)

`score ≈ adj_edge_jaccard + 0.1 · division_jaccard`

Optimize both edges and divisions. Public LB ≈ partial test.

## What to do next (ordered)

### 1) Finish DE training (NOW)

```powershell
$env:KAGGLE_CONFIG_DIR = "$env:USERPROFILE\.kaggle_trainF"
python -m kaggle kernels status jiiiiiiiim/bh-lineage-realde-f1
# when COMPLETE:
python lineage_v2/wait_de_f1.py   # or manual kernels output -p out_stageDE_f1
```

Pull export → read `train_meta.json` / logs:

- steps advanced past 32913?
- best_loss vs 0.521?
- NaN / crash / early stop?
- dual GPU samples/s healthy?

**If ERROR/CANCEL:** diagnose log, re-push DE (or single-GPU fallback) on trainF — still no LB submit.

**If COMPLETE + healthy:** publish DE export as new warm dataset on the account that will run next train/infer (usually primary for submit path, or same worker for more train).

### 2) Do **not** burn submit slots on ≤0.915 clones

Gate: only submit if **honest expected public > 0.915**.

Pure Yusuke 0.914 COMPLETE is a **floor reference**, not an auto-submit.

### 3) After DE healthy — build real-weight ultimate submit

Path that can beat 0.914/0.915:

1. Working **CELLECT** (or matching-arch) **inference + association + fork** notebook  
2. Load **DE best_exact.pt** (else C) with DataParallel `module.` strip  
3. Steal **only PP** from Yusuke 0.914 that is arch-agnostic: harmonic det, bidir edge costs, frame retention, division rules — **not** force-load UNet weights into CELLECT  
4. Optional track-space ensemble: Yusuke detections ⊕ CELLECT tracks  
5. Offline sanity / stem coverage / schema checks  
6. Push on **primary** T4×2, internet OFF, ≤11.5h  
7. Submit **only** if conviction **>0.915**

### 4) Optional further train (if DE weak)

- Real-GEFF Stage C (not synthetic) if stems available on worker  
- Second DE seed / lower LR polish  
- Still train-only on workers

### 5) Watch cadence

Every ~5–15 min while DE runs:

```text
python -m kaggle kernels status jiiiiiiiim/bh-lineage-realde-f1
python -m kaggle competitions submissions -c biohub-cell-tracking-during-development
python -m kaggle kernels list --mine -v
```

Primary config for submissions; trainF for DE.

## Repo map (training essentials in git)

```text
lineage_v2/
  lineage_v2/           # package (models, checkpoint, metrics)
  dataset_src_trainF/   # DE train script + package snapshot
  kernel_realDE_f1/     # live DE kernel
  kernel_realC_e1/      # Stage C kernel
  kernel_stageC_c1/     # train_stage_c.py
  kernel_train11h/      # A/B/C/DE train scripts
  kernel_yusuke_pure914/
  kernel_hybrid_submit/
  kernel_stagec_infer/  # broken CELLECT infer — fix later
  wait_de_f1.py         # DE watch, pull on COMPLETE, NO submit
  NO_AUTO_SUBMIT.flag
  HANDOFF_*.md / PLAN_*.md / this file
```

**Not in git (by design):** `*.pt`, `out_*`, warm dumps, large logs. Weights live on Kaggle datasets.

## Push / pull helpers

```powershell
# DE status + pull
$env:KAGGLE_CONFIG_DIR = "$env:USERPROFILE\.kaggle_trainF"
python -m kaggle kernels status jiiiiiiiim/bh-lineage-realde-f1
python -m kaggle kernels output jiiiiiiiim/bh-lineage-realde-f1 -p lineage_v2/out_stageDE_f1 --force

# Primary submissions
$env:KAGGLE_CONFIG_DIR = "$env:USERPROFILE\.kaggle"
python -m kaggle competitions submissions -c biohub-cell-tracking-during-development
```

## Success criteria for “training finished → submit time”

Training complete **≠** submit time.

Submit only when:

1. DE (or best real) export healthy  
2. CELLECT (or fused) infer notebook runs end-to-end on a smoke stem  
3. Expected public **> 0.915** vs Yusuke 0.914 / our 0.909 bank  
4. User gate still respected (`NO_AUTO_SUBMIT.flag`)
