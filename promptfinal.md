# promptfinal.md — Continue Biohub training + watch submissions

You are taking over an in-progress Kaggle campaign for:

**Competition:** [Biohub – Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)

**Code repo (already pushed):** https://github.com/KHHH2312/vibecode  
**Local workspace:** `C:\Users\Khalid\Desktop\New_folder\vibecode`  
**Training stack path:** `lineage_v2/`  
**Read first in repo:** `lineage_v2/CONTINUATION_PLAN.md`, `lineage_v2/HANDOFF_TRAINE_DE_STATUS.md`, `lineage_v2/NO_AUTO_SUBMIT.flag`

---

## Your mission (in order)

1. **Watch Stage DE training** until COMPLETE/ERROR/CANCEL  
2. **Watch primary submissions / kernels** (do not waste daily slots)  
3. **Finish the train path** (pull DE, health-check, publish weights if good)  
4. **Only then** build a real-weight ultimate submit path that can beat **0.915**  
5. **Never auto-submit** unless expected public score **> 0.915**

This is a **train-finish + watch** handoff, not a “spam submit” handoff.

---

## Non-negotiable rules

| Rule | Detail |
|------|--------|
| Submit gate | Expected public **> 0.915** only. File: `lineage_v2/NO_AUTO_SUBMIT.flag` |
| Who submits | **Only** primary `khalid000000` (`~/.kaggle`). **≤3** competition submits/day |
| Train workers | `jiiiiiiiim` (trainF / kaggle 4), `ikoooooooooooop` (trainE) — **train only, no LB submit** |
| Runtime | ≤11–12h/session; prefer stop ~11h45 |
| AMP | **OFF** for CELLECT train (NaNs with AMP on Stage C) |
| Narrative | Notebook text: training on **main account only**. Never mention multi-account |
| Code competition | Internet **OFF** at submit; valid submission artifact; use kernel submit path not raw 400 CSV mistakes |
| No token burn | No 30s busy-wait loops that spam the agent; use a background watcher or poll every 1–5 min |

---

## Accounts (local Kaggle API dirs)

| Username | Config | Use |
|----------|--------|-----|
| `khalid000000` | `~/.kaggle` | Primary LB kernels + submissions |
| `jiiiiiiiim` | `~/.kaggle_trainF` ← Downloads `kaggle (4).json` | **Stage DE live** |
| `ikoooooooooooop` | `~/.kaggle_trainE` ← kaggle (3) | Stage C done |
| Others (mimiiiii0, …) | trainB–D | Historical A/B; do not restart A/B unless needed |

```powershell
# Always set config dir before kaggle CLI
$env:KAGGLE_CONFIG_DIR = "$env:USERPROFILE\.kaggle_trainF"   # DE
python -m kaggle kernels status jiiiiiiiim/bh-lineage-realde-f1

$env:KAGGLE_CONFIG_DIR = "$env:USERPROFILE\.kaggle"          # primary
python -m kaggle competitions submissions -c biohub-cell-tracking-during-development
```

Use `python -m kaggle` (Python 3.14 on this machine has the package). Plain `kaggle` may be missing from PATH.

---

## LIVE: Stage DE (finish training)

| Field | Value |
|-------|--------|
| Kernel | `jiiiiiiiim/bh-lineage-realde-f1` |
| Account | `jiiiiiiiim` / `~/.kaggle_trainF` |
| Warm start | Stage C **step 32913**, **best_loss 0.521**, dual GPU batch=4, AMP off |
| Datasets | `jiiiiiiiim/lineage-v2-src` + `jiiiiiiiim/lineage-v2-warm` |
| Env | SESSION `realde_f1`, STEPS 200000, TIME **11.85h**, STOP_REMAIN 0.12, DUAL=1, BATCH=4, LR 3e-5, AMP=0 |
| Expected | ~**18–20k steps** in ~11h @ ~0.49 step/s (time-limited) |
| Submit? | **NO** |

Local script (re-arm if needed):

```powershell
cd C:\Users\Khalid\Desktop\New_folder\vibecode
$env:KAGGLE_CONFIG_DIR = "$env:USERPROFILE\.kaggle_trainF"
python lineage_v2/wait_de_f1.py
```

`wait_de_f1.py` polls ~60s, on COMPLETE pulls to `lineage_v2/out_stageDE_f1`, **never** competition-submits.  
Prior watcher run **timed out** after DNS failures to `api.kaggle.com` while kernel was still RUNNING — **do not trust timeout = done**. Re-check status immediately.

### When DE reaches COMPLETE

1. Confirm pull of export (`best_exact.pt`, `last.pt`, `train_meta.json`, logs)
2. Health report:
   - final/best step vs 32913
   - best_loss vs 0.521 (improved? flat? NaN?)
   - samples/s, dual VRAM, early exit reason
3. If healthy: version/publish DE weights as Kaggle dataset for primary infer
4. If ERROR: read log, fix, re-push DE on trainF (still no LB submit)
5. Update `lineage_v2/HANDOFF_TRAINE_DE_STATUS.md` + commit/push to vibecode if useful

### Training complete ≠ submit time

Only after DE health + a **working real-weight infer** with honest expectation **>0.915** do you burn a primary submit.

---

## Watch: primary submissions & kernels

**Best banked public (primary):** ~**0.909** (edge-TTA clean path)  
**Recent hybrids:** hyb-a **0.896**, hyb-b **0.908** (public 350ep, not CELLECT DE)  
**Public reference to beat:** Yusuke-style **0.914**  
**Gate:** do not submit clones expected ≤0.915

| Kernel | Status | Action |
|--------|--------|--------|
| `khalid000000/bh-yusuke-pure914` | COMPLETE | Reference only — **do not submit** under gate |
| `khalid000000/bh-hyb-a-yusuke350` | COMPLETE | Scored 0.896 — dead path for >0.915 |
| `khalid000000/bh-hyb-b-ult1yusuke` | COMPLETE | Scored 0.908 — dead path for >0.915 |
| Stage C CELLECT infer | ERROR | Arch mismatch vs Yusuke UNet — needs real CELLECT infer rewrite |

Poll:

```powershell
$env:KAGGLE_CONFIG_DIR = "$env:USERPROFILE\.kaggle"
python -m kaggle competitions submissions -c biohub-cell-tracking-during-development
python -m kaggle kernels list --mine -v
python -m kaggle kernels status khalid000000/bh-yusuke-pure914
```

If any **new** primary kernel is RUNNING, watch it to COMPLETE, read score/logs, **do not** auto-submit weak outputs.

---

## Why we are not at 0.914+ yet (important)

- Yusuke **0.914** = harmonic detection + bidirectional edge (~0.20) + DeepCenter OFF + strong public UNet edge weights (~350ep lineage), ~229k rows style pipeline  
- Our **CELLECT** Stage C/DE is a **different architecture** (center/offset/flow/mitosis/embed + assoc/fork)  
- Hybrids that stuffed **public 350ep** into Yusuke PP got **0.896/0.908** — not an improvement  
- Forcing C `best_exact.pt` into Yusuke UNet load = fail / nonsense  
- **Winning path:** finish DE → real CELLECT (or track-fusion) infer + best PP ideas from 0.914 **without** wrong weight load

Score: `≈ adj_edge + 0.1·divJ`. Care about divisions, not only edges.

---

## Plan after DE is healthy

### A. Package weights

- Export DE `best_exact.pt` / `last.pt` + `train_meta.json`
- Publish dataset on primary (or attach from worker if policy allows — private datasets do **not** cross accounts; you must re-upload to primary)

### B. Fix inference (blocking)

Build a notebook that:

1. Loads CELLECT-lite + assoc + fork from our package (`lineage_v2/`)
2. Handles `DataParallel` `module.` prefixes
3. Runs full test stems, internet OFF, T4×2, ≤11.5h
4. Writes valid competition submission
5. Optionally fuses Yusuke det/tracks in **graph space** if time allows
6. Adopts arch-agnostic PP from 0.914: retention, bidir costs, division rules, harmonic peak logic **adapted** to our heads

Broken reference to fix/replace: `lineage_v2/kernel_stagec_infer/`

### C. Gate then submit (primary only)

- Offline / smoke checks green  
- Honest expected public **> 0.915**  
- Push kernel on `khalid000000`  
- Watch to COMPLETE  
- Submit **one** high-conviction version (kernel output submit for code comp)  
- Stop if score ≤0.915; analyze before burning slot 2/3  

### D. Optional more train

Only if DE weak or infer shows underfit:

- Real-GEFF C (if available)  
- Second DE seed  
- Still on train workers, no LB submit  

---

## Repo contents you need (already on GitHub `vibecode`)

```text
lineage_v2/
  CONTINUATION_PLAN.md          # full plan
  HANDOFF_TRAINE_DE_STATUS.md   # DE live card
  NO_AUTO_SUBMIT.flag
  wait_de_f1.py
  lineage_v2/                   # package
  dataset_src_trainF/           # train_stage_de.py + package
  kernel_realDE_f1/             # live DE notebook + metadata
  kernel_realC_e1/
  kernel_stageC_c1/ train_stage_c.py
  kernel_train11h/              # stage A/B/C/DE scripts
  kernel_yusuke_pure914/
  kernel_hybrid_submit/
  chain_*.py, build_hybrid_notebooks.py, PLAN_AFTER_A_D2.md
```

**Not in git:** `*.pt` weights, `out_*` downloads, warm dumps — pull from Kaggle.

Clone/update:

```powershell
cd C:\Users\Khalid\Desktop\New_folder\vibecode
git pull
# or: gh repo clone KHHH2312/vibecode
```

---

## Immediate checklist (do this first)

- [ ] `kernels status jiiiiiiiim/bh-lineage-realde-f1` under trainF  
- [ ] If RUNNING: start/re-arm `wait_de_f1.py` (background)  
- [ ] If COMPLETE: pull output + health report (no submit)  
- [ ] List primary submissions + any RUNNING primary kernels  
- [ ] Confirm pure914 still unsubmitted under gate  
- [ ] Report status to user in one tight table (DE / primary / next action)  
- [ ] Only after DE healthy: start real CELLECT ultimate infer work  

---

## Success definition

| Milestone | Done when |
|-----------|-----------|
| Training finished | DE COMPLETE + healthy export past Stage C |
| Ready to submit | Working real-weight infer + expected **>0.915** |
| Campaign win | Public **>0.915** (beat Yusuke 0.914 / our 0.909) without burning slots on junk |

**Do not** “finish” by submitting pure Yusuke or weak hybrids. **Do** watch DE, finish train, then build the real path.
