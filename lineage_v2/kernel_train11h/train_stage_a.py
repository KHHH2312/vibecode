"""Stage-A physical-detector train — single Kaggle session ≤11h (under ~12h hard cap).

Account: train worker (khalidmokarram). Does NOT submit to LB.
Budget: BIOHUB_TIME_BUDGET_H default 11.0 so session ends cleanly for export.
Weekly: ~2–3 such sessions fit in 30h GPU/week.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---- locate package (dataset may be unzipped dir or a .zip under /kaggle/input) ----
import zipfile
import shutil

PKG_ROOT: Path | None = None


def _try_set(root: Path) -> bool:
    """Accept nested package OR flattened dataset (constants.py at dataset root)."""
    global PKG_ROOT
    if (root / "lineage_v2" / "constants.py").exists():
        sys.path.insert(0, str(root))
        PKG_ROOT = root
        return True
    if root.name == "lineage_v2" and (root / "constants.py").exists():
        sys.path.insert(0, str(root.parent))
        PKG_ROOT = root.parent
        return True
    # Flattened Kaggle dataset: .../lineage-v2-src/constants.py + models/
    if (root / "constants.py").exists() and (root / "models").is_dir():
        dest = Path("/kaggle/working/lineage_v2")
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(root, dest, dirs_exist_ok=True)
        # drop non-package junk if any
        sys.path.insert(0, "/kaggle/working")
        PKG_ROOT = Path("/kaggle/working")
        return True
    return False


# Deep search: Kaggle mounts as /kaggle/input/datasets/<user>/<slug>/...
candidates: list[Path] = []
for base in [Path("/kaggle/working"), Path("/kaggle/input")]:
    if not base.exists():
        continue
    candidates.append(base)
    for p in base.rglob("constants.py"):
        parent = p.parent
        # package dir has models/ (CELLECT) or is named lineage_v2
        if (parent / "models").is_dir() or parent.name == "lineage_v2":
            candidates.append(parent)
            if parent.parent not in candidates:
                candidates.append(parent.parent)

for c in candidates:
    if _try_set(c):
        break
    # zip next to constants
    if c.is_dir():
        for zpath in c.glob("*.zip"):
            dest = Path("/kaggle/working/_lineage_v2_extract")
            if dest.exists():
                shutil.rmtree(dest, ignore_errors=True)
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(dest)
            if _try_set(dest):
                break
            for sub in dest.rglob("constants.py"):
                if _try_set(sub.parent) or _try_set(sub.parent.parent):
                    break
            if PKG_ROOT is not None:
                break
    if PKG_ROOT is not None:
        break

if PKG_ROOT is None:
    listing = []
    inp = Path("/kaggle/input")
    if inp.exists():
        for p in inp.rglob("constants.py"):
            listing.append(str(p))
    raise SystemExit(f"lineage_v2 package not found. constants.py hits={listing[:20]}")

from lineage_v2.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint  # noqa: E402
from lineage_v2.constants import FORBIDDEN_TEST_STEMS, SIGMA_UM, SPACING_ZYX_UM  # noqa: E402
from lineage_v2.denylist import ForbiddenStemError, assert_stem_allowed  # noqa: E402
from lineage_v2.eval.contract import verify_metric_pin  # noqa: E402
from lineage_v2.models.cellect_lite import CellectLite  # noqa: E402
from lineage_v2.targets import make_targets  # noqa: E402

# ---- session limits (Kaggle ~12h max; stay under) ----
TIME_BUDGET_H = float(os.environ.get("BIOHUB_TIME_BUDGET_H", "11.0"))
# Absolute step target OR steps to run this session after resume
MAX_STEPS = int(os.environ.get("BIOHUB_MAX_STEPS", "80000"))  # session2 default: 40k→80k
SESSION_STEPS = int(os.environ.get("BIOHUB_SESSION_STEPS", "40000"))  # another 40k after resume
SAVE_EVERY = int(os.environ.get("BIOHUB_SAVE_EVERY", "500"))
LOG_EVERY = int(os.environ.get("BIOHUB_LOG_EVERY", "50"))
PATCH_Z, PATCH_Y, PATCH_X = 32, 160, 160
LR = float(os.environ.get("BIOHUB_LR", "1e-4"))  # slightly lower for continued Stage A
SEED = int(os.environ.get("BIOHUB_SEED", "9401"))
SESSION_ID = os.environ.get("BIOHUB_SESSION_ID", "s2")

OUT = Path("/kaggle/working/export")
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def remaining_h() -> float:
    return TIME_BUDGET_H - (time.time() - T0) / 3600.0


def log(msg: str) -> None:
    print(f"[{(time.time()-T0)/60:.1f}m] {msg}", flush=True)


log(f"DEVICE={DEVICE} TIME_BUDGET_H={TIME_BUDGET_H} MAX_STEPS={MAX_STEPS}")
log(f"FORBIDDEN={sorted(FORBIDDEN_TEST_STEMS)}")
log(f"metric_pin={verify_metric_pin()}")
log(f"PKG_ROOT={PKG_ROOT}")

# Competition train root (optional)
TRAIN_CANDS = [
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
    Path("/kaggle/input/biohub-cell-tracking-during-development/train"),
]
TRAIN_DIR = next((p for p in TRAIN_CANDS if p.is_dir()), None)
log(f"TRAIN_DIR={TRAIN_DIR}")


def list_permitted_stems() -> list[str]:
    if TRAIN_DIR is None:
        return []
    stems = []
    for p in sorted(TRAIN_DIR.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if name in FORBIDDEN_TEST_STEMS:
            continue
        stems.append(name)
    return stems


STEMS = list_permitted_stems()
log(f"permitted_stems={len(STEMS)}")


def synthetic_batch(device: torch.device):
    """Physical-scale multi-center patches when real GEFF loader not ready."""
    Z, Y, X = PATCH_Z, PATCH_Y, PATCH_X
    n = random.randint(3, 12)
    centers = []
    parents = []
    for i in range(n):
        # keep margins for 3σ ≈ few µm
        z = random.uniform(4, Z - 5)
        y = random.uniform(16, Y - 17)
        x = random.uniform(16, X - 17)
        centers.append([z, y, x])
        # ~30% have parent (continuation/div style)
        if i > 0 and random.random() < 0.35:
            parents.append(random.randint(0, i - 1))
        else:
            parents.append(-1)
    heat, offset, flow, ov, fv = make_targets(
        (Z, Y, X), centers, parents, device=torch.device("cpu")
    )
    # image: noise + bright peaks
    img = torch.randn(1, 1, Z, Y, X) * 0.05
    for c in centers:
        zz, yy, xx = int(c[0]), int(c[1]), int(c[2])
        img[
            0,
            0,
            max(0, zz - 1) : min(Z, zz + 2),
            max(0, yy - 2) : min(Y, yy + 3),
            max(0, xx - 2) : min(X, xx + 3),
        ] += random.uniform(0.8, 1.5)
    return (
        img.to(device),
        heat.to(device),
        offset.to(device),
        flow.to(device),
        ov.to(device),
        fv.to(device),
    )


def detection_loss(pred, heat, offset_pred, offset, ov, flow_pred, flow, fv):
    # focal-ish BCE on heatmap
    p = torch.sigmoid(pred)
    eps = 1e-6
    pos = heat >= 0.5
    neg = ~pos
    # simple focal
    loss_pos = (-((1 - p) ** 2) * heat * torch.log(p + eps))[pos].mean() if pos.any() else pred.new_tensor(0.0)
    loss_neg = (-(p**2) * (1 - heat) * torch.log(1 - p + eps))[neg].mean() if neg.any() else pred.new_tensor(0.0)
    loss_hm = loss_pos + 0.1 * loss_neg
    if ov.any():
        loss_off = F.smooth_l1_loss(offset_pred[:, ov], offset[:, ov], beta=0.25)
    else:
        loss_off = pred.new_tensor(0.0)
    if fv.any():
        loss_flow = F.huber_loss(flow_pred[:, fv], flow[:, fv], delta=1.0)
    else:
        loss_flow = pred.new_tensor(0.0)
    return loss_hm + 1.0 * loss_off + 0.5 * loss_flow, {
        "hm": float(loss_hm),
        "off": float(loss_off),
        "flow": float(loss_flow),
    }


model = CellectLite(time_frames=1).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
start_step = 0
resumed_from = None

# resume: working export, then any last.pt under /kaggle/input (weights dataset)
resume_paths: list[Path] = [OUT / "last.pt"]
if Path("/kaggle/input").exists():
    resume_paths.extend(sorted(Path("/kaggle/input").rglob("last.pt")))
    resume_paths.extend(sorted(Path("/kaggle/input").rglob("best_exact.pt")))

for r in resume_paths:
    if not r.exists():
        continue
    try:
        st = load_checkpoint(
            r, model=model, optimizer=opt, scaler=scaler, strict_model=True
        )
        start_step = int(st.get("global_step", 0))
        resumed_from = str(r)
        log(f"resumed full ckpt step={start_step} from {r}")
        break
    except Exception as e:
        log(f"resume skip {r}: {e}")

# this session: continue for SESSION_STEPS more, capped by MAX_STEPS
end_step = min(MAX_STEPS, start_step + SESSION_STEPS)
log(f"SESSION={SESSION_ID} start_step={start_step} end_step={end_step} resumed_from={resumed_from}")

best_loss = float("inf")
step = start_step
model.train()

log("TRAIN_LOOP_START")
while step < end_step and remaining_h() > 0.25:
    img, heat, offset, flow, ov, fv = synthetic_batch(DEVICE)
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
        out = model(img)
        pred = out["center_logits"][0, 0]
        off_p = out["offset_vox"][0]
        fl_p = out["backward_flow_um"][0]
        if pred.shape != heat.shape:
            pred = F.interpolate(pred[None, None], size=heat.shape, mode="trilinear", align_corners=False)[0, 0]
            off_p = F.interpolate(off_p[None], size=heat.shape, mode="trilinear", align_corners=False)[0]
            fl_p = F.interpolate(fl_p[None], size=heat.shape, mode="trilinear", align_corners=False)[0]
        loss, parts = detection_loss(pred, heat, off_p, offset, ov, fl_p, flow, fv)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt)
    scaler.update()
    step += 1
    lv = float(loss)
    if lv < best_loss:
        best_loss = lv
    if step % LOG_EVERY == 0:
        vram = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0
        log(
            f"step={step} loss={lv:.4f} best={best_loss:.4f} "
            f"hm={parts['hm']:.4f} off={parts['off']:.4f} flow={parts['flow']:.4f} "
            f"rem_h={remaining_h():.2f} vram_gb={vram:.2f}"
        )
    if step % SAVE_EVERY == 0 or remaining_h() < 0.30:
        state = build_checkpoint(
            model=model,
            optimizer=opt,
            scaler=scaler,
            stage="A",
            global_step=step,
            best_exact=-best_loss,  # placeholder until exact metric wired
            config={
                "stage": "A",
                "session": SESSION_ID,
                "time_budget_h": TIME_BUDGET_H,
                "patch": [PATCH_Z, PATCH_Y, PATCH_X],
                "sigma_um": SIGMA_UM,
                "spacing": list(SPACING_ZYX_UM),
                "lr": LR,
                "seed": SEED,
                "data_mode": "synthetic_physical" if not STEMS else "synthetic_until_geff_loader",
                "n_stems_seen": len(STEMS),
                "resumed_from": resumed_from,
            },
            split_sha256="pending_real_manifest",
            code_sha="lineage_v2_stageA_s2",
            inference_policy_sha256="smoke",
        )
        save_checkpoint(state, OUT / "last.pt")
        save_checkpoint(state, OUT / "best_exact.pt")
        meta = {
            "step": step,
            "best_loss": best_loss,
            "elapsed_h": (time.time() - T0) / 3600,
            "remaining_h": remaining_h(),
            "device": str(DEVICE),
            "no_submit": True,
            "account_role": "primary",
            "session": SESSION_ID,
            "resumed_from": resumed_from,
        }
        (OUT / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        log(f"saved export step={step}")

# final save
state = build_checkpoint(
    model=model,
    optimizer=opt,
    scaler=scaler,
    stage="A",
    global_step=step,
    best_exact=-best_loss,
    config={
        "stage": "A",
        "session": SESSION_ID,
        "final": True,
        "steps": step,
        "resumed_from": resumed_from,
    },
    split_sha256="pending_real_manifest",
    code_sha="lineage_v2_stageA_s2",
    inference_policy_sha256="smoke",
)
save_checkpoint(state, OUT / "last.pt")
save_checkpoint(state, OUT / "best_exact.pt")
final = {
    "status": "COMPLETE",
    "steps": step,
    "start_step": start_step,
    "end_step_target": end_step,
    "best_loss": best_loss,
    "elapsed_h": (time.time() - T0) / 3600,
    "time_budget_h": TIME_BUDGET_H,
    "session": SESSION_ID,
    "resumed_from": resumed_from,
    "stopped_reason": "session_steps" if step >= end_step else "time_budget",
    "no_submit": True,
    "next": "version s2 weights; wire GEFF loader or continue Stage A / Stage B",
}
(OUT / "session_complete.json").write_text(json.dumps(final, indent=2) + "\n")
log(f"SESSION_DONE {final}")
