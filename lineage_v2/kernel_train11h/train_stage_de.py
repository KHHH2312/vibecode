"""Stage D+E combined — joint refine + simple fork head (plan D 20k + E 15k ≈ 35k).

Resume C weights. 11h cap. No submit.

Dual-GPU hack (measure real speedup on Kaggle T4 x2):
  - BIOHUB_DUAL_GPU=1 (default): if torch sees 2+ GPUs, wrap detector in DataParallel
  - BIOHUB_BATCH auto = 2 * n_gpu when dual (so each card gets work); else 1
  - Logs n_gpu, batch, ms/step, samples/s, vram0/vram1 for A/B vs Stage C single-GPU
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PKG_ROOT: Path | None = None


def _try_set(root: Path) -> bool:
    global PKG_ROOT
    if (root / "lineage_v2" / "constants.py").exists():
        sys.path.insert(0, str(root))
        PKG_ROOT = root
        return True
    if root.name == "lineage_v2" and (root / "constants.py").exists():
        sys.path.insert(0, str(root.parent))
        PKG_ROOT = root.parent
        return True
    if (root / "constants.py").exists() and (root / "models").is_dir():
        dest = Path("/kaggle/working/lineage_v2")
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(root, dest, dirs_exist_ok=True)
        sys.path.insert(0, "/kaggle/working")
        PKG_ROOT = Path("/kaggle/working")
        return True
    return False


for base in [Path("/kaggle/working"), Path("/kaggle/input")]:
    if not base.exists():
        continue
    if _try_set(base):
        break
    for p in base.rglob("constants.py"):
        if _try_set(p.parent) or _try_set(p.parent.parent):
            break
    if PKG_ROOT:
        break
    for zpath in list(base.rglob("*.zip"))[:5] if base.exists() else []:
        dest = Path("/kaggle/working/_lv2")
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        dest.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zpath, "r") as zf:
                zf.extractall(dest)
        except Exception:
            continue
        if _try_set(dest):
            break
        for p in dest.rglob("constants.py"):
            if _try_set(p.parent) or _try_set(p.parent.parent):
                break
        if PKG_ROOT:
            break
    if PKG_ROOT:
        break

if PKG_ROOT is None:
    raise SystemExit("lineage_v2 not found")

from lineage_v2.constants import SPACING_ZYX_UM  # noqa: E402
from lineage_v2.eval.contract import verify_metric_pin  # noqa: E402
from lineage_v2.models.assoc import ParentalAssocHead  # noqa: E402
from lineage_v2.models.cellect_lite import CellectLite  # noqa: E402
from lineage_v2.targets import make_targets  # noqa: E402

TIME_BUDGET_H = float(os.environ.get("BIOHUB_TIME_BUDGET_H", "11.0"))
SESSION_STEPS = int(os.environ.get("BIOHUB_SESSION_STEPS", "35000"))
SAVE_EVERY = 500
LOG_EVERY = 50
LR = float(os.environ.get("BIOHUB_LR", "3e-5"))
SEED = 9401
SESSION_ID = os.environ.get("BIOHUB_SESSION_ID", "de1")
PATCH_Z, PATCH_Y, PATCH_X = 32, 160, 160
SPACING = torch.tensor(list(SPACING_ZYX_UM), dtype=torch.float32)

OUT = Path("/kaggle/working/export")
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count() if DEVICE.type == "cuda" else 0
# Dual-GPU abandoned; single GPU + batch=4 (same as Stage C)
USE_DUAL = os.environ.get("BIOHUB_DUAL_GPU", "0") == "1" and N_GPU >= 2
BATCH = max(1, int(os.environ.get("BIOHUB_BATCH", "4")))
if USE_DUAL and BATCH < N_GPU:
    BATCH = N_GPU
if not USE_DUAL and DEVICE.type == "cuda":
    torch.cuda.set_device(0)
    DEVICE = torch.device("cuda:0")
    N_GPU = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)


def remaining_h() -> float:
    return TIME_BUDGET_H - (time.time() - T0) / 3600.0


def log(msg: str) -> None:
    print(f"[{(time.time()-T0)/60:.1f}m] {msg}", flush=True)


def vram_gb(idx: int = 0) -> float:
    if DEVICE.type != "cuda" or idx >= N_GPU:
        return 0.0
    try:
        return torch.cuda.max_memory_allocated(idx) / 1e9
    except Exception:
        return 0.0


class ForkHead(nn.Module):
    def __init__(self, emb_dim: int = 64, hidden: int = 128):
        super().__init__()
        in_dim = emb_dim * 3 + 4
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, ep, e1, e2, geom: torch.Tensor) -> torch.Tensor:
        x = torch.cat([ep, e1 + e2, (e1 - e2).abs(), geom], dim=-1)
        return self.mlp(x).squeeze(-1)


def unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m


def det_state_dict(m: nn.Module) -> dict:
    return unwrap(m).state_dict()


log(f"STAGE D+E session={SESSION_ID} DEVICE={DEVICE} n_gpu={N_GPU} dual={USE_DUAL} batch={BATCH}")
log(f"metric={verify_metric_pin()}")
if DEVICE.type == "cuda":
    for i in range(N_GPU):
        log(f"cuda:{i} name={torch.cuda.get_device_name(i)}")

detector = CellectLite(time_frames=1).to(DEVICE)
assoc = ParentalAssocHead(emb_dim=64).to(DEVICE)
fork = ForkHead().to(DEVICE)
resumed_from = None
for r in list(Path("/kaggle/input").rglob("last.pt")) if Path("/kaggle/input").exists() else []:
    try:
        st = torch.load(r, map_location="cpu", weights_only=False)
        if not isinstance(st, dict) or "model" not in st:
            continue
        detector.load_state_dict(st["model"], strict=False)
        if "assoc" in st:
            assoc.load_state_dict(st["assoc"], strict=False)
        if "fork" in st:
            fork.load_state_dict(st["fork"], strict=False)
        resumed_from = str(r)
        log(f"loaded {r}")
        break
    except Exception as e:
        log(f"skip {e}")

if USE_DUAL:
    detector = nn.DataParallel(detector)
    log(f"DataParallel ON devices={list(range(N_GPU))} batch={BATCH} (measure dual-GPU speedup)")

opt = torch.optim.AdamW(
    list(detector.parameters()) + list(assoc.parameters()) + list(fork.parameters()),
    lr=LR,
    weight_decay=1e-4,
)
# AMP off — NaN failure mode with GradScaler on this stack
USE_AMP = os.environ.get("BIOHUB_USE_AMP", "0") == "1"
scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))
log(f"AMP={USE_AMP}")


def sample_embed_at(emb_map, centers_zyx):
    # emb_map: (C,D,H,W) for one sample
    if emb_map.ndim == 4:
        emb_map = emb_map.unsqueeze(0)
    _, C, D, H, W = emb_map.shape
    z = centers_zyx[:, 0] / max(D - 1, 1) * 2 - 1
    y = centers_zyx[:, 1] / max(H - 1, 1) * 2 - 1
    x = centers_zyx[:, 2] / max(W - 1, 1) * 2 - 1
    out = []
    for i in range(centers_zyx.shape[0]):
        g = torch.stack([x[i : i + 1], y[i : i + 1], z[i : i + 1]], dim=-1).view(1, 1, 1, 1, 3)
        out.append(F.grid_sample(emb_map, g, align_corners=True, mode="bilinear").view(C))
    return torch.stack(out, 0)


def make_one_scene():
    """One synthetic fork scene (CPU lists + heat); tensors built after batching."""
    n = random.randint(8, 16)
    Z, Y, X = PATCH_Z, PATCH_Y, PATCH_X
    centers, parents = [], []
    fork_triples = []
    for i in range(n):
        centers.append(
            [
                random.uniform(4, Z - 5),
                random.uniform(16, Y - 17),
                random.uniform(16, X - 17),
            ]
        )
        parents.append(-1)
    if n >= 3:
        p, c1, c2 = 0, 1, 2
        parents[c1] = p
        parents[c2] = p
        centers[c1] = [
            centers[p][0] + 0.3,
            centers[p][1] + 3.0,
            centers[p][2] + 2.0,
        ]
        centers[c2] = [
            centers[p][0] - 0.2,
            centers[p][1] - 2.5,
            centers[p][2] + 2.5,
        ]
        fork_triples.append((p, c1, c2))
    for i in range(3, n):
        if random.random() < 0.4:
            parents[i] = random.randint(0, i - 1)

    img = torch.randn(1, Z, Y, X) * 0.05
    for c in centers:
        zz, yy, xx = int(c[0]), int(c[1]), int(c[2])
        img[
            0, max(0, zz - 1) : zz + 2, max(0, yy - 2) : yy + 3, max(0, xx - 2) : xx + 3
        ] += 1.0
    heat, _, _, _, _ = make_targets((Z, Y, X), centers, parents)
    return {
        "img": img,  # (1,Z,Y,X)
        "heat": heat,
        "centers": centers,
        "parents": parents,
        "fork_triples": fork_triples,
    }


def scene_losses(out_b: dict, scenes: list, b_idx: int):
    """Assoc + fork + det losses for one batch index (outputs already on DEVICE)."""
    emb_map = out_b["embedding_map"][b_idx]
    flow_vol = out_b["backward_flow_um"][b_idx]
    pred = out_b["center_logits"][b_idx, 0]
    sc = scenes[b_idx]
    centers_t = torch.tensor(sc["centers"], dtype=torch.float32, device=DEVICE)
    parents_t = torch.tensor(sc["parents"], dtype=torch.long, device=DEVICE)
    heat = sc["heat"].to(DEVICE)
    fork_triples = sc["fork_triples"]

    _, ez, ey, ex = emb_map.shape
    scale = torch.tensor(
        [ez / PATCH_Z, ey / PATCH_Y, ex / PATCH_X], device=DEVICE, dtype=centers_t.dtype
    )
    embs = sample_embed_at(emb_map, centers_t * scale)
    flows = []
    for c in centers_t:
        az = int(min(max(c[0].item(), 0), PATCH_Z - 1))
        ay = int(min(max(c[1].item(), 0), PATCH_Y - 1))
        ax = int(min(max(c[2].item(), 0), PATCH_X - 1))
        flows.append(flow_vol[:, az, ay, ax])
    flows = torch.stack(flows, 0)

    spacing = SPACING.to(DEVICE)
    um = centers_t * spacing
    loss_a = centers_t.new_tensor(0.0)
    na = 0
    for i in range(len(sc["centers"])):
        d = (um - um[i : i + 1]).norm(dim=-1)
        d[i] = 1e9
        k = min(8, len(sc["centers"]) - 1)
        if k < 1:
            continue
        _, idx = torch.topk(d, k=k, largest=False)
        tp = int(parents_t[i].item())
        if tp >= 0 and tp not in idx.tolist():
            idx = torch.cat([idx[:-1], parents_t.new_tensor([tp])])
            k = idx.numel()
        el = assoc.edge_logit(
            embs[i : i + 1].expand(k, -1),
            embs[idx],
            um[i : i + 1].expand(k, -1),
            um[idx],
            flows[i : i + 1].expand(k, -1),
        )
        ql = assoc.quiet_logit(embs[i : i + 1], flows[i : i + 1])
        logits = torch.cat([ql, el], 0)
        if tp < 0:
            tgt = 0
        else:
            pos = (idx == tp).nonzero(as_tuple=False)
            tgt = int(pos[0, 0].item()) + 1 if pos.numel() else 0
        loss_a = loss_a + F.cross_entropy(
            logits.unsqueeze(0), logits.new_tensor([tgt], dtype=torch.long)
        )
        na += 1
    loss_a = loss_a / max(na, 1)

    loss_f = centers_t.new_tensor(0.0)
    if fork_triples:
        p, c1, c2 = fork_triples[0]
        sister = (um[c1] - um[c2]).norm()
        mid = 0.5 * (um[c1] + um[c2])
        mid_res = (mid - um[p]).norm()
        geom = torch.stack(
            [sister, mid_res, um.new_tensor(float(len(sc["centers"]))), um.new_tensor(1.0)]
        )
        pos_logit = fork(embs[p : p + 1], embs[c1 : c1 + 1], embs[c2 : c2 + 1], geom.unsqueeze(0))
        neg_logits = []
        for _ in range(8):
            a, b, c = random.sample(range(len(sc["centers"])), 3)
            sister = (um[b] - um[c]).norm()
            mid = 0.5 * (um[b] + um[c])
            mid_res = (mid - um[a]).norm()
            geom_n = torch.stack(
                [sister, mid_res, um.new_tensor(float(len(sc["centers"]))), um.new_tensor(0.0)]
            )
            neg_logits.append(
                fork(embs[a : a + 1], embs[b : b + 1], embs[c : c + 1], geom_n.unsqueeze(0))
            )
        neg_stack = torch.cat(neg_logits, 0)
        loss_f = F.binary_cross_entropy_with_logits(
            pos_logit, pos_logit.new_ones(1)
        ) + F.binary_cross_entropy_with_logits(neg_stack, neg_stack.new_zeros(neg_stack.shape))

    if pred.shape != heat.shape:
        pred = F.interpolate(
            pred[None, None], size=heat.shape, mode="trilinear", align_corners=False
        )[0, 0]
    loss_d = F.binary_cross_entropy_with_logits(pred, heat.clamp(0, 1))
    loss = loss_a + 0.5 * loss_f + 0.15 * loss_d
    return loss, loss_a, loss_f, loss_d


step = 0
best_loss = float("inf")
nan_streak = 0
end_step = SESSION_STEPS
detector.train()
assoc.train()
fork.train()
log(f"TRAIN D+E end={end_step} resumed={resumed_from} dual={USE_DUAL} batch={BATCH}")
# timing for speedup report (skip first LOG_EVERY for warmup)
t_window0 = time.time()
steps_window0 = 0

while step < end_step and remaining_h() > 0.25:
    scenes = [make_one_scene() for _ in range(BATCH)]
    # img batch: (B, 1, Z, Y, X)
    img = torch.stack([s["img"] for s in scenes], dim=0).to(DEVICE)

    opt.zero_grad(set_to_none=True)
    t_step0 = time.time()
    with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
        out = detector(img)
        losses = []
        la_s, lf_s, ld_s = [], [], []
        for bi in range(BATCH):
            loss_i, la, lf, ld = scene_losses(out, scenes, bi)
            losses.append(loss_i)
            la_s.append(la)
            lf_s.append(lf)
            ld_s.append(ld)
        loss = torch.stack(losses).mean()
        loss_a = torch.stack(la_s).mean()
        loss_f = torch.stack(lf_s).mean()
        loss_d = torch.stack(ld_s).mean()

    if not torch.isfinite(loss):
        nan_streak += 1
        if nan_streak <= 3 or nan_streak % 10 == 0:
            log(f"SKIP nonfinite loss nan_streak={nan_streak}")
        if nan_streak >= 50:
            log("ABORT: 50 consecutive nonfinite losses")
            break
        continue
    nan_streak = 0

    if USE_AMP:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            list(detector.parameters()) + list(assoc.parameters()) + list(fork.parameters()), 1.0
        )
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(detector.parameters()) + list(assoc.parameters()) + list(fork.parameters()), 1.0
        )
        opt.step()

    step += 1
    steps_window0 += 1
    step_ms = (time.time() - t_step0) * 1000.0
    lv = float(loss.detach())
    if lv < best_loss:
        best_loss = lv

    if step % LOG_EVERY == 0:
        elapsed_w = max(time.time() - t_window0, 1e-6)
        steps_s = steps_window0 / elapsed_w
        samples_s = steps_s * BATCH
        v0 = vram_gb(0)
        v1 = vram_gb(1) if N_GPU > 1 else 0.0
        log(
            f"step={step} loss={lv:.4f} best={best_loss:.4f} "
            f"a={float(loss_a.detach()):.4f} f={float(loss_f.detach()):.4f} d={float(loss_d.detach()):.4f} "
            f"rem={remaining_h():.2f} batch={BATCH} n_gpu={N_GPU} dual={int(USE_DUAL)} "
            f"ms/step={step_ms:.0f} steps/s={steps_s:.3f} samples/s={samples_s:.3f} "
            f"vram0={v0:.2f} vram1={v1:.2f}"
        )
        t_window0 = time.time()
        steps_window0 = 0

    if step % SAVE_EVERY == 0 or remaining_h() < 0.3:
        joint = {
            "schema_version": 2,
            "stage": "DE",
            "session": SESSION_ID,
            "global_step": step,
            "model": det_state_dict(detector),
            "assoc": assoc.state_dict(),
            "fork": fork.state_dict(),
            "best_exact": -best_loss if math.isfinite(best_loss) else 0.0,
            "config": {
                "stage": "DE",
                "resumed_from": resumed_from,
                "use_amp": USE_AMP,
                "dual_gpu": USE_DUAL,
                "n_gpu": N_GPU,
                "batch": BATCH,
            },
            "code_sha": "lineage_v2_stageDE_de1_dualgpu_amp_off",
        }
        torch.save(joint, OUT / "last.pt")
        if math.isfinite(best_loss):
            torch.save(joint, OUT / "best_exact.pt")
        log(f"saved {step}")

joint = {
    "schema_version": 2,
    "stage": "DE",
    "session": SESSION_ID,
    "global_step": step,
    "model": det_state_dict(detector),
    "assoc": assoc.state_dict(),
    "fork": fork.state_dict(),
    "best_exact": -best_loss if math.isfinite(best_loss) else 0.0,
    "config": {
        "stage": "DE",
        "final": True,
        "resumed_from": resumed_from,
        "dual_gpu": USE_DUAL,
        "n_gpu": N_GPU,
        "batch": BATCH,
    },
    "code_sha": "lineage_v2_stageDE_de1_dualgpu_amp_off",
}
torch.save(joint, OUT / "last.pt")
torch.save(joint, OUT / "best_exact.pt")
final = {
    "status": "COMPLETE",
    "stage": "DE",
    "session": SESSION_ID,
    "steps": step,
    "best_loss": best_loss,
    "elapsed_h": (time.time() - T0) / 3600,
    "resumed_from": resumed_from,
    "no_submit": True,
    "dual_gpu": USE_DUAL,
    "n_gpu": N_GPU,
    "batch": BATCH,
    "next": "READY_FOR_PRIMARY — compare samples/s vs Stage C single-GPU baseline",
}
(OUT / "session_complete.json").write_text(json.dumps(final, indent=2) + "\n")
log(f"DONE {final}")
