"""Stage C — association on predicted/noisy nodes (plan 50k). Resume B weights. No submit.

Dual-GPU (default ON): DataParallel detector + batch=2*n_gpu so both T4s work.
Logs samples/s + vram0/vram1. AMP off.
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


candidates: list[Path] = []
for base in [Path("/kaggle/working"), Path("/kaggle/input")]:
    if not base.exists():
        continue
    candidates.append(base)
    for p in base.rglob("constants.py"):
        parent = p.parent
        if (parent / "models").is_dir() or parent.name == "lineage_v2":
            candidates.append(parent)
            candidates.append(parent.parent)

for c in candidates:
    if _try_set(c):
        break
    if c.is_dir():
        for zpath in list(c.glob("*.zip"))[:3]:
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
    raise SystemExit("lineage_v2 package not found")

from lineage_v2.constants import FORBIDDEN_TEST_STEMS, SPACING_ZYX_UM  # noqa: E402
from lineage_v2.eval.contract import verify_metric_pin  # noqa: E402
from lineage_v2.models.assoc import ParentalAssocHead  # noqa: E402
from lineage_v2.models.cellect_lite import CellectLite  # noqa: E402
from lineage_v2.targets import make_targets  # noqa: E402

# Kaggle hard cap ~12h; aim wall finish ~11h45 → budget 11.85h, stop with ~7min for final save
TIME_BUDGET_H = float(os.environ.get("BIOHUB_TIME_BUDGET_H", "11.85"))
# High step cap so TIME budget is the real limiter (maximize samples in session)
SESSION_STEPS = int(os.environ.get("BIOHUB_SESSION_STEPS", "200000"))
SAVE_EVERY = int(os.environ.get("BIOHUB_SAVE_EVERY", "500"))
LOG_EVERY = int(os.environ.get("BIOHUB_LOG_EVERY", "50"))
# Leave this much budget for final checkpoint write (hours)
STOP_REMAIN_H = float(os.environ.get("BIOHUB_STOP_REMAIN_H", "0.12"))
LR = float(os.environ.get("BIOHUB_LR", "1e-4"))
SEED = int(os.environ.get("BIOHUB_SEED", "9401"))
SESSION_ID = os.environ.get("BIOHUB_SESSION_ID", "c1")
PATCH_Z, PATCH_Y, PATCH_X = 32, 160, 160
K_CAND = 8
SPACING = torch.tensor(list(SPACING_ZYX_UM), dtype=torch.float32)

OUT = Path("/kaggle/working/export")
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_GPU = torch.cuda.device_count() if DEVICE.type == "cuda" else 0
N_GPU_VISIBLE = N_GPU
# Dual ON by default — max samples/hour for score; batch=4 measured ~2.25 samples/s
USE_DUAL = os.environ.get("BIOHUB_DUAL_GPU", "1") == "1" and N_GPU >= 2
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


def unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m


def det_state_dict(m: nn.Module) -> dict:
    return unwrap(m).state_dict()


log(
    f"STAGE C session={SESSION_ID} DEVICE={DEVICE} n_gpu_use={N_GPU} "
    f"cuda_visible={N_GPU_VISIBLE} dual={USE_DUAL} batch={BATCH} "
    f"budget={TIME_BUDGET_H}h stop_remain={STOP_REMAIN_H}h steps={SESSION_STEPS} "
    f"(time-gated for ~11h45 finish under 12h cap)"
)
log(f"metric_pin={verify_metric_pin()}")
log(f"FORBIDDEN={sorted(FORBIDDEN_TEST_STEMS)}")
if DEVICE.type == "cuda":
    log(f"cuda:0 name={torch.cuda.get_device_name(0)} (single-GPU mode, batch={BATCH})")


def sample_embed_at(emb_map: torch.Tensor, centers_zyx: torch.Tensor) -> torch.Tensor:
    if emb_map.ndim == 4:
        emb_map = emb_map.unsqueeze(0)
    _, C, D, H, W = emb_map.shape
    z = centers_zyx[:, 0] / max(D - 1, 1) * 2 - 1
    y = centers_zyx[:, 1] / max(H - 1, 1) * 2 - 1
    x = centers_zyx[:, 2] / max(W - 1, 1) * 2 - 1
    out = []
    for i in range(centers_zyx.shape[0]):
        g = torch.stack([x[i : i + 1], y[i : i + 1], z[i : i + 1]], dim=-1).view(1, 1, 1, 1, 3)
        v = F.grid_sample(emb_map, g, align_corners=True, mode="bilinear")
        out.append(v.view(C))
    return torch.stack(out, dim=0)


def make_noisy_scene():
    """One synthetic scene on CPU (tensors moved after batching)."""
    n_gt = random.randint(6, 14)
    Z, Y, X = PATCH_Z, PATCH_Y, PATCH_X
    centers, parents = [], []
    for i in range(n_gt):
        centers.append(
            [
                random.uniform(4, Z - 5),
                random.uniform(16, Y - 17),
                random.uniform(16, X - 17),
            ]
        )
        parents.append(random.randint(0, i - 1) if i > 0 and random.random() < 0.55 else -1)
    centers = [
        [
            min(Z - 2, max(1, c[0] + random.uniform(-0.8, 0.8))),
            min(Y - 2, max(1, c[1] + random.uniform(-1.5, 1.5))),
            min(X - 2, max(1, c[2] + random.uniform(-1.5, 1.5))),
        ]
        for c in centers
    ]
    keep = [i for i in range(n_gt) if random.random() > 0.12]
    if len(keep) < 3:
        keep = list(range(min(3, n_gt)))
    id_map = {old: new for new, old in enumerate(keep)}
    centers = [centers[i] for i in keep]
    parents = [id_map[parents[i]] if parents[i] in id_map else -1 for i in keep]
    n_sup = len(centers)
    for _ in range(random.randint(1, 4)):
        centers.append(
            [
                random.uniform(4, Z - 5),
                random.uniform(16, Y - 17),
                random.uniform(16, X - 17),
            ]
        )
        parents.append(-1)
    if n_sup > 0 and random.random() < 0.4:
        j = random.randint(0, n_sup - 1)
        c = centers[j]
        centers.append([c[0] + 0.5, c[1] + 1.0, c[2] - 0.8])
        parents.append(-1)
    img = torch.randn(1, Z, Y, X) * 0.05
    for c in centers:
        zz, yy, xx = int(c[0]), int(c[1]), int(c[2])
        img[
            0, max(0, zz - 1) : zz + 2, max(0, yy - 2) : yy + 3, max(0, xx - 2) : xx + 3
        ] += random.uniform(0.7, 1.4)
    heat, _, _, _, _ = make_targets((Z, Y, X), centers[:n_sup], parents[:n_sup])
    return {
        "img": img,
        "centers": centers,
        "parents": parents,
        "n_sup": n_sup,
        "heat": heat,
    }


def assoc_loss(embs, centers_zyx, parents, flows_at, n_sup, div_w: float):
    n = centers_zyx.shape[0]
    spacing = SPACING.to(centers_zyx.device)
    centers_um = centers_zyx * spacing
    total = centers_zyx.new_tensor(0.0)
    n_terms = 0
    for i in range(n):
        true_p = int(parents[i].item())
        d = (centers_um - centers_um[i : i + 1]).norm(dim=-1)
        d[i] = 1e9
        k = min(K_CAND, n - 1)
        if k < 1:
            continue
        _, idx = torch.topk(d, k=k, largest=False)
        if true_p >= 0 and true_p not in idx.tolist():
            idx = torch.cat([idx[:-1], parents.new_tensor([true_p])])
            k = idx.numel()
        emb_c = embs[i : i + 1].expand(k, -1)
        emb_p = embs[idx]
        c_um = centers_um[i : i + 1].expand(k, -1)
        p_um = centers_um[idx]
        flow = flows_at[i : i + 1].expand(k, -1)
        edge_logits = assoc.edge_logit(emb_c, emb_p, c_um, p_um, flow)
        q_logit = assoc.quiet_logit(embs[i : i + 1], flows_at[i : i + 1])
        logits = torch.cat([q_logit, edge_logits], dim=0)
        if true_p < 0:
            target = 0
            w = 1.2 if i >= n_sup else 1.0
        else:
            pos = (idx == true_p).nonzero(as_tuple=False)
            if pos.numel() == 0:
                target = 0
                w = 1.0
            else:
                target = int(pos[0, 0].item()) + 1
                w = div_w if random.random() < 0.25 else 1.0
        if true_p >= 0 and logits.numel() > 2:
            wrong = edge_logits.clone()
            for j, ind in enumerate(idx.tolist()):
                if ind == true_p:
                    wrong[j] = -1e4
            margin = F.relu(wrong.max() - edge_logits[max(0, target - 1)] + 0.2)
        else:
            margin = logits.new_tensor(0.0)
        total = total + w * F.cross_entropy(
            logits.unsqueeze(0), logits.new_tensor([target], dtype=torch.long)
        )
        total = total + 0.1 * margin
        n_terms += 1
    return total / max(n_terms, 1)


def scene_losses(out_b: dict, scenes: list, b_idx: int, div_w: float):
    emb_map = out_b["embedding_map"][b_idx]
    flow_vol = out_b["backward_flow_um"][b_idx]
    pred = out_b["center_logits"][b_idx, 0]
    sc = scenes[b_idx]
    centers = torch.tensor(sc["centers"], dtype=torch.float32, device=DEVICE)
    parents = torch.tensor(sc["parents"], dtype=torch.long, device=DEVICE)
    n_sup = sc["n_sup"]
    heat = sc["heat"].to(DEVICE)

    _, ez, ey, ex = emb_map.shape
    scale = torch.tensor(
        [ez / PATCH_Z, ey / PATCH_Y, ex / PATCH_X], device=DEVICE, dtype=centers.dtype
    )
    embs = sample_embed_at(emb_map, centers * scale)
    flows_at = []
    for c in centers:
        az = int(min(max(c[0].item(), 0), PATCH_Z - 1))
        ay = int(min(max(c[1].item(), 0), PATCH_Y - 1))
        ax = int(min(max(c[2].item(), 0), PATCH_X - 1))
        flows_at.append(flow_vol[:, az, ay, ax])
    flows_at = torch.stack(flows_at, dim=0)
    loss_a = assoc_loss(embs, centers, parents, flows_at, n_sup, div_w)
    if pred.shape != heat.shape:
        pred = F.interpolate(
            pred[None, None], size=heat.shape, mode="trilinear", align_corners=False
        )[0, 0]
    loss_d = F.binary_cross_entropy_with_logits(pred, heat.clamp(0, 1))
    loss = loss_a + 0.15 * loss_d
    return loss, loss_a, loss_d


detector = CellectLite(time_frames=1).to(DEVICE)
assoc = ParentalAssocHead(emb_dim=64).to(DEVICE)
resumed_from = None
resume_step = 0
resume_best = float("inf")
if Path("/kaggle/input").exists():
    for r in list(Path("/kaggle/input").rglob("last.pt")) + list(
        Path("/kaggle/input").rglob("best_exact.pt")
    ):
        try:
            st = torch.load(r, map_location="cpu", weights_only=False)
            if not isinstance(st, dict) or "model" not in st:
                continue
            detector.load_state_dict(st["model"], strict=False)
            if "assoc" in st:
                assoc.load_state_dict(st["assoc"], strict=False)
                log(f"loaded detector+assoc from {r}")
            else:
                log(f"loaded detector only from {r}")
            resumed_from = str(r)
            if isinstance(st.get("global_step"), int):
                resume_step = int(st["global_step"])
            if st.get("best_exact") is not None:
                try:
                    # best_exact stored as -best_loss in our checkpoints
                    be = float(st["best_exact"])
                    resume_best = -be if be < 0 else float("inf")
                except Exception:
                    pass
            break
        except Exception as e:
            log(f"load skip {r}: {e}")

if USE_DUAL:
    detector = nn.DataParallel(detector)
    log(f"DataParallel ON devices={list(range(N_GPU))} batch={BATCH}")
else:
    log(f"SINGLE-GPU mode device={DEVICE} batch={BATCH} (no DataParallel)")

opt = torch.optim.AdamW(
    list(detector.parameters()) + list(assoc.parameters()), lr=LR, weight_decay=1e-4
)
# AMP off — same NaN failure mode as real Stage A (fp16 GradScaler)
USE_AMP = os.environ.get("BIOHUB_USE_AMP", "0") == "1"
scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and DEVICE.type == "cuda"))
step = resume_step
best_loss = resume_best if math.isfinite(resume_best) else float("inf")
end_step = SESSION_STEPS
nan_streak = 0
detector.train()
assoc.train()
log(
    f"TRAIN_LOOP Stage C start_step={step} end={end_step} resumed={resumed_from} "
    f"AMP={USE_AMP} dual={USE_DUAL} batch={BATCH} n_gpu_use={N_GPU}"
)

t_window0 = time.time()
steps_window0 = 0

while step < end_step and remaining_h() > STOP_REMAIN_H:
    scenes = [make_noisy_scene() for _ in range(BATCH)]
    img = torch.stack([s["img"] for s in scenes], dim=0).to(DEVICE)  # (B,1,Z,Y,X)

    opt.zero_grad(set_to_none=True)
    t_step0 = time.time()
    div_w = 2.0 + min(2.0, step / 15000.0)
    with torch.cuda.amp.autocast(enabled=(USE_AMP and DEVICE.type == "cuda")):
        out = detector(img)
        losses, la_s, ld_s = [], [], []
        for bi in range(BATCH):
            loss_i, la, ld = scene_losses(out, scenes, bi, div_w)
            losses.append(loss_i)
            la_s.append(la)
            ld_s.append(ld)
        loss = torch.stack(losses).mean()
        loss_a = torch.stack(la_s).mean()
        loss_d = torch.stack(ld_s).mean()

    if not torch.isfinite(loss):
        nan_streak += 1
        if step % LOG_EVERY == 0 or nan_streak <= 3:
            log(f"step={step} SKIP nonfinite loss nan_streak={nan_streak}")
        if nan_streak >= 50:
            log("ABORT: 50 consecutive nonfinite losses")
            break
        continue
    nan_streak = 0
    if USE_AMP:
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            list(detector.parameters()) + list(assoc.parameters()), 1.0
        )
        scaler.step(opt)
        scaler.update()
    else:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(detector.parameters()) + list(assoc.parameters()), 1.0
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
            f"a={float(loss_a.detach()):.4f} d={float(loss_d.detach()):.4f} "
            f"rem_h={remaining_h():.2f} batch={BATCH} dual={int(USE_DUAL)} "
            f"ms/step={step_ms:.0f} steps/s={steps_s:.3f} samples/s={samples_s:.3f} "
            f"vram0={v0:.2f}"
        )
        t_window0 = time.time()
        steps_window0 = 0
    if step % SAVE_EVERY == 0 or remaining_h() < (STOP_REMAIN_H + 0.05):
        joint = {
            "schema_version": 2,
            "stage": "C",
            "session": SESSION_ID,
            "global_step": step,
            "model": det_state_dict(detector),
            "assoc": assoc.state_dict(),
            "optimizer": opt.state_dict(),
            "best_exact": -best_loss if math.isfinite(best_loss) else 0.0,
            "config": {
                "stage": "C",
                "session": SESSION_ID,
                "resumed_from": resumed_from,
                "use_amp": USE_AMP,
                "dual_gpu": USE_DUAL,
                "n_gpu": N_GPU,
                "batch": BATCH,
            },
            "code_sha": "lineage_v2_stageC_c1_dualgpu_amp_off",
        }
        torch.save(joint, OUT / "last.pt")
        if math.isfinite(best_loss):
            torch.save(joint, OUT / "best_exact.pt")
        (OUT / "train_meta.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "best_loss": best_loss if math.isfinite(best_loss) else None,
                    "session": SESSION_ID,
                    "stage": "C",
                    "elapsed_h": (time.time() - T0) / 3600,
                    "no_submit": True,
                    "use_amp": USE_AMP,
                    "dual_gpu": USE_DUAL,
                    "n_gpu": N_GPU,
                    "batch": BATCH,
                },
                indent=2,
            )
            + "\n"
        )
        log(f"saved step={step}")

joint = {
    "schema_version": 2,
    "stage": "C",
    "session": SESSION_ID,
    "global_step": step,
    "model": det_state_dict(detector),
    "assoc": assoc.state_dict(),
    "best_exact": -best_loss if math.isfinite(best_loss) else 0.0,
    "config": {
        "stage": "C",
        "final": True,
        "resumed_from": resumed_from,
        "dual_gpu": USE_DUAL,
        "n_gpu": N_GPU,
        "batch": BATCH,
    },
    "code_sha": "lineage_v2_stageC_c1_dualgpu_amp_off",
}
torch.save(joint, OUT / "last.pt")
if math.isfinite(best_loss):
    torch.save(joint, OUT / "best_exact.pt")
final = {
    "status": "COMPLETE",
    "stage": "C",
    "session": SESSION_ID,
    "steps": step,
    "best_loss": best_loss if math.isfinite(best_loss) else None,
    "elapsed_h": (time.time() - T0) / 3600,
    "resumed_from": resumed_from,
    "no_submit": True,
    "dual_gpu": USE_DUAL,
    "n_gpu": N_GPU,
    "batch": BATCH,
}
(OUT / "session_complete.json").write_text(json.dumps(final, indent=2) + "\n")
log(f"DONE {final}")
