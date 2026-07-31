"""Stage B — parental association (session b1). Resume Stage A detector weights.

11h cap. No LB submit. Synthetic multi-node GT parents until GEFF loader exists.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---- package locate (same as Stage A) ----
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
    raise SystemExit("lineage_v2 package not found")

from lineage_v2.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint  # noqa: E402
from lineage_v2.constants import FORBIDDEN_TEST_STEMS, SPACING_ZYX_UM  # noqa: E402
from lineage_v2.eval.contract import verify_metric_pin  # noqa: E402
from lineage_v2.models.assoc import ParentalAssocHead  # noqa: E402
from lineage_v2.models.cellect_lite import CellectLite  # noqa: E402
from lineage_v2.targets import make_targets  # noqa: E402

TIME_BUDGET_H = float(os.environ.get("BIOHUB_TIME_BUDGET_H", "11.0"))
MAX_STEPS = int(os.environ.get("BIOHUB_MAX_STEPS", "30000"))  # Stage B plan 30k
SESSION_STEPS = int(os.environ.get("BIOHUB_SESSION_STEPS", "30000"))
SAVE_EVERY = int(os.environ.get("BIOHUB_SAVE_EVERY", "500"))
LOG_EVERY = int(os.environ.get("BIOHUB_LOG_EVERY", "50"))
LR_LINKER = float(os.environ.get("BIOHUB_LR_LINKER", "2e-4"))
LR_BACKBONE = float(os.environ.get("BIOHUB_LR_BACKBONE", "2e-5"))
SEED = int(os.environ.get("BIOHUB_SEED", "9401"))
SESSION_ID = os.environ.get("BIOHUB_SESSION_ID", "b1")
PATCH_Z, PATCH_Y, PATCH_X = 32, 160, 160
K_CAND = 8
SPACING = torch.tensor(list(SPACING_ZYX_UM), dtype=torch.float32)

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


log(f"STAGE B session={SESSION_ID} DEVICE={DEVICE} budget={TIME_BUDGET_H}h")
log(f"metric_pin={verify_metric_pin()}")
log(f"FORBIDDEN={sorted(FORBIDDEN_TEST_STEMS)}")

detector = CellectLite(time_frames=1).to(DEVICE)
assoc = ParentalAssocHead(emb_dim=64).to(DEVICE)

# freeze early detector for first 5k (plan: last two levels trainable later)
for p in detector.parameters():
    p.requires_grad = False
# unfreeze dec0/heads/embed for light finetune after step 5k handled in loop
for name, p in detector.named_parameters():
    if any(k in name for k in ("dec0", "head_", "dec1", "up0", "up1")):
        p.requires_grad = True

opt = torch.optim.AdamW(
    [
        {"params": [p for p in detector.parameters() if p.requires_grad], "lr": LR_BACKBONE},
        {"params": assoc.parameters(), "lr": LR_LINKER},
    ],
    weight_decay=1e-4,
)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# resume detector from Stage A weights
start_step = 0
resumed_from = None
resume_paths = [OUT / "last.pt"]
if Path("/kaggle/input").exists():
    resume_paths.extend(sorted(Path("/kaggle/input").rglob("last.pt")))
    resume_paths.extend(sorted(Path("/kaggle/input").rglob("best_exact.pt")))

for r in resume_paths:
    if not r.exists():
        continue
    try:
        st = torch.load(r, map_location="cpu", weights_only=False)
        if isinstance(st, dict) and "model" in st:
            missing, unexpected = detector.load_state_dict(st["model"], strict=False)
            log(f"loaded detector from {r} missing={len(missing)} unexpected={len(unexpected)}")
            # Stage B global_step starts at 0 for assoc; keep A step in meta
            start_step = 0
            resumed_from = str(r)
            break
    except Exception as e:
        log(f"resume skip {r}: {e}")

end_step = min(MAX_STEPS, start_step + SESSION_STEPS)
log(f"start_step={start_step} end_step={end_step} resumed_from={resumed_from}")


def sample_um_batch(device: torch.device):
    """Synthetic nodes in a small 3D crop with true parents (GT-node assoc)."""
    n = random.randint(6, 16)
    Z, Y, X = PATCH_Z, PATCH_Y, PATCH_X
    centers = []
    parents = []
    for i in range(n):
        centers.append(
            [
                random.uniform(4, Z - 5),
                random.uniform(16, Y - 17),
                random.uniform(16, X - 17),
            ]
        )
        if i > 0 and random.random() < 0.55:
            parents.append(random.randint(0, i - 1))
        else:
            parents.append(-1)  # quiet / birth
    centers_t = torch.tensor(centers, dtype=torch.float32)
    # image blobs
    img = torch.randn(1, 1, Z, Y, X) * 0.05
    for c in centers:
        zz, yy, xx = int(c[0]), int(c[1]), int(c[2])
        img[0, 0, max(0, zz - 1) : zz + 2, max(0, yy - 2) : yy + 3, max(0, xx - 2) : xx + 3] += 1.0
    heat, offset, flow, ov, fv = make_targets((Z, Y, X), centers, parents)
    return (
        img.to(device),
        centers_t.to(device),
        torch.tensor(parents, dtype=torch.long, device=device),
        heat.to(device),
        offset.to(device),
        flow.to(device),
        ov.to(device),
        fv.to(device),
    )


def sample_embed_at(emb_map: torch.Tensor, centers_zyx: torch.Tensor) -> torch.Tensor:
    """Trilinear sample embedding map at float zyx (N,3) → (N,D). emb_map (D,z,y,x) or (1,D,z,y,x)."""
    if emb_map.ndim == 4:
        emb_map = emb_map.unsqueeze(0)
    # grid_sample expects (N,C,D,H,W) and grid xyz in [-1,1]
    _, C, D, H, W = emb_map.shape
    z = centers_zyx[:, 0] / max(D - 1, 1) * 2 - 1
    y = centers_zyx[:, 1] / max(H - 1, 1) * 2 - 1
    x = centers_zyx[:, 2] / max(W - 1, 1) * 2 - 1
    # grid: (N,1,1,1,3) with order x,y,z
    grid = torch.stack([x, y, z], dim=-1).view(-1, 1, 1, 1, 3)
    # sample each point: expand map
    out = []
    for i in range(centers_zyx.shape[0]):
        g = grid[i : i + 1]
        v = F.grid_sample(emb_map, g, align_corners=True, mode="bilinear")
        out.append(v.view(C))
    return torch.stack(out, dim=0)


def assoc_loss(embs, centers_zyx, parents, flows_at, div_w: float):
    """Parental softmax over K candidates + quiet. parents[i]=-1 → quiet."""
    n = centers_zyx.shape[0]
    spacing = SPACING.to(centers_zyx.device)
    centers_um = centers_zyx * spacing
    total = centers_zyx.new_tensor(0.0)
    n_terms = 0
    for i in range(n):
        true_p = int(parents[i].item())
        # candidate parents: all others by distance
        d = (centers_um - centers_um[i : i + 1]).norm(dim=-1)
        d[i] = 1e9
        k = min(K_CAND, n - 1)
        if k < 1:
            continue
        _, idx = torch.topk(d, k=k, largest=False)
        # force-insert true parent
        if true_p >= 0 and true_p not in idx.tolist():
            idx = torch.cat([idx[:-1], parents.new_tensor([true_p])])
        emb_c = embs[i : i + 1].expand(k, -1)
        emb_p = embs[idx]
        c_um = centers_um[i : i + 1].expand(k, -1)
        p_um = centers_um[idx]
        flow = flows_at[i : i + 1].expand(k, -1)
        edge_logits = assoc.edge_logit(emb_c, emb_p, c_um, p_um, flow)
        q_logit = assoc.quiet_logit(embs[i : i + 1], flows_at[i : i + 1])
        logits = torch.cat([q_logit, edge_logits], dim=0)  # quiet + k
        if true_p < 0:
            target = 0  # quiet
            w = 1.0
        else:
            # index of true parent in idx → +1 for quiet offset
            pos = (idx == true_p).nonzero(as_tuple=False)
            if pos.numel() == 0:
                target = 0
                w = 1.0
            else:
                target = int(pos[0, 0].item()) + 1
                w = div_w if random.random() < 0.2 else 1.0  # mild div ramp proxy
        total = total + w * F.cross_entropy(logits.unsqueeze(0), logits.new_tensor([target], dtype=torch.long))
        n_terms += 1
    if n_terms == 0:
        return total
    return total / n_terms


step = 0
best_loss = float("inf")
detector.train()
assoc.train()
log("TRAIN_LOOP_START Stage B")

while step < end_step and remaining_h() > 0.25:
    # unfreeze more backbone after 5k
    if step == 5000:
        for p in detector.parameters():
            p.requires_grad = True
        opt = torch.optim.AdamW(
            [
                {"params": detector.parameters(), "lr": LR_BACKBONE},
                {"params": assoc.parameters(), "lr": LR_LINKER},
            ],
            weight_decay=1e-4,
        )
        log("unfroze full detector at step 5000")

    img, centers, parents, heat, offset, flow, ov, fv = sample_um_batch(DEVICE)
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
        out = detector(img)
        emb_map = out["embedding_map"][0]  # C,z,y,x
        # scale centers to emb map resolution
        _, ez, ey, ex = emb_map.shape
        scale = torch.tensor(
            [ez / PATCH_Z, ey / PATCH_Y, ex / PATCH_X],
            device=DEVICE,
            dtype=centers.dtype,
        )
        centers_e = centers * scale
        embs = sample_embed_at(emb_map, centers_e)
        # flow at integer anchors
        flows_at = []
        for c in centers:
            az, ay, ax = [int(v.item()) for v in c]
            az = min(max(az, 0), PATCH_Z - 1)
            ay = min(max(ay, 0), PATCH_Y - 1)
            ax = min(max(ax, 0), PATCH_X - 1)
            flows_at.append(out["backward_flow_um"][0, :, az, ay, ax])
        flows_at = torch.stack(flows_at, dim=0)
        div_w = 2.0 + min(2.0, step / 10000.0)  # ramp 2→4
        loss_a = assoc_loss(embs, centers, parents, flows_at, div_w)
        # light det keep-alive
        pred = out["center_logits"][0, 0]
        if pred.shape != heat.shape:
            pred = F.interpolate(pred[None, None], size=heat.shape, mode="trilinear", align_corners=False)[0, 0]
        loss_d = F.binary_cross_entropy_with_logits(pred, heat.clamp(0, 1))
        loss = loss_a + 0.2 * loss_d
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(
        list(detector.parameters()) + list(assoc.parameters()), 1.0
    )
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
            f"assoc={float(loss_a):.4f} det={float(loss_d):.4f} "
            f"div_w={div_w:.2f} rem_h={remaining_h():.2f} vram={vram:.2f}"
        )
    if step % SAVE_EVERY == 0 or remaining_h() < 0.30:
        # save joint state: detector + assoc
        joint = {
            "schema_version": 2,
            "stage": "B",
            "session": SESSION_ID,
            "global_step": step,
            "model": detector.state_dict(),
            "assoc": assoc.state_dict(),
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "best_exact": -best_loss,
            "config": {
                "stage": "B",
                "session": SESSION_ID,
                "resumed_detector_from": resumed_from,
            },
            "split_sha256": "pending_real_manifest",
            "code_sha": "lineage_v2_stageB_b1",
            "inference_policy_sha256": "smoke",
        }
        torch.save(joint, OUT / "last.pt")
        torch.save(joint, OUT / "best_exact.pt")
        (OUT / "train_meta.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "best_loss": best_loss,
                    "session": SESSION_ID,
                    "stage": "B",
                    "elapsed_h": (time.time() - T0) / 3600,
                    "no_submit": True,
                },
                indent=2,
            )
            + "\n"
        )
        log(f"saved export step={step}")

joint = {
    "schema_version": 2,
    "stage": "B",
    "session": SESSION_ID,
    "global_step": step,
    "model": detector.state_dict(),
    "assoc": assoc.state_dict(),
    "optimizer": opt.state_dict(),
    "best_exact": -best_loss,
    "config": {"stage": "B", "session": SESSION_ID, "final": True},
    "code_sha": "lineage_v2_stageB_b1",
}
torch.save(joint, OUT / "last.pt")
torch.save(joint, OUT / "best_exact.pt")
final = {
    "status": "COMPLETE",
    "stage": "B",
    "session": SESSION_ID,
    "steps": step,
    "best_loss": best_loss,
    "elapsed_h": (time.time() - T0) / 3600,
    "time_budget_h": TIME_BUDGET_H,
    "resumed_detector_from": resumed_from,
    "stopped_reason": "session_steps" if step >= end_step else "time_budget",
    "no_submit": True,
    "next": "Stage B session b2 or GEFF GT-node association",
}
(OUT / "session_complete.json").write_text(json.dumps(final, indent=2) + "\n")
log(f"SESSION_DONE {final}")
