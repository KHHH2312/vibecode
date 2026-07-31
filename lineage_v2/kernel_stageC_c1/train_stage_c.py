import os
os.environ.setdefault('BIOHUB_SESSION_ID','c1')
os.environ.setdefault('BIOHUB_SESSION_STEPS','50000')
os.environ.setdefault('BIOHUB_TIME_BUDGET_H','11.0')
os.environ.setdefault('BIOHUB_LR','1e-4')
print('STAGE C c1')
"""Stage C — association on predicted/noisy nodes (plan 50k). Resume B weights. No submit."""
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

TIME_BUDGET_H = float(os.environ.get("BIOHUB_TIME_BUDGET_H", "11.0"))
SESSION_STEPS = int(os.environ.get("BIOHUB_SESSION_STEPS", "50000"))
SAVE_EVERY = int(os.environ.get("BIOHUB_SAVE_EVERY", "500"))
LOG_EVERY = int(os.environ.get("BIOHUB_LOG_EVERY", "50"))
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
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def remaining_h() -> float:
    return TIME_BUDGET_H - (time.time() - T0) / 3600.0


def log(msg: str) -> None:
    print(f"[{(time.time()-T0)/60:.1f}m] {msg}", flush=True)


log(f"STAGE C session={SESSION_ID} DEVICE={DEVICE} budget={TIME_BUDGET_H}h steps={SESSION_STEPS}")
log(f"metric_pin={verify_metric_pin()}")
log(f"FORBIDDEN={sorted(FORBIDDEN_TEST_STEMS)}")

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


def make_noisy_batch(device: torch.device):
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
    img = torch.randn(1, 1, Z, Y, X) * 0.05
    for c in centers:
        zz, yy, xx = int(c[0]), int(c[1]), int(c[2])
        img[
            0, 0, max(0, zz - 1) : zz + 2, max(0, yy - 2) : yy + 3, max(0, xx - 2) : xx + 3
        ] += random.uniform(0.7, 1.4)
    heat, _, _, _, _ = make_targets((Z, Y, X), centers[:n_sup], parents[:n_sup])
    return (
        img.to(device),
        torch.tensor(centers, dtype=torch.float32, device=device),
        torch.tensor(parents, dtype=torch.long, device=device),
        n_sup,
        heat.to(device),
    )


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


detector = CellectLite(time_frames=1).to(DEVICE)
assoc = ParentalAssocHead(emb_dim=64).to(DEVICE)
resumed_from = None
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
            break
        except Exception as e:
            log(f"load skip {r}: {e}")

opt = torch.optim.AdamW(
    list(detector.parameters()) + list(assoc.parameters()), lr=LR, weight_decay=1e-4
)
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == "cuda"))
step = 0
best_loss = float("inf")
end_step = SESSION_STEPS
detector.train()
assoc.train()
log(f"TRAIN_LOOP Stage C end={end_step} resumed={resumed_from}")

while step < end_step and remaining_h() > 0.25:
    img, centers, parents, n_sup, heat = make_noisy_batch(DEVICE)
    opt.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled=(DEVICE.type == "cuda")):
        out = detector(img)
        emb_map = out["embedding_map"][0]
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
            flows_at.append(out["backward_flow_um"][0, :, az, ay, ax])
        flows_at = torch.stack(flows_at, dim=0)
        div_w = 2.0 + min(2.0, step / 15000.0)
        loss_a = assoc_loss(embs, centers, parents, flows_at, n_sup, div_w)
        pred = out["center_logits"][0, 0]
        if pred.shape != heat.shape:
            pred = F.interpolate(
                pred[None, None], size=heat.shape, mode="trilinear", align_corners=False
            )[0, 0]
        loss_d = F.binary_cross_entropy_with_logits(pred, heat.clamp(0, 1))
        loss = loss_a + 0.15 * loss_d
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
            f"a={float(loss_a):.4f} d={float(loss_d):.4f} rem_h={remaining_h():.2f} vram={vram:.2f}"
        )
    if step % SAVE_EVERY == 0 or remaining_h() < 0.30:
        joint = {
            "schema_version": 2,
            "stage": "C",
            "session": SESSION_ID,
            "global_step": step,
            "model": detector.state_dict(),
            "assoc": assoc.state_dict(),
            "optimizer": opt.state_dict(),
            "best_exact": -best_loss,
            "config": {"stage": "C", "session": SESSION_ID, "resumed_from": resumed_from},
            "code_sha": "lineage_v2_stageC_c1",
        }
        torch.save(joint, OUT / "last.pt")
        torch.save(joint, OUT / "best_exact.pt")
        (OUT / "train_meta.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "best_loss": best_loss,
                    "session": SESSION_ID,
                    "stage": "C",
                    "elapsed_h": (time.time() - T0) / 3600,
                    "no_submit": True,
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
    "model": detector.state_dict(),
    "assoc": assoc.state_dict(),
    "best_exact": -best_loss,
    "config": {"stage": "C", "session": SESSION_ID, "final": True},
    "code_sha": "lineage_v2_stageC_c1",
}
torch.save(joint, OUT / "last.pt")
torch.save(joint, OUT / "best_exact.pt")
final = {
    "status": "COMPLETE",
    "stage": "C",
    "session": SESSION_ID,
    "steps": step,
    "best_loss": best_loss,
    "elapsed_h": (time.time() - T0) / 3600,
    "time_budget_h": TIME_BUDGET_H,
    "resumed_from": resumed_from,
    "stopped_reason": "session_steps" if step >= end_step else "time_budget",
    "no_submit": True,
    "next": "Stage D joint then E forks; GEFF real data needed for #1",
}
(OUT / "session_complete.json").write_text(json.dumps(final, indent=2) + "\n")
log(f"SESSION_DONE {final}")

