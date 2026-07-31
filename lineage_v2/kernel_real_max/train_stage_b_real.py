"""Stage B REAL — parental association on GEFF GT nodes.

Resume real Stage A detector. 11h cap. No LB submit.
AMP OFF by default (real GEFF stability). Fast startup (no full-input rglob).
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def _offline_install_zarr() -> None:
    """Prefer known dataset wheel path — never rglob full competition tree."""
    try:
        import zarr  # noqa: F401

        return
    except ImportError:
        pass
    preferred = []
    for user in ("mijuuu8", "kokiiii"):
        preferred.extend(
            [
                Path(f"/kaggle/input/datasets/{user}/lineage-v2-src/wheels"),
                Path(f"/kaggle/input/{user}/lineage-v2-src/wheels"),
            ]
        )
    preferred.append(Path("/kaggle/input/lineage-v2-src/wheels"))
    wheel_dirs = [p for p in preferred if p.is_dir()]
    if not wheel_dirs and Path("/kaggle/input/datasets").exists():
        wheel_dirs = [p for p in Path("/kaggle/input/datasets").rglob("wheels") if p.is_dir()]
    if not wheel_dirs:
        raise SystemExit("zarr missing and no wheels/")
    wdir = wheel_dirs[0]
    pkgs = []
    for name in ("packaging", "typing_extensions", "pyyaml", "donfig", "google_crc32c", "numcodecs", "zarr"):
        matches = sorted(wdir.glob(f"{name}-*.whl")) + sorted(
            wdir.glob(f"{name.replace('_', '-')}-*.whl")
        )
        if matches:
            pkgs.append(str(matches[0]))
    print("OFFLINE_PIP", pkgs, flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--quiet", *pkgs]
    )
    import zarr  # noqa: F401

    print(f"zarr_ok version={zarr.__version__}", flush=True)


_offline_install_zarr()
import zarr  # noqa: E402

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


_PKG_CANDS = []
for user in ("mijuuu8", "kokiiii"):
    _PKG_CANDS.extend(
        [
            Path(f"/kaggle/input/datasets/{user}/lineage-v2-src"),
            Path(f"/kaggle/input/{user}/lineage-v2-src"),
        ]
    )
_PKG_CANDS.extend([Path("/kaggle/input/lineage-v2-src"), Path("/kaggle/working")])
for base in _PKG_CANDS:
    if base.exists() and _try_set(base):
        break
if PKG_ROOT is None and Path("/kaggle/input/datasets").exists():
    for p in Path("/kaggle/input/datasets").rglob("constants.py"):
        if _try_set(p.parent) or _try_set(p.parent.parent):
            break
if PKG_ROOT is None:
    raise SystemExit("lineage_v2 not found")

from lineage_v2.checkpoint import build_checkpoint, save_checkpoint  # noqa: E402
from lineage_v2.constants import FORBIDDEN_TEST_STEMS, SPACING_ZYX_UM  # noqa: E402
from lineage_v2.denylist import assert_stem_allowed  # noqa: E402
from lineage_v2.eval.contract import verify_metric_pin  # noqa: E402
from lineage_v2.models.assoc import ParentalAssocHead  # noqa: E402
from lineage_v2.models.cellect_lite import CellectLite  # noqa: E402

TIME_BUDGET_H = float(os.environ.get("BIOHUB_TIME_BUDGET_H", "11.0"))
SESSION_STEPS = int(os.environ.get("BIOHUB_SESSION_STEPS", "35000"))
MAX_STEPS = int(os.environ.get("BIOHUB_MAX_STEPS", "35000"))
SAVE_EVERY = int(os.environ.get("BIOHUB_SAVE_EVERY", "500"))
LOG_EVERY = int(os.environ.get("BIOHUB_LOG_EVERY", "50"))
LR_LINKER = float(os.environ.get("BIOHUB_LR_LINKER", "2e-4"))
LR_BACKBONE = float(os.environ.get("BIOHUB_LR_BACKBONE", "2e-5"))
SEED = int(os.environ.get("BIOHUB_SEED", "94017"))
SESSION_ID = os.environ.get("BIOHUB_SESSION_ID", "realB1")
K_CAND = int(os.environ.get("BIOHUB_K_CAND", "8"))
USE_AMP = os.environ.get("BIOHUB_USE_AMP", "0") == "1"
MAX_CONSEC_NONFINITE = int(os.environ.get("BIOHUB_MAX_CONSEC_NONFINITE", "80"))
PATCH_Z, PATCH_Y, PATCH_X = 32, 128, 128

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
    print(f"[{(time.time() - T0) / 60:.1f}m] {msg}", flush=True)


log(f"STAGE B REAL session={SESSION_ID} DEVICE={DEVICE} USE_AMP={USE_AMP}")
log(f"metric_pin={verify_metric_pin()}")
log(f"FORBIDDEN={sorted(FORBIDDEN_TEST_STEMS)}")
log(f"PKG_ROOT={PKG_ROOT}")

TRAIN_CANDS = [
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
    Path("/kaggle/input/biohub-cell-tracking-during-development/train"),
]
TRAIN_DIR = next((p for p in TRAIN_CANDS if p.is_dir()), None)
log(f"TRAIN_DIR={TRAIN_DIR}")


def discover_stems(train_dir: Path | None) -> list[dict]:
    if train_dir is None:
        return []
    out = []
    for zp in sorted(train_dir.glob("*.zarr")):
        stem = zp.name[: -len(".zarr")] if zp.name.endswith(".zarr") else zp.stem
        if stem in FORBIDDEN_TEST_STEMS:
            log(f"SKIP forbidden stem {stem}")
            continue
        assert_stem_allowed(stem, context="discover")
        gp = train_dir / f"{stem}.geff"
        if gp.exists():
            out.append({"stem": stem, "zarr": zp, "geff": gp})
    return out


STEM_RECS = discover_stems(TRAIN_DIR)
log(f"n_stems={len(STEM_RECS)}")
if not STEM_RECS:
    raise SystemExit("no real stems for Stage B")


class StemCache:
    def __init__(self, rec: dict):
        self.stem = rec["stem"]
        assert_stem_allowed(self.stem, context="Bcache")
        g = zarr.open(str(rec["geff"]), mode="r")
        self.ids = np.asarray(g["nodes/ids"][:])
        self.t = np.asarray(g["nodes/props/t/values"][:], dtype=np.int64)
        self.z = np.asarray(g["nodes/props/z/values"][:], dtype=np.float64)
        self.y = np.asarray(g["nodes/props/y/values"][:], dtype=np.float64)
        self.x = np.asarray(g["nodes/props/x/values"][:], dtype=np.float64)
        edges = np.asarray(g["edges/ids"][:])
        id_to_i = {int(i): k for k, i in enumerate(self.ids)}
        self.parent_of = np.full(len(self.ids), -1, dtype=np.int64)
        for a, b in edges:
            ia, ib = id_to_i.get(int(a)), id_to_i.get(int(b))
            if ia is not None and ib is not None:
                self.parent_of[ib] = ia
        self.by_t = {int(ti): np.where(self.t == ti)[0] for ti in np.unique(self.t)}
        zg = zarr.open(str(rec["zarr"]), mode="r")
        self.img = zg["0"] if "0" in zg else zg[list(zg.keys())[0]]
        self.T, self.Z, self.Y, self.X = map(int, self.img.shape[:4])


_CACHE: dict[str, StemCache] = {}


def get_stem(rec: dict) -> StemCache:
    if rec["stem"] not in _CACHE:
        log(f"load {rec['stem']}")
        _CACHE[rec["stem"]] = StemCache(rec)
    return _CACHE[rec["stem"]]


def normalize(vol: np.ndarray) -> np.ndarray:
    v = np.ascontiguousarray(vol, dtype=np.float32)
    lo, hi = np.percentile(v, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((v - lo) / (hi - lo), 0, 1).astype(np.float32)


detector = CellectLite(time_frames=1).to(DEVICE)
assoc = ParentalAssocHead(emb_dim=64).to(DEVICE)
opt = torch.optim.AdamW(
    [
        {"params": detector.parameters(), "lr": LR_BACKBONE},
        {"params": assoc.parameters(), "lr": LR_LINKER},
    ],
    weight_decay=1e-4,
)
scaler = torch.amp.GradScaler("cuda", enabled=(USE_AMP and DEVICE.type == "cuda"))

# Prefer known resume datasets — avoid competition rglob
warm_from = None
warm_paths: list[Path] = []
for name in (
    "lineage-v2-warm",
    "lineage-v2-reala3",
    "lineage-v2-realA3",
    "lineage-v2-reala2",
    "lineage-v2-realA2",
    "lineage-v2-reala1",
    "lineage-v2-realA1",
    "lineage-v2-realb1",
    "lineage-v2-realB1",
):
    for user in ("mijuuu8", "kokiiii"):
        for base in (
            Path(f"/kaggle/input/datasets/{user}/{name}"),
            Path(f"/kaggle/input/{user}/{name}"),
        ):
            if (base / "last.pt").exists():
                warm_paths.append(base / "last.pt")
            if (base / "best_exact.pt").exists():
                warm_paths.append(base / "best_exact.pt")
    base = Path(f"/kaggle/input/{name}")
    if (base / "last.pt").exists():
        warm_paths.append(base / "last.pt")
    if (base / "best_exact.pt").exists():
        warm_paths.append(base / "best_exact.pt")
if Path("/kaggle/input/datasets").exists() and not warm_paths:
    warm_paths.extend(sorted(Path("/kaggle/input/datasets").rglob("last.pt")))

for r in warm_paths:
    try:
        st = torch.load(r, map_location="cpu", weights_only=False)
        if not isinstance(st, dict) or "model" not in st:
            continue
        detector.load_state_dict(st["model"], strict=False)
        if "assoc" in st:
            try:
                assoc.load_state_dict(st["assoc"], strict=False)
            except Exception:
                pass
        warm_from = str(r)
        log(f"WARM from {r}")
        break
    except Exception as e:
        log(f"warm skip {r}: {e}")

step = 0
end_step = min(MAX_STEPS, SESSION_STEPS)
best_loss = float("inf")
consecutive_nonfinite = 0
log(f"end_step={end_step} warm_from={warm_from}")


def sample_batch():
    rec = random.choice(STEM_RECS)
    sc = get_stem(rec)
    child_idxs = [i for i in range(len(sc.ids)) if sc.parent_of[i] >= 0 and 0 <= sc.t[i] < sc.T]
    if not child_idxs:
        raise RuntimeError("no parented nodes")
    ci = int(random.choice(child_idxs))
    pi = int(sc.parent_of[ci])
    t_c = int(sc.t[ci])
    t_p = int(sc.t[pi])
    cz, cy, cx = float(sc.z[ci]), float(sc.y[ci]), float(sc.x[ci])
    z0 = int(np.clip(round(cz - PATCH_Z / 2), 0, max(0, sc.Z - PATCH_Z)))
    y0 = int(np.clip(round(cy - PATCH_Y / 2), 0, max(0, sc.Y - PATCH_Y)))
    x0 = int(np.clip(round(cx - PATCH_X / 2), 0, max(0, sc.X - PATCH_X)))
    patch = np.asarray(sc.img[t_c, z0 : z0 + PATCH_Z, y0 : y0 + PATCH_Y, x0 : x0 + PATCH_X])
    if patch.shape != (PATCH_Z, PATCH_Y, PATCH_X):
        pad = np.zeros((PATCH_Z, PATCH_Y, PATCH_X), dtype=np.float32)
        pad[: patch.shape[0], : patch.shape[1], : patch.shape[2]] = patch
        patch = pad
    img = torch.from_numpy(normalize(patch))[None, None].to(DEVICE, dtype=torch.float32)
    img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

    p_idxs = list(sc.by_t.get(t_p, []))
    if pi not in p_idxs:
        p_idxs.append(pi)

    def dist_um(i):
        dz = (float(sc.z[i]) - cz) * SPACING_ZYX_UM[0]
        dy = (float(sc.y[i]) - cy) * SPACING_ZYX_UM[1]
        dx = (float(sc.x[i]) - cx) * SPACING_ZYX_UM[2]
        return (dz * dz + dy * dy + dx * dx) ** 0.5

    p_idxs = sorted(set(int(i) for i in p_idxs), key=dist_um)[:K_CAND]
    if pi not in p_idxs:
        p_idxs = [pi] + p_idxs[: K_CAND - 1]
    true_k = p_idxs.index(pi)

    with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
        out = detector(img)
        emb_map = out["embedding_map"]
        if emb_map.ndim == 5:
            emb_map = emb_map[0]
        flow = out["backward_flow_um"][0]

        def sample_emb(wz, wy, wx):
            ez = int(np.clip((wz - z0) / PATCH_Z * emb_map.shape[-3], 0, emb_map.shape[-3] - 1))
            ey = int(np.clip((wy - y0) / PATCH_Y * emb_map.shape[-2], 0, emb_map.shape[-2] - 1))
            ex = int(np.clip((wx - x0) / PATCH_X * emb_map.shape[-1], 0, emb_map.shape[-1] - 1))
            return emb_map[:, ez, ey, ex]

        emb_c = sample_emb(cz, cy, cx).float()
        child_um = torch.tensor(
            [cz * SPACING_ZYX_UM[0], cy * SPACING_ZYX_UM[1], cx * SPACING_ZYX_UM[2]],
            device=DEVICE,
            dtype=torch.float32,
        )
        if flow.shape[-3:] != (PATCH_Z, PATCH_Y, PATCH_X):
            flow_u = F.interpolate(
                flow[None], size=(PATCH_Z, PATCH_Y, PATCH_X), mode="trilinear", align_corners=False
            )[0]
        else:
            flow_u = flow
        child_flow = flow_u[
            :,
            int(np.clip(cz - z0, 0, PATCH_Z - 1)),
            int(np.clip(cy - y0, 0, PATCH_Y - 1)),
            int(np.clip(cx - x0, 0, PATCH_X - 1)),
        ].float()

        logits = []
        for pj in p_idxs:
            emb_p = sample_emb(float(sc.z[pj]), float(sc.y[pj]), float(sc.x[pj])).float()
            parent_um = torch.tensor(
                [
                    float(sc.z[pj]) * SPACING_ZYX_UM[0],
                    float(sc.y[pj]) * SPACING_ZYX_UM[1],
                    float(sc.x[pj]) * SPACING_ZYX_UM[2],
                ],
                device=DEVICE,
                dtype=torch.float32,
            )
            logits.append(
                assoc.edge_logit(
                    emb_c[None], emb_p[None], child_um[None], parent_um[None], child_flow[None]
                )[0]
            )
        q = assoc.quiet_logit(emb_c[None], child_flow[None])[0]
        logits.append(q)
        logits_t = torch.stack(logits).float()
        target = torch.tensor(true_k, device=DEVICE, dtype=torch.long)
        n_sib = int((sc.parent_of == pi).sum())
        w = 3.0 if n_sib >= 2 else 1.0
        loss = F.cross_entropy(logits_t[None], target[None]) * w
    return loss.float(), true_k, len(p_idxs), sc.stem, w


detector.train()
assoc.train()
log("TRAIN_LOOP_START data_mode=geff_real_assoc USE_AMP=" + str(USE_AMP))
while step < end_step and remaining_h() > 0.25:
    try:
        loss, true_k, nk, stem, w = sample_batch()
    except Exception as e:
        log(f"batch_fail: {e}")
        consecutive_nonfinite += 1
        if consecutive_nonfinite >= MAX_CONSEC_NONFINITE:
            log(f"ABORT too many batch fails at step={step}")
            break
        continue
    if not torch.isfinite(loss):
        consecutive_nonfinite += 1
        if consecutive_nonfinite <= 5 or consecutive_nonfinite % 20 == 0:
            log(f"nonfinite loss at step {step}, skip (consec={consecutive_nonfinite})")
        if consecutive_nonfinite >= MAX_CONSEC_NONFINITE:
            log(f"ABORT: {MAX_CONSEC_NONFINITE} consecutive nonfinite")
            break
        continue
    opt.zero_grad(set_to_none=True)
    try:
        if USE_AMP and DEVICE.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(list(detector.parameters()) + list(assoc.parameters()), 0.5)
        grads_ok = True
        for p in list(detector.parameters()) + list(assoc.parameters()):
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grads_ok = False
                break
        if not grads_ok:
            consecutive_nonfinite += 1
            opt.zero_grad(set_to_none=True)
            if USE_AMP:
                scaler.update()
            continue
        if USE_AMP and DEVICE.type == "cuda":
            scaler.step(opt)
            scaler.update()
        else:
            opt.step()
    except RuntimeError as e:
        log(f"step_runtime_fail: {e}")
        consecutive_nonfinite += 1
        opt.zero_grad(set_to_none=True)
        if consecutive_nonfinite >= MAX_CONSEC_NONFINITE:
            break
        continue
    consecutive_nonfinite = 0
    step += 1
    lv = float(loss.detach())
    if lv < best_loss:
        best_loss = lv
    if step % LOG_EVERY == 0:
        log(
            f"step={step} loss={lv:.4f} best={best_loss:.4f} true_k={true_k} nk={nk} "
            f"w={w} stem={stem} rem_h={remaining_h():.2f}"
        )
    if step % SAVE_EVERY == 0 or remaining_h() < 0.30:
        pack = {
            "schema_version": 2,
            "stage": "B",
            "session": SESSION_ID,
            "global_step": step,
            "model": detector.state_dict(),
            "assoc": assoc.state_dict(),
            "best_exact": -best_loss,
            "config": {
                "stage": "B",
                "session": SESSION_ID,
                "data_mode": "geff_real_assoc",
                "warm_from": warm_from,
                "n_stems": len(STEM_RECS),
                "use_amp": USE_AMP,
            },
        }
        torch.save(pack, OUT / "last.pt")
        torch.save(pack, OUT / "best_exact.pt")
        meta = {
            "step": step,
            "best_loss": best_loss,
            "elapsed_h": (time.time() - T0) / 3600,
            "remaining_h": remaining_h(),
            "session": SESSION_ID,
            "data_mode": "geff_real_assoc",
            "warm_from": warm_from,
            "no_submit": True,
        }
        (OUT / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        log(f"saved step={step}")

# final save
pack = {
    "schema_version": 2,
    "stage": "B",
    "session": SESSION_ID,
    "global_step": step,
    "model": detector.state_dict(),
    "assoc": assoc.state_dict(),
    "best_exact": -best_loss,
    "config": {
        "stage": "B",
        "session": SESSION_ID,
        "final": True,
        "data_mode": "geff_real_assoc",
        "warm_from": warm_from,
        "use_amp": USE_AMP,
    },
}
torch.save(pack, OUT / "last.pt")
torch.save(pack, OUT / "best_exact.pt")
final = {
    "status": "COMPLETE",
    "stage": "B",
    "session": SESSION_ID,
    "steps": step,
    "best_loss": best_loss,
    "elapsed_h": (time.time() - T0) / 3600,
    "data_mode": "geff_real_assoc",
    "n_stems": len(STEM_RECS),
    "warm_from": warm_from,
    "use_amp": USE_AMP,
    "no_submit": True,
    "next": "realA3 detector polish or Stage C",
}
(OUT / "session_complete.json").write_text(json.dumps(final, indent=2) + "\n")
log(f"DONE {final}")
