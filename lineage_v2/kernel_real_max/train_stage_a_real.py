"""Stage-A REAL GEFF train — first data session on train account.

- Loads competition train .zarr + .geff (denylist fail-closed)
- Warm-starts detector from synthetic DE weights if present
- 11h budget, no LB submit
"""
from __future__ import annotations

import json
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def _offline_install_zarr() -> None:
    """Install zarr from wheels bundled in attached datasets (internet OFF safe).

    Prefer known paths — full /kaggle/input rglob walks the competition train tree
    and costs ~10+ minutes on Kaggle NFS.
    """
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
    wheel_dirs: list[Path] = [p for p in preferred if p.is_dir()]
    if not wheel_dirs and Path("/kaggle/input/datasets").exists():
        wheel_dirs = [p for p in Path("/kaggle/input/datasets").rglob("wheels") if p.is_dir()]
    if not wheel_dirs:
        raise SystemExit(
            "zarr not installed and no lineage-v2-src/wheels found. "
            "Re-version lineage-v2-src with wheels/."
        )
    wdir = wheel_dirs[0]
    # Prefer pure-python / manylinux wheels; do NOT force reinstall numpy (Kaggle already has it).
    pkgs = []
    for name in ("packaging", "typing_extensions", "pyyaml", "donfig", "google_crc32c", "numcodecs", "zarr"):
        matches = sorted(wdir.glob(f"{name}-*.whl")) + sorted(wdir.glob(f"{name.replace('_','-')}-*.whl"))
        if matches:
            pkgs.append(str(matches[0]))
    if not pkgs:
        raise SystemExit(f"no usable wheels in {wdir}")
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-index",
        "--no-deps",
        "--quiet",
        *pkgs,
    ]
    print("OFFLINE_PIP", cmd, flush=True)
    subprocess.check_call(cmd)
    import zarr  # noqa: F401

    print(f"zarr_ok version={zarr.__version__}", flush=True)


_offline_install_zarr()

import numpy as np
import torch
import torch.nn.functional as F

# ---- locate package ----
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
for user in ("mijuuu8", "kokiiii"):
    candidates.extend(
        [
            Path(f"/kaggle/input/datasets/{user}/lineage-v2-src"),
            Path(f"/kaggle/input/{user}/lineage-v2-src"),
        ]
    )
candidates.extend([Path("/kaggle/input/lineage-v2-src"), Path("/kaggle/working")])
# Only scan datasets/ for package (never competitions/)
if Path("/kaggle/input/datasets").exists():
    for p in Path("/kaggle/input/datasets").rglob("constants.py"):
        parent = p.parent
        if (parent / "models").is_dir() or parent.name == "lineage_v2":
            candidates.append(parent)
            if parent.parent not in candidates:
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
    listing = []
    inp = Path("/kaggle/input")
    if inp.exists():
        for p in inp.rglob("constants.py"):
            listing.append(str(p))
    raise SystemExit(f"lineage_v2 package not found. constants.py hits={listing[:20]}")

from lineage_v2.checkpoint import build_checkpoint, save_checkpoint  # noqa: E402
from lineage_v2.constants import FORBIDDEN_TEST_STEMS, SIGMA_UM, SPACING_ZYX_UM  # noqa: E402
from lineage_v2.denylist import ForbiddenStemError, assert_stem_allowed  # noqa: E402
from lineage_v2.eval.contract import verify_metric_pin  # noqa: E402
from lineage_v2.models.cellect_lite import CellectLite  # noqa: E402
from lineage_v2.targets import make_targets  # noqa: E402

import zarr  # installed via _offline_install_zarr above

# ---- session ----
TIME_BUDGET_H = float(os.environ.get("BIOHUB_TIME_BUDGET_H", "11.0"))
# Max-score defaults: fill ~11h on real I/O (override via env per session)
MAX_STEPS = int(os.environ.get("BIOHUB_MAX_STEPS", "70000"))
SESSION_STEPS = int(os.environ.get("BIOHUB_SESSION_STEPS", "70000"))
SAVE_EVERY = int(os.environ.get("BIOHUB_SAVE_EVERY", "500"))
LOG_EVERY = int(os.environ.get("BIOHUB_LOG_EVERY", "50"))
PATCH_Z, PATCH_Y, PATCH_X = 32, 160, 160
LR = float(os.environ.get("BIOHUB_LR", "5e-5"))  # lower LR: real data + warm start
SEED = int(os.environ.get("BIOHUB_SEED", "94017"))
SESSION_ID = os.environ.get("BIOHUB_SESSION_ID", "realA_max")
REQUIRE_REAL = os.environ.get("BIOHUB_REQUIRE_REAL", "1") != "0"
# Plan-style sampler mix: ordinary / dense / division
P_DENSE = float(os.environ.get("BIOHUB_P_DENSE", "0.30"))
P_DIV = float(os.environ.get("BIOHUB_P_DIV", "0.25"))

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


log(f"DEVICE={DEVICE} TIME_BUDGET_H={TIME_BUDGET_H} SESSION={SESSION_ID}")
log(f"FORBIDDEN={sorted(FORBIDDEN_TEST_STEMS)}")
log(f"metric_pin={verify_metric_pin()}")
log(f"PKG_ROOT={PKG_ROOT}")

# ---- competition train root ----
TRAIN_CANDS = [
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development/train"),
    Path("/kaggle/input/biohub-cell-tracking-during-development/train"),
    Path("/kaggle/input/competitions/biohub-cell-tracking-during-development"),
    Path("/kaggle/input/biohub-cell-tracking-during-development"),
]
TRAIN_DIR = None
for p in TRAIN_CANDS:
    if not p.is_dir():
        continue
    # prefer a dir that actually has .zarr or train subdir
    if any(x.suffix == ".zarr" or x.name.endswith(".zarr") for x in p.iterdir()):
        TRAIN_DIR = p
        break
    if (p / "train").is_dir():
        TRAIN_DIR = p / "train"
        break
    TRAIN_DIR = p  # last resort

log(f"TRAIN_DIR={TRAIN_DIR}")


def discover_stems(train_dir: Path | None) -> list[dict]:
    """Return [{stem, zarr, geff}, ...] excluding forbidden test stems."""
    if train_dir is None or not train_dir.is_dir():
        return []
    out: list[dict] = []
    # Case A: flat {stem}.zarr + {stem}.geff
    zarrs = list(train_dir.glob("*.zarr"))
    for zp in sorted(zarrs):
        stem = zp.name[: -len(".zarr")] if zp.name.endswith(".zarr") else zp.stem
        if stem in FORBIDDEN_TEST_STEMS:
            log(f"SKIP forbidden stem {stem}")
            continue
        assert_stem_allowed(stem, context="discover")
        gp = train_dir / f"{stem}.geff"
        if not gp.exists():
            log(f"SKIP {stem}: no .geff")
            continue
        out.append({"stem": stem, "zarr": zp, "geff": gp})
    # Case B: per-stem directories
    if not out:
        for d in sorted(train_dir.iterdir()):
            if not d.is_dir():
                continue
            stem = d.name
            if stem in FORBIDDEN_TEST_STEMS:
                continue
            assert_stem_allowed(stem, context="discover_dir")
            zp = d if (d / "0").exists() else (d / f"{stem}.zarr")
            gp = d / f"{stem}.geff" if (d / f"{stem}.geff").exists() else d.with_suffix(".geff")
            # also sibling
            if not gp.exists():
                gp = train_dir / f"{stem}.geff"
            if zp.exists() and gp.exists():
                out.append({"stem": stem, "zarr": zp, "geff": gp})
    return out


STEM_RECS = discover_stems(TRAIN_DIR)
log(f"permitted_stems_with_gt={len(STEM_RECS)}")
if STEM_RECS:
    log(f"sample_stems={[r['stem'] for r in STEM_RECS[:5]]}")

if REQUIRE_REAL and not STEM_RECS:
    # dump input tree for debug then fail — do NOT silently fall back to synthetic
    listing = []
    for base in Path("/kaggle/input").rglob("*"):
        if base.is_dir() and base.parent.name in ("input", "competitions", "datasets"):
            listing.append(str(base))
        if len(listing) > 40:
            break
    raise SystemExit(
        "REAL DATA REQUIRED but no permitted stem.zarr+.geff found. "
        f"TRAIN_DIR={TRAIN_DIR} listing_sample={listing}"
    )


# ---- GEFF cache ----
class StemCache:
    __slots__ = ("stem", "zarr_path", "geff_path", "ids", "t", "z", "y", "x", "parent_of", "by_t", "img_arr", "T", "Z", "Y", "X")

    def __init__(self, rec: dict):
        self.stem = rec["stem"]
        assert_stem_allowed(self.stem, context="StemCache")
        self.zarr_path = Path(rec["zarr"])
        self.geff_path = Path(rec["geff"])
        g = zarr.open(str(self.geff_path), mode="r")
        self.ids = np.asarray(g["nodes/ids"][:])
        self.t = np.asarray(g["nodes/props/t/values"][:], dtype=np.int64)
        self.z = np.asarray(g["nodes/props/z/values"][:], dtype=np.float64)
        self.y = np.asarray(g["nodes/props/y/values"][:], dtype=np.float64)
        self.x = np.asarray(g["nodes/props/x/values"][:], dtype=np.float64)
        edges = np.asarray(g["edges/ids"][:])
        id_to_i = {int(i): k for k, i in enumerate(self.ids)}
        # parent_of[child_index] = parent_index or -1
        self.parent_of = np.full(len(self.ids), -1, dtype=np.int64)
        for a, b in edges:
            ia, ib = id_to_i.get(int(a)), id_to_i.get(int(b))
            if ia is None or ib is None:
                continue
            # edge (parent, child), dt=1
            self.parent_of[ib] = ia
        self.by_t: dict[int, np.ndarray] = {}
        for ti in np.unique(self.t):
            self.by_t[int(ti)] = np.where(self.t == ti)[0]
        # open image zarr lazily
        zg = zarr.open(str(self.zarr_path), mode="r")
        if "0" in zg:
            self.img_arr = zg["0"]
        else:
            # some layouts store array at root
            keys = list(zg.keys()) if hasattr(zg, "keys") else []
            if keys:
                self.img_arr = zg[keys[0]]
            else:
                self.img_arr = zg
        shape = self.img_arr.shape
        if len(shape) == 4:
            self.T, self.Z, self.Y, self.X = map(int, shape)
        elif len(shape) == 5:
            # maybe (1,T,Z,Y,X)
            self.T, self.Z, self.Y, self.X = map(int, shape[1:])
            self.img_arr = self.img_arr[0]
        else:
            raise RuntimeError(f"unexpected image shape {shape} for {self.stem}")


_STEM_CACHE: dict[str, StemCache] = {}


def get_stem(rec: dict) -> StemCache:
    s = rec["stem"]
    if s not in _STEM_CACHE:
        log(f"loading stem {s} ...")
        _STEM_CACHE[s] = StemCache(rec)
        sc = _STEM_CACHE[s]
        log(f"  nodes={len(sc.ids)} T={sc.T} ZYX={sc.Z},{sc.Y},{sc.X} frames_with_nodes={len(sc.by_t)}")
    return _STEM_CACHE[s]


def normalize_patch(vol: np.ndarray) -> np.ndarray:
    # Always materialize float32 — zarr often yields float64; AMP requires fp32 inputs.
    v = np.ascontiguousarray(vol, dtype=np.float32)
    lo, hi = np.percentile(v, [1.0, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    v = (v - lo) / (hi - lo)
    return np.clip(v, 0.0, 1.0).astype(np.float32, copy=False)


def _frame_stats(sc: "StemCache") -> tuple[list[int], list[int], list[int]]:
    """Return (all_frames, dense_frames, div_frames) in-range."""
    frames = [t for t in sc.by_t if 0 <= t < sc.T]
    if not frames:
        return [], [], []
    counts = {t: len(sc.by_t[t]) for t in frames}
    med = float(np.median(list(counts.values()))) if counts else 0.0
    dense = [t for t in frames if counts[t] >= max(med * 1.25, med + 2)]
    # division: parent at t has 2+ children at t+1
    child_count: dict[int, int] = {}
    for ci, p in enumerate(sc.parent_of):
        if p < 0:
            continue
        child_count[int(p)] = child_count.get(int(p), 0) + 1
    div_parents = {p for p, c in child_count.items() if c >= 2}
    div_frames = []
    for t in frames:
        for i in sc.by_t[t]:
            if int(i) in div_parents:
                div_frames.append(t)
                break
    return frames, dense, div_frames


def real_batch(device: torch.device):
    """Sample a real GEFF-supervised patch (dense / division biased)."""
    if not STEM_RECS:
        raise RuntimeError("no real stems")
    rec = random.choice(STEM_RECS)
    sc = get_stem(rec)
    frames, dense_frames, div_frames = _frame_stats(sc)
    if not frames:
        raise RuntimeError(f"no in-range frames for {sc.stem}")
    r = random.random()
    if r < P_DIV and div_frames:
        t = random.choice(div_frames)
        mode = "div"
    elif r < P_DIV + P_DENSE and dense_frames:
        t = random.choice(dense_frames)
        mode = "dense"
    else:
        t = random.choice(frames)
        mode = "ord"
    idxs = sc.by_t[t]
    # pick anchor cell for patch center
    anchor = int(random.choice(idxs))
    cz0, cy0, cx0 = float(sc.z[anchor]), float(sc.y[anchor]), float(sc.x[anchor])
    # random jitter so we don't always center exactly on a cell
    cz0 += random.uniform(-2, 2)
    cy0 += random.uniform(-8, 8)
    cx0 += random.uniform(-8, 8)
    z0 = int(np.clip(round(cz0 - PATCH_Z / 2), 0, max(0, sc.Z - PATCH_Z)))
    y0 = int(np.clip(round(cy0 - PATCH_Y / 2), 0, max(0, sc.Y - PATCH_Y)))
    x0 = int(np.clip(round(cx0 - PATCH_X / 2), 0, max(0, sc.X - PATCH_X)))
    z1, y1, x1 = z0 + PATCH_Z, y0 + PATCH_Y, x0 + PATCH_X
    # load patch (T slice)
    try:
        patch = np.asarray(sc.img_arr[t, z0:z1, y0:y1, x0:x1])
    except Exception:
        # fallback full frame then crop
        frame = np.asarray(sc.img_arr[t])
        patch = frame[z0:z1, y0:y1, x0:x1]
    # pad if at border
    if patch.shape != (PATCH_Z, PATCH_Y, PATCH_X):
        pad = np.zeros((PATCH_Z, PATCH_Y, PATCH_X), dtype=patch.dtype)
        zz, yy, xx = patch.shape
        pad[:zz, :yy, :xx] = patch
        patch = pad
    img_np = normalize_patch(patch)

    # centers in patch coords + parents among in-patch nodes
    centers = []
    local_ids = []  # original indices
    for i in idxs:
        lz = float(sc.z[i]) - z0
        ly = float(sc.y[i]) - y0
        lx = float(sc.x[i]) - x0
        if 1 <= lz < PATCH_Z - 1 and 2 <= ly < PATCH_Y - 2 and 2 <= lx < PATCH_X - 2:
            centers.append([lz, ly, lx])
            local_ids.append(int(i))

    parents = []
    # map original index -> local center index
    o2l = {oid: li for li, oid in enumerate(local_ids)}
    for oid in local_ids:
        p = int(sc.parent_of[oid])
        if p >= 0 and p in o2l:
            # parent only valid if parent was also same-frame? NO — parent is previous frame
            # for Stage A flow: parent_index among centers list only works for same-patch same-frame siblings
            # For backward flow we need parent center in THIS patch at THIS frame? make_targets expects
            # parent among the centers list of THIS frame for flow from parent pos.
            # Real parent is at t-1, so for flow we approximate using same-frame only if parent also listed
            # Better: place synthetic parent offset using previous-frame parent position projected into patch
            parents.append(-1)  # fill below
        else:
            parents.append(-1)

    # recompute parents using prev-frame parent projected into patch (if in bounds)
    parents = []
    for oid in local_ids:
        p = int(sc.parent_of[oid])
        if p < 0:
            parents.append(-1)
            continue
        # parent world coords; if parent time is t-1, project into same patch space
        pz = float(sc.z[p]) - z0
        py = float(sc.y[p]) - y0
        px = float(sc.x[p]) - x0
        if 0 <= pz < PATCH_Z and 0 <= py < PATCH_Y and 0 <= px < PATCH_X:
            # add parent as a phantom center if not already present as a node this frame
            # find nearest local center to parent for parent_index; else append parent as extra center
            best_j, best_d = -1, 1e9
            for j, c in enumerate(centers):
                d = (c[0] - pz) ** 2 + (c[1] - py) ** 2 + (c[2] - px) ** 2
                if d < best_d:
                    best_d, best_j = d, j
            if best_d < 4.0:  # ~2 vox
                parents.append(best_j)
            else:
                centers.append([pz, py, px])
                parents.append(len(centers) - 1)
                # the child index already fixed; parent index is last
                # wait: parents list is aligned with local_ids only so far
                # fix: we need parents length == len(centers) at the end
                pass
        else:
            parents.append(-1)

    # parents currently length = len(local_ids); pad for any extra parent centers added
    while len(parents) < len(centers):
        parents.append(-1)

    if not centers:
        # empty patch — still train background
        centers = [[PATCH_Z / 2, PATCH_Y / 2, PATCH_X / 2]]
        parents = [-1]
        # zero out image slightly so heatmap empty-ish still has signal? keep image

    heat, offset, flow, ov, fv = make_targets(
        (PATCH_Z, PATCH_Y, PATCH_X), centers, parents, device=torch.device("cpu")
    )
    img = torch.from_numpy(np.ascontiguousarray(img_np, dtype=np.float32))[
        None, None
    ]  # 1,1,Z,Y,X
    # Explicit fp32 on device — never let float64 hit AMP/half conv weights
    return (
        img.to(device=device, dtype=torch.float32),
        heat.to(device=device, dtype=torch.float32),
        offset.to(device=device, dtype=torch.float32),
        flow.to(device=device, dtype=torch.float32),
        ov.to(device=device),
        fv.to(device=device),
        sc.stem,
        t,
        len(local_ids),
    )


def detection_loss(pred, heat, offset_pred, offset, ov, flow_pred, flow, fv):
    """Stable fp32 loss. Soft heads down-weighted; drop nonfinite terms."""
    pred = pred.float().clamp(-20.0, 20.0)
    heat = heat.float().clamp(0.0, 1.0)
    offset_pred = offset_pred.float().clamp(-8.0, 8.0)
    offset = offset.float().clamp(-8.0, 8.0)
    flow_pred = flow_pred.float().clamp(-30.0, 30.0)
    flow = flow.float().clamp(-30.0, 30.0)
    p = torch.sigmoid(pred).clamp(1e-4, 1.0 - 1e-4)
    eps = 1e-6
    pos = heat >= 0.5
    neg = ~pos
    # balanced BCE-focal (mean over all voxels — avoids empty-pos NaN)
    loss_hm = (
        -((1 - p) ** 2) * heat * torch.log(p + eps)
        - (p**2) * (1 - heat) * torch.log(1 - p + eps) * 0.1
    ).mean()
    loss_off = pred.new_tensor(0.0)
    loss_flow = pred.new_tensor(0.0)
    if ov.any():
        loss_off = F.smooth_l1_loss(offset_pred[:, ov], offset[:, ov], beta=0.25)
    if fv.any():
        loss_flow = F.huber_loss(flow_pred[:, fv], flow[:, fv], delta=1.0)
    # soft heads: small weight — heatmap drives adjEdge score
    w_off = 0.25 if torch.isfinite(loss_off) else 0.0
    w_flow = 0.10 if torch.isfinite(loss_flow) else 0.0
    if not torch.isfinite(loss_hm):
        # last resort: L2 to heatmap
        loss_hm = F.mse_loss(p, heat)
    total = loss_hm + w_off * loss_off + w_flow * loss_flow
    return total, {
        "hm": float(loss_hm.detach()) if torch.isfinite(loss_hm) else -1.0,
        "off": float(loss_off.detach()) if torch.isfinite(loss_off) else -1.0,
        "flow": float(loss_flow.detach()) if torch.isfinite(loss_flow) else -1.0,
    }


model = CellectLite(time_frames=1).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
start_step = 0
warm_from = None
resumed_from = None

# Resume / warm: known dataset roots only (never rglob competition)
warm_paths: list[Path] = []
_warm_names = (
    "lineage-v2-warm-a3",
    "lineage-v2-warm",
    "lineage-v2-reala3",
    "lineage-v2-realA3",
    "lineage-v2-realb1",
    "lineage-v2-realB1",
    "lineage-v2-reala2",
    "lineage-v2-realA2",
    "lineage-v2-reala1",
    "lineage-v2-realA1",
    "lineage-v2-warm-de1",
    "lineage-v2-stagede-de1",
)
for name in _warm_names:
    for user in ("mijuuu8", "kokiiii"):
        for base in (
            Path(f"/kaggle/input/datasets/{user}/{name}"),
            Path(f"/kaggle/input/{user}/{name}"),
        ):
            for fname in ("last.pt", "best_exact.pt"):
                p = base / fname
                if p.exists():
                    warm_paths.append(p)
    for base in (Path(f"/kaggle/input/{name}"),):
        for fname in ("last.pt", "best_exact.pt"):
            p = base / fname
            if p.exists():
                warm_paths.append(p)
if not warm_paths and Path("/kaggle/input/datasets").exists():
    warm_paths.extend(sorted(Path("/kaggle/input/datasets").rglob("last.pt")))
    warm_paths.extend(sorted(Path("/kaggle/input/datasets").rglob("best_exact.pt")))
warm_paths = sorted(
    warm_paths,
    key=lambda p: (
        0 if "reala" in str(p).lower() or "real-a" in str(p).lower() else 1,
        0 if "realb" in str(p).lower() else 2,
        0 if "geff" in str(p).lower() else 1,
        0 if "stagede" in str(p).lower() or "de1" in str(p).lower() else 3,
        0 if p.name == "last.pt" else 1,
        str(p),
    ),
)

for r in warm_paths:
    if not r.exists():
        continue
    try:
        state = torch.load(r, map_location="cpu", weights_only=False)
        if not isinstance(state, dict) or "model" not in state:
            continue
        missing, unexpected = model.load_state_dict(state["model"], strict=False)
        warm_from = str(r)
        cfg = state.get("config") or {}
        # Continue step count if prior real GEFF train — always FRESH optimizer
        # (resumed Adam moments + NaN history destabilizes real GEFF)
        if cfg.get("data_mode") == "geff_real" or "reala" in str(r).lower():
            start_step = int(state.get("global_step", 0))
            resumed_from = str(r)
            log(
                f"RESUME weights only step={start_step} from {r} "
                f"missing={len(missing)} unexpected={len(unexpected)} (fresh optimizer)"
            )
        else:
            start_step = 0
            log(
                f"WARM_START model from {r} missing={len(missing)} unexpected={len(unexpected)}"
            )
        break
    except Exception as e:
        log(f"warm skip {r}: {e}")

# working dir resume (rerun same kernel)
for r in [OUT / "last.pt"]:
    if not r.exists():
        continue
    try:
        state = torch.load(r, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and state.get("config", {}).get("data_mode") == "geff_real":
            model.load_state_dict(state["model"], strict=True)
            start_step = int(state.get("global_step", 0))
            resumed_from = str(r)
            log(f"RESUMED working export step={start_step}")
    except Exception as e:
        log(f"resume skip {r}: {e}")

end_step = min(MAX_STEPS, start_step + SESSION_STEPS)
log(
    f"SESSION={SESSION_ID} start_step={start_step} end_step={end_step} "
    f"warm_from={warm_from} resumed_from={resumed_from} n_stems={len(STEM_RECS)}"
)

best_loss = float("inf")
step = start_step
model.train()
stems_seen: set[str] = set()
consecutive_nonfinite = 0
MAX_CONSEC_NONFINITE = int(os.environ.get("BIOHUB_MAX_CONSEC_NONFINITE", "80"))
# Default AMP OFF on real GEFF — half precision was primary NaN source
USE_AMP = os.environ.get("BIOHUB_USE_AMP", "0") == "1"
log(f"TRAIN_LOOP_START data_mode=geff_real USE_AMP={USE_AMP}")
while step < end_step and remaining_h() > 0.25:
    try:
        img, heat, offset, flow, ov, fv, stem, t, n_cent = real_batch(DEVICE)
        stems_seen.add(stem)
    except ForbiddenStemError as e:
        raise
    except Exception as e:
        log(f"batch_fail: {e}")
        consecutive_nonfinite += 1
        if consecutive_nonfinite >= MAX_CONSEC_NONFINITE:
            log(f"ABORT too many batch fails at step={step}")
            break
        continue
    # Guard dtypes + sanitize
    img = torch.nan_to_num(img.float(), nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    heat = torch.nan_to_num(heat.float(), nan=0.0).clamp(0.0, 1.0)
    offset = torch.nan_to_num(offset.float(), nan=0.0).clamp(-8.0, 8.0)
    flow = torch.nan_to_num(flow.float(), nan=0.0, posinf=20.0, neginf=-20.0).clamp(-30.0, 30.0)
    if (not torch.isfinite(img).all()) or (not torch.isfinite(heat).all()):
        consecutive_nonfinite += 1
        continue
    opt.zero_grad(set_to_none=True)
    try:
        # Forward (optional AMP) — loss ALWAYS outside AMP in fp32
        with torch.amp.autocast("cuda", enabled=(USE_AMP and DEVICE.type == "cuda")):
            out = model(img)
            pred = out["center_logits"][0, 0]
            off_p = out["offset_vox"][0]
            fl_p = out["backward_flow_um"][0]
        pred = pred.float()
        off_p = off_p.float()
        fl_p = fl_p.float()
        if pred.shape != heat.shape:
            pred = F.interpolate(
                pred[None, None], size=heat.shape, mode="trilinear", align_corners=False
            )[0, 0]
            off_p = F.interpolate(
                off_p[None], size=heat.shape, mode="trilinear", align_corners=False
            )[0]
            fl_p = F.interpolate(
                fl_p[None], size=heat.shape, mode="trilinear", align_corners=False
            )[0]
        loss, parts = detection_loss(pred, heat, off_p, offset, ov, fl_p, flow, fv)
        if not torch.isfinite(loss):
            consecutive_nonfinite += 1
            if consecutive_nonfinite <= 5 or consecutive_nonfinite % 20 == 0:
                log(f"nonfinite loss at step {step}, skip (consec={consecutive_nonfinite})")
            if consecutive_nonfinite >= MAX_CONSEC_NONFINITE:
                log(f"ABORT: {MAX_CONSEC_NONFINITE} consecutive nonfinite — saving and exit")
                break
            continue
        if USE_AMP and DEVICE.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
        else:
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        grads_ok = True
        for p in model.parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all():
                grads_ok = False
                break
        if not grads_ok:
            consecutive_nonfinite += 1
            opt.zero_grad(set_to_none=True)
            if USE_AMP:
                scaler.update()
            if consecutive_nonfinite >= MAX_CONSEC_NONFINITE:
                log("ABORT nonfinite grads")
                break
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
        vram = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0
        log(
            f"step={step} loss={lv:.4f} best={best_loss:.4f} "
            f"hm={parts['hm']:.4f} off={parts['off']:.4f} flow={parts['flow']:.4f} "
            f"stem={stem} t={t} n_cent={n_cent} stems_seen={len(stems_seen)} "
            f"rem_h={remaining_h():.2f} vram_gb={vram:.2f}"
        )
    if step % SAVE_EVERY == 0 or remaining_h() < 0.30:
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
                "time_budget_h": TIME_BUDGET_H,
                "patch": [PATCH_Z, PATCH_Y, PATCH_X],
                "sigma_um": SIGMA_UM,
                "spacing": list(SPACING_ZYX_UM),
                "lr": LR,
                "seed": SEED,
                "data_mode": "geff_real",
                "n_stems": len(STEM_RECS),
                "stems_seen": len(stems_seen),
                "warm_from": warm_from,
                "resumed_from": resumed_from,
                "account_role": "train_worker",
            },
            split_sha256="real_train_denylist_v1",
            code_sha="lineage_v2_stageA_realA1",
            inference_policy_sha256="train_only_no_submit",
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
            "account_role": "train_worker",
            "session": SESSION_ID,
            "data_mode": "geff_real",
            "n_stems": len(STEM_RECS),
            "stems_seen": sorted(stems_seen)[:50],
            "warm_from": warm_from,
        }
        (OUT / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        log(f"saved export step={step}")

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
        "data_mode": "geff_real",
        "steps": step,
        "warm_from": warm_from,
    },
    split_sha256="real_train_denylist_v1",
    code_sha="lineage_v2_stageA_realA1",
    inference_policy_sha256="train_only_no_submit",
)
save_checkpoint(state, OUT / "last.pt")
save_checkpoint(state, OUT / "best_exact.pt")
final = {
    "status": "COMPLETE",
    "stage": "A",
    "session": SESSION_ID,
    "steps": step,
    "start_step": start_step,
    "end_step_target": end_step,
    "best_loss": best_loss,
    "elapsed_h": (time.time() - T0) / 3600,
    "time_budget_h": TIME_BUDGET_H,
    "data_mode": "geff_real",
    "n_stems": len(STEM_RECS),
    "stems_seen": len(stems_seen),
    "warm_from": warm_from,
    "resumed_from": resumed_from,
    "stopped_reason": "session_steps" if step >= end_step else "time_budget",
    "no_submit": True,
    "account_role": "train_worker",
    "next": "version realA1 weights → continue realA2 or Stage B on real GT nodes",
}
(OUT / "session_complete.json").write_text(json.dumps(final, indent=2) + "\n")
log(f"SESSION_DONE {final}")
