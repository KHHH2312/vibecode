#!/usr/bin/env python3
"""Generate bh-ens-screen-v1.ipynb: A/B screen of an edge-logit ensemble.

Reuses the proven infra cells from kernel_verify/bh-v100b-verify.ipynb
(constants, deps+repo materialize, patched-metric bundle, localval symlink)
and adds:
  - an ensemble text-patch to predict_unet_transformer.py (load N edge
    predictors; detect with model[0]; average sigmoid(predict_edges) across
    all models), and
  - an A/B driver that runs predict once single (50ep) and once ensemble
    ({50ep,350ep,v34}) on the SAME localval train movies, then scores the
    raw-ILP geffs directly with the patched metric (edge_jaccard).

The node set is identical across arms (detection = model[0] only), so the
edge_jaccard delta isolates the edge predictor.
"""
import json
from pathlib import Path

REPO = Path("/home/user/vibecode")
VERIFY_NB = REPO / "kernel_verify" / "bh-v100b-verify.ipynb"
LOCAL_PREDICT = Path(
    "/tmp/claude-0/-home-user-vibecode/27038f89-9d1c-5417-ac20-81af2582a6f1"
    "/scratchpad/divsweep_out/tracking_repo/scripts/predict_unet_transformer.py"
)
OUT_DIR = REPO / "kernel_ensemble"
OUT_NB = OUT_DIR / "bh-ens-screen-v1.ipynb"

# 6 train movies with GT (3x 44b6 + 3x 6bba), big->small spread.
# Excludes the 4 forbidden test stems and the overlap 6bba_07e24132.
SCREEN_IDS = [
    "6bba_57b7cc1e",  # 75802 nodes (dominant)
    "44b6_12dfb391",  # 46434
    "44b6_d5e7d891",  # 43824
    "6bba_337b1b3a",  # 29021
    "44b6_0c582fdc",  # 27382
    "6bba_062c8d37",  # 5976 (small)
]

# ----------------------------------------------------------------------------
# Ensemble text-patch anchors (must match predict_unet_transformer.py exactly).
# ----------------------------------------------------------------------------
ANCHOR_GLOBAL = "_DEFAULT_CONFIG = {"
REPLACE_GLOBAL = "ENSEMBLE_MODELS = None\n\n\n_DEFAULT_CONFIG = {"

ANCHOR_LOAD = "    model, window_size, downsample = load_model(weights_path, device)\n"
REPLACE_LOAD = (
    "    model, window_size, downsample = load_model(weights_path, device)\n"
    "    global ENSEMBLE_MODELS\n"
    "    ENSEMBLE_MODELS = [model]\n"
    "    _ens_env = os.environ.get('BIOHUB_ENSEMBLE_WEIGHTS', '').strip()\n"
    "    if _ens_env:\n"
    "        for _wp in _ens_env.split(','):\n"
    "            _wp = _wp.strip()\n"
    "            if not _wp:\n"
    "                continue\n"
    "            _m, _ws, _ds = load_model(Path(_wp), device)\n"
    "            assert _ws == window_size and tuple(_ds) == tuple(downsample), \\\n"
    "                f'ensemble arch mismatch: {_wp} ws={_ws} ds={_ds}'\n"
    "            ENSEMBLE_MODELS.append(_m)\n"
    "        print(f'[ensemble] {len(ENSEMBLE_MODELS)} models (detector=base); extra={_ens_env}', flush=True)\n"
)

ANCHOR_DELIMGS = (
    "        del imgs\n\n"
    "        # --- Detect cells in each frame (dedup across windows) ---\n"
)
REPLACE_DELIMGS = (
    "        if ENSEMBLE_MODELS is None or len(ENSEMBLE_MODELS) <= 1:\n"
    "            del imgs\n\n"
    "        # --- Detect cells in each frame (dedup across windows) ---\n"
)

ANCHOR_EDGE = (
    "            unet_feat_src = model._index_features(\n"
    "                unet_out[:, f_idx], p_coords_src, p_mask_src,\n"
    "            )\n"
    "            unet_feat_tgt = model._index_features(\n"
    "                unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt,\n"
    "            )\n"
    "            edge_logits_pair = model.predict_edges(\n"
    "                unet_feat_src, unet_feat_tgt,\n"
    "                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n"
    "                p_pos_src, p_pos_tgt,\n"
    "                p_mask_src, p_mask_tgt,\n"
    "            )  # (1, n_src, n_tgt)\n"
    "\n"
    "            raw = edge_logits_pair[0]\n"
    "            if cfg.edge_activation == \"softmax\":\n"
    "                probs = torch.softmax(raw, dim=0).cpu().numpy()\n"
    "            else:\n"
    "                probs = torch.sigmoid(raw).cpu().numpy()\n"
)
REPLACE_EDGE = (
    "            _ens_models = ENSEMBLE_MODELS if ENSEMBLE_MODELS else [model]\n"
    "            _probs_sum = None\n"
    "            for _mi, _m in enumerate(_ens_models):\n"
    "                if _mi == 0:\n"
    "                    _uo_s = unet_out[:, f_idx]\n"
    "                    _uo_t = unet_out[:, f_idx + 1]\n"
    "                else:\n"
    "                    _uo_full = _m.encode(imgs)[0]\n"
    "                    _uo_s = _uo_full[:, f_idx]\n"
    "                    _uo_t = _uo_full[:, f_idx + 1]\n"
    "                _feat_src = _m._index_features(_uo_s, p_coords_src, p_mask_src)\n"
    "                _feat_tgt = _m._index_features(_uo_t, p_coords_tgt, p_mask_tgt)\n"
    "                _lg = _m.predict_edges(\n"
    "                    _feat_src, _feat_tgt,\n"
    "                    p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n"
    "                    p_pos_src, p_pos_tgt,\n"
    "                    p_mask_src, p_mask_tgt,\n"
    "                )\n"
    "                _raw = _lg[0]\n"
    "                if cfg.edge_activation == \"softmax\":\n"
    "                    _p = torch.softmax(_raw, dim=0)\n"
    "                else:\n"
    "                    _p = torch.sigmoid(_raw)\n"
    "                _probs_sum = _p if _probs_sum is None else _probs_sum + _p\n"
    "                if _mi != 0:\n"
    "                    del _uo_full\n"
    "            probs = (_probs_sum / len(_ens_models)).cpu().numpy()\n"
)

PATCHES = [
    ("module global ENSEMBLE_MODELS", ANCHOR_GLOBAL, REPLACE_GLOBAL),
    ("load ensemble models", ANCHOR_LOAD, REPLACE_LOAD),
    ("keep imgs alive", ANCHOR_DELIMGS, REPLACE_DELIMGS),
    ("average edge probs", ANCHOR_EDGE, REPLACE_EDGE),
]


def validate_anchors():
    src = LOCAL_PREDICT.read_text()
    ok = True
    for name, anchor, _ in PATCHES:
        n = src.count(anchor)
        status = "OK" if n == 1 else f"!! FOUND {n}x"
        if n != 1:
            ok = False
        print(f"  [{status}] {name}")
    return ok


# ----------------------------------------------------------------------------
# New cells.
# ----------------------------------------------------------------------------
CELL_CONFIG = f'''# =============================================================================
# bh-ens-screen-v1 - edge-logit ENSEMBLE A/B screen (NO exploit)
# =============================================================================
# Question: does averaging edge probabilities across the 3 distinct public
# edge predictors (50ep / 350ep / v34, all same architecture) raise edge_jaccard
# vs the single 50ep model?  Detection is fixed to model[0] (50ep) so the node
# set is identical across arms and the edge_jaccard delta isolates the edge
# predictor.  Scored raw-ILP (pre-PP) with the bundled PATCHED metric.
# =============================================================================
import os

# --- localval: score on TRAIN movies with GT (never the 4 test stems) ---
os.environ["BIOHUB_TEST_DIR"] = "/kaggle/working/localval"
LOCALVAL_IDS = {SCREEN_IDS!r}

# --- detection / ILP: match the bank (edge predictor is the only variable) ---
os.environ["BIOHUB_DET_THRESHOLD"] = "0.9725"
os.environ["BIOHUB_USE_ILP"] = "1"
os.environ["BIOHUB_ILP_EDGE_WEIGHT"] = "-1.0"
os.environ["BIOHUB_ILP_APPEARANCE_WEIGHT"] = "0.1"
os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "0.1"
os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "1.0"
os.environ["BIOHUB_UNET_BATCH_SIZE"] = "4"
os.environ["BIOHUB_ALLOW_PIP_INSTALL"] = "0"
os.environ["BIOHUB_ALLOW_ARTIFACT_FALLBACK"] = "0"
os.environ["BIOHUB_RUN_VISUAL_EDA"] = "0"
os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "0"

print("=" * 72)
print("bh-ens-screen-v1 | ENSEMBLE A/B | detector=50ep | edges=avg(50ep,350ep,v34)")
print("localval movies:", LOCALVAL_IDS)
print("=" * 72)
'''

CELL_PATCH = '''# ==================== ENSEMBLE PATCH to predict_unet_transformer.py ====================
# Loads N edge predictors from BIOHUB_ENSEMBLE_WEIGHTS (comma-separated), keeps
# detection on model[0], and averages sigmoid(predict_edges) across all models.
# When BIOHUB_ENSEMBLE_WEIGHTS is unset -> identical to the stock single-model path.
_ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"
_s = _ps.read_text()

_PATCHES = [
%PATCHLIST%
]
for _name, _anchor, _repl in _PATCHES:
    _n = _s.count(_anchor)
    assert _n == 1, f"ENSEMBLE PATCH anchor not unique ({_n}x): {_name}"
    _s = _s.replace(_anchor, _repl)
_ps.write_text(_s)
print("ensemble patch applied:", [p[0] for p in _PATCHES])

# sanity: the module now defines ENSEMBLE_MODELS and the averaging loop
assert "ENSEMBLE_MODELS = None" in _s and "_probs_sum" in _s, "ensemble patch incomplete"
'''

CELL_AB = '''# ==================== A/B: run predict single vs ensemble, score raw-ILP ====================
import subprocess, sys, time, shutil, json as _json
from pathlib import Path

def _find_weight(slug_substr):
    root = Path("/kaggle/input")
    if not root.exists():
        return None
    for p in sorted(root.rglob("edge_predictor_best.pth")):
        if slug_substr in str(p):
            return p
    return None

W_350 = _find_weight("350ep-checkpoint-pin")
W_V34 = _find_weight("v34-retrain-weights-mirror")
print("[ens] 350ep weights:", W_350)
print("[ens] v34   weights:", W_V34)
assert W_350 is not None and W_V34 is not None, "missing 350ep/v34 ensemble weights"
ENS_WEIGHTS = f"{W_350},{W_V34}"

SCREEN_STEMS = list(LOCALVAL_IDS)
splits_path = REPO_DIR / "kaggle_screen_splits.json"
splits_path.write_text(_json.dumps([{"split": 0, "train": [], "test": SCREEN_STEMS}], indent=2))

def _build_cmd():
    cmd = [
        sys.executable, "scripts/predict_unet_transformer.py",
        "--data-dir", str(TEST_DIR),
        "--splits", splits_path.name,
        "--split", "0",
        "--weights", WEIGHTS_RELATIVE,
        "--unet-batch-size", str(UNET_BATCH_SIZE),
        "--det-threshold", str(DET_THRESHOLD),
        "--ilp-edge-weight", str(ILP_EDGE_WEIGHT),
        "--ilp-appearance-weight", str(ILP_APPEARANCE_WEIGHT),
        "--ilp-disappearance-weight", str(ILP_DISAPPEARANCE_WEIGHT),
        "--ilp-division-weight", str(ILP_DIVISION_WEIGHT),
    ]
    if USE_ILP:
        cmd.append("--use-ilp")
    return cmd

def run_predict(tag, ens_weights):
    env = {**os.environ, "PYTHONPATH": "src"}
    if ens_weights:
        env["BIOHUB_ENSEMBLE_WEIGHTS"] = ens_weights
    else:
        env.pop("BIOHUB_ENSEMBLE_WEIGHTS", None)
    t0 = time.time()
    print(f"\\n[{tag}] launching predict (ensemble={bool(ens_weights)}) ...", flush=True)
    subprocess.run(_build_cmd(), cwd=REPO_DIR, env=env, check=True)
    dt = (time.time() - t0) / 60.0
    geffs = sorted(Path("/kaggle/working").rglob("*/split_0/*.geff"))
    if not geffs:
        geffs = sorted(REPO_DIR.rglob("*/split_0/*.geff"))
    dst = Path(f"/kaggle/working/geffs_{tag}")
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    for g in geffs:
        d = dst / g.name
        if g.is_dir():
            shutil.copytree(g, d)
        else:
            shutil.copy2(g, d)
    print(f"[{tag}] predict done in {dt:.1f} min | {len(geffs)} geffs -> {dst}", flush=True)
    return dst

DIR_SINGLE = run_predict("single", "")
DIR_ENS = run_predict("ens", ENS_WEIGHTS)
print("\\nA/B predict complete.")
'''

CELL_SCORE = '''# ==================== SCORE raw-ILP geffs with PATCHED metric ====================
import importlib, sys as _sys, math, traceback
for _m in [m for m in list(_sys.modules) if m == "tracking_cellmot" or m.startswith("tracking_cellmot.")]:
    _sys.modules.pop(_m, None)
from tracking_cellmot.metrics import (
    evaluate as P_eval,
    per_sample_metrics as P_psm,
    summarise as P_sum,
    node_recall as P_nr,
)
from geff import GeffMetadata
import tracksdata as td
import polars as pl

VOXEL_SCALE_UM = (1.625, 0.40625, 0.40625)
_TRAIN = COMP_DIR / "train"

def graph_from_geff(path):
    g = td.graph.IndexedRXGraph.from_geff(path)
    return g[0] if isinstance(g, tuple) else g

def build_inmem(pred_geff):
    gp = graph_from_geff(pred_geff)
    g = td.graph.InMemoryGraph()
    for key in ["z", "y", "x"]:
        g.add_node_attr_key(key, pl.Float64, -999999.0)
    nlist = []
    csvids = []
    for row in gp.node_attrs().iter_rows(named=True):
        csvids.append(int(row["node_id"]))
        nlist.append({"t": int(row["t"]), "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"])})
    gids = g.bulk_add_nodes(nlist)
    id2g = dict(zip(csvids, gids))
    elist = []
    for row in gp.edge_attrs().iter_rows(named=True):
        elist.append({"source_id": id2g[int(row["source_id"])], "target_id": id2g[int(row["target_id"])]})
    if elist:
        g.bulk_add_edges(elist)
    return g

def score_dir(geff_dir):
    rows = []
    per = {}
    for stem in LOCALVAL_IDS:
        try:
            pg = geff_dir / f"{stem}.geff"
            gt_geff = _TRAIN / f"{stem}.geff"
            if not pg.exists() or not gt_geff.exists():
                print(f"  {stem}: missing pred/gt, skip", flush=True)
                continue
            gt = graph_from_geff(gt_geff)
            n_total = float(GeffMetadata.read(str(gt_geff)).extra["estimated_number_of_nodes"])
            g = build_inmem(pg)
            er = P_eval(g, gt, scale=VOXEL_SCALE_UM, max_distance=7.0)
            rec = P_nr(g, gt)
            psm = P_psm(er=er, n_total=n_total, node_recall=rec)
            eden = er.edge_tp + er.edge_fp + er.edge_fn
            edgeJ = er.edge_tp / eden if eden > 0 else float("nan")
            per[stem] = {
                "edgeJ": edgeJ,
                "adjEdgeJ": psm["adj_edge_jaccard"],
                "eTP": er.edge_tp, "eFP": er.edge_fp, "eFN": er.edge_fn,
                "Npred": er.num_pred_nodes,
            }
            rows.append(psm)
        except Exception as e:
            print(f"  {stem}: FAIL {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
    s = P_sum(rows) if rows else None
    return per, s

print("scoring SINGLE ...", flush=True)
PER_S, S_S = score_dir(DIR_SINGLE)
print("scoring ENSEMBLE ...", flush=True)
PER_E, S_E = score_dir(DIR_ENS)

print("\\n" + "=" * 78)
print("ENSEMBLE SCREEN - raw-ILP edge_jaccard (patched metric), detector fixed=50ep")
print("=" * 78)
print(f"{'stem':20s} {'single':>9s} {'ensemble':>9s} {'delta':>8s}   (eTP/eFP/eFN single -> ens)")
for stem in LOCALVAL_IDS:
    if stem in PER_S and stem in PER_E:
        a = PER_S[stem]; b = PER_E[stem]
        print(f"{stem:20s} {a['edgeJ']:9.4f} {b['edgeJ']:9.4f} {b['edgeJ']-a['edgeJ']:+8.4f}   "
              f"{a['eTP']}/{a['eFP']}/{a['eFN']} -> {b['eTP']}/{b['eFP']}/{b['eFN']}")
if S_S and S_E:
    print("-" * 78)
    print(f"MICRO-AVG edge_jaccard : single={S_S['edge_jaccard']:.4f}  ensemble={S_E['edge_jaccard']:.4f}  "
          f"delta={S_E['edge_jaccard']-S_S['edge_jaccard']:+.4f}")
    print(f"MICRO-AVG adj_edge_jac : single={S_S['adj_edge_jaccard']:.4f}  ensemble={S_E['adj_edge_jaccard']:.4f}")
    print(f"MICRO-AVG score        : single={S_S['score']:.4f}  ensemble={S_E['score']:.4f}")
    print("-" * 78)
    _d = S_E['edge_jaccard'] - S_S['edge_jaccard']
    if _d > 0.001:
        print(f">>> ENSEMBLE HELPS edge_jaccard by {_d:+.4f}. Worth a full bank-PP submit run.")
    elif _d < -0.001:
        print(f">>> ENSEMBLE HURTS edge_jaccard by {_d:+.4f}. Dead lever; keep single 50ep.")
    else:
        print(f">>> ENSEMBLE NEUTRAL ({_d:+.4f}). Not worth the extra compute.")
print("SCREEN DONE")
'''


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def main():
    print("Validating ensemble patch anchors against local predict script:")
    if not validate_anchors():
        raise SystemExit("ANCHOR VALIDATION FAILED - fix anchors before generating")

    verify = json.loads(VERIFY_NB.read_text())
    vcells = verify["cells"]
    # verify cell indices: 3=constants, 5=deps+repo, 6=metric bundle, 7=symlink
    c_constants = "".join(vcells[3]["source"])
    c_deps = "".join(vcells[5]["source"])
    c_metric = "".join(vcells[6]["source"])
    c_symlink = "".join(vcells[7]["source"])

    # build the patch-list literal for CELL_PATCH
    def pyrepr(s):
        return repr(s)

    patchlist_lines = []
    for name, anchor, repl in PATCHES:
        patchlist_lines.append(f"    ({pyrepr(name)}, {pyrepr(anchor)}, {pyrepr(repl)}),")
    patch_cell = CELL_PATCH.replace("%PATCHLIST%", "\n".join(patchlist_lines))

    cells = [
        code(CELL_CONFIG),
        md("## Constants (reused verbatim from verified 0.900 notebook)"),
        code(c_constants),
        md("## Dependencies + inference repo + 50ep weights (reused verbatim)"),
        code(c_deps),
        md("## Patched metric bundle == Monday re-score (reused verbatim)"),
        code(c_metric),
        md("## Symlink localval TRAIN movies (GT available)"),
        code(c_symlink),
        md("## Apply ensemble patch to predict script"),
        code(patch_cell),
        md("## A/B: single (50ep) vs ensemble (50ep+350ep+v34)"),
        code(CELL_AB),
        md("## Score raw-ILP geffs with the patched metric"),
        code(CELL_SCORE),
    ]

    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1))
    print(f"\nWrote {OUT_NB} ({len(cells)} cells)")

    meta = {
        "id": "khalid000000/bh-ens-screen-v1",
        "title": "bh-ens-screen-v1",
        "code_file": "bh-ens-screen-v1.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "hongdaekim/biohub-350ep-checkpoint-pin-v1",
            "subinium/biohub-v34-retrain-weights-mirror",
        ],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_DIR / 'kernel-metadata.json'}")


if __name__ == "__main__":
    main()
