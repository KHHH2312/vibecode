#!/usr/bin/env python3
"""Generate bh-edgetta-screen-v1.ipynb: A/B screen of EDGE-level TTA.

The multi-model ensemble failed by dilution (350ep/v34 are weaker). This screen
tests the non-diluting alternative: sharpen the single best 50ep edge predictor
by averaging its edge probabilities over spatial-flip augmentations (y, x, xy).
Coords live in the downsampled unet_out grid, so a flip is coord_y -> (Yd-1-y)
etc., applied consistently to imgs, indexing coords, scaled coords, and pos
features. Detection stays single-pass 50ep (same node set across arms), so the
edge_jaccard delta isolates the edge-TTA effect.

Reuses the screen infra cells (constants, deps+repo, metric bundle, symlink)
from bh-ens-screen-v1 style; A/B toggles BIOHUB_EDGE_TTA_FLIPS.
"""
import json
from pathlib import Path

import gen_ensemble_screen as screen  # reuse infra cell extraction + ANCHOR_EDGE/DELIMGS

REPO = Path("/home/user/vibecode")
VERIFY_NB = REPO / "kernel_verify" / "bh-v100b-verify.ipynb"
LOCAL_PREDICT = Path(
    "/tmp/claude-0/-home-user-vibecode/27038f89-9d1c-5417-ac20-81af2582a6f1"
    "/scratchpad/divsweep_out/tracking_repo/scripts/predict_unet_transformer.py"
)
OUT_DIR = REPO / "kernel_edgetta"
OUT_NB = OUT_DIR / "bh-edgetta-screen-v1.ipynb"

SCREEN_IDS = [
    "6bba_57b7cc1e", "44b6_12dfb391", "44b6_d5e7d891",
    "6bba_337b1b3a", "44b6_0c582fdc", "6bba_062c8d37",
]

# --- patch 1: keep imgs alive when edge-TTA is on (reuse ensemble anchor) ---
ANCHOR_DELIMGS = screen.ANCHOR_DELIMGS
REPLACE_DELIMGS = (
    "        if not os.environ.get('BIOHUB_EDGE_TTA_FLIPS', '').strip():\n"
    "            del imgs\n\n"
    "        # --- Detect cells in each frame (dedup across windows) ---\n"
)

# --- patch 2: edge block -> average model edge probs over spatial flips ---
ANCHOR_EDGE = screen.ANCHOR_EDGE
REPLACE_EDGE = (
    "            _tta_env = os.environ.get('BIOHUB_EDGE_TTA_FLIPS', '').strip()\n"
    "            _flip_list = [f.strip() for f in _tta_env.split(',') if f.strip()] if _tta_env else []\n"
    "            _Yd = unet_out.shape[-2]\n"
    "            _Xd = unet_out.shape[-1]\n"
    "            # identity view (== stock single-model path)\n"
    "            _uf_s = model._index_features(unet_out[:, f_idx], p_coords_src, p_mask_src)\n"
    "            _uf_t = model._index_features(unet_out[:, f_idx + 1], p_coords_tgt, p_mask_tgt)\n"
    "            _lg = model.predict_edges(\n"
    "                _uf_s, _uf_t,\n"
    "                p_coords_src * ds_arr_t, p_coords_tgt * ds_arr_t,\n"
    "                p_pos_src, p_pos_tgt, p_mask_src, p_mask_tgt,\n"
    "            )\n"
    "            _rw = _lg[0]\n"
    "            _probs_sum = torch.softmax(_rw, dim=0) if cfg.edge_activation == \"softmax\" else torch.sigmoid(_rw)\n"
    "            _nviews = 1\n"
    "            for _fl in _flip_list:\n"
    "                _fy = 'y' in _fl\n"
    "                _fx = 'x' in _fl\n"
    "                _imgs_f = imgs\n"
    "                if _fy:\n"
    "                    _imgs_f = _imgs_f.flip(-2)\n"
    "                if _fx:\n"
    "                    _imgs_f = _imgs_f.flip(-1)\n"
    "                _uo_f = model.encode(_imgs_f)[0]\n"
    "                _cs = c_src.copy()\n"
    "                _ct = c_tgt.copy()\n"
    "                if _fy:\n"
    "                    _cs[:, 2] = (_Yd - 1) - _cs[:, 2]\n"
    "                    _ct[:, 2] = (_Yd - 1) - _ct[:, 2]\n"
    "                if _fx:\n"
    "                    _cs[:, 3] = (_Xd - 1) - _cs[:, 3]\n"
    "                    _ct[:, 3] = (_Xd - 1) - _ct[:, 3]\n"
    "                _pcs = torch.from_numpy(_cs[:, 1:].astype(np.float32)).unsqueeze(0).to(device)\n"
    "                _pct = torch.from_numpy(_ct[:, 1:].astype(np.float32)).unsqueeze(0).to(device)\n"
    "                _csr = _cs.copy(); _csr[:, 0] = f_idx\n"
    "                _ctr = _ct.copy(); _ctr[:, 0] = f_idx + 1\n"
    "                _pps = torch.from_numpy(extract_pos_features(_csr, window_shape)).unsqueeze(0).to(device)\n"
    "                _ppt = torch.from_numpy(extract_pos_features(_ctr, window_shape)).unsqueeze(0).to(device)\n"
    "                _uf_s = model._index_features(_uo_f[:, f_idx], _pcs, p_mask_src)\n"
    "                _uf_t = model._index_features(_uo_f[:, f_idx + 1], _pct, p_mask_tgt)\n"
    "                _lg = model.predict_edges(\n"
    "                    _uf_s, _uf_t,\n"
    "                    _pcs * ds_arr_t, _pct * ds_arr_t,\n"
    "                    _pps, _ppt, p_mask_src, p_mask_tgt,\n"
    "                )\n"
    "                _rw = _lg[0]\n"
    "                _pp = torch.softmax(_rw, dim=0) if cfg.edge_activation == \"softmax\" else torch.sigmoid(_rw)\n"
    "                _probs_sum = _probs_sum + _pp\n"
    "                _nviews += 1\n"
    "                del _uo_f\n"
    "            probs = (_probs_sum / _nviews).cpu().numpy()\n"
)

PATCHES = [
    ("keep imgs alive (edge-TTA)", ANCHOR_DELIMGS, REPLACE_DELIMGS),
    ("edge-TTA average", ANCHOR_EDGE, REPLACE_EDGE),
]


def validate_anchors():
    src = LOCAL_PREDICT.read_text()
    ok = True
    for name, anchor, _ in PATCHES:
        n = src.count(anchor)
        print(f"  [{'OK' if n == 1 else f'!! {n}x'}] {name}")
        if n != 1:
            ok = False
    return ok


CELL_CONFIG = f'''# =============================================================================
# bh-edgetta-screen-v1 - EDGE-level TTA A/B screen (NO exploit)
# =============================================================================
# The multi-model ensemble failed by dilution. This tests the non-diluting
# alternative: average the single 50ep edge predictor's probabilities over
# spatial-flip augmentations (y, x, xy). Detection stays single-pass 50ep so the
# node set is identical across arms; the edge_jaccard delta isolates edge-TTA.
# =============================================================================
import os
os.environ["BIOHUB_TEST_DIR"] = "/kaggle/working/localval"
LOCALVAL_IDS = {SCREEN_IDS!r}
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
print("bh-edgetta-screen-v1 | single 50ep vs 50ep + edge-TTA(y,x,xy)")
print("localval:", LOCALVAL_IDS)
'''

CELL_PATCH = '''# ==================== EDGE-TTA PATCH to predict_unet_transformer.py ====================
_ps = REPO_DIR / "scripts" / "predict_unet_transformer.py"
_s = _ps.read_text()
_PATCHES = [
%PATCHLIST%
]
for _name, _anchor, _repl in _PATCHES:
    _n = _s.count(_anchor)
    assert _n == 1, f"EDGE-TTA anchor not unique ({_n}x): {_name}"
    _s = _s.replace(_anchor, _repl)
_ps.write_text(_s)
print("edge-TTA patch applied:", [p[0] for p in _PATCHES])
assert "_flip_list" in _s and "BIOHUB_EDGE_TTA_FLIPS" in _s, "edge-TTA patch incomplete"
'''

CELL_AB = '''# ==================== A/B: single vs edge-TTA, score raw-ILP ====================
import subprocess, sys, time, shutil, json as _json
from pathlib import Path

SCREEN_STEMS = list(LOCALVAL_IDS)
splits_path = REPO_DIR / "kaggle_edgetta_splits.json"
splits_path.write_text(_json.dumps([{"split": 0, "train": [], "test": SCREEN_STEMS}], indent=2))

def _build_cmd():
    cmd = [
        sys.executable, "scripts/predict_unet_transformer.py",
        "--data-dir", str(TEST_DIR), "--splits", splits_path.name, "--split", "0",
        "--weights", WEIGHTS_RELATIVE, "--unet-batch-size", str(UNET_BATCH_SIZE),
        "--det-threshold", str(DET_THRESHOLD),
        "--ilp-edge-weight", str(ILP_EDGE_WEIGHT),
        "--ilp-appearance-weight", str(ILP_APPEARANCE_WEIGHT),
        "--ilp-disappearance-weight", str(ILP_DISAPPEARANCE_WEIGHT),
        "--ilp-division-weight", str(ILP_DIVISION_WEIGHT),
    ]
    if USE_ILP:
        cmd.append("--use-ilp")
    return cmd

def run_predict(tag, tta_flips):
    env = {**os.environ, "PYTHONPATH": "src"}
    if tta_flips:
        env["BIOHUB_EDGE_TTA_FLIPS"] = tta_flips
    else:
        env.pop("BIOHUB_EDGE_TTA_FLIPS", None)
    t0 = time.time()
    print(f"\\n[{tag}] predict (edge_tta={tta_flips or 'off'}) ...", flush=True)
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
        shutil.copytree(g, d) if g.is_dir() else shutil.copy2(g, d)
    print(f"[{tag}] done {dt:.1f} min | {len(geffs)} geffs -> {dst}", flush=True)
    return dst

DIR_SINGLE = run_predict("single", "")
DIR_TTA = run_predict("tta", "y,x,xy")
print("\\nA/B predict complete.")
'''

CELL_SCORE = '''# ==================== SCORE raw-ILP geffs with PATCHED metric ====================
import sys as _sys, traceback
for _m in [m for m in list(_sys.modules) if m == "tracking_cellmot" or m.startswith("tracking_cellmot.")]:
    _sys.modules.pop(_m, None)
from tracking_cellmot.metrics import (
    evaluate as P_eval, per_sample_metrics as P_psm, summarise as P_sum, node_recall as P_nr,
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
    nlist, csvids = [], []
    for row in gp.node_attrs().iter_rows(named=True):
        csvids.append(int(row["node_id"]))
        nlist.append({"t": int(row["t"]), "z": float(row["z"]), "y": float(row["y"]), "x": float(row["x"])})
    gids = g.bulk_add_nodes(nlist)
    id2g = dict(zip(csvids, gids))
    elist = [{"source_id": id2g[int(r["source_id"])], "target_id": id2g[int(r["target_id"])]}
             for r in gp.edge_attrs().iter_rows(named=True)]
    if elist:
        g.bulk_add_edges(elist)
    return g

def score_dir(geff_dir):
    rows, per = [], {}
    for stem in LOCALVAL_IDS:
        try:
            pg = geff_dir / f"{stem}.geff"
            gt_geff = _TRAIN / f"{stem}.geff"
            if not pg.exists() or not gt_geff.exists():
                print(f"  {stem}: missing, skip", flush=True); continue
            gt = graph_from_geff(gt_geff)
            n_total = float(GeffMetadata.read(str(gt_geff)).extra["estimated_number_of_nodes"])
            g = build_inmem(pg)
            er = P_eval(g, gt, scale=VOXEL_SCALE_UM, max_distance=7.0)
            rec = P_nr(g, gt)
            psm = P_psm(er=er, n_total=n_total, node_recall=rec)
            eden = er.edge_tp + er.edge_fp + er.edge_fn
            per[stem] = {"edgeJ": er.edge_tp / eden if eden else float("nan"),
                         "eTP": er.edge_tp, "eFP": er.edge_fp, "eFN": er.edge_fn}
            rows.append(psm)
        except Exception as e:
            print(f"  {stem}: FAIL {type(e).__name__}: {e}", flush=True); traceback.print_exc()
    return per, (P_sum(rows) if rows else None)

print("scoring SINGLE ...", flush=True); PER_S, S_S = score_dir(DIR_SINGLE)
print("scoring EDGE-TTA ...", flush=True); PER_T, S_T = score_dir(DIR_TTA)

print("\\n" + "=" * 78)
print("EDGE-TTA SCREEN - raw-ILP edge_jaccard (patched metric), detector fixed=50ep")
print("=" * 78)
print(f"{'stem':20s} {'single':>9s} {'edgeTTA':>9s} {'delta':>8s}   (eTP/eFP/eFN single -> tta)")
for stem in LOCALVAL_IDS:
    if stem in PER_S and stem in PER_T:
        a, b = PER_S[stem], PER_T[stem]
        print(f"{stem:20s} {a['edgeJ']:9.4f} {b['edgeJ']:9.4f} {b['edgeJ']-a['edgeJ']:+8.4f}   "
              f"{a['eTP']}/{a['eFP']}/{a['eFN']} -> {b['eTP']}/{b['eFP']}/{b['eFN']}")
if S_S and S_T:
    print("-" * 78)
    d = S_T['edge_jaccard'] - S_S['edge_jaccard']
    print(f"MICRO-AVG edge_jaccard : single={S_S['edge_jaccard']:.4f}  edgeTTA={S_T['edge_jaccard']:.4f}  delta={d:+.4f}")
    print(f"MICRO-AVG adj_edge_jac : single={S_S['adj_edge_jaccard']:.4f}  edgeTTA={S_T['adj_edge_jaccard']:.4f}")
    print("-" * 78)
    if d > 0.001:
        print(f">>> EDGE-TTA HELPS by {d:+.4f}. Worth a full bank-PP submit run.")
    elif d < -0.001:
        print(f">>> EDGE-TTA HURTS by {d:+.4f} (if catastrophic, suspect a coord-flip bug).")
    else:
        print(f">>> EDGE-TTA NEUTRAL ({d:+.4f}).")
print("SCREEN DONE")
'''


def md(t):
    return {"cell_type": "markdown", "metadata": {}, "source": t.splitlines(keepends=True)}


def code(t):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": t.splitlines(keepends=True)}


def main():
    print("Validating edge-TTA anchors:")
    if not validate_anchors():
        raise SystemExit("ANCHOR VALIDATION FAILED")

    verify = json.loads(VERIFY_NB.read_text())
    v = verify["cells"]
    c_constants = "".join(v[3]["source"])
    c_deps = "".join(v[5]["source"])
    c_metric = "".join(v[6]["source"])
    c_symlink = "".join(v[7]["source"])

    patchlist = "\n".join(f"    ({n!r}, {a!r}, {r!r})," for n, a, r in PATCHES)
    patch_cell = CELL_PATCH.replace("%PATCHLIST%", patchlist)

    cells = [
        code(CELL_CONFIG),
        md("## Constants (reused verbatim)"), code(c_constants),
        md("## Dependencies + inference repo + 50ep weights (reused verbatim)"), code(c_deps),
        md("## Patched metric bundle == Monday re-score (reused verbatim)"), code(c_metric),
        md("## Symlink localval TRAIN movies (GT available)"), code(c_symlink),
        md("## Apply edge-TTA patch"), code(patch_cell),
        md("## A/B: single vs edge-TTA(y,x,xy)"), code(CELL_AB),
        md("## Score raw-ILP geffs"), code(CELL_SCORE),
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {OUT_NB} ({len(cells)} cells)")

    meta = {
        "id": "khalid000000/bh-edgetta-screen-v1", "title": "bh-edgetta-screen-v1",
        "code_file": "bh-edgetta-screen-v1.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1"],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_DIR / 'kernel-metadata.json'}")


if __name__ == "__main__":
    main()
