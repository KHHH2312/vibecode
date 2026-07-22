#!/usr/bin/env python3
"""Build bh-divsweep-v1.ipynb: a CPU-fast safe-division sweep.

Reuses the PROVEN cells from the divrich-postlink-cache notebook (config env,
setup/constants, offline-deps + repo, post-processing definitions) + the v101
patched-metric bundle. Skips prediction entirely: loads the pre-safe cache
(uploaded as dataset khalid000000/biohub-divrich-presafe-cache-v1), replays the
post-safe TAIL per config (add_safe_divisions_postlink -> prune-isolated ->
short-track -> linefit -> CSV rounding), builds an InMemoryGraph, and scores
with the bundled PATCHED metric (== Monday's re-score) against local GT.

Optimises the honest aggregate score = adj_edge_jaccard + 0.1*division_jaccard.
"""
import json

CACHE_NB = "/root/.claude/uploads/27038f89-9d1c-5417-ac20-81af2582a6f1/eecc008f-bhdivrichpostlinkcache.ipynb"
V101     = "/home/user/vibecode/notebooks/bh-v101-sweep.ipynb"
DST_KERNEL = "/home/user/vibecode/kernel_divsweep/bh-divsweep-v1.ipynb"
DST_NB     = "/home/user/vibecode/notebooks/bh-divsweep-v1.ipynb"

def src(c): return "".join(c.get("source", []))
def code(s): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [s]}

cnb = json.load(open(CACHE_NB))
v101 = json.load(open(V101))

# code cells by index in the cache notebook: 7=config env, 8=setup, 10=deps+repo, 14=post-proc defs
cell_config = cnb["cells"][7]
cell_setup  = cnb["cells"][8]
cell_deps   = cnb["cells"][10]
cell_post   = cnb["cells"][14]
assert 'BIOHUB_SAFE_DIV_MAX_UM' in src(cell_config), "cell7 not config"
assert 'COMP_DIR = Path' in src(cell_setup), "cell8 not setup"
assert 'Offline' in src(cell_deps) or 'wheels' in src(cell_deps) or 'pip' in src(cell_deps), "cell10 not deps"
assert 'def add_safe_divisions_postlink' in src(cell_post), "cell14 not post-proc"

# patched-metric bundle (v101 cell 2)
metric_bundle = src(v101["cells"][2])
assert "tracking_cellmot" in metric_bundle and "b64decode" in metric_bundle, "metric bundle missing"

# ---- sweep driver ----
sweep = r'''# ==================== SAFE-DIVISION SWEEP (CPU, from pre-safe cache) ====================
import json as _json, zipfile as _zip, time as _time, traceback
from pathlib import Path as _P
from collections import defaultdict as _dd
import numpy as _np
import polars as pl
import pandas as pd

# --- patched metric (== Monday re-score) ---
import sys as _sys
for _m in [m for m in list(_sys.modules) if m == "tracking_cellmot" or m.startswith("tracking_cellmot.")]:
    _sys.modules.pop(_m, None)
from tracking_cellmot.metrics import (
    evaluate as P_eval, per_sample_metrics as P_psm, summarise as P_sum, node_recall as P_nr,
)
from geff import GeffMetadata

FORBIDDEN = {"44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1"}

# --- locate pre-safe cache (Kaggle auto-extracts uploaded zips -> prefer the folder) ---
_ins = _P("/kaggle/input")
_man_path = next(_ins.rglob("presafe_manifest.json"))
_state_jsons = list(_ins.rglob("presafe_states/*.json"))
if _state_jsons:
    CACHE_DIR = _state_jsons[0].parent
else:
    _zip_path = next(_ins.rglob("presafe_states.zip"))
    CACHE_DIR = _P("/kaggle/working/presafe_states"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with _zip.ZipFile(_zip_path) as zf: zf.extractall(CACHE_DIR)
manifest = _json.loads(_man_path.read_text())
RICH = list(dict.fromkeys(manifest["division_rich_stems"]))
CONTROL = [s for s in manifest["held_out_control_stems"] if s not in set(RICH)]  # drop overlap 6bba_07e24132
ALLV = manifest["all_validation_stems"]
assert not (set(ALLV) & FORBIDDEN), "forbidden test stem in validation pool!"
print(f"[sweep] rich={len(RICH)} control={len(CONTROL)} all={len(ALLV)}  (overlap dropped: {set(manifest['held_out_control_stems'])&set(RICH)})", flush=True)

# --- preload cache states + GT graphs + n_total (once) ---
_TRAIN = COMP_DIR / "train"
STATES, GT = {}, {}
_t0 = _time.time()
for stem in ALLV:
    d = _json.loads((CACHE_DIR / f"{stem}.json").read_text())
    nbid = {int(n["node_id"]): dict(n) for n in d["nodes"]}
    edges = [dict(e) for e in d["edges"]]
    STATES[stem] = (nbid, edges)
    geff = _TRAIN / f"{stem}.geff"
    gt = graph_from_geff(geff)
    meta = GeffMetadata.read(str(geff))
    GT[stem] = (gt, float(meta.extra["estimated_number_of_nodes"]))
print(f"[sweep] preloaded {len(STATES)} states + {len(GT)} GT graphs in {_time.time()-_t0:.1f}s", flush=True)

# --- config application (rebind module globals the post-proc reads at call time) ---
def apply_config(cfg):
    global SAFE_DIV_MAX_UM, SAFE_DIV_SISTER_MAX_UM, SAFE_DIV_EXISTING_CHILD_MAX_UM
    global SAFE_DIV_FRAME_FRAC_CAP, SAFE_DIV_GLOBAL_FRAC_CAP, OUTPUT_SAFE_DIVISIONS
    SAFE_DIV_MAX_UM = float(cfg["max_um"])
    SAFE_DIV_SISTER_MAX_UM = float(cfg["sister_max_um"])
    SAFE_DIV_EXISTING_CHILD_MAX_UM = float(cfg["existing_child_max_um"])
    SAFE_DIV_FRAME_FRAC_CAP = float(cfg["frame_frac_cap"])
    SAFE_DIV_GLOBAL_FRAC_CAP = float(cfg["global_frac_cap"])
    OUTPUT_SAFE_DIVISIONS = bool(cfg.get("safe_div_on", True))

# --- faithful post-safe TAIL (mirrors filter_output_graph after the safe-div call) ---
def score_stem(stem):
    nbid0, edges0 = STATES[stem]
    nbid = {k: dict(v) for k, v in nbid0.items()}   # copy: linefit mutates coords
    edges = [dict(e) for e in edges0]
    stats = _dd(int)
    edges = add_safe_divisions_postlink(nbid, edges, stats, dataset=stem)
    # OUTPUT_DIVISION_GEOMETRY_FILTER is False in this preset -> skip
    if OUTPUT_PRUNE_ISOLATED:
        incident = {int(e["source_id"]) for e in edges} | {int(e["target_id"]) for e in edges}
        if incident:
            nbid = {nid: n for nid, n in nbid.items() if nid in incident}
            edges = [e for e in edges if int(e["source_id"]) in nbid and int(e["target_id"]) in nbid]
    nbid, edges = filter_short_track_components(nbid, edges, stats)
    nbid = linefit_smooth_output_graph(nbid, edges, stats)
    # build scored graph with the exact CSV writer rounding: max(0, int(round(coord)))
    g = td.graph.InMemoryGraph()
    for key in ("z", "y", "x"):
        g.add_node_attr_key(key, pl.Float64, -999999.0)
    ids = sorted(nbid)
    gids = g.bulk_add_nodes([
        {"t": int(nbid[i]["t"]),
         "z": float(max(0, int(round(float(nbid[i]["z"]))))),
         "y": float(max(0, int(round(float(nbid[i]["y"]))))),
         "x": float(max(0, int(round(float(nbid[i]["x"])))))}
        for i in ids])
    id2g = dict(zip(ids, gids))
    if edges:
        g.bulk_add_edges([
            {"source_id": id2g[int(e["source_id"])], "target_id": id2g[int(e["target_id"])]}
            for e in edges if int(e["source_id"]) in id2g and int(e["target_id"]) in id2g])
    gt, n_total = GT[stem]
    er = P_eval(g, gt, scale=VOXEL_SCALE_UM, max_distance=7.0)
    rec = P_nr(g, gt) if (g.num_edges() > 0 and g.num_nodes() > 0) else 0.0
    psm = P_psm(er=er, n_total=n_total, node_recall=rec)
    return er, psm

def agg(psms):
    if not psms: return None
    s = P_sum(psms)
    return dict(score=s["score"], adj=s["adj_edge_jaccard"], edgeJ=s["edge_jaccard"],
               divJ=s["division_jaccard"], dTP=s["division_tp"], dFP=s["division_fp"], dFN=s["division_fn"])

# --- config grid ---
BASE = dict(max_um=4.66, sister_max_um=8.5, existing_child_max_um=7.65,
            frame_frac_cap=0.0076, global_frac_cap=0.00375, safe_div_on=True)
configs = []
configs.append(("baseline", dict(BASE)))
configs.append(("safe_OFF", {**BASE, "safe_div_on": False}))
for gc in [0.0075, 0.015, 0.03, 0.06, 0.10]:
    configs.append((f"gcap_{gc}", {**BASE, "global_frac_cap": gc}))
for fc in [0.02, 0.05, 0.10]:
    configs.append((f"fcap_{fc}", {**BASE, "frame_frac_cap": fc, "global_frac_cap": 0.03}))
for mu in [5.5, 6.5, 7.5]:
    configs.append((f"maxum_{mu}", {**BASE, "max_um": mu, "global_frac_cap": 0.03, "frame_frac_cap": 0.05}))
for sm in [10.0, 12.0]:
    configs.append((f"sister_{sm}", {**BASE, "sister_max_um": sm, "global_frac_cap": 0.03, "frame_frac_cap": 0.05}))
for ec in [9.0, 11.0]:
    configs.append((f"exchild_{ec}", {**BASE, "existing_child_max_um": ec, "global_frac_cap": 0.03, "frame_frac_cap": 0.05}))
configs.append(("combo_A", {**BASE, "global_frac_cap": 0.05, "frame_frac_cap": 0.05, "max_um": 6.5, "sister_max_um": 10.0}))
configs.append(("combo_B", {**BASE, "global_frac_cap": 0.08, "frame_frac_cap": 0.08, "max_um": 6.5, "sister_max_um": 10.0, "existing_child_max_um": 9.0}))
configs.append(("combo_C", {**BASE, "global_frac_cap": 0.12, "frame_frac_cap": 0.10, "max_um": 7.0, "sister_max_um": 11.0, "existing_child_max_um": 9.0}))
print(f"[sweep] {len(configs)} configs over {len(ALLV)} stems\n", flush=True)

# --- run (incremental save + wall-clock budget so partial results always survive) ---
T_START = _time.time()
BUDGET_S = 7200  # stop launching new configs after 2h; save what we have
rows_summary, rows_perstem = [], []
_RICH_SET, _CTRL_SET = set(RICH), set(CONTROL)
for ci, (name, cfg) in enumerate(configs):
    if _time.time() - T_START > BUDGET_S:
        print(f"[sweep] BUDGET {BUDGET_S}s reached after {ci} configs — stopping early", flush=True)
        break
    apply_config(cfg)
    _tc = _time.time()
    psm_all, psm_rich, psm_ctrl = [], [], []
    errs = 0
    for stem in ALLV:
        try:
            er, psm = score_stem(stem)
        except Exception as e:
            errs += 1
            print(f"  [{name}] {stem} FAIL {type(e).__name__}: {e}", flush=True)
            continue
        psm_all.append(psm)
        if stem in _RICH_SET: psm_rich.append(psm)
        if stem in _CTRL_SET: psm_ctrl.append(psm)
        rows_perstem.append(dict(config=name, stem=stem, **{k: psm.get(k) for k in
            ("edge_jaccard","adj_edge_jaccard","division_jaccard","division_tp","division_fp","division_fn")}))
    A, R, C = agg(psm_all), agg(psm_rich), agg(psm_ctrl)
    row = dict(config=name, **{f"p_{k}": v for k, v in cfg.items()}, errs=errs,
               all_score=A["score"], all_adj=A["adj"], all_edgeJ=A["edgeJ"], all_divJ=A["divJ"],
               all_dTP=A["dTP"], all_dFP=A["dFP"], all_dFN=A["dFN"],
               rich_score=R["score"], rich_divJ=R["divJ"], rich_edgeJ=R["edgeJ"],
               ctrl_score=C["score"], ctrl_divJ=C["divJ"], ctrl_edgeJ=C["edgeJ"],
               secs=round(_time.time()-_tc, 1))
    rows_summary.append(row)
    print(f"[{ci+1:2}/{len(configs)}] {name:14} | ALL score={A['score']:.4f} adj={A['adj']:.4f} edgeJ={A['edgeJ']:.4f} divJ={A['divJ']:.4f} (d TP/FP/FN={A['dTP']}/{A['dFP']}/{A['dFN']}) | RICH div={R['divJ']:.4f} | CTRL edge={C['edgeJ']:.4f} div={C['divJ']:.4f} | {row['secs']}s errs={errs}", flush=True)
    # incremental save so a timeout/early-stop still yields ranked results
    pd.DataFrame(rows_summary).sort_values("all_score", ascending=False).to_csv("/kaggle/working/sweep_results.csv", index=False)
    pd.DataFrame(rows_perstem).to_csv("/kaggle/working/sweep_perstem.csv", index=False)

# --- save + rank ---
df = pd.DataFrame(rows_summary).sort_values("all_score", ascending=False).reset_index(drop=True)
df.to_csv("/kaggle/working/sweep_results.csv", index=False)
pd.DataFrame(rows_perstem).to_csv("/kaggle/working/sweep_perstem.csv", index=False)
base_row = df[df.config == "baseline"].iloc[0]
print("\n================= SWEEP RANKING (by ALL aggregate score, patched metric) =================", flush=True)
print(f"baseline: ALL score={base_row.all_score:.4f}  edgeJ={base_row.all_edgeJ:.4f}  divJ={base_row.all_divJ:.4f}", flush=True)
for _, r in df.head(12).iterrows():
    d = r.all_score - base_row.all_score
    print(f"  {r.config:14} ALL={r.all_score:.4f} ({d:+.4f})  adj={r.all_adj:.4f} edgeJ={r.all_edgeJ:.4f} divJ={r.all_divJ:.4f}  CTRLedge={r.ctrl_edgeJ:.4f}", flush=True)
print("\n[sweep] wrote sweep_results.csv + sweep_perstem.csv", flush=True)
print("SWEEP DONE", flush=True)
'''

new_cells = [cell_config, cell_setup, cell_deps, cell_post, code(metric_bundle), code(sweep)]
cnb_out = dict(cnb)
cnb_out["cells"] = new_cells

import os
os.makedirs(os.path.dirname(DST_KERNEL), exist_ok=True)
json.dump(cnb_out, open(DST_KERNEL, "w"), indent=1)
json.dump(cnb_out, open(DST_NB, "w"), indent=1)
print(f"wrote {DST_KERNEL} ({len(new_cells)} cells)")
for i, c in enumerate(new_cells):
    first = (src(c).strip().splitlines() or ["(empty)"])[0][:70]
    print(f"  cell {i} [{c['cell_type']:5}] {len(src(c)):6} chars | {first}")
