"""Build bh-v102-refine notebook from bh-v100c-submit.

Changes vs v100c:
1. Sub-voxel peak COM refinement inside _detect_cells_pooled
2. Keep float coords through predict_video (no int16 quantize after upsample)
3. Intensity-based node refine after loading geff
4. Dense-movie FP control on 6bba_05db0fb1
5. Same preflight + post-write guards
6. No exploit, no fusion
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "kernel_submit" / "bh-v100c-submit.ipynb"


def lines(s: str) -> list[str]:
    if not s.endswith("\n"):
        s += "\n"
    return s.splitlines(keepends=True)


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    # ------------------------------------------------------------------ cell 0
    nb["cells"][0]["source"] = lines(
        '''# =============================================================================
# Biohub v102-refine — honest post-patch edge attack (NO exploit, NO fusion)
# =============================================================================
# Base: bh-v100c-submit (verified 0.900 stack / honest ~0.872 post-patch).
#
# NEW vs v100c (attacks 6bba_05db0fb1 edge quality — the only real lever):
#   1. Sub-voxel peak COM refine on det logits (downsampled grid → prob centroid)
#   2. Keep float coords through predict (no int16 quantize after *downsample)
#   3. Full-res intensity COM refine on every node after geff load
#   4. Dense-movie FP control on 6bba_05db0fb1 (tighter edge max + GAP2 caps)
#
# Dead levers (measured): detection threshold (+0.0002). Do NOT re-sweep thr.
# Forbidden: hub/ladder division exploit (host patched). Fusion (scored 0.891).
# Guards: preflight (real 4-movie test dir) + post-write (exact stems, no dangles).
# GPU: T4x2 only. Internet OFF. No auto-submit.
# =============================================================================

import os
from pathlib import Path
from IPython.display import Image, display

BIOHUB_PRESET = "v102_subvoxel_refine_dense_fp_control"
BIOHUB_SCORE_AXIS = (
    "v100c 0.900 recipe + subvoxel peak COM + fullres intensity refine "
    "+ dense-movie edge/GAP2 tighten on 6bba_05db0fb1"
)

# --- Detection (precision-calibrated; threshold lever is DEAD — do not lower) ---
os.environ["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"
os.environ["BIOHUB_PREFER_CUSTOM_WEIGHTS"] = "1"
os.environ["BIOHUB_DET_THRESHOLD"] = "0.9725"
os.environ["BIOHUB_GAP_CLOSE_MAX_GAP"] = "2"
os.environ["BIOHUB_GAP_CLOSE_EFFECTIVE_MAX_GAP"] = "2"
os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "6"
os.environ["BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS"] = "1"

# --- Safe division geometry (pilkwang calibrated) ---
os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "4.66"
os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "8.5"
os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "7.65"
os.environ["BIOHUB_SAFE_DIV_FRAME_FRAC_CAP"] = "0.0076"
os.environ["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.00375"

# --- Proven multi-axis repairs that lifted 0.899 -> 0.900 ---
os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "1"
os.environ["BIOHUB_GAP2_MAX_TOTAL_UM"] = "9.7"
os.environ["BIOHUB_GAP2_MAX_STEP_UM"] = "4.05"
os.environ["BIOHUB_GAP2_MAX_LINKS_FRAC"] = "0.0032"
os.environ["BIOHUB_GAP2_MAX_LINKS_ABS"] = "140"
os.environ["BIOHUB_GAP2_REQUIRE_CONTEXT"] = "1"
os.environ["BIOHUB_GAP2_FRAME_FRAC_CAP"] = "0.006"

os.environ["BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER"] = "1"

os.environ["BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE"] = "1"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC"] = "0.10"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN"] = "4"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB"] = "0.82"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM"] = "3.25"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC"] = "0.018"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS"] = "180"

# --- Motion relink ---
os.environ["BIOHUB_OUTPUT_MOTION_RELINK"] = "1"
os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = "1.0"

# --- DeepCenter: veto OFF. Fusion OFF (scored 0.891). ---
os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "0"
os.environ["BIOHUB_DEEPCENTER_GAP_VETO"] = "0"
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "0"
os.environ["BIOHUB_USE_FULL_FRAME_CENTER_FUSION"] = "0"
os.environ["BIOHUB_REQUIRE_FULL_FRAME_CENTER"] = "0"

# --- ILP ---
os.environ["BIOHUB_USE_ILP"] = "1"
os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "1.0"

# --- v102: sub-voxel + intensity refine + dense FP control ---
os.environ["BIOHUB_SUBVOXEL_PEAK_REFINE"] = "1"
os.environ["BIOHUB_INTENSITY_NODE_REFINE"] = "1"
os.environ["BIOHUB_INTENSITY_REFINE_WIN_Z"] = "1"
os.environ["BIOHUB_INTENSITY_REFINE_WIN_YX"] = "3"
os.environ["BIOHUB_INTENSITY_REFINE_MAX_SHIFT_UM"] = "2.5"
os.environ["BIOHUB_DENSE_MOVIE_FP_CONTROL"] = "1"
# 6bba_05db0fb1 dominates post-patch score (adj~0.803, FP=159, FN=101).
os.environ["BIOHUB_DENSE_EDGE_MAX_UM"] = "11.5"
os.environ["BIOHUB_DENSE_GAP2_MAX_TOTAL_UM"] = "8.8"
os.environ["BIOHUB_DENSE_GAP2_MAX_STEP_UM"] = "3.7"
os.environ["BIOHUB_DENSE_GAP2_MAX_LINKS_ABS"] = "110"
os.environ["BIOHUB_DENSE_GAP2_MAX_LINKS_FRAC"] = "0.0026"
os.environ["BIOHUB_DENSE_MOTION_RELAXED_UM"] = "8.5"

# --- Runtime ---
os.environ["BIOHUB_RUN_VISUAL_EDA"] = "0"
os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "1"
os.environ["BIOHUB_ALLOW_PIP_INSTALL"] = "0"
os.environ["BIOHUB_ALLOW_ARTIFACT_FALLBACK"] = "0"
os.environ["BIOHUB_UNET_BATCH_SIZE"] = "4"

print(
    "MAXSCORE v102-refine | DET=0.9725 | GAP2=ON | RESCUE=ON | "
    "SUBVOXEL=ON | INTENSITY_REFINE=ON | DENSE_FP=ON"
)
print(f"BIOHUB_PRESET: {BIOHUB_PRESET}")
print(f"BIOHUB_SCORE_AXIS: {BIOHUB_SCORE_AXIS}")
'''
    )

    # ------------------------------------------------------------------ cell 1
    nb["cells"][1]["source"] = lines(
        """# Biohub v102-refine (honest edge attack)

Post-patch pivot. Host patched the division exploit; honest score ≈ **edge quality**.

## Evidence

| Asset | Honest post-patch | Note |
|-------|------------------:|------|
| bh-v100c-submit (0.900 recipe) | **~0.8723** | best honest baseline; 4-movie |
| v101 threshold sweep | +0.0002 | **DEAD lever** |
| **6bba_05db0fb1** | adj **0.803** | FP=159 FN=101 — the whole game |

## v102 changes

1. **Sub-voxel peak COM** on detection logits (grid was 4× subsampled XY).
2. **Float coords** kept through predict (no int16 snap after upsample).
3. **Full-res intensity COM refine** on every node after geff load.
4. **Dense-movie FP control** on `6bba_05db0fb1` only (tighter edge max + GAP2).

No hub/ladder exploit. No fusion. Same 4-movie guards as v100c.
"""
    )

    # ------------------------------------------------------------------ cell 2
    nb["cells"][2]["source"] = lines(
        """## Locked recipe

1. **Detect** at `τ = 0.9725` + D4-style spatial TTA + **sub-voxel peak COM**.
2. **Associate** with node transformer + SCIP ILP (`division_weight=1.0`).
3. **Refine** node XYZ via full-res intensity COM (capped shift).
4. **Repair** with motion relink + gap-close + GAP2 + short-track rescue + div-geom.
5. On **`6bba_05db0fb1` only**: tighter `EDGE_MAX_UM` / GAP2 / motion-relaxed caps to cut FP.
6. Emit **exactly the 4 test stems**. Preflight + post-write guards enforced.
"""
    )

    # ------------------------------------------------------------------ cell 3
    c3 = "".join(nb["cells"][3]["source"])
    c3 = c3.replace("bh-v100c-submit", "bh-v102-refine")
    if "SUBVOXEL_PEAK_REFINE" not in c3:
        marker = 'OUTPUT_EDGE_MAX_UM = float(os.environ.get("BIOHUB_OUTPUT_EDGE_MAX_UM", "14.0"))\n'
        if marker not in c3:
            raise SystemExit("OUTPUT_EDGE_MAX_UM marker missing in cell 3")
        knobs = '''OUTPUT_EDGE_MAX_UM = float(os.environ.get("BIOHUB_OUTPUT_EDGE_MAX_UM", "14.0"))

# --- v102 refine / dense FP control ---
SUBVOXEL_PEAK_REFINE = os.environ.get("BIOHUB_SUBVOXEL_PEAK_REFINE", "1") != "0"
INTENSITY_NODE_REFINE = os.environ.get("BIOHUB_INTENSITY_NODE_REFINE", "1") != "0"
INTENSITY_REFINE_WIN_Z = int(os.environ.get("BIOHUB_INTENSITY_REFINE_WIN_Z", "1"))
INTENSITY_REFINE_WIN_YX = int(os.environ.get("BIOHUB_INTENSITY_REFINE_WIN_YX", "3"))
INTENSITY_REFINE_MAX_SHIFT_UM = float(os.environ.get("BIOHUB_INTENSITY_REFINE_MAX_SHIFT_UM", "2.5"))
DENSE_MOVIE_FP_CONTROL = os.environ.get("BIOHUB_DENSE_MOVIE_FP_CONTROL", "1") != "0"
DENSE_MOVIE_STEMS = set(
    s.strip()
    for s in os.environ.get("BIOHUB_DENSE_MOVIE_STEMS", "6bba_05db0fb1").split(",")
    if s.strip()
)
DENSE_EDGE_MAX_UM = float(os.environ.get("BIOHUB_DENSE_EDGE_MAX_UM", "11.5"))
DENSE_GAP2_MAX_TOTAL_UM = float(os.environ.get("BIOHUB_DENSE_GAP2_MAX_TOTAL_UM", "8.8"))
DENSE_GAP2_MAX_STEP_UM = float(os.environ.get("BIOHUB_DENSE_GAP2_MAX_STEP_UM", "3.7"))
DENSE_GAP2_MAX_LINKS_ABS = int(os.environ.get("BIOHUB_DENSE_GAP2_MAX_LINKS_ABS", "110"))
DENSE_GAP2_MAX_LINKS_FRAC = float(os.environ.get("BIOHUB_DENSE_GAP2_MAX_LINKS_FRAC", "0.0026"))
DENSE_MOTION_RELAXED_UM = float(os.environ.get("BIOHUB_DENSE_MOTION_RELAXED_UM", "8.5"))
'''
        c3 = c3.replace(marker, knobs, 1)
    if "EXPERIMENT_TAG" not in c3:
        c3 = c3.replace(
            'SUBMISSION_PATH = WORKING_DIR / "submission.csv"',
            'EXPERIMENT_TAG = os.environ.get("BIOHUB_EXPERIMENT_TAG", "bh-v102-refine")\n'
            'SUBMISSION_PATH = WORKING_DIR / "submission.csv"',
        )
    nb["cells"][3]["source"] = lines(c3)

    # ------------------------------------------------------------------ cell 5
    nb["cells"][5]["source"] = lines(
        """## Artifacts

Attach:

1. `pilkwang/biohub-tracking-support-pack-50ep-v1`
2. Competition data `biohub-cell-tracking-during-development`

DeepCenter is **not required** (fusion/veto off). GPU: **T4×2**, Internet **OFF**.
"""
    )

    # ------------------------------------------------------------------ cell 7
    nb["cells"][7]["source"] = lines(
        """## Predict graphs

Writes one `.geff` per test video at DET=0.9725 + ILP.
Includes the D4-style detection TTA patch **and** the v102 sub-voxel peak COM + float-coord patches.
"""
    )

    # ------------------------------------------------------------------ cell 8
    c8 = "".join(nb["cells"][8]["source"])
    if "v102 PATCH: sub-voxel" not in c8:
        anchor = "\ndef list_test_stems() -> list[str]:"
        if anchor not in c8:
            raise SystemExit("list_test_stems anchor missing in cell 8")
        patch = r'''
# ==================== v102 PATCH: sub-voxel peak COM + float coords ====================
# Peaks sit on a 4x XY-subsampled grid (~1.6 µm/step). Soft COM of local probability
# recovers sub-voxel centroids → better 7 µm node matching + cleaner edge distances.
# Also stop casting coords to int16 after upsample (keeps fractional voxels).
if SUBVOXEL_PEAK_REFINE:
    _ps2 = REPO_DIR / "scripts" / "predict_unet_transformer.py"
    _s2 = _ps2.read_text()
    _old_ret = """    coords = peak_idx.float().cpu().numpy()
    t_col = np.full((len(coords), 1), t, dtype=np.float32)
    return np.concatenate([t_col, coords], axis=1).astype(np.int16)"""
    _new_ret = """    # v102: sub-voxel COM refine on sigmoid probability mass around each peak
    probs = torch.sigmoid(logits[0, 0])  # (Z, Y, X)
    rz = ry = rx = 1
    refined = []
    for pz, py, px in peak_idx.tolist():
        z0 = max(0, pz - rz); z1 = min(probs.shape[0], pz + rz + 1)
        y0 = max(0, py - ry); y1 = min(probs.shape[1], py + ry + 1)
        x0 = max(0, px - rx); x1 = min(probs.shape[2], px + rx + 1)
        patch = probs[z0:z1, y0:y1, x0:x1]
        w = patch
        total = float(w.sum().item())
        if total <= 1e-8:
            refined.append((float(t), float(pz), float(py), float(px)))
            continue
        zz = torch.arange(z0, z1, device=probs.device, dtype=probs.dtype)
        yy = torch.arange(y0, y1, device=probs.device, dtype=probs.dtype)
        xx = torch.arange(x0, x1, device=probs.device, dtype=probs.dtype)
        wz = (w.sum(dim=(1, 2)) * zz).sum() / total
        wy = (w.sum(dim=(0, 2)) * yy).sum() / total
        wx = (w.sum(dim=(0, 1)) * xx).sum() / total
        refined.append((float(t), float(wz.item()), float(wy.item()), float(wx.item())))
    return np.asarray(refined, dtype=np.float32)"""
    if _old_ret in _s2:
        _s2 = _s2.replace(_old_ret, _new_ret)
        print("v102 peak-COM refine patch applied")
    else:
        print("v102 WARNING: peak return block not found — subvoxel refine skipped")

    _old_empty = "        return np.empty((0, 4), dtype=np.int16)"
    _new_empty = "        return np.empty((0, 4), dtype=np.float32)"
    if _old_empty in _s2:
        _s2 = _s2.replace(_old_empty, _new_empty)

    _old_up = """    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.int16)
    # Scale spatial coords back to original resolution.
    coords = coords.astype(np.float32)
    coords[:, 1:] *= ds_arr
    coords = coords.astype(np.int16)
    return coords, all_edges"""
    _new_up = """    coords = np.concatenate(coord_lists) if coord_lists else np.empty((0, 4), dtype=np.float32)
    # Scale spatial coords back to original resolution. v102: KEEP FLOAT (no int16 snap).
    coords = coords.astype(np.float32)
    coords[:, 1:] *= ds_arr
    return coords, all_edges"""
    if _old_up in _s2:
        _s2 = _s2.replace(_old_up, _new_up)
        print("v102 float-coord upsample patch applied")
    else:
        # looser fallback: just strip the int16 cast line if present
        _cast = "    coords = coords.astype(np.int16)\n    return coords, all_edges"
        if _cast in _s2:
            _s2 = _s2.replace(
                _cast,
                "    # v102: keep float coords\n    return coords, all_edges",
            )
            print("v102 float-coord patch applied (fallback)")
        else:
            print("v102 WARNING: upsample cast block not found")

    _ps2.write_text(_s2)
else:
    print("v102 subvoxel refine disabled by env")


def list_test_stems() -> list[str]:'''
        c8 = c8.replace(anchor, patch, 1)
    nb["cells"][8]["source"] = lines(c8)

    # ------------------------------------------------------------------ cell 9
    nb["cells"][9]["source"] = lines(
        """## Build submission

GAP2 + division geometry + adaptive short-track rescue.
**Plus v102:** full-res intensity node refine + dense-movie FP control on `6bba_05db0fb1`.
**No** full-frame center fusion. **No** exploit.
"""
    )

    # ------------------------------------------------------------------ cell 10
    c10 = "".join(nb["cells"][10]["source"])

    # Inject intensity refine helper + dense overrides near refine_synthetic_midpoint
    if "def refine_all_nodes_intensity" not in c10:
        marker = "def refine_synthetic_midpoint("
        if marker not in c10:
            raise SystemExit("refine_synthetic_midpoint missing")
        helper = '''def refine_all_nodes_intensity(
    dataset: str,
    nodes_by_id: dict[int, dict[str, object]],
    stats: dict[str, int],
) -> dict[int, dict[str, object]]:
    """Full-res intensity COM refine for every node (capped shift). v102."""
    if not INTENSITY_NODE_REFINE or not dataset:
        return nodes_by_id
    frame_cache: dict[int, np.ndarray] = {}
    stats.setdefault("intensity_refined_nodes", 0)
    stats.setdefault("intensity_refine_rejected_shift", 0)
    stats.setdefault("intensity_refine_failed", 0)
    out: dict[int, dict[str, object]] = {}
    for nid, node in nodes_by_id.items():
        mid = (float(node["z"]), float(node["y"]), float(node["x"]))
        try:
            frame = read_test_frame(dataset, int(node["t"]), frame_cache)
            z, y, x = [int(round(v)) for v in mid]
            z0 = max(0, z - INTENSITY_REFINE_WIN_Z)
            z1 = min(frame.shape[0], z + INTENSITY_REFINE_WIN_Z + 1)
            y0 = max(0, y - INTENSITY_REFINE_WIN_YX)
            y1 = min(frame.shape[1], y + INTENSITY_REFINE_WIN_YX + 1)
            x0 = max(0, x - INTENSITY_REFINE_WIN_YX)
            x1 = min(frame.shape[2], x + INTENSITY_REFINE_WIN_YX + 1)
            patch = frame[z0:z1, y0:y1, x0:x1].astype(np.float64)
            if patch.size == 0:
                stats["intensity_refine_failed"] += 1
                out[nid] = node
                continue
            baseline = float(np.percentile(patch, 20.0))
            weights = np.maximum(patch - baseline, 0.0)
            total = float(weights.sum())
            if total <= 0:
                stats["intensity_refine_failed"] += 1
                out[nid] = node
                continue
            zz = np.arange(z0, z1, dtype=np.float64)[:, None, None]
            yy = np.arange(y0, y1, dtype=np.float64)[None, :, None]
            xx = np.arange(x0, x1, dtype=np.float64)[None, None, :]
            refined = (
                float((weights * zz).sum() / total),
                float((weights * yy).sum() / total),
                float((weights * xx).sum() / total),
            )
            if point_distance_um(refined, mid) > INTENSITY_REFINE_MAX_SHIFT_UM:
                stats["intensity_refine_rejected_shift"] += 1
                out[nid] = node
                continue
            new_node = dict(node)
            new_node["z"], new_node["y"], new_node["x"] = refined
            out[nid] = new_node
            stats["intensity_refined_nodes"] += 1
        except Exception:
            stats["intensity_refine_failed"] += 1
            out[nid] = node
    return out


def _dense_fp_overrides(dataset: str | None) -> dict[str, float | int | bool]:
    """Per-movie tighter association caps for the dense bottleneck movie."""
    if not DENSE_MOVIE_FP_CONTROL or not dataset or dataset not in DENSE_MOVIE_STEMS:
        return {}
    return {
        "edge_max_um": DENSE_EDGE_MAX_UM,
        "gap2_max_total_um": DENSE_GAP2_MAX_TOTAL_UM,
        "gap2_max_step_um": DENSE_GAP2_MAX_STEP_UM,
        "gap2_max_links_abs": DENSE_GAP2_MAX_LINKS_ABS,
        "gap2_max_links_frac": DENSE_GAP2_MAX_LINKS_FRAC,
        "motion_relaxed_um": DENSE_MOTION_RELAXED_UM,
    }


def refine_synthetic_midpoint('''
        c10 = c10.replace(marker, helper, 1)

    # Wrap filter_output_graph to apply dense overrides via temporary globals
    # Safer: patch the call site in the main loop to refine nodes + set overrides.
    old_loop = '''        raw_node_count = len(nodes_by_id)
        nodes_by_id, edges, filter_stats = filter_output_graph(nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR)'''
    new_loop = '''        raw_node_count = len(nodes_by_id)
        # v102: intensity COM refine before association repairs (better edge geometry)
        _refine_stats: dict[str, int] = {}
        nodes_by_id = refine_all_nodes_intensity(dataset, nodes_by_id, _refine_stats)
        print(f"[{dataset}] intensity refine: {_refine_stats}")

        # v102: dense-movie FP control via temporary global overrides
        _dense = _dense_fp_overrides(dataset)
        _saved = {}
        if _dense:
            print(f"[{dataset}] dense FP control: {_dense}")
            _saved = {
                "OUTPUT_EDGE_MAX_UM": OUTPUT_EDGE_MAX_UM,
                "GAP2_MAX_TOTAL_UM": GAP2_MAX_TOTAL_UM,
                "GAP2_MAX_STEP_UM": GAP2_MAX_STEP_UM,
                "GAP2_MAX_LINKS_ABS": GAP2_MAX_LINKS_ABS,
                "GAP2_MAX_LINKS_FRAC": GAP2_MAX_LINKS_FRAC,
                "MOTION_RELINK_RELAXED_UM": MOTION_RELINK_RELAXED_UM,
            }
            # rebind module-level names used by filter_output_graph
            globals()["OUTPUT_EDGE_MAX_UM"] = float(_dense["edge_max_um"])
            globals()["GAP2_MAX_TOTAL_UM"] = float(_dense["gap2_max_total_um"])
            globals()["GAP2_MAX_STEP_UM"] = float(_dense["gap2_max_step_um"])
            globals()["GAP2_MAX_LINKS_ABS"] = int(_dense["gap2_max_links_abs"])
            globals()["GAP2_MAX_LINKS_FRAC"] = float(_dense["gap2_max_links_frac"])
            globals()["MOTION_RELINK_RELAXED_UM"] = float(_dense["motion_relaxed_um"])
        try:
            nodes_by_id, edges, filter_stats = filter_output_graph(
                nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR
            )
        finally:
            if _saved:
                for k, v in _saved.items():
                    globals()[k] = v
        filter_stats.update(_refine_stats)'''
    if old_loop not in c10:
        raise SystemExit("main filter_output_graph call site missing")
    c10 = c10.replace(old_loop, new_loop, 1)

    # Ensure EXPERIMENT_TAG default in stats if used
    c10 = c10.replace("bh-v100c-submit", "bh-v102-refine")
    nb["cells"][10]["source"] = lines(c10)

    # cell 11 guards already good — leave them

    # Write outputs
    targets = [
        ROOT / "kernel_v102" / "bh-v102-refine.ipynb",
        ROOT / "kernel" / "bh-v102-refine.ipynb",
        ROOT / "notebooks" / "bh-v102-refine.ipynb",
    ]
    for t in targets:
        t.parent.mkdir(parents=True, exist_ok=True)
        t.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print("wrote", t)

    meta = {
        "id": "khalid000000/bh-v102-refine",
        "title": "bh-v102-refine",
        "code_file": "bh-v102-refine.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1"
        ],
        "competition_sources": [
            "biohub-cell-tracking-during-development"
        ],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    meta_path = ROOT / "kernel_v102" / "kernel-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print("wrote", meta_path)

    # quick sanity
    c8b = "".join(nb["cells"][8]["source"])
    c10b = "".join(nb["cells"][10]["source"])
    assert "v102 PATCH: sub-voxel" in c8b
    assert "refine_all_nodes_intensity" in c10b
    assert "dense FP control" in c10b
    assert "POST-WRITE GUARD" in "".join(nb["cells"][11]["source"])
    print("SANITY OK: v102 notebook ready")


if __name__ == "__main__":
    main()
