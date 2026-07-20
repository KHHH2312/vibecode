"""Build bh-v103-assoc from bh-v102-refine.

Honest association / FN recovery on 6bba_05db0fb1:
1. Keep subvoxel + intensity refine
2. Moderate dense EDGE_MAX (12.8) + higher GAP2 budget (FN recovery)
3. Stronger motion relink (learned bonus + velocity)
4. Quality edge prune: drop low-prob AND long edges only
5. Slightly looser short-track rescue
No exploit, no fusion.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "kernel_v102" / "bh-v102-refine.ipynb"


def lines(s: str) -> list[str]:
    if not s.endswith("\n"):
        s += "\n"
    return s.splitlines(keepends=True)


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    nb["cells"][0]["source"] = lines(
        '''# =============================================================================
# Biohub v103-assoc — honest association / FN recovery on 6bba (NO exploit)
# =============================================================================
# Base: bh-v102-refine (subvoxel + intensity refine).
#
# NEW vs v102 (attacks 6bba_05db0fb1 FN while controlling FP):
#   1. Keep subvoxel peak COM + intensity refine (better 7um matching)
#   2. Dense-movie: moderate EDGE_MAX (12.8) + HIGHER GAP2 budget for FN recovery
#   3. Motion relink: higher learned-prob bonus + velocity weight
#   4. Quality edge prune: drop low-prob long edges (prob<0.40 and dist>9.5um)
#   5. Slightly more aggressive short-track rescue
#
# Forbidden: hub/ladder exploit. Fusion OFF. 4-movie guards. T4x2 only.
# =============================================================================

import os
from pathlib import Path
from IPython.display import Image, display

BIOHUB_PRESET = "v103_assoc_fn_recovery"
BIOHUB_SCORE_AXIS = (
    "v102 refine + moderate dense edge + boosted GAP2/motion + quality edge prune "
    "on 6bba_05db0fb1"
)

# --- Detection (precision-calibrated; threshold lever is DEAD) ---
os.environ["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"
os.environ["BIOHUB_PREFER_CUSTOM_WEIGHTS"] = "1"
os.environ["BIOHUB_DET_THRESHOLD"] = "0.9725"
os.environ["BIOHUB_GAP_CLOSE_MAX_GAP"] = "2"
os.environ["BIOHUB_GAP_CLOSE_EFFECTIVE_MAX_GAP"] = "2"
os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "6"
os.environ["BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS"] = "1"

# --- Safe division geometry ---
os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "4.66"
os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "8.5"
os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "7.65"
os.environ["BIOHUB_SAFE_DIV_FRAME_FRAC_CAP"] = "0.0076"
os.environ["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.00375"

# --- Proven multi-axis repairs ---
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
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB"] = "0.80"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM"] = "3.5"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC"] = "0.022"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS"] = "220"

# --- Motion relink (v103: stronger learned + velocity) ---
os.environ["BIOHUB_OUTPUT_MOTION_RELINK"] = "1"
os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = "1.25"
os.environ["BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT"] = "0.65"
os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "6.5"
os.environ["BIOHUB_MOTION_RELINK_RELAXED_UM"] = "11.0"

# --- DeepCenter / fusion OFF ---
os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "0"
os.environ["BIOHUB_DEEPCENTER_GAP_VETO"] = "0"
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "0"
os.environ["BIOHUB_USE_FULL_FRAME_CENTER_FUSION"] = "0"
os.environ["BIOHUB_REQUIRE_FULL_FRAME_CENTER"] = "0"

# --- ILP ---
os.environ["BIOHUB_USE_ILP"] = "1"
os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "1.0"
os.environ["BIOHUB_ILP_EDGE_WEIGHT"] = "-1.0"
os.environ["BIOHUB_ILP_APPEARANCE_WEIGHT"] = "0.1"
os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "0.1"

# --- v102/v103 refine ---
os.environ["BIOHUB_SUBVOXEL_PEAK_REFINE"] = "1"
os.environ["BIOHUB_INTENSITY_NODE_REFINE"] = "1"
os.environ["BIOHUB_INTENSITY_REFINE_WIN_Z"] = "1"
os.environ["BIOHUB_INTENSITY_REFINE_WIN_YX"] = "3"
os.environ["BIOHUB_INTENSITY_REFINE_MAX_SHIFT_UM"] = "2.5"

# --- Dense movie: FN recovery (less tight than v102) + quality prune ---
os.environ["BIOHUB_DENSE_MOVIE_FP_CONTROL"] = "1"
os.environ["BIOHUB_DENSE_EDGE_MAX_UM"] = "12.8"
os.environ["BIOHUB_DENSE_GAP2_MAX_TOTAL_UM"] = "10.5"
os.environ["BIOHUB_DENSE_GAP2_MAX_STEP_UM"] = "4.4"
os.environ["BIOHUB_DENSE_GAP2_MAX_LINKS_ABS"] = "170"
os.environ["BIOHUB_DENSE_GAP2_MAX_LINKS_FRAC"] = "0.0038"
os.environ["BIOHUB_DENSE_MOTION_RELAXED_UM"] = "10.5"

# Quality prune (honest FP control without distance-only cut)
os.environ["BIOHUB_QUALITY_EDGE_PRUNE"] = "1"
os.environ["BIOHUB_QUALITY_EDGE_MIN_PROB"] = "0.40"
os.environ["BIOHUB_QUALITY_EDGE_MIN_DIST_UM"] = "9.5"

# --- Runtime ---
os.environ["BIOHUB_RUN_VISUAL_EDA"] = "0"
os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "1"
os.environ["BIOHUB_ALLOW_PIP_INSTALL"] = "0"
os.environ["BIOHUB_ALLOW_ARTIFACT_FALLBACK"] = "0"
os.environ["BIOHUB_UNET_BATCH_SIZE"] = "4"
os.environ["BIOHUB_EXPERIMENT_TAG"] = "bh-v103-assoc"

print(
    "MAXSCORE v103-assoc | DET=0.9725 | GAP2=ON | RESCUE=ON | "
    "SUBVOXEL=ON | INTENSITY=ON | DENSE_FN_RECOVERY=ON | QUALITY_PRUNE=ON"
)
print(f"BIOHUB_PRESET: {BIOHUB_PRESET}")
print(f"BIOHUB_SCORE_AXIS: {BIOHUB_SCORE_AXIS}")
'''
    )

    nb["cells"][1]["source"] = lines(
        """# Biohub v103-assoc (honest association / FN recovery)

Post-patch pivot. Honest score ≈ **edge quality** on `6bba_05db0fb1`.

## v103 vs v102

| Change | Why |
|--------|-----|
| Dense EDGE_MAX 11.5 → **12.8** | v102 may have cut true long edges (FN) |
| Dense GAP2 **higher** budget | Recover missed 1–2 frame links on dense movie |
| Motion bonus 1.0 → **1.25**, vel **0.65** | Prefer high-prob + ballistic true links |
| **Quality edge prune** | Drop only low-prob AND long edges (FP without FN) |
| Rescue slightly looser | Keep legitimate short components |

No hub/ladder. No fusion. 4-movie guards.
"""
    )

    nb["cells"][2]["source"] = lines(
        """## Locked recipe

1. Detect τ=0.9725 + D4 TTA + sub-voxel peak COM.
2. Associate ILP + intensity COM refine.
3. Motion relink (boosted) + gap-close + GAP2 (dense-boosted) + rescue.
4. Quality-prune low-prob long edges on all movies.
5. Emit exactly 4 test stems. Preflight + post-write guards.
"""
    )

    c3 = "".join(nb["cells"][3]["source"])
    c3 = c3.replace("bh-v102-refine", "bh-v103-assoc")
    if "QUALITY_EDGE_PRUNE" not in c3:
        needle = (
            'DENSE_MOTION_RELAXED_UM = float(os.environ.get("BIOHUB_DENSE_MOTION_RELAXED_UM", "8.5"))\n'
        )
        if needle not in c3:
            raise SystemExit("DENSE_MOTION marker missing")
        c3 = c3.replace(
            needle,
            needle
            + """QUALITY_EDGE_PRUNE = os.environ.get("BIOHUB_QUALITY_EDGE_PRUNE", "1") != "0"
QUALITY_EDGE_MIN_PROB = float(os.environ.get("BIOHUB_QUALITY_EDGE_MIN_PROB", "0.40"))
QUALITY_EDGE_MIN_DIST_UM = float(os.environ.get("BIOHUB_QUALITY_EDGE_MIN_DIST_UM", "9.5"))
""",
            1,
        )
    nb["cells"][3]["source"] = lines(c3)

    nb["cells"][9]["source"] = lines(
        """## Build submission

v103: intensity refine + dense FN recovery (moderate edge max + boosted GAP2) +
quality edge prune + boosted motion.
No fusion. No exploit.
"""
    )

    c10 = "".join(nb["cells"][10]["source"])
    if "def quality_prune_edges" not in c10:
        marker = "def filter_output_graph("
        if marker not in c10:
            raise SystemExit("filter_output_graph missing")
        helper = '''def quality_prune_edges(
    nodes_by_id: dict[int, dict[str, object]],
    edges: list[dict[str, object]],
    stats: dict[str, int],
) -> list[dict[str, object]]:
    """Drop low-confidence long edges (honest FP control). v103."""
    if not QUALITY_EDGE_PRUNE or not edges:
        return edges
    stats.setdefault("quality_pruned_edges", 0)
    kept: list[dict[str, object]] = []
    for e in edges:
        try:
            prob = float(e.get("edge_prob", e.get("prob", 1.0)))
        except Exception:
            prob = 1.0
        try:
            dist = float(e.get("edge_dist", e.get("distance_um", 0.0)))
        except Exception:
            s = nodes_by_id.get(int(e.get("source", e.get("source_id", -1))), None)
            t = nodes_by_id.get(int(e.get("target", e.get("target_id", -1))), None)
            if s is None or t is None:
                kept.append(e)
                continue
            dist = point_distance_um(
                (float(s["z"]), float(s["y"]), float(s["x"])),
                (float(t["z"]), float(t["y"]), float(t["x"])),
            )
        if prob < QUALITY_EDGE_MIN_PROB and dist > QUALITY_EDGE_MIN_DIST_UM:
            stats["quality_pruned_edges"] += 1
            continue
        kept.append(e)
    return kept


def filter_output_graph('''
        c10 = c10.replace(marker, helper, 1)

    old = """        try:
            nodes_by_id, edges, filter_stats = filter_output_graph(
                nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR
            )
        finally:"""
    new = """        try:
            nodes_by_id, edges, filter_stats = filter_output_graph(
                nodes_by_id, raw_edges, dataset=dataset, deepcenter_bundle=DEEPCENTER_VETO_DETECTOR
            )
            edges = quality_prune_edges(nodes_by_id, edges, filter_stats)
            if filter_stats.get("quality_pruned_edges", 0):
                print(f"[{dataset}] quality prune: dropped {filter_stats['quality_pruned_edges']} low-prob long edges")
        finally:"""
    if old in c10:
        c10 = c10.replace(old, new, 1)
    else:
        raise SystemExit("filter try-block not found for quality prune inject")

    nb["cells"][10]["source"] = lines(c10)

    out_dir = ROOT / "kernel_v103"
    out_dir.mkdir(exist_ok=True)
    out_nb = out_dir / "bh-v103-assoc.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    meta = {
        "id": "khalid000000/bh-v103-assoc",
        "title": "bh-v103-assoc",
        "code_file": "bh-v103-assoc.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1"],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (out_dir / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy(out_nb, ROOT / "kernel" / "bh-v103-assoc.ipynb")
    shutil.copy(out_nb, ROOT / "notebooks" / "bh-v103-assoc.ipynb")
    print("Wrote", out_nb)

    blob = out_nb.read_text(encoding="utf-8")
    assert "quality_prune_edges" in blob
    assert "QUALITY_EDGE_PRUNE" in blob
    assert "v103" in blob
    assert "hub_id" not in blob
    assert "DUAL_LADDERS" not in blob
    print("sanity OK")


if __name__ == "__main__":
    main()
