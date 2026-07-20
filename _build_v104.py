"""Build bh-v104-ilp from bh-v103-assoc.

Honest ILP track-persistence probe:
- Lower appearance/disappearance costs (harder to break tracks → fewer FN)
- Slightly more negative edge weight (prefer real associations)
- Keep v103 refine + quality prune + dense FN recovery knobs
No exploit, no fusion.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "kernel_v103" / "bh-v103-assoc.ipynb"


def lines(s: str) -> list[str]:
    if not s.endswith("\n"):
        s += "\n"
    return s.splitlines(keepends=True)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"missing {SRC}")
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    nb["cells"][0]["source"] = lines(
        '''# =============================================================================
# Biohub v104-ilp — honest ILP track-persistence (NO exploit)
# =============================================================================
# Base: bh-v103-assoc (refine + dense FN recovery + quality prune).
#
# NEW vs v103:
#   1. ILP appearance/disappearance 0.1 → 0.05 (harder to open/close tracks)
#   2. ILP edge weight -1.0 → -1.15 (prefer keeping real edges)
#   3. ILP division weight 1.0 → 1.25 (mild real-fork preference; no synthetic)
#   4. Keep subvoxel + intensity + quality prune + dense FN recovery
#
# Forbidden: hub/ladder. Fusion OFF. 4-movie guards. T4x2 only.
# =============================================================================

import os
from pathlib import Path
from IPython.display import Image, display

BIOHUB_PRESET = "v104_ilp_track_persist"
BIOHUB_SCORE_AXIS = (
    "v103 stack + ILP appearance/disappearance 0.05 + edge_w -1.15 + div_w 1.25"
)

os.environ["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"
os.environ["BIOHUB_PREFER_CUSTOM_WEIGHTS"] = "1"
os.environ["BIOHUB_DET_THRESHOLD"] = "0.9725"
os.environ["BIOHUB_GAP_CLOSE_MAX_GAP"] = "2"
os.environ["BIOHUB_GAP_CLOSE_EFFECTIVE_MAX_GAP"] = "2"
os.environ["BIOHUB_OUTPUT_MIN_TRACK_LEN"] = "6"
os.environ["BIOHUB_OUTPUT_KEEP_DIVISION_COMPONENTS"] = "1"

os.environ["BIOHUB_SAFE_DIV_MAX_UM"] = "4.66"
os.environ["BIOHUB_SAFE_DIV_SISTER_MAX_UM"] = "8.5"
os.environ["BIOHUB_SAFE_DIV_EXISTING_CHILD_MAX_UM"] = "7.65"
os.environ["BIOHUB_SAFE_DIV_FRAME_FRAC_CAP"] = "0.0076"
os.environ["BIOHUB_SAFE_DIV_GLOBAL_FRAC_CAP"] = "0.00375"

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

os.environ["BIOHUB_OUTPUT_MOTION_RELINK"] = "1"
os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = "1.25"
os.environ["BIOHUB_MOTION_RELINK_VELOCITY_WEIGHT"] = "0.65"
os.environ["BIOHUB_MOTION_RELINK_TIGHT_UM"] = "6.5"
os.environ["BIOHUB_MOTION_RELINK_RELAXED_UM"] = "11.0"

os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "0"
os.environ["BIOHUB_DEEPCENTER_GAP_VETO"] = "0"
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "0"
os.environ["BIOHUB_USE_FULL_FRAME_CENTER_FUSION"] = "0"
os.environ["BIOHUB_REQUIRE_FULL_FRAME_CENTER"] = "0"

# --- v104 ILP track persistence ---
os.environ["BIOHUB_USE_ILP"] = "1"
os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "1.25"
os.environ["BIOHUB_ILP_EDGE_WEIGHT"] = "-1.15"
os.environ["BIOHUB_ILP_APPEARANCE_WEIGHT"] = "0.05"
os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "0.05"

os.environ["BIOHUB_SUBVOXEL_PEAK_REFINE"] = "1"
os.environ["BIOHUB_INTENSITY_NODE_REFINE"] = "1"
os.environ["BIOHUB_INTENSITY_REFINE_WIN_Z"] = "1"
os.environ["BIOHUB_INTENSITY_REFINE_WIN_YX"] = "3"
os.environ["BIOHUB_INTENSITY_REFINE_MAX_SHIFT_UM"] = "2.5"

os.environ["BIOHUB_DENSE_MOVIE_FP_CONTROL"] = "1"
os.environ["BIOHUB_DENSE_EDGE_MAX_UM"] = "12.8"
os.environ["BIOHUB_DENSE_GAP2_MAX_TOTAL_UM"] = "10.5"
os.environ["BIOHUB_DENSE_GAP2_MAX_STEP_UM"] = "4.4"
os.environ["BIOHUB_DENSE_GAP2_MAX_LINKS_ABS"] = "170"
os.environ["BIOHUB_DENSE_GAP2_MAX_LINKS_FRAC"] = "0.0038"
os.environ["BIOHUB_DENSE_MOTION_RELAXED_UM"] = "10.5"

os.environ["BIOHUB_QUALITY_EDGE_PRUNE"] = "1"
os.environ["BIOHUB_QUALITY_EDGE_MIN_PROB"] = "0.40"
os.environ["BIOHUB_QUALITY_EDGE_MIN_DIST_UM"] = "9.5"

os.environ["BIOHUB_RUN_VISUAL_EDA"] = "0"
os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "1"
os.environ["BIOHUB_ALLOW_PIP_INSTALL"] = "0"
os.environ["BIOHUB_ALLOW_ARTIFACT_FALLBACK"] = "0"
os.environ["BIOHUB_UNET_BATCH_SIZE"] = "4"
os.environ["BIOHUB_EXPERIMENT_TAG"] = "bh-v104-ilp"

print(
    "MAXSCORE v104-ilp | DET=0.9725 | ILP app/dis=0.05 edge=-1.15 div=1.25 | "
    "SUBVOXEL=ON | QUALITY_PRUNE=ON"
)
print(f"BIOHUB_PRESET: {BIOHUB_PRESET}")
print(f"BIOHUB_SCORE_AXIS: {BIOHUB_SCORE_AXIS}")
'''
    )

    nb["cells"][1]["source"] = lines(
        """# Biohub v104-ilp (honest ILP track persistence)

Same refine/GAP2/quality stack as v103, with ILP knobs that prefer continuous tracks:

- appearance / disappearance cost **0.05** (was 0.1)
- edge weight **-1.15** (was -1.0)
- division weight **1.25** (mild real-fork preference; no synthetic hubs)

No exploit. No fusion. 4-movie guards. T4×2.
"""
    )

    nb["cells"][2]["source"] = lines(
        """## Locked recipe

1. Detect τ=0.9725 + D4 TTA + sub-voxel peak COM.
2. **ILP** with persistence-biased costs.
3. Intensity refine + motion + GAP2 + quality prune.
4. Emit exactly 4 test stems.
"""
    )

    c3 = "".join(nb["cells"][3]["source"])
    c3 = c3.replace("bh-v103-assoc", "bh-v104-ilp")
    c3 = c3.replace("bh-v102-refine", "bh-v104-ilp")
    nb["cells"][3]["source"] = lines(c3)

    nb["cells"][9]["source"] = lines(
        """## Build submission

v104: v103 stack + ILP track-persistence weights.
No fusion. No exploit.
"""
    )

    out_dir = ROOT / "kernel_v104"
    out_dir.mkdir(exist_ok=True)
    out_nb = out_dir / "bh-v104-ilp.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    meta = {
        "id": "khalid000000/bh-v104-ilp",
        "title": "bh-v104-ilp",
        "code_file": "bh-v104-ilp.ipynb",
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
    shutil.copy(out_nb, ROOT / "kernel" / "bh-v104-ilp.ipynb")
    shutil.copy(out_nb, ROOT / "notebooks" / "bh-v104-ilp.ipynb")
    print("Wrote", out_nb)
    blob = out_nb.read_text(encoding="utf-8")
    assert "ILP_APPEARANCE" in blob or "ilp_appearance" in blob.lower() or "0.05" in blob
    assert "hub_id" not in blob
    assert "DUAL_LADDERS" not in blob
    print("sanity OK")


if __name__ == "__main__":
    main()
