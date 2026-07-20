"""Build bh-v105-peak: subvoxel only control (no intensity, no dense tighten)."""
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
# Biohub v105-peak — peak COM only, no intensity, no dense tighten (NO exploit)
# =============================================================================
# Control axis: isolate subvoxel peak COM without intensity refine or dense FP
# control that may have over-pruned 6bba edges in v102.
# Base recipe = v100c 0.900 + SUBVOXEL only.
# Forbidden: hub/ladder. Fusion OFF. 4-movie guards. T4x2 only.
# =============================================================================

import os
from pathlib import Path
from IPython.display import Image, display

BIOHUB_PRESET = "v105_peak_only"
BIOHUB_SCORE_AXIS = "v100c recipe + subvoxel peak COM only (no intensity, no dense FP)"

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
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB"] = "0.82"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM"] = "3.25"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC"] = "0.018"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS"] = "180"

os.environ["BIOHUB_OUTPUT_MOTION_RELINK"] = "1"
os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = "1.0"

os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "0"
os.environ["BIOHUB_DEEPCENTER_GAP_VETO"] = "0"
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "0"
os.environ["BIOHUB_USE_FULL_FRAME_CENTER_FUSION"] = "0"
os.environ["BIOHUB_REQUIRE_FULL_FRAME_CENTER"] = "0"

os.environ["BIOHUB_USE_ILP"] = "1"
os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "1.0"

os.environ["BIOHUB_SUBVOXEL_PEAK_REFINE"] = "1"
os.environ["BIOHUB_INTENSITY_NODE_REFINE"] = "0"
os.environ["BIOHUB_DENSE_MOVIE_FP_CONTROL"] = "0"

os.environ["BIOHUB_RUN_VISUAL_EDA"] = "0"
os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "1"
os.environ["BIOHUB_ALLOW_PIP_INSTALL"] = "0"
os.environ["BIOHUB_ALLOW_ARTIFACT_FALLBACK"] = "0"
os.environ["BIOHUB_UNET_BATCH_SIZE"] = "4"
os.environ["BIOHUB_EXPERIMENT_TAG"] = "bh-v105-peak"

print("MAXSCORE v105-peak | DET=0.9725 | SUBVOXEL=ON | INTENSITY=OFF | DENSE=OFF")
print(f"BIOHUB_PRESET: {BIOHUB_PRESET}")
print(f"BIOHUB_SCORE_AXIS: {BIOHUB_SCORE_AXIS}")
'''
    )

    nb["cells"][1]["source"] = lines(
        """# Biohub v105-peak (subvoxel only control)

Isolates the sub-voxel peak COM lever without intensity refine or dense FP
control. Use if v102/v103 intensity/dense regress vs bank 0.902.

No exploit. No fusion. 4-movie guards.
"""
    )

    c3 = "".join(nb["cells"][3]["source"])
    c3 = c3.replace("bh-v102-refine", "bh-v105-peak")
    nb["cells"][3]["source"] = lines(c3)

    out = ROOT / "kernel_v105"
    out.mkdir(exist_ok=True)
    out_nb = out / "bh-v105-peak.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {
        "id": "khalid000000/bh-v105-peak",
        "title": "bh-v105-peak",
        "code_file": "bh-v105-peak.ipynb",
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
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    shutil.copy(out_nb, ROOT / "kernel" / "bh-v105-peak.ipynb")
    shutil.copy(out_nb, ROOT / "notebooks" / "bh-v105-peak.ipynb")
    print("Wrote", out_nb)
    blob = out_nb.read_text(encoding="utf-8")
    assert "hub_id" not in blob
    assert "DUAL_LADDERS" not in blob
    print("sanity OK")


if __name__ == "__main__":
    main()
