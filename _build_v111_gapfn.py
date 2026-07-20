"""Build bh-v111-gapfn: bank 0.900 PP + 350ep weights + mild GAP2 FN boost (NO exploit).

Second 0.920-aim delta if v110 alone is insufficient. Keeps bank stack; only
raises GAP2 budgets slightly to recover 6bba FN without intensity/dense/prune.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "kernel_v110" / "bh-v110-350ep.ipynb"


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
# Biohub v111-gapfn — bank PP + 350ep weights + mild GAP2 FN boost (NO exploit)
# =============================================================================
# On top of v110: slightly higher GAP2 budgets to cut 6bba FN without the
# intensity/dense/quality knobs that scored 0.895.
# Forbidden: hub/ladder, fusion. 4-movie guards. T4x2.
# =============================================================================

import os
from pathlib import Path
from IPython.display import Image, display

BIOHUB_PRESET = "v111_bank_350ep_gapfn"
BIOHUB_SCORE_AXIS = "v110 + mild GAP2 FN boost for 6bba; aim 0.920+"

os.environ["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"
os.environ["BIOHUB_PREFER_CUSTOM_WEIGHTS"] = "1"
os.environ["BIOHUB_PREFER_350EP"] = "1"
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

# Mild FN-oriented GAP2 lift vs bank (9.7/4.05/140/0.0032)
os.environ["BIOHUB_OUTPUT_GAP2_RECOVERY"] = "1"
os.environ["BIOHUB_GAP2_MAX_TOTAL_UM"] = "10.4"
os.environ["BIOHUB_GAP2_MAX_STEP_UM"] = "4.35"
os.environ["BIOHUB_GAP2_MAX_LINKS_FRAC"] = "0.0040"
os.environ["BIOHUB_GAP2_MAX_LINKS_ABS"] = "175"
os.environ["BIOHUB_GAP2_REQUIRE_CONTEXT"] = "1"
os.environ["BIOHUB_GAP2_FRAME_FRAC_CAP"] = "0.007"

os.environ["BIOHUB_OUTPUT_DIVISION_GEOMETRY_FILTER"] = "1"

os.environ["BIOHUB_ADAPTIVE_SHORT_TRACK_RESCUE"] = "1"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_TRIGGER_REMOVED_FRAC"] = "0.10"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_LEN"] = "4"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MIN_MEAN_EDGE_PROB"] = "0.82"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_MEAN_EDGE_DIST_UM"] = "3.25"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_FRAC"] = "0.018"
os.environ["BIOHUB_SHORT_TRACK_RESCUE_MAX_NODES_ABS"] = "180"

os.environ["BIOHUB_OUTPUT_MOTION_RELINK"] = "1"
os.environ["BIOHUB_MOTION_RELINK_LEARNED_BONUS"] = "1.1"
os.environ["BIOHUB_MOTION_RELINK_RELAXED_UM"] = "10.5"

os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "0"
os.environ["BIOHUB_DEEPCENTER_GAP_VETO"] = "0"
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "0"
os.environ["BIOHUB_USE_FULL_FRAME_CENTER_FUSION"] = "0"
os.environ["BIOHUB_REQUIRE_FULL_FRAME_CENTER"] = "0"

os.environ["BIOHUB_USE_ILP"] = "1"
os.environ["BIOHUB_ILP_DIVISION_WEIGHT"] = "1.0"

os.environ["BIOHUB_RUN_VISUAL_EDA"] = "0"
os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "1"
os.environ["BIOHUB_ALLOW_PIP_INSTALL"] = "0"
os.environ["BIOHUB_ALLOW_ARTIFACT_FALLBACK"] = "0"
os.environ["BIOHUB_UNET_BATCH_SIZE"] = "4"
os.environ["BIOHUB_EXPERIMENT_TAG"] = "bh-v111-gapfn"

print("MAXSCORE v111-gapfn | BANK+350ep | mild GAP2 FN boost | NO EXPLOIT | aim 0.920+")
print("BIOHUB_PRESET:", BIOHUB_PRESET)
'''
    )

    nb["cells"][1]["source"] = lines(
        """# Biohub v111-gapfn

v110 (bank PP + 350ep) plus mild GAP2/motion FN recovery. No intensity/dense/prune.
No exploit. Aim honest 0.920+.
"""
    )

    c3 = "".join(nb["cells"][3]["source"])
    c3 = c3.replace("bh-v110-350ep", "bh-v111-gapfn")
    nb["cells"][3]["source"] = lines(c3)

    out = ROOT / "kernel_v111"
    out.mkdir(exist_ok=True)
    out_nb = out / "bh-v111-gapfn.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = {
        "id": "khalid000000/bh-v111-gapfn",
        "title": "bh-v111-gapfn",
        "code_file": "bh-v111-gapfn.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "hongdaekim/biohub-350ep-checkpoint-pin-v1",
            "shehailrs/biohub-tracking-350ep-public-weight-snapshot",
            "subinium/biohub-v34-retrain-weights-mirror",
        ],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    shutil.copy(out_nb, ROOT / "kernel" / "bh-v111-gapfn.ipynb")
    shutil.copy(out_nb, ROOT / "notebooks" / "bh-v111-gapfn.ipynb")
    blob = out_nb.read_text(encoding="utf-8")
    assert "hub_id" not in blob and "DUAL_LADDERS" not in blob
    assert "_install_stronger_public_weights" in blob
    print("Wrote", out_nb)


if __name__ == "__main__":
    main()
