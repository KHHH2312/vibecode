"""Build bh-v112-v34: bank 0.900 PP + v34-retrain public edge weights (NO exploit).

Rationale (verified against live CLI 2026-07-21):
  * Bank honest floor = 0.903 (ref 54748675 lineage); yesterday re-floored 0.902.
  * v110 350ep REGRESSED to 0.901 -> MORE epochs of the public checkpoint hurt.
    So the lever is NOT "more training" but a *different training lineage*.
  * Handoff §10.2 item 2: A/B alternate checkpoints (300ep, v34-retrain) on
    IDENTICAL bank PP. The v34-retrain mirror is a distinct lineage (2026-07-08),
    the single untested checkpoint with a real chance to differ from bank.

This reuses the exact v110 bank recipe and its writable-weight-install snippet
(never overwrites hardlinked pack weights). Only differences vs v110:
  * dataset_sources: support pack + v34 mirror ONLY (no 350ep -> no ambiguity
    about which edge_predictor_best.pth the ranker picks).
  * BIOHUB_PREFER_350EP=0 so the ranker prefers the v34/retrain checkpoint.
  * id/title/tag -> bh-v112-v34.

No hub/ladder, FORKS, fusion, intensity-refine, dense FP, or quality prune.
Guards: 4-movie preflight + SAFE post-write. T4x2. Internet OFF.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

# Reuse the proven pieces from the v110 builder so PP stays byte-identical.
import _build_v110_350ep as v110

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "kernel_submit" / "bh-v100c-submit.ipynb"

KID = "bh-v112-v34"


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    # ---- cell 0: env preset (copy v110's, retarget to v34-retrain lineage) ----
    cell0 = "".join(v110_cell0())
    nb["cells"][0]["source"] = v110.lines(cell0)

    nb["cells"][1]["source"] = v110.lines(
        """# Biohub v112-v34 — alternate-lineage weights A/B

## Why
- Live CLI 2026-07-21: bank floor **0.903**; v110 **350ep regressed to 0.901**.
- More epochs of the public checkpoint HURT -> try a *different lineage*, not more epochs.
- v34-retrain mirror (2026-07-08) on **identical bank PP**. Honest, no exploit.

## Locked
Bank PP only + D4 TTA. No hub/ladder. No fusion. 4-movie guards. T4x2.
"""
    )

    nb["cells"][2]["source"] = v110.lines(
        """## Recipe
1. Materialize support pack (repo + wheels + default 50ep weights).
2. Install v34-retrain `edge_predictor_best.pth` into a WRITABLE override path
   and retarget WEIGHTS_RELATIVE (never overwrite hardlinked pack weights).
3. Detect tau=0.9725 + D4 TTA + ILP + bank repairs (GAP2, rescue, motion, div-geom).
4. Emit exactly 4 test stems; SAFE guard.
"""
    )

    # cell 3: experiment tag
    c3 = "".join(nb["cells"][3]["source"])
    c3 = c3.replace("maxscore_bank_0900_no_fusion", KID)
    c3 = c3.replace("bh-v100c-submit", KID)
    nb["cells"][3]["source"] = v110.lines(c3)

    # cell 6: inject the writable weight-install snippet after materialize
    c6 = "".join(nb["cells"][6]["source"])
    marker = "materialize_inference_repo(ARTIFACTS)"
    if marker not in c6:
        raise SystemExit("materialize call marker missing")
    if "_install_stronger_public_weights" not in c6:
        c6 = c6.replace(marker, marker + "\n" + v110.WEIGHT_OVERRIDE_SNIPPET, 1)
    nb["cells"][6]["source"] = v110.lines(c6)

    out = ROOT / "kernel_v112"
    out.mkdir(exist_ok=True)
    out_nb = out / f"{KID}.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    meta = {
        "id": f"khalid000000/{KID}",
        "title": KID,
        "code_file": f"{KID}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        # ONLY support pack + v34 mirror -> deterministic weight pick.
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "subinium/biohub-v34-retrain-weights-mirror",
        ],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (out / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )

    # sanity: no exploit sentinels, weight install present, v34 lineage targeted
    blob = out_nb.read_text(encoding="utf-8")
    assert "_install_stronger_public_weights" in blob
    assert "hub_id" not in blob
    assert "DUAL_LADDERS" not in blob
    assert "FORKS=" not in blob
    assert "BIOHUB_INTENSITY_NODE_REFINE" not in blob
    # (.ipynb JSON escapes quotes, so match the escaped form)
    assert 'BIOHUB_PREFER_350EP\\"] = \\"0\\"' in blob
    assert "bh-v112-v34" in blob
    print("Wrote", out_nb)
    print("sanity OK")


def v110_cell0() -> list[str]:
    """v110 cell-0 env preset, retargeted to the v34-retrain lineage."""
    return v110.lines(
        '''# =============================================================================
# Biohub v112-v34 — bank 0.900 PP + v34-retrain weights (NO exploit)
# =============================================================================
# Diagnosis: v110 350ep REGRESSED to 0.901 (bank floor 0.903). More epochs hurt.
# Strategy: keep proven bank post-process; swap ONLY the edge weights to the
# v34-retrain lineage (a different training run, not more epochs).
#
# Locked bank recipe (DET=0.9725, GAP2, rescue, motion, div-geom, fusion OFF).
# Weights: attach support pack + v34-retrain mirror; auto-install v34 edge_predictor.
#
# Forbidden: hub/ladder, FORKS, fusion, intensity-refine, dense FP, quality prune.
# Guards: 4-movie preflight + SAFE post-write. T4x2. Internet OFF.
# =============================================================================

import os
from pathlib import Path
from IPython.display import Image, display

BIOHUB_PRESET = "v112_bank_pp_v34_weights"
BIOHUB_SCORE_AXIS = (
    "bank 0.900 PP + v34-retrain edge_predictor; honest alternate-lineage A/B"
)

os.environ["BIOHUB_OUTPUT_FILTER_SHORT_TRACKS"] = "1"
os.environ["BIOHUB_PREFER_CUSTOM_WEIGHTS"] = "1"
os.environ["BIOHUB_PREFER_350EP"] = "0"
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

os.environ["BIOHUB_RUN_VISUAL_EDA"] = "0"
os.environ["BIOHUB_RUN_OUTPUT_DIAGNOSTICS"] = "1"
os.environ["BIOHUB_ALLOW_PIP_INSTALL"] = "0"
os.environ["BIOHUB_ALLOW_ARTIFACT_FALLBACK"] = "0"
os.environ["BIOHUB_UNET_BATCH_SIZE"] = "4"
os.environ["BIOHUB_EXPERIMENT_TAG"] = "bh-v112-v34"

print("=" * 72)
print("MAXSCORE v112-v34 | BANK PP | v34-RETRAIN WEIGHTS | NO EXPLOIT")
print("BIOHUB_PRESET:", BIOHUB_PRESET)
print("BIOHUB_SCORE_AXIS:", BIOHUB_SCORE_AXIS)
print("=" * 72)
'''
    )


if __name__ == "__main__":
    main()
