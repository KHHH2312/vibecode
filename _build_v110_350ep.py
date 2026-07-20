"""Build bh-v110-350ep: pure bank 0.900 PP + 350ep public edge weights (NO exploit).

Restores the honest bank recipe that scored 0.902 (v102/v103 refine stacks
regressed to 0.895). Swaps only the edge-predictor checkpoint for the public
350ep pin (hongdaekim / shehailrs layout). No intensity refine, dense FP,
quality prune, hub/ladder, or fusion.
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


WEIGHT_OVERRIDE_SNIPPET = r'''
# ==================== v110: prefer attached 350ep public weights ====================
# Support-pack weights are often hardlinked/read-only under REPO_DIR/weights.
# Install stronger public weights to a WRITABLE path and retarget WEIGHTS_RELATIVE.
def _install_stronger_public_weights() -> None:
    global WEIGHTS_RELATIVE
    prefer_350 = os.environ.get("BIOHUB_PREFER_350EP", "1") != "0"
    override = os.environ.get("BIOHUB_WEIGHTS_OVERRIDE", "").strip()

    # Always write under a fresh writable tree (not the linked pack path).
    alt_rel = "weights_override/unet_transformer/split_0/edge_predictor_best.pth"
    alt_abs = REPO_DIR / alt_rel
    alt_abs.parent.mkdir(parents=True, exist_ok=True)

    def _pick_source() -> Path | None:
        if override:
            src = Path(override)
            if not src.exists():
                raise FileNotFoundError(f"BIOHUB_WEIGHTS_OVERRIDE not found: {src}")
            return src
        roots = [Path("/kaggle/input"), Path("/kaggle/input/datasets")]
        cands: list[Path] = []
        for root in roots:
            if not root.exists():
                continue
            try:
                cands.extend(root.rglob("edge_predictor_best.pth"))
            except Exception as exc:
                print(f"v110 weight scan skip {root}: {exc}")
        if not cands:
            return None

        def rank(p: Path) -> tuple:
            s = str(p).lower().replace("\\", "/")
            if "deepcenter" in s or "full_frame" in s or "weights_override" in s:
                return (9, len(s), s)
            if prefer_350 and ("350ep" in s or "350" in s):
                return (0, len(s), s)
            if "300ep" in s or "300" in s:
                return (1, len(s), s)
            if "retrain" in s or "v34" in s:
                return (2, len(s), s)
            if "unet_transformer" in s:
                return (3, len(s), s)
            if "biohub-tracking-support-pack" in s:
                return (8, len(s), s)
            return (5, len(s), s)

        cands = sorted(set(cands), key=rank)
        print("v110 candidates (top 8):")
        for p in cands[:8]:
            print(f"  rank={rank(p)} {p}")
        return cands[0]

    src = _pick_source()
    if src is None:
        print("v110: no alternate edge_predictor_best.pth found; keeping support-pack weights")
        return

    # Copy bytes into writable working path (never overwrite linked pack path).
    data = Path(src).read_bytes()
    alt_abs.write_bytes(data)
    WEIGHTS_RELATIVE = alt_rel.replace("\\", "/")
    print(f"v110 installed stronger weights: {src} -> {alt_abs}")
    print(f"v110 WEIGHTS_RELATIVE retargeted to {WEIGHTS_RELATIVE} ({len(data)} bytes)")


_install_stronger_public_weights()
# ==================== end v110 weight override ====================
'''


def main() -> None:
    nb = json.loads(SRC.read_text(encoding="utf-8"))

    nb["cells"][0]["source"] = lines(
        '''# =============================================================================
# Biohub v110-350ep — bank 0.900 PP + public 350ep weights (NO exploit)
# =============================================================================
# Diagnosis: v102/v103 refine stacks scored 0.895 (below bank 0.902).
# Strategy toward 0.920+: restore proven bank post-process; upgrade weights only.
#
# Locked bank recipe (DET=0.9725, GAP2, rescue, motion, div-geom, fusion OFF).
# Weights: attach 350ep pin / 350ep snapshot; auto-install best edge_predictor.
#
# Forbidden: hub/ladder, FORKS, fusion, intensity-refine, dense FP, quality prune.
# Guards: 4-movie preflight + SAFE post-write. T4x2. Internet OFF.
# =============================================================================

import os
from pathlib import Path
from IPython.display import Image, display

BIOHUB_PRESET = "v110_bank_pp_350ep_weights"
BIOHUB_SCORE_AXIS = (
    "bank 0.900 PP + public 350ep edge_predictor; aim honest 0.920+ via weights"
)

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
os.environ["BIOHUB_EXPERIMENT_TAG"] = "bh-v110-350ep"

print("=" * 72)
print("MAXSCORE v110-350ep | BANK PP | 350ep WEIGHTS | NO EXPLOIT | aim 0.920+")
print("BIOHUB_PRESET:", BIOHUB_PRESET)
print("BIOHUB_SCORE_AXIS:", BIOHUB_SCORE_AXIS)
print("=" * 72)
'''
    )

    nb["cells"][1]["source"] = lines(
        """# Biohub v110-350ep — honest weights upgrade

## Why
- v102/v103 landed **0.895** (bank −0.007). Refine knobs hurt.
- Bank recipe remains the honest floor (**0.902**).
- Aim **0.920+** via **better public edge weights** (350ep pin), not metric exploits.

## Locked
Bank PP only + D4 TTA. No hub/ladder. No fusion. 4-movie guards. T4×2.
"""
    )

    nb["cells"][2]["source"] = lines(
        """## Recipe
1. Materialize support pack (repo + wheels + default weights).
2. **Overwrite** `edge_predictor_best.pth` with attached 350ep public checkpoint when present.
3. Detect τ=0.9725 + D4 TTA + ILP + bank repairs (GAP2, rescue, motion, div-geom).
4. Emit exactly 4 test stems; SAFE guard.
"""
    )

    # cell 3: experiment tag
    c3 = "".join(nb["cells"][3]["source"])
    c3 = c3.replace("maxscore_bank_0900_no_fusion", "bh-v110-350ep")
    c3 = c3.replace("bh-v100c-submit", "bh-v110-350ep")
    nb["cells"][3]["source"] = lines(c3)

    # cell 6: after materialize_inference_repo(ARTIFACTS) inject weight install
    c6 = "".join(nb["cells"][6]["source"])
    marker = "materialize_inference_repo(ARTIFACTS)"
    if marker not in c6:
        raise SystemExit("materialize call marker missing")
    if "_install_stronger_public_weights" not in c6:
        # call may be last line without trailing newline
        c6 = c6.replace(marker, marker + "\n" + WEIGHT_OVERRIDE_SNIPPET, 1)
    nb["cells"][6]["source"] = lines(c6)

    out = ROOT / "kernel_v110"
    out.mkdir(exist_ok=True)
    out_nb = out / "bh-v110-350ep.ipynb"
    out_nb.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    meta = {
        "id": "khalid000000/bh-v110-350ep",
        "title": "bh-v110-350ep",
        "code_file": "bh-v110-350ep.ipynb",
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
    (out / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy(out_nb, ROOT / "kernel" / "bh-v110-350ep.ipynb")
    shutil.copy(out_nb, ROOT / "notebooks" / "bh-v110-350ep.ipynb")

    blob = out_nb.read_text(encoding="utf-8")
    assert "_install_stronger_public_weights" in blob
    assert "hub_id" not in blob
    assert "DUAL_LADDERS" not in blob
    assert "FORKS=" not in blob
    assert "BIOHUB_INTENSITY_NODE_REFINE" not in blob
    print("Wrote", out_nb)
    print("sanity OK")


if __name__ == "__main__":
    main()
