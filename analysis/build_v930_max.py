#!/usr/bin/env python3
"""Build bh-v930-max: best honest public-asset stack aimed at 0.93+.

Research synthesis (post-rescore 2026-07-24):
- Exploit (hub/ladder) is dead; honest LB top ~0.929.
- Clean Yusuke-style base ~0.908 (disap=1.5–1.575, app=0.0, det~0.969).
- Edge-TTA (y/x/xy flip avg on edge head) lifted clean to **0.909** LB.
- Dual-seed + 350ep (bh-dsc-350ep-cv) fixed-8 CV **0.88163** ≈ ~0.911 LB extrap.
- m2div graft did NOT transfer (CV 0.88062, divJ=0) — omit.
- Intensity/dense/quality-prune stack scored **0.894** — omit.
- Path to 0.93: stack proven edge levers; real 0.93 likely needs private training
  or real divisions — we still ship the maximum public stack.

Stack:
1. dsc dual-seed + 350ep primary weights (Downloads/bhdsc350epcv)
2. Edge-TTA BIOHUB_EDGE_TTA_FLIPS=y,x,xy (proven +0.001 LB on clean)
3. Yusuke ILP disap=1.575 (clean-approach calibration)
4. No hub/ladder; no fusion; 4-movie guards + fixed-8 CV cell retained

T4, internet off.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_edgetta_screen as ett  # noqa: E402

SRC_CANDIDATES = [
    REPO / "kernel_dsC" / "bh-dsc-350ep-cv.ipynb",
    Path(r"C:\Users\Khalid\Downloads\bhdsc350epcv.ipynb"),
    REPO / "kernel" / "bh-dsc-350ep-cv.ipynb",
]
OUT_DIR = REPO / "kernel_v930"
OUT_NB = OUT_DIR / "bh-v930-max.ipynb"

ANCHOR_AFTER_TTA = '    print("TTA WARNING: block not found - using default 4-way")\n'
# alternate anchors that may appear
ALT_TTA_ANCHORS = [
    ANCHOR_AFTER_TTA,
    'print("TTA WARNING: block not found - using default 4-way")\n',
    'print("TTA patch applied (400ep spatial D4-style)")\n',
]


def normalize_nb(nb: dict) -> dict:
    for c in nb["cells"]:
        src = c.get("source", [])
        if isinstance(src, str):
            c["source"] = src.splitlines(keepends=True) if src else []
        c.setdefault("metadata", {})
        if c.get("cell_type") == "code":
            c.setdefault("outputs", [])
            c.setdefault("execution_count", None)
    nb.setdefault("metadata", {})
    nb["metadata"].setdefault(
        "kernelspec",
        {"display_name": "Python 3", "language": "python", "name": "python3"},
    )
    return nb


def lines(s: str) -> list[str]:
    if not s.endswith("\n"):
        s += "\n"
    return s.splitlines(keepends=True)


def make_ett_patch_block() -> str:
    lines_out = [
        "",
        "# EDGE-TTA PATCH (v930): average edge predictor over y/x/xy flips",
        "_es = _ps.read_text()",
        "_ett_patches = [",
    ]
    for name, anchor, repl in ett.PATCHES:
        lines_out.append(f"    ({name!r}, {anchor!r}, {repl!r}),")
    lines_out += [
        "]",
        "for _nm, _a, _r in _ett_patches:",
        "    _cnt = _es.count(_a)",
        "    if _cnt == 1:",
        "        _es = _es.replace(_a, _r)",
        "        print(f'edge-TTA applied: {_nm}')",
        "    else:",
        "        print(f'edge-TTA SKIP {_nm}: anchor count={_cnt}')",
        "_ps.write_text(_es)",
        "print('edge-TTA done; has flip list marker:', '_flip_list' in _es)",
    ]
    return "\n".join(lines_out) + "\n"


def inject_env_overrides(cell_src: str) -> str:
    """Append high-EV env overrides after existing env block (later assignment wins)."""
    extra = '''
# === v930-max research stack overrides (after base env; later wins) ===
os.environ["BIOHUB_EDGE_TTA_FLIPS"] = "y,x,xy"
os.environ["BIOHUB_PREFER_350EP"] = "1"
# Yusuke clean-approach ILP birth/death calibration (public ~0.908 baseline)
os.environ["BIOHUB_ILP_APPEARANCE_WEIGHT"] = "0.0"
os.environ["BIOHUB_ILP_DISAPPEARANCE_WEIGHT"] = "1.575"
# Keep DeepCenter OFF (epoch mismatch)
os.environ["BIOHUB_USE_DEEPCENTER_VETO"] = "0"
os.environ["BIOHUB_REQUIRE_DEEPCENTER_VETO"] = "0"
os.environ["BIOHUB_DEEPCENTER_GAP_VETO"] = "0"
os.environ["BIOHUB_DEEPCENTER_SAFE_DIV_VETO"] = "0"
print("[v930-max] EDGE_TTA=y,x,xy | ILP app=0 disap=1.575 | 350ep preferred | NO exploit")
'''
    if "v930-max" in cell_src:
        return cell_src
    return cell_src.rstrip() + "\n" + extra


def main() -> None:
    src_path = next((p for p in SRC_CANDIDATES if p.exists()), None)
    if src_path is None:
        raise SystemExit(f"no source notebook found in {SRC_CANDIDATES}")
    nb = normalize_nb(json.loads(src_path.read_text(encoding="utf-8")))

    # cell 0 banner
    nb["cells"][0]["source"] = lines(
        "# bh-v930-max — honest public-asset max stack (aim 0.93+)\n"
        "# Base: dual-seed + 350ep (dsc CV 0.88163). Plus edge-TTA + Yusuke ILP.\n"
        "# NO hub/ladder. Fixed-8 CV retained if present.\n"
    )

    # find env cell (BIOHUB_DET_THRESHOLD set)
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"])
        if 'os.environ["BIOHUB_DET_THRESHOLD"]' in src or "BIOHUB_DET_THRESHOLD" in src:
            c["source"] = lines(inject_env_overrides(src))
            print(f"env overrides injected cell {i}")
            break

    # find TTA/predict cell
    injected = False
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c["source"])
        if "predict_unet_transformer.py" not in src and "TTA" not in src:
            continue
        if "edge-TTA" in src or "EDGE-TTA" in src:
            print(f"edge-TTA already present cell {i}")
            injected = True
            break
        # inject after det TTA block
        for anc in ALT_TTA_ANCHORS:
            if anc in src:
                src = src.replace(anc, anc + make_ett_patch_block(), 1)
                # ensure env set before predict
                if "start_time = time.time()" in src and "BIOHUB_EDGE_TTA_FLIPS" not in src:
                    src = src.replace(
                        "start_time = time.time()",
                        'os.environ.setdefault("BIOHUB_EDGE_TTA_FLIPS", "y,x,xy")\n'
                        'print("[v930] EDGE_TTA_FLIPS", os.environ.get("BIOHUB_EDGE_TTA_FLIPS"))\n'
                        "start_time = time.time()",
                        1,
                    )
                c["source"] = lines(src)
                print(f"edge-TTA injected cell {i} via anchor")
                injected = True
                break
        if injected:
            break
        # looser: after _ps.write_text for TTA
        if "_ps.write_text" in src and "predict_cmd" in src:
            marker = "print("
            # append patch block before predict_cmd
            if "predict_cmd" in src and "EDGE-TTA PATCH" not in src:
                src = src.replace(
                    "predict_cmd = [",
                    make_ett_patch_block()
                    + 'os.environ.setdefault("BIOHUB_EDGE_TTA_FLIPS", "y,x,xy")\n'
                    + "predict_cmd = [",
                    1,
                )
                c["source"] = lines(src)
                print(f"edge-TTA injected before predict_cmd cell {i}")
                injected = True
                break

    if not injected:
        print("WARNING: edge-TTA not injected — check predict cell structure")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    meta = {
        "id": "khalid000000/bh-v930-max",
        "title": "bh-v930-max",
        "code_file": "bh-v930-max.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "hongdaekim/biohub-350ep-checkpoint-pin-v1",
            "hongdaekim/biohub-300ep-checkpoint-pin-v1",
            "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
            "pilkwang/biohub-temporal-unet3d-seed314159-v1",
            "shehailrs/biohub-tracking-350ep-public-weight-snapshot",
            "subinium/biohub-v34-retrain-weights-mirror",
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "pilkwang/pilkwang-public-dataset-for-notebooks-figures",
            "beyondlogic/biohub-training-code-b",
        ],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy(OUT_NB, REPO / "kernel" / "bh-v930-max.ipynb")

    blob = OUT_NB.read_text(encoding="utf-8")
    assert "hub_id" not in blob
    assert "augment_dataset" not in blob
    assert "DUAL_LADDERS" not in blob
    print("Wrote", OUT_NB)
    print("edge-TTA in nb:", "EDGE-TTA" in blob or "edge-TTA" in blob)
    print("v930-max env:", "v930-max" in blob)


if __name__ == "__main__":
    main()
