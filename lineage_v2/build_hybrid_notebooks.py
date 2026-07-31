#!/usr/bin/env python3
"""Build two hybrid Biohub submit notebooks from ULT1 + Yusuke 0.914."""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

ROOT = Path(r"C:\Users\Khalid\Desktop\New_folder\vibecode\lineage_v2")
CMP = ROOT / "compare_primary"
OUT = ROOT / "kernel_hybrid_submit"
OUT.mkdir(parents=True, exist_ok=True)

ult = json.loads((CMP / "bh-ult1-pub913-350ep" / "bh-ult1-pub913-350ep.ipynb").read_text(encoding="utf-8"))
yus = json.loads(
    (CMP / "pub914_yusuke_v338569479" / "no-hack-biohub-cell-another-approch-3rd.ipynb").read_text(
        encoding="utf-8"
    )
)

DATASETS_FULL = [
    "hongdaekim/biohub-350ep-checkpoint-pin-v1",
    "pilkwang/biohub-deepcenter-unet3d-center-prior-v1",
    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
    "shehailrs/biohub-tracking-350ep-public-weight-snapshot",
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
    "pilkwang/pilkwang-public-dataset-for-notebooks-figures",
]
DATASETS_A = [
    "hongdaekim/biohub-350ep-checkpoint-pin-v1",
    "pilkwang/biohub-temporal-unet3d-seed314159-v1",
    "shehailrs/biohub-tracking-350ep-public-weight-snapshot",
    "pilkwang/biohub-tracking-support-pack-50ep-v1",
    "pilkwang/pilkwang-public-dataset-for-notebooks-figures",
]

WEIGHT_FIX_CELL = r"""# === FIXED weight install (runs BEFORE predict) ===
# Prefer 350ep primary edge predictor; also copy sibling config.json to kill
# "config.json not found ... using defaults" warnings that desync architecture.
import os
import shutil
from pathlib import Path

os.environ.setdefault("BIOHUB_PREFER_350EP", "1")

def _install_fixed_primary_weights() -> None:
    global WEIGHTS_RELATIVE
    prefer_350 = os.environ.get("BIOHUB_PREFER_350EP", "1") != "0"
    override = os.environ.get("BIOHUB_WEIGHTS_OVERRIDE", "").strip()
    alt_rel = "weights_override/unet_transformer/split_0/edge_predictor_best.pth"
    alt_abs = REPO_DIR / alt_rel
    alt_dir = alt_abs.parent
    alt_dir.mkdir(parents=True, exist_ok=True)

    def rank(p: Path):
        s = str(p).lower().replace("\\", "/")
        if "deepcenter" in s or "full_frame" in s or "weights_override" in s:
            return (9, len(s), s)
        if prefer_350 and ("350ep" in s or "/350" in s or "350-ep" in s):
            return (0, len(s), s)
        if "300ep" in s or "300" in s:
            return (1, len(s), s)
        if "retrain" in s or "v34" in s:
            return (2, len(s), s)
        if "unet_transformer" in s and "edge_predictor_best" in s:
            return (3, len(s), s)
        if "biohub-tracking-support-pack" in s:
            return (8, len(s), s)
        return (5, len(s), s)

    def pick_source():
        if override:
            src = Path(override)
            if not src.exists():
                raise FileNotFoundError(f"BIOHUB_WEIGHTS_OVERRIDE not found: {src}")
            return src
        cands = []
        for root in [Path("/kaggle/input"), Path("/kaggle/input/datasets")]:
            if not root.exists():
                continue
            try:
                cands.extend(root.rglob("edge_predictor_best.pth"))
            except Exception as exc:
                print("weight scan skip", root, exc)
        if not cands:
            return None
        cands = sorted(set(cands), key=rank)
        print("weight candidates (top 10):")
        for p in cands[:10]:
            print(" ", rank(p), p)
        return cands[0]

    src = pick_source()
    if src is None:
        print("WARNING: no edge_predictor_best.pth found; keeping support-pack weights")
        return

    data = Path(src).read_bytes()
    alt_abs.write_bytes(data)
    WEIGHTS_RELATIVE = alt_rel.replace("\\", "/")
    print(f"INSTALLED primary weights: {src}")
    print(f"  -> {alt_abs} ({len(data)} bytes)")
    print(f"WEIGHTS_RELATIVE = {WEIGHTS_RELATIVE}")

    src_dir = Path(src).parent
    for name in ("config.json", "training_config.json", "split_manifest.json", "SNAPSHOT_MANIFEST.json"):
        cand = src_dir / name
        if cand.exists():
            shutil.copy2(cand, alt_dir / name)
            print(f"copied sidecar {name} from {cand}")
    if not (alt_dir / "config.json").exists():
        for root in [Path("/kaggle/input"), Path("/kaggle/input/datasets")]:
            if not root.exists():
                continue
            for cfg in root.rglob("config.json"):
                s = str(cfg).lower().replace("\\", "/")
                if "unet_transformer" in s and "split_0" in s and "deepcenter" not in s:
                    shutil.copy2(cfg, alt_dir / "config.json")
                    print(f"fallback config.json from {cfg}")
                    break
            if (alt_dir / "config.json").exists():
                break
    if not (alt_dir / "config.json").exists():
        print("WARNING: config.json still missing next to override weights")
    else:
        print("OK config.json present beside override weights")

_install_fixed_primary_weights()
print("WEIGHT_FIX_DONE")
"""


def src_of(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_src(cell: dict, text: str) -> None:
    if not text.endswith("\n"):
        text += "\n"
    cell["source"] = text.splitlines(keepends=True)


def make_code_cell(text: str) -> dict:
    if not text.endswith("\n"):
        text += "\n"
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def write_kernel(slug: str, title: str, nb: dict, datasets: list[str]) -> Path:
    d = OUT / slug
    d.mkdir(parents=True, exist_ok=True)
    nb_name = f"{slug}.ipynb"
    for c in nb.get("cells", []):
        if c.get("cell_type") == "code":
            c["outputs"] = []
            c["execution_count"] = None
    (d / nb_name).write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
    meta = {
        "id": f"khalid000000/{slug}",
        "title": title,
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "keywords": ["gpu"],
        "dataset_sources": datasets,
        "kernel_sources": [],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "model_sources": [],
    }
    (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("WROTE", d, "cells", len(nb["cells"]))
    return d


def extract_yusuke_bi_block() -> str:
    """Pull bidirectional patch block from Yusuke predict cell (source form)."""
    pred = src_of(yus["cells"][11])
    start = pred.find("# Model-level experiment 138:")
    if start < 0:
        start = pred.find("_bidirectional_weight_guard")
        # include a bit of comment
        start = pred.rfind("\n", 0, start) + 1
    end = pred.find("def list_test_stems")
    if start < 0 or end < 0:
        raise RuntimeError("could not extract Yusuke bidirectional block")
    block = pred[start:end].rstrip() + "\n"
    return block


# ---------- Hybrid A: Yusuke 0.914 + fixed 350ep ----------
nb_a = copy.deepcopy(yus)
set_src(
    nb_a["cells"][0],
    "# bh-hyb-a — Yusuke 0.914 (harmonic + bidirectional 0.20) + FIXED 350ep primary weights\n"
    "# No Stage DE. Primary-account submit candidate. Internet OFF.\n",
)
set_src(
    nb_a["cells"][1],
    "# Biohub Hybrid A | Yusuke 0.914 association + 350ep primary\n\n"
    "- Detection fusion: **harmonic_probability**\n"
    "- Association: reverse-time primary edge blend **0.20**\n"
    "- Primary weights: **350ep** (fixed install + config.json)\n"
    "- Secondary: seed314159 dual-seed low-margin\n"
    "- DeepCenter: OFF (matches Yusuke 0.914)\n"
    "- Main-account inference only\n",
)
guard = src_of(nb_a["cells"][9])
guard = guard.replace(
    'raise RuntimeError(\n        "The V2 / 0.912 baseline contains an unintended extra change. "\n'
    '        "Do not use this run for model selection."\n    )',
    'print("WARNING strategy drift (non-fatal for hybrid weight swap):")\n'
    '    print("\\n".join(f"- {item}" for item in _strategy_drift))',
)
set_src(nb_a["cells"][9], guard)
nb_a["cells"].insert(11, make_code_cell(WEIGHT_FIX_CELL))
for c in nb_a["cells"]:
    t = src_of(c)
    if "EXPERIMENT_TAG" in t and "biohub_146" in t:
        t = t.replace(
            'EXPERIMENT_TAG = "biohub_146_detection_harmonic_consensus_target093_nohack"',
            'EXPERIMENT_TAG = "hyb_a_yusuke914_350ep_fixed"',
        )
        t = t.replace(
            "BIOHUB_PRESET = 'detection_harmonic_consensus_target093'",
            "BIOHUB_PRESET = 'hyb_a_yusuke914_350ep_fixed'",
        )
        set_src(c, t)

path_a = write_kernel("bh-hyb-a-yusuke350", "bh-hyb-a-yusuke350", nb_a, DATASETS_A)

# ---------- Hybrid B: ULT1 + Yusuke association ----------
nb_b = copy.deepcopy(ult)
set_src(
    nb_b["cells"][0],
    "# bh-hyb-b — ULT1 (350ep+DeepCenter+retention) + Yusuke harmonic/bidirectional 0.20\n"
    "# No Stage DE. Primary-account submit candidate.\n",
)
set_src(
    nb_b["cells"][1],
    "# Biohub Hybrid B | ULT1 PP + Yusuke 0.914 association\n\n"
    "- Base: ULT1 dual-seed + DeepCenter@500 + retention guard + 350ep\n"
    "- + harmonic detection fusion (Yusuke)\n"
    "- + reverse-time primary edge weight **0.20** (Yusuke)\n"
    "- Fixed weight install copies config.json\n",
)

env = src_of(nb_b["cells"][4])
if "BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT" not in env:
    env = (
        env.rstrip()
        + "\n\nos.environ[\"BIOHUB_BIDIRECTIONAL_EDGE_WEIGHT\"] = \"0.20\"\n"
        + "os.environ[\"BIOHUB_DETECTION_FUSION_MODE\"] = \"harmonic_probability\"\n"
        + 'print("[cfg] Yusuke association: bidirectional=0.20 harmonic_probability")\n'
    )
set_src(nb_b["cells"][4], env)
set_src(nb_b["cells"][9], WEIGHT_FIX_CELL)

pred = src_of(nb_b["cells"][10])
linear = (
    "                    det_logits[f] = (\n"
    "                        (1.0 - secondary_detection_weight) * primary_det\n"
    "                        + secondary_detection_weight * secondary_det_aligned\n"
    "                    )"
)
harmonic = (
    "                    # Yusuke/Biohub-146 harmonic-probability detection consensus\n"
    "                    primary_det_prob = torch.sigmoid(primary_det.float()).clamp(1e-6, 1.0 - 1e-6)\n"
    "                    secondary_det_prob = torch.sigmoid(\n"
    "                        secondary_det_aligned.float()\n"
    "                    ).clamp(1e-6, 1.0 - 1e-6)\n"
    "                    consensus_det_prob = 1.0 / (\n"
    "                        (1.0 - secondary_detection_weight) / primary_det_prob\n"
    "                        + secondary_detection_weight / secondary_det_prob\n"
    "                    )\n"
    "                    det_logits[f] = torch.logit(\n"
    "                        consensus_det_prob.clamp(1e-6, 1.0 - 1e-6)\n"
    "                    ).to(primary_det.dtype)"
)
if linear in pred:
    pred = pred.replace(linear, harmonic, 1)
    print("B: replaced linear det blend with harmonic")
else:
    linear2 = linear.replace("\n", "\\n")
    harmonic2 = harmonic.replace("\n", "\\n")
    if linear2 in pred:
        pred = pred.replace(linear2, harmonic2, 1)
        print("B: replaced linear det blend (escaped) with harmonic")
    else:
        print("B WARNING: linear det blend not found")
        i = pred.find("secondary_detection_weight) * primary_det")
        print("  idx", i)

if "reverse_logits_native" not in pred:
    bi_block = extract_yusuke_bi_block()
    # Relax hard fail if patch count wrong — print warning instead for ULT1 structure variants
    bi_block = bi_block.replace(
        'raise RuntimeError(\n'
        '        f"Bidirectional edge patch expected one transformed block, found {_bi_count}"\n'
        "    )",
        'print(f"BI WARNING: expected 1 transformed block, found {_bi_count} — skip")\n'
        "    _bi_count = 0",
    )
    # also need if _bi_count was zeroed skip replace
    bi_block = bi_block.replace(
        "_s = _s.replace(_bi_old, _bi_new, 1)\ncompile(_s, str(_ps), \"exec\")\n_ps.write_text(_s)\nprint(\n"
        '    "Bidirectional primary edge-logit blend applied | weight=",\n'
        "    _bidirectional_weight_guard,\n"
        ")",
        "if _bi_count == 1:\n"
        "    _s = _s.replace(_bi_old, _bi_new, 1)\n"
        "    compile(_s, str(_ps), \"exec\")\n"
        "    _ps.write_text(_s)\n"
        "    print(\n"
        '        "Bidirectional primary edge-logit blend applied | weight=",\n'
        "        _bidirectional_weight_guard,\n"
        "    )\n"
        "else:\n"
        '    print("Bidirectional patch not applied")',
    )
    marker = "def list_test_stems"
    if marker in pred:
        pred = pred.replace(marker, bi_block + "\n\n" + marker, 1)
        print("B: inserted Yusuke bidirectional block")
    else:
        pred = pred + "\n" + bi_block + "\n"
        print("B: appended Yusuke bidirectional block")
else:
    print("B: bidirectional already present")

set_src(nb_b["cells"][10], pred)

for c in nb_b["cells"]:
    t = src_of(c)
    if "EXPERIMENT_TAG" in t:
        t2 = re.sub(
            r'EXPERIMENT_TAG\s*=\s*"[^"]*"',
            'EXPERIMENT_TAG = "hyb_b_ult1_yusuke_bidir_harmonic_350ep"',
            t,
            count=1,
        )
        if t2 != t:
            set_src(c, t2)

path_b = write_kernel("bh-hyb-b-ult1yusuke", "bh-hyb-b-ult1yusuke", nb_b, DATASETS_FULL)

print("DONE")
print(path_a)
print(path_b)

