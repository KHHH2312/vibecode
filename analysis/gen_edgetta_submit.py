#!/usr/bin/env python3
"""Generate bh-edgetta-submit-v1.ipynb: full bank-PP test submission + edge-TTA.

Clones kernel_submit/bh-v100c-submit.ipynb (verified 0.902 bank pipeline) and
injects the edge-TTA patch (average the 50ep edge predictor over y/x/xy flips)
plus BIOHUB_EDGE_TTA_FLIPS=y,x,xy. Detection keeps the bank's 8-way D4 TTA; the
bank post-processing and post-write guard are untouched. Edge-TTA raised raw-ILP
edge_jaccard by +0.0084 in the screen (bh-edgetta-screen-v1).
"""
import json
from pathlib import Path

import gen_edgetta_screen as ett  # reuse the exact edge-TTA patch tuples

REPO = Path("/home/user/vibecode")
SUBMIT_NB = REPO / "kernel_submit" / "bh-v100c-submit.ipynb"
OUT_DIR = REPO / "kernel_ett_submit"
OUT_NB = OUT_DIR / "bh-edgetta-submit-v1.ipynb"

ANCHOR_AFTER_TTA = '    print("TTA WARNING: block not found - using default 4-way")\n'
ANCHOR_BEFORE_PREDICT = "start_time = time.time()\n"


def make_ett_patch_block():
    lines = ["", "# EDGE-TTA PATCH: average 50ep edge predictor over spatial flips (y,x,xy)"]
    lines.append("_es = _ps.read_text()")
    lines.append("_ett_patches = [")
    for name, anchor, repl in ett.PATCHES:
        lines.append(f"    ({name!r}, {anchor!r}, {repl!r}),")
    lines.append("]")
    lines.append("for _nm, _a, _r in _ett_patches:")
    lines.append("    _cnt = _es.count(_a)")
    lines.append("    assert _cnt == 1, f'edge-TTA anchor not unique ({_cnt}x): {_nm}'")
    lines.append("    _es = _es.replace(_a, _r)")
    lines.append("_ps.write_text(_es)")
    lines.append("print('edge-TTA patch applied:', [p[0] for p in _ett_patches])")
    lines.append("assert '_flip_list' in _es and 'BIOHUB_EDGE_TTA_FLIPS' in _es, 'edge-TTA patch incomplete'")
    return "\n".join(lines) + "\n"


ETT_ENV_BLOCK = '''# EDGE-TTA: enable y/x/xy flip averaging of the 50ep edge predictor
os.environ["BIOHUB_EDGE_TTA_FLIPS"] = "y,x,xy"
print("[edge-TTA] BIOHUB_EDGE_TTA_FLIPS =", os.environ["BIOHUB_EDGE_TTA_FLIPS"])
start_time = time.time()
'''


def main():
    nb = json.loads(SUBMIT_NB.read_text())
    cells = nb["cells"]
    pc = next((i for i, c in enumerate(cells)
               if c["cell_type"] == "code" and "".join(c["source"]).startswith("# PATCH: 400ep spatial D4-style")), None)
    assert pc is not None, "could not find TTA-patch/predict cell"
    src = "".join(cells[pc]["source"])
    assert src.count(ANCHOR_AFTER_TTA) == 1 and src.count(ANCHOR_BEFORE_PREDICT) == 1

    src = src.replace(ANCHOR_AFTER_TTA, ANCHOR_AFTER_TTA + make_ett_patch_block())
    src = src.replace(ANCHOR_BEFORE_PREDICT, ETT_ENV_BLOCK)
    cells[pc]["source"] = src.splitlines(keepends=True)

    c0 = "".join(cells[0]["source"]).replace(
        "# Biohub MAXSCORE — verified 0.900 stack (NO fusion)",
        "# Biohub MAXSCORE — bank 0.902 PP + EDGE-TTA (50ep, y/x/xy flips)", 1)
    cells[0]["source"] = c0.splitlines(keepends=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {OUT_NB} ({len(cells)} cells); injected edge-TTA into cell {pc}")

    meta = {
        "id": "khalid000000/bh-edgetta-submit-v1", "title": "bh-edgetta-submit-v1",
        "code_file": "bh-edgetta-submit-v1.ipynb", "language": "python", "kernel_type": "notebook",
        "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": ["pilkwang/biohub-tracking-support-pack-50ep-v1"],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [], "model_sources": [], "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_DIR / 'kernel-metadata.json'}")


if __name__ == "__main__":
    main()
