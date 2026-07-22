#!/usr/bin/env python3
"""Generate bh-ens-submit-v1.ipynb: full bank-PP test submission with the
edge-logit ensemble.

Clones kernel_submit/bh-v100c-submit.ipynb (the verified 0.900/0.902 bank
pipeline that writes submission.csv on the 4 test movies) and injects the
SAME ensemble patch used by the screen kernel:
  - patches predict_unet_transformer.py to load N edge predictors and average
    sigmoid(predict_edges) across them (detection stays on model[0]), and
  - sets BIOHUB_ENSEMBLE_WEIGHTS to the 350ep + v34 checkpoints before predict.

The bank post-processing (cell 10) and post-write safety guard (cell 11) are
untouched: the ensemble only changes the ILP edge graph, which the PP consumes
unchanged. Do NOT push/submit this until the screen (bh-ens-screen-v1) confirms
the ensemble raises edge_jaccard.
"""
import json
from pathlib import Path

import gen_ensemble_screen as screen  # reuse the exact ensemble patch tuples

REPO = Path("/home/user/vibecode")
SUBMIT_NB = REPO / "kernel_submit" / "bh-v100c-submit.ipynb"
OUT_DIR = REPO / "kernel_ens_submit"
OUT_NB = OUT_DIR / "bh-ens-submit-v1.ipynb"

# Injection anchors inside submit cell 8 (the TTA-patch + predict cell).
ANCHOR_AFTER_TTA = '    print("TTA WARNING: block not found - using default 4-way")\n'
ANCHOR_BEFORE_PREDICT = "start_time = time.time()\n"


def make_ens_patch_block():
    lines = ["", "# ENSEMBLE PATCH: load N edge predictors; detect with model[0]; average edge probs"]
    lines.append("_es = _ps.read_text()")
    lines.append("_ens_patches = [")
    for name, anchor, repl in screen.PATCHES:
        lines.append(f"    ({name!r}, {anchor!r}, {repl!r}),")
    lines.append("]")
    lines.append("for _nm, _a, _r in _ens_patches:")
    lines.append("    _cnt = _es.count(_a)")
    lines.append("    assert _cnt == 1, f'ensemble anchor not unique ({_cnt}x): {_nm}'")
    lines.append("    _es = _es.replace(_a, _r)")
    lines.append("_ps.write_text(_es)")
    lines.append("print('ensemble patch applied:', [p[0] for p in _ens_patches])")
    lines.append("assert 'ENSEMBLE_MODELS = None' in _es and '_probs_sum' in _es, 'ensemble patch incomplete'")
    return "\n".join(lines) + "\n"


ENS_ENV_BLOCK = '''# ENSEMBLE: resolve extra edge-predictor weights and enable averaging
def _find_w(sub):
    from pathlib import Path as _P
    r = _P("/kaggle/input")
    if not r.exists():
        return None
    for p in sorted(r.rglob("edge_predictor_best.pth")):
        if sub in str(p):
            return p
    return None
_w350 = _find_w("350ep-checkpoint-pin")
_wv34 = _find_w("v34-retrain-weights-mirror")
print("[ensemble] 350ep:", _w350, " v34:", _wv34)
assert _w350 is not None and _wv34 is not None, "missing ensemble weights (attach hongdaekim 350ep + subinium v34)"
os.environ["BIOHUB_ENSEMBLE_WEIGHTS"] = f"{_w350},{_wv34}"
print("[ensemble] BIOHUB_ENSEMBLE_WEIGHTS =", os.environ["BIOHUB_ENSEMBLE_WEIGHTS"])
start_time = time.time()
'''


def main():
    nb = json.loads(SUBMIT_NB.read_text())
    cells = nb["cells"]

    # locate the predict cell (cell 8: starts with the TTA PATCH comment)
    pc = None
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "".join(c["source"]).startswith("# PATCH: 400ep spatial D4-style"):
            pc = i
            break
    assert pc is not None, "could not find TTA-patch/predict cell in submit notebook"
    src = "".join(cells[pc]["source"])

    assert src.count(ANCHOR_AFTER_TTA) == 1, "TTA-warning anchor not unique"
    assert src.count(ANCHOR_BEFORE_PREDICT) == 1, "predict-start anchor not unique"

    src = src.replace(ANCHOR_AFTER_TTA, ANCHOR_AFTER_TTA + make_ens_patch_block())
    src = src.replace(ANCHOR_BEFORE_PREDICT, ENS_ENV_BLOCK)
    cells[pc]["source"] = src.splitlines(keepends=True)

    # Tag cell 0 header so it's obvious this is the ensemble variant.
    c0 = "".join(cells[0]["source"])
    c0 = c0.replace(
        "# Biohub MAXSCORE — verified 0.900 stack (NO fusion)",
        "# Biohub MAXSCORE — bank 0.900 PP + EDGE-LOGIT ENSEMBLE (50ep+350ep+v34)",
        1,
    )
    cells[0]["source"] = c0.splitlines(keepends=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_NB.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {OUT_NB} ({len(cells)} cells); injected ensemble into cell {pc}")

    meta = {
        "id": "khalid000000/bh-ens-submit-v1",
        "title": "bh-ens-submit-v1",
        "code_file": "bh-ens-submit-v1.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            "pilkwang/biohub-tracking-support-pack-50ep-v1",
            "hongdaekim/biohub-350ep-checkpoint-pin-v1",
            "subinium/biohub-v34-retrain-weights-mirror",
        ],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_DIR / 'kernel-metadata.json'}")


if __name__ == "__main__":
    main()
