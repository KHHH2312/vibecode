#!/usr/bin/env python3
"""Inject edge-TTA into kernel_v930/bh-v930-max.ipynb predict cell."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_edgetta_screen as ett  # noqa: E402

NB = REPO / "kernel_v930" / "bh-v930-max.ipynb"


def lines(s: str) -> list[str]:
    if not s.endswith("\n"):
        s += "\n"
    return s.splitlines(keepends=True)


def main() -> None:
    nb = json.loads(NB.read_text(encoding="utf-8"))
    # find predict cell
    idx = None
    for i, c in enumerate(nb["cells"]):
        if c.get("cell_type") != "code":
            continue
        src = "".join(c.get("source") or [])
        if "predict_unet_transformer.py" in src and "TTA" in src:
            idx = i
            break
    if idx is None:
        raise SystemExit("predict/TTA cell not found")
    src = "".join(nb["cells"][idx]["source"])
    if "_ett_patches" in src:
        print("already injected")
        return

    block_lines = [
        "",
        "# EDGE-TTA PATCH (v930): average edge predictor over y/x/xy flips",
        "_es = _ps.read_text()",
        "_ett_patches = [",
    ]
    for name, anchor, repl in ett.PATCHES:
        block_lines.append(f"    ({name!r}, {anchor!r}, {repl!r}),")
    block_lines += [
        "]",
        "for _nm, _a, _r in _ett_patches:",
        "    _cnt = _es.count(_a)",
        "    if _cnt == 1:",
        "        _es = _es.replace(_a, _r)",
        "        print(f'edge-TTA applied: {_nm}')",
        "    else:",
        "        print(f'edge-TTA SKIP {_nm}: count={_cnt}')",
        "_ps.write_text(_es)",
        "print('edge-TTA complete; flip-list marker:', '_flip_list' in _es)",
        'os.environ.setdefault("BIOHUB_EDGE_TTA_FLIPS", "y,x,xy")',
        'print("[v930] BIOHUB_EDGE_TTA_FLIPS=", os.environ.get("BIOHUB_EDGE_TTA_FLIPS"))',
        "",
    ]
    block = "\n".join(block_lines) + "\n"

    injected = False
    for m in ("predict_cmd = [", "def list_test_stems", "Using single-process prediction"):
        if m in src:
            src = src.replace(m, block + m, 1)
            injected = True
            print("injected before", m)
            break
    if not injected:
        # after TTA if/else
        key = "TTA WARNING: block not found"
        if key not in src:
            key = "TTA patch applied"
        pos = src.find(key)
        if pos < 0:
            raise SystemExit("no injection point")
        nl = src.find("\n", pos)
        src = src[: nl + 1] + block + src[nl + 1 :]
        print("injected after", key)

    nb["cells"][idx]["source"] = lines(src)
    text = json.dumps(nb, indent=1, ensure_ascii=False) + "\n"
    NB.write_text(text, encoding="utf-8")
    (REPO / "kernel" / "bh-v930-max.ipynb").write_text(text, encoding="utf-8")
    assert "_ett_patches" in src
    print("OK", NB, "cell", idx)


if __name__ == "__main__":
    main()
