#!/usr/bin/env python3
"""Build a placeholder stratified manifest until full stem census is available.

Replace with real train stem list from Kaggle (excluding FORBIDDEN_TEST_STEMS).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lineage_v2.constants import FORBIDDEN_TEST_STEMS  # noqa: E402
from lineage_v2.splits import build_manifest, save_manifest  # noqa: E402


def synthetic_stems(n44: int = 69, n6b: int = 126) -> list[str]:
    # plan claims 69 44b6 + 126 6bba permitted after excluding 4 test
    stems = [f"44b6_synth{i:04d}" for i in range(n44)]
    stems += [f"6bba_synth{i:04d}" for i in range(n6b)]
    assert not (set(stems) & FORBIDDEN_TEST_STEMS)
    return stems


def main() -> int:
    out = ROOT / "manifests" / "placeholder_seed94017.json"
    m = build_manifest(synthetic_stems())
    h = save_manifest(m, out)
    print("wrote", out)
    print("sha256", h)
    print("meta", m.meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
