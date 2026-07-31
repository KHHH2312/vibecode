"""Session 0 tests that do not require torch."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lineage_v2.constants import FORBIDDEN_TEST_STEMS, METRIC_PIN_SHA256  # noqa: E402
from lineage_v2.denylist import ForbiddenStemError, assert_path_allowed_for_gt  # noqa: E402
from lineage_v2.eval.contract import official_score, verify_metric_pin  # noqa: E402
from lineage_v2.splits import build_manifest  # noqa: E402


def main() -> None:
    verify_metric_pin()
    assert abs(official_score(0.92, 0.20) - 0.94) < 1e-12
    for s in FORBIDDEN_TEST_STEMS:
        try:
            assert_path_allowed_for_gt(f"/train/{s}/t.geff")
            raise SystemExit("denylist fail")
        except ForbiddenStemError:
            pass
    stems = [f"44b6_{i:08x}" for i in range(20)] + [f"6bba_{i:08x}" for i in range(40)]
    m = build_manifest(stems, n_confirmation=12, n_conf_44b6=4)
    assert len(m.confirmation) == 12
    assert set(METRIC_PIN_SHA256)
    print("ALL_PASS session0_notorch")


if __name__ == "__main__":
    main()
