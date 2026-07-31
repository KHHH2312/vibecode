"""Fixed constants for the 0.94 program (plan.md)."""
from __future__ import annotations

from typing import Final

# Public test stems — GT must never be used for train/cal/metric/cache.
FORBIDDEN_TEST_STEMS: Final[frozenset[str]] = frozenset(
    {
        "44b6_0113de3b",
        "44b6_0b24845f",
        "6bba_05b6850b",
        "6bba_05db0fb1",
    }
)

# Physical spacing (z, y, x) µm
SPACING_ZYX_UM: Final[tuple[float, float, float]] = (1.625, 0.40625, 0.40625)

# Detection defaults (plan)
SIGMA_UM: Final[float] = 2.4
NMS_UM: Final[float] = 3.0
CENTER_PRE_ILP_THRESH: Final[float] = 0.10
MAX_PEAKS_PER_FRAME: Final[int] = 2000

# Checkpoint schema
CHECKPOINT_SCHEMA_VERSION: Final[int] = 2

# Split seed from plan
SPLIT_SEED: Final[int] = 94017

# Metric pin fingerprints (sha256 of pinned source files)
METRIC_PIN_SHA256: Final[dict[str, str]] = {
    "metrics_pinned.py": "ab11310db0ada78ebb408fcd913bd001aed0dd5f8c1179e0a45dbb7152d6ebde",
    "division_metrics_pinned.py": "ef7472347a06842982bda795f896fd60fd7cb2e429fb87a72c6b9ede50c49d29",
}

# Official score weights (must match host)
ADJUSTMENT_ALPHA: Final[float] = 0.1
SCORE_DIVISION_WEIGHT: Final[float] = 0.1
