"""Fail-closed evaluation contract + metric pin verification."""
from __future__ import annotations

import hashlib
from pathlib import Path

from ..constants import (
    ADJUSTMENT_ALPHA,
    FORBIDDEN_TEST_STEMS,
    METRIC_PIN_SHA256,
    SCORE_DIVISION_WEIGHT,
)
from ..denylist import ForbiddenStemError, reject_if_any_forbidden


class MetricContractError(RuntimeError):
    pass


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_metric_pin(eval_dir: Path | None = None) -> dict[str, str]:
    """Ensure pinned metric sources match frozen fingerprints."""
    eval_dir = eval_dir or Path(__file__).resolve().parent
    out = {}
    for name, expected in METRIC_PIN_SHA256.items():
        p = eval_dir / name
        if not p.exists():
            raise MetricContractError(f"missing pinned metric file {p}")
        got = _sha256_file(p)
        if got != expected:
            raise MetricContractError(
                f"metric pin mismatch for {name}: got {got[:16]}… expected {expected[:16]}…"
            )
        out[name] = got
    # sanity constants
    if abs(ADJUSTMENT_ALPHA - 0.1) > 1e-12 or abs(SCORE_DIVISION_WEIGHT - 0.1) > 1e-12:
        raise MetricContractError("score constants drifted from host formula")
    return out


def assert_score_finite(score: float, *, context: str = "") -> float:
    if score != score or score is None:  # NaN
        raise MetricContractError(f"NaN score {context}")
    if score < 0:
        raise MetricContractError(f"negative score {score} {context}")
    return float(score)


def official_score(adj_edge_jaccard: float, division_jaccard: float) -> float:
    if any(x != x for x in (adj_edge_jaccard, division_jaccard)):
        raise MetricContractError("NaN component in official_score")
    return float(adj_edge_jaccard) + SCORE_DIVISION_WEIGHT * float(division_jaccard)


def validate_stems_for_metric(stems: list[str]) -> None:
    """Offline metric on GT must never include public-test stems."""
    reject_if_any_forbidden(stems, context="metric stems")
    overlap = set(stems) & FORBIDDEN_TEST_STEMS
    if overlap:
        raise ForbiddenStemError(f"test stems in metric set: {overlap}")
