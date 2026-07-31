"""Hard denylist for public-test GT stems (fail closed)."""
from __future__ import annotations

from pathlib import Path

from .constants import FORBIDDEN_TEST_STEMS


class ForbiddenStemError(RuntimeError):
    """Raised when a path or stem touches denylisted test GT."""


def stem_from_path(path: str | Path) -> str | None:
    """Best-effort stem id from a path (directory or .geff name)."""
    p = Path(path)
    name = p.name
    if name.endswith(".geff"):
        name = name[: -len(".geff")]
    # e.g. 44b6_0113de3b or nested .../44b6_0113de3b/...
    parts = list(p.parts) + [name]
    for part in reversed(parts):
        if part in FORBIDDEN_TEST_STEMS:
            return part
        if part.endswith(".geff") and part[: -len(".geff")] in FORBIDDEN_TEST_STEMS:
            return part[: -len(".geff")]
    return None


def assert_stem_allowed(stem: str, *, context: str = "") -> None:
    if stem in FORBIDDEN_TEST_STEMS:
        raise ForbiddenStemError(
            f"FORBIDDEN test stem {stem!r} in {context or 'operation'} — "
            "must not use GT for train/cal/metric/cache"
        )


def assert_path_allowed_for_gt(path: str | Path, *, context: str = "") -> None:
    """Call before opening any GT .geff / annotation for training or offline metric."""
    stem = stem_from_path(path)
    if stem is not None:
        assert_stem_allowed(stem, context=context or str(path))


def filter_stems(stems: list[str]) -> list[str]:
    """Drop forbidden stems; raise if any present when strict=False not set — drops silently for listing."""
    return [s for s in stems if s not in FORBIDDEN_TEST_STEMS]


def reject_if_any_forbidden(stems: list[str], *, context: str = "") -> None:
    bad = [s for s in stems if s in FORBIDDEN_TEST_STEMS]
    if bad:
        raise ForbiddenStemError(f"forbidden stems in {context}: {bad}")
