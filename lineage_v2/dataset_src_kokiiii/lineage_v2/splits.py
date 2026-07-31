"""Deterministic leak-free manifests (plan.md seed 94017)."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .constants import FORBIDDEN_TEST_STEMS, SPLIT_SEED
from .denylist import ForbiddenStemError, reject_if_any_forbidden


@dataclass
class Manifest:
    seed: int
    n_permitted: int
    confirmation: list[str]
    folds: list[list[str]]  # 3 outer folds (validation stems for each fold)
    train_pools: list[list[str]]  # train = all_dev \\ fold_i
    calibration: list[list[str]]  # 10% of each train pool
    meta: dict[str, Any]

    def sha256(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()


def _prefix(stem: str) -> str:
    return stem.split("_", 1)[0]


def _division_bin(n_div: int) -> str:
    if n_div <= 0:
        return "0"
    if n_div == 1:
        return "1"
    return "ge2"


def build_manifest(
    stems: list[str],
    *,
    division_counts: dict[str, int] | None = None,
    edge_counts: dict[str, int] | None = None,
    seed: int = SPLIT_SEED,
    n_confirmation: int = 24,
    n_conf_44b6: int = 8,
    n_folds: int = 3,
) -> Manifest:
    """Build stratified confirmation + 3 OOF folds.

    If division_counts/edge_counts missing, uses prefix-only stratification.
    """
    stems = sorted(set(stems))
    reject_if_any_forbidden(stems, context="build_manifest input")
    for s in stems:
        if s in FORBIDDEN_TEST_STEMS:
            raise ForbiddenStemError(s)

    division_counts = division_counts or {s: 0 for s in stems}
    edge_counts = edge_counts or {s: 0 for s in stems}

    rng = random.Random(seed)

    # Stratification key
    def key(s: str) -> tuple:
        edges = edge_counts.get(s, 0)
        return (_prefix(s), _division_bin(division_counts.get(s, 0)), edges)

    # Confirmation: exactly 8 44b6 + 16 6bba when available
    by_pref: dict[str, list[str]] = {"44b6": [], "6bba": [], "other": []}
    for s in stems:
        p = _prefix(s)
        by_pref.setdefault(p if p in ("44b6", "6bba") else "other", []).append(s)
    for p in by_pref:
        by_pref[p].sort(key=lambda s: (key(s), s))
        rng.shuffle(by_pref[p])

    conf: list[str] = []
    take_44 = min(n_conf_44b6, len(by_pref.get("44b6", [])))
    take_6b = min(n_confirmation - take_44, len(by_pref.get("6bba", [])))
    conf.extend(by_pref["44b6"][:take_44])
    conf.extend(by_pref["6bba"][:take_6b])
    # fill if short
    remaining_pool = [s for s in stems if s not in conf]
    rng.shuffle(remaining_pool)
    while len(conf) < n_confirmation and remaining_pool:
        conf.append(remaining_pool.pop())
    conf = sorted(conf[:n_confirmation])

    dev = sorted(s for s in stems if s not in conf)
    # 3 outer folds — stratified round-robin by prefix+div bin
    buckets: dict[tuple, list[str]] = {}
    for s in dev:
        buckets.setdefault((_prefix(s), _division_bin(division_counts.get(s, 0))), []).append(s)
    for b in buckets.values():
        rng.shuffle(b)

    folds: list[list[str]] = [[] for _ in range(n_folds)]
    # round-robin each bucket
    for b in sorted(buckets.keys()):
        items = buckets[b]
        for i, s in enumerate(items):
            folds[i % n_folds].append(s)
    folds = [sorted(f) for f in folds]

    train_pools: list[list[str]] = []
    calibration: list[list[str]] = []
    for i in range(n_folds):
        val = set(folds[i])
        train = sorted(s for s in dev if s not in val)
        cal_n = max(1, int(round(0.10 * len(train))))
        train_shuffled = train[:]
        rng.shuffle(train_shuffled)
        cal = sorted(train_shuffled[:cal_n])
        train_pools.append(train)
        calibration.append(cal)

    m = Manifest(
        seed=seed,
        n_permitted=len(stems),
        confirmation=conf,
        folds=folds,
        train_pools=train_pools,
        calibration=calibration,
        meta={
            "n_confirmation": len(conf),
            "n_dev": len(dev),
            "n_44b6_conf": sum(1 for s in conf if s.startswith("44b6")),
            "n_6bba_conf": sum(1 for s in conf if s.startswith("6bba")),
            "forbidden_excluded": sorted(FORBIDDEN_TEST_STEMS),
        },
    )
    return m


def save_manifest(m: Manifest, path: Path) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = asdict(m)
    d["manifest_sha256"] = m.sha256()
    path.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")
    return d["manifest_sha256"]


def load_manifest(path: Path) -> Manifest:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    d.pop("manifest_sha256", None)
    meta = d.pop("meta", {})
    m = Manifest(meta=meta, **d)
    reject_if_any_forbidden(
        m.confirmation + [s for f in m.folds for s in f],
        context="loaded manifest",
    )
    return m
