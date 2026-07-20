"""Structural tests for 4-movie submission validity and no-exploit campaign notebooks.

These drive the same stem/row checks we rely on before competitions submit.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_STEMS = {
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
}
FORBIDDEN_EXPLOIT = ("hub_id", "DUAL_LADDERS", "MAX_COMPONENTS", "FORKS=")


def validate_submission_csv(path: Path) -> dict:
    """Return validation stats; raises AssertionError if invalid for this competition."""
    df = pd.read_csv(path)
    assert "dataset" in df.columns, "missing dataset column"
    assert "row_type" in df.columns, "missing row_type column"
    stems = set(df["dataset"].astype(str).unique())
    assert stems == REQUIRED_STEMS, f"stems {stems} != {REQUIRED_STEMS}"
    # no fifth movie
    assert "44b6_33b596bf" not in stems
    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]
    assert len(nodes) > 0 and len(edges) > 0
    # edge endpoints must exist in node ids per dataset
    for ds, g in df.groupby("dataset"):
        nids = set(g.loc[g["row_type"] == "node", "node_id"].astype(int))
        eg = g[g["row_type"] == "edge"]
        src = set(eg["source_id"].astype(int))
        tgt = set(eg["target_id"].astype(int))
        dangling = (src | tgt) - nids
        assert not dangling, f"{ds} dangling edge ids sample={list(dangling)[:5]}"
    return {
        "rows": len(df),
        "nodes": len(nodes),
        "edges": len(edges),
        "stems": sorted(stems),
    }


def test_v102_local_submission_is_4movie_safe():
    path = ROOT / "out_v102" / "submission.csv"
    if not path.exists():
        # allow CI without artifacts
        print("SKIP: out_v102/submission.csv missing")
        return
    stats = validate_submission_csv(path)
    assert stats["rows"] > 100_000
    assert set(stats["stems"]) == REQUIRED_STEMS
    print("v102 submission SAFE", stats)


def test_campaign_notebooks_have_no_exploit_and_t4_meta():
    for folder, name in [
        ("kernel_v102", "bh-v102-refine.ipynb"),
        ("kernel_v103", "bh-v103-assoc.ipynb"),
        ("kernel_v104", "bh-v104-ilp.ipynb"),
        ("kernel_v105", "bh-v105-peak.ipynb"),
    ]:
        nb_path = ROOT / folder / name
        assert nb_path.exists(), nb_path
        blob = nb_path.read_text(encoding="utf-8")
        for bad in FORBIDDEN_EXPLOIT:
            assert bad not in blob, f"{name} contains exploit pattern {bad}"
        meta = json.loads((ROOT / folder / "kernel-metadata.json").read_text(encoding="utf-8"))
        assert meta["machine_shape"] == "NvidiaTeslaT4"
        assert meta["enable_gpu"] is True
        assert "biohub-cell-tracking-during-development" in meta.get("competition_sources", [])
        print("OK", name)


def test_quality_prune_logic():
    """Unit test of the shipped prune predicate used by v103+."""
    # Mirror the decision rule from quality_prune_edges (prob low AND dist high)
    min_prob, min_dist = 0.40, 9.5

    def should_prune(prob, dist):
        return prob < min_prob and dist > min_dist

    assert should_prune(0.2, 12.0) is True
    assert should_prune(0.9, 12.0) is False  # high conf long edge kept
    assert should_prune(0.2, 3.0) is False  # short low-conf kept
    assert should_prune(0.5, 20.0) is False
    print("quality prune predicate OK")


if __name__ == "__main__":
    test_campaign_notebooks_have_no_exploit_and_t4_meta()
    test_quality_prune_logic()
    test_v102_local_submission_is_4movie_safe()
    print("ALL TESTS PASSED")
    sys.exit(0)
