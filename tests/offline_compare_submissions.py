"""Offline structural compare of two competition submissions (4-movie SAFE).

Does NOT claim publicScore. Used as better-gate evidence when LB is PENDING.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REQUIRED = {
    "44b6_0113de3b",
    "44b6_0b24845f",
    "6bba_05b6850b",
    "6bba_05db0fb1",
}
DOMINANT = "6bba_05db0fb1"


def summarize(path: Path) -> dict:
    df = pd.read_csv(path)
    assert "dataset" in df.columns and "row_type" in df.columns
    stems = set(df["dataset"].astype(str).unique())
    assert stems == REQUIRED, f"stems {stems} != {REQUIRED}"
    assert "44b6_33b596bf" not in stems
    nodes = df[df["row_type"] == "node"]
    edges = df[df["row_type"] == "edge"]
    assert len(nodes) > 0 and len(edges) > 0
    per: dict[str, dict[str, int]] = {}
    for ds, g in df.groupby("dataset"):
        nids = set(g.loc[g["row_type"] == "node", "node_id"].astype(int))
        eg = g[g["row_type"] == "edge"]
        dangling = (set(eg["source_id"].astype(int)) | set(eg["target_id"].astype(int))) - nids
        assert not dangling, f"{ds} dangling={list(dangling)[:5]}"
        per[str(ds)] = {
            "nodes": int((g["row_type"] == "node").sum()),
            "edges": int((g["row_type"] == "edge").sum()),
        }
    return {
        "path": str(path),
        "rows": int(len(df)),
        "nodes": int(len(nodes)),
        "edges": int(len(edges)),
        "per": per,
        "stems": sorted(stems),
    }


def compare(cand: Path, bank: Path) -> dict:
    c = summarize(cand)
    b = summarize(bank)
    deltas = {}
    for ds in REQUIRED:
        deltas[ds] = {
            "d_nodes": c["per"][ds]["nodes"] - b["per"][ds]["nodes"],
            "d_edges": c["per"][ds]["edges"] - b["per"][ds]["edges"],
            "cand_nodes": c["per"][ds]["nodes"],
            "bank_nodes": b["per"][ds]["nodes"],
            "cand_edges": c["per"][ds]["edges"],
            "bank_edges": b["per"][ds]["edges"],
        }
    report = {
        "candidate": c,
        "bank": b,
        "deltas": deltas,
        "dominant_movie": DOMINANT,
        "dominant_d_nodes": deltas[DOMINANT]["d_nodes"],
        "dominant_d_edges": deltas[DOMINANT]["d_edges"],
        # Heuristic only: fewer edges on dominant with similar nodes may cut FP
        # or FN — cannot decide better without metric. Structural identity => not better.
        "structurally_identical": c["rows"] == b["rows"]
        and c["nodes"] == b["nodes"]
        and c["edges"] == b["edges"]
        and all(
            deltas[ds]["d_nodes"] == 0 and deltas[ds]["d_edges"] == 0 for ds in REQUIRED
        ),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cand", required=True, type=Path)
    p.add_argument("--bank", required=True, type=Path)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)
    rep = compare(args.cand, args.bank)
    text = json.dumps(rep, indent=2)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
