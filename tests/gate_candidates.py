"""Validate candidates and emit better-gate decisions (offline)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from offline_compare_submissions import compare  # noqa: E402
from test_submission_guards import validate_submission_csv  # noqa: E402


def sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def main() -> int:
    scratch = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    scratch.mkdir(parents=True, exist_ok=True)

    stats = {}
    for name in ["v102", "v103", "v104", "v105"]:
        p = ROOT / f"out_{name}" / "submission.csv"
        s = validate_submission_csv(p)
        s["sha16"] = sha16(p)
        stats[name] = s
        print(name, "SAFE", s)

    # pairwise compares
    for name in ["v104", "v105"]:
        for bank in ["v102", "v103"]:
            rep = compare(
                ROOT / f"out_{name}" / "submission.csv",
                ROOT / f"out_{bank}" / "submission.csv",
            )
            out = scratch / f"compare_{name}_{bank}.json"
            out.write_text(json.dumps(rep, indent=2) + "\n", encoding="utf-8")
            print(
                f"{name} vs {bank}: identical={rep['structurally_identical']} "
                f"dom_dn={rep['dominant_d_nodes']} dom_de={rep['dominant_d_edges']} "
                f"rows={rep['candidate']['rows']}"
            )

    # better-gate logic:
    # baseline honest real = 0.902; v102/v103 PENDING so not usable yet
    # Without real score or local patched metric on GT, only clear SKIP cases:
    # - structurally identical to already-submitted lineage => not better
    # - small unmeasured ILP knob churn without score feedback => SKIP (no positive better case)
    baseline = 0.902
    decisions = []

    # v104: ILP tweak of v103 stack — differs from v102/v103 but no metric proof of gain
    r104_v102 = json.loads((scratch / "compare_v104_v102.json").read_text(encoding="utf-8"))
    r104_v103 = json.loads((scratch / "compare_v104_v103.json").read_text(encoding="utf-8"))
    v104_decision = {
        "candidate": "bh-v104-ilp",
        "COMPLETE": True,
        "SAFE_4movie": True,
        "baseline": baseline,
        "baseline_ref": "54830671",
        "structurally_identical_to_v102": r104_v102["structurally_identical"],
        "structurally_identical_to_v103": r104_v103["structurally_identical"],
        "dominant_d_edges_vs_v103": r104_v103["dominant_d_edges"],
        "dominant_d_nodes_vs_v103": r104_v103["dominant_d_nodes"],
        "real_publicScore": None,
        "expected_better_than_baseline": False,
        "decision": "SKIP",
        "reason": (
            "No real LB score yet for v104; offline counts only show small ILP-driven "
            "graph deltas vs v103 (already submitted PENDING) without patched-metric proof "
            f"that adj_edge_jaccard > {baseline}. Positive better case not established; "
            "do not burn a submit slot."
        ),
    }
    decisions.append(v104_decision)

    r105_v102 = json.loads((scratch / "compare_v105_v102.json").read_text(encoding="utf-8"))
    r105_v103 = json.loads((scratch / "compare_v105_v103.json").read_text(encoding="utf-8"))
    # v105 is peak-only control — closer to bank; may equal/worse than full v102 refine
    v105_decision = {
        "candidate": "bh-v105-peak",
        "COMPLETE": True,
        "SAFE_4movie": True,
        "baseline": baseline,
        "baseline_ref": "54830671",
        "structurally_identical_to_v102": r105_v102["structurally_identical"],
        "structurally_identical_to_v103": r105_v103["structurally_identical"],
        "dominant_d_edges_vs_v102": r105_v102["dominant_d_edges"],
        "dominant_d_nodes_vs_v102": r105_v102["dominant_d_nodes"],
        "real_publicScore": None,
        "expected_better_than_baseline": False,
        "decision": "SKIP",
        "reason": (
            "Peak-only ablated control (intensity OFF, dense OFF). Designed to isolate "
            "subvoxel COM, not to beat bank. Offline deltas vs v102 do not prove higher "
            f"adj_edge_jaccard than {baseline}; v102 already covers peak COM + more. "
            "No positive better case → SKIP."
        ),
    }
    decisions.append(v105_decision)

    # v106: not pushed/run — skip unless future score feedback
    decisions.append(
        {
            "candidate": "bh-v106-hybrid",
            "COMPLETE": False,
            "SAFE_4movie": False,
            "baseline": baseline,
            "decision": "SKIP",
            "reason": (
                "Not run on Kaggle; hybrid of v103 levers already submitted as v103. "
                "No evidence it beats baseline beyond v103; wait for v102/v103 real scores."
            ),
        }
    )

    for d in decisions:
        name = d["candidate"].replace("bh-", "").replace("-", "_")
        path = scratch / f"better_gate_{name}.md"
        lines = [
            f"# Better-gate: {d['candidate']}",
            "",
            f"- baseline honest real: **{d.get('baseline', baseline)}** (ref {d.get('baseline_ref', '54830671')})",
            f"- decision: **{d['decision']}**",
            f"- reason: {d['reason']}",
            "",
            "```json",
            json.dumps(d, indent=2),
            "```",
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        print("WROTE", path, d["decision"])

    summary = {
        "baseline": baseline,
        "pending_campaign_refs": ["54840764", "54844836"],
        "stats": stats,
        "decisions": decisions,
        "submits_this_goal": [],
    }
    (scratch / "campaign_selective.md").write_text(
        "# Campaign selective submit log\n\n"
        f"Baseline best honest real: **{baseline}** (54830671)\n\n"
        "| candidate | COMPLETE | SAFE | better? | submitted | ref | publicScore | notes |\n"
        "|-----------|----------|------|---------|-----------|-----|-------------|-------|\n"
        "| bh-v102-refine | yes | yes | unknown (PENDING) | yes (prior) | 54840764 | PENDING | already submitted |\n"
        "| bh-v103-assoc | yes | yes | unknown (PENDING) | yes (prior) | 54844836 | PENDING | already submitted |\n"
        f"| bh-v104-ilp | yes | yes | no | **no** | — | — | {v104_decision['reason'][:80]}... |\n"
        f"| bh-v105-peak | yes | yes | no | **no** | — | — | {v105_decision['reason'][:80]}... |\n"
        "| bh-v106-hybrid | no | — | no | **no** | — | — | not run; wait for scores |\n"
        "\n"
        f"Best honest real score this window so far: **{baseline}** (prior day); campaign refs still PENDING.\n",
        encoding="utf-8",
    )
    (scratch / "gate_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print("ALL GATES WRITTEN; no submits (policy)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
