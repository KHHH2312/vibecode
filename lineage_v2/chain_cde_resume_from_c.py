#!/usr/bin/env python3
"""Resume trainE CDE from already-pushed Stage C (bh-lineage-realc-e1).

Skips A_d2. Waits C COMPLETE → warm → push DE → wait DE → READY_FOR_PRIMARY.
NO SUBMIT from trainE.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cde", REPO / "chain_cde_trainE.py")
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)

# use fixed slugs from module
C_SLUG = h.C_SLUG
DE_SLUG = h.DE_SLUG


def main() -> int:
    h.log("=== RESUME trainE from Stage C (correct slug realc-e1) ===")
    state = {
        "utc_start": h.utc(),
        "user_e": h.USER_E,
        "no_submit": True,
        "no_more_ab": True,
        "phase": "resume_wait_C",
        "pushed": f"{h.USER_E}/{C_SLUG}",
        "note": "Resumed after slug fix; A_d2 already handoffed",
    }
    h.write_status(state)

    oc = h.wait_kernel(h.USER_E, C_SLUG, h.CFG_E, h.MAX_WAIT_H, "Stage C resume")
    state["C_outcome"] = oc
    h.write_status(state)
    if oc != "COMPLETE":
        try:
            h.pull_kernel(h.USER_E, C_SLUG, REPO / "out_stageC_e1", h.CFG_E)
        except Exception as e:
            h.log(f"pull C: {e}")
        state["phase"] = f"stopped_C_{oc}"
        state["utc_end"] = h.utc()
        h.write_status(state)
        return 0 if oc in ("ERROR", "CANCEL", "TIMEOUT", "MISSING") else 3

    out_c = h.pull_kernel(h.USER_E, C_SLUG, REPO / "out_stageC_e1", h.CFG_E)
    hc = h.analyze(out_c, min_step=3000)
    state["C_health"] = hc
    h.write_status(state)
    h.log(f"health C: {hc}")
    if not hc.get("ok") or not hc.get("last_pt"):
        state["phase"] = "C_health_fail_no_DE"
        state["utc_end"] = h.utc()
        h.write_status(state)
        return 4

    h.ensure_warm_on_e(Path(hc["last_pt"]), f"C step~{hc.get('final_step')}")
    h.log(f"gates OK — push Stage DE {h.USER_E}/{DE_SLUG}")
    h.push_kernel_e(DE_SLUG, "realDE_e1", h.DE_SCRIPT, h.DE_STEPS, "StageDE")
    state["pushed"] = f"{h.USER_E}/{DE_SLUG}"
    state["phase"] = "running_stageDE_e1"
    h.write_status(state)

    ode = h.wait_kernel(h.USER_E, DE_SLUG, h.CFG_E, h.MAX_WAIT_H, "Stage DE")
    state["DE_outcome"] = ode
    if ode == "COMPLETE":
        out_de = h.pull_kernel(h.USER_E, DE_SLUG, REPO / "out_stageDE_e1", h.CFG_E)
        hde = h.analyze(out_de, min_step=2000)
        state["DE_health"] = hde
        if hde.get("ok") and hde.get("last_pt"):
            h.ensure_warm_on_e(Path(hde["last_pt"]), f"DE step~{hde.get('final_step')}")
            state["phase"] = "CDE_complete_ready_for_primary"
            ready = {
                "utc": h.utc(),
                "weights": hde["last_pt"],
                "train_account": h.USER_E,
                "next": "Build ultimate dual-seed notebook on khalid000000; ≤3 submits",
                "no_submit_from_train": True,
            }
            (REPO / "READY_FOR_PRIMARY_SUBMIT.json").write_text(
                json.dumps(ready, indent=2) + "\n", encoding="utf-8"
            )
            h.log("READY_FOR_PRIMARY_SUBMIT written")
        else:
            state["phase"] = "DE_complete_health_warn"
    else:
        try:
            h.pull_kernel(h.USER_E, DE_SLUG, REPO / "out_stageDE_e1", h.CFG_E)
        except Exception as e:
            h.log(f"pull DE: {e}")
        state["phase"] = f"stopped_DE_{ode}"

    state["utc_end"] = h.utc()
    state["no_submit"] = True
    h.write_status(state)
    h.log(f"=== RESUME CDE DONE phase={state.get('phase')} ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        h.log(f"FATAL {e}")
        raise
