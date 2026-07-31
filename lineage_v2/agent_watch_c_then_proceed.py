#!/usr/bin/env python3
"""Watch Stage C until terminal; report chain DE handoff. Exit so agent wakes.

Does NOT push. chain_cde_resume_from_c.py owns C→DE push.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
REF = "ikoooooooooooop/bh-lineage-realc-e1"
DE_REF = "ikoooooooooooop/bh-lineage-realde-e1"
CFG = Path.home() / ".kaggle_trainE"
LOG = REPO / "agent_watch_c.log"
INTERVAL = 30


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def kstatus(ref: str) -> str:
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(CFG)
    r = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "status", ref],
        capture_output=True,
        text=True,
        timeout=90,
        env=env,
    )
    out = ((r.stdout or "") + (r.stderr or "")).strip()
    return out.splitlines()[-1] if out else "EMPTY"


def short_status(st: str) -> str:
    if "KernelWorkerStatus." in st:
        return st.split("KernelWorkerStatus.")[-1].strip().strip('"')
    return st[:160]


def terminal_kind(st: str) -> str | None:
    u = st.upper()
    if "KERNELWORKERSTATUS" not in u:
        return None
    if "COMPLETE" in u:
        return "COMPLETE"
    if "ERROR" in u:
        return "ERROR"
    if "CANCEL" in u:
        return "CANCEL"
    return None


def main() -> int:
    if not (CFG / "kaggle.json").exists():
        log(f"missing {CFG}/kaggle.json")
        return 2

    log(f"AGENT_WATCH {REF} every {INTERVAL}s")
    last = ""
    n = 0
    t0 = time.time()
    while True:
        n += 1
        st = kstatus(REF)
        short = short_status(st)
        if n == 1 or short != last or n % 20 == 0:
            log(f"poll#{n} +{(time.time() - t0) / 60:.1f}m -> {short}")
        last = short

        kind = terminal_kind(st)
        if not kind:
            time.sleep(INTERVAL)
            continue

        log(f"TERMINAL Stage C: {kind} ({short})")
        wake = {
            "utc": utc(),
            "kernel": REF,
            "outcome": kind,
            "status": st,
            "action": "Agent: verify C health; chain should push DE if ok; then DE→primary",
        }
        (REPO / "WAKE_AGENT_STAGE_C.json").write_text(
            json.dumps(wake, indent=2) + "\n", encoding="utf-8"
        )

        if kind != "COMPLETE":
            log("C not COMPLETE — agent must decide (no blind DE)")
            return 3 if kind == "ERROR" else 4

        log("C COMPLETE — wait up to 20m for chain warm+DE push")
        for i in range(40):
            time.sleep(INTERVAL)
            de_st = kstatus(DE_REF)
            de_short = short_status(de_st)
            ready = (REPO / "READY_FOR_PRIMARY_SUBMIT.json").exists()
            handoff = ""
            hp = REPO / "HANDOFF_TRAINE_CDE_STATUS.md"
            if hp.exists():
                handoff = hp.read_text(encoding="utf-8", errors="replace")
            phase = "?"
            if '"phase"' in handoff:
                try:
                    # rough extract
                    idx = handoff.find('"phase"')
                    phase = handoff[idx : idx + 80].replace("\n", " ")
                except Exception:
                    pass
            log(
                f"postC#{i + 1} DE={de_short} ready={ready} {phase}"
            )
            u = de_st.upper()
            advanced = (
                "RUNNING" in u
                or "COMPLETE" in u
                or "QUEUED" in u
                or ready
                or "running_stageDE" in handoff
                or "C_health_fail" in handoff
                or "stopped_C" in handoff
                or "CDE_complete" in handoff
            )
            if advanced:
                log("chain advanced or stopped — agent continue plan")
                break
        else:
            log("WARN chain did not advance in 20m — agent must investigate")

        # dump tails for agent
        for name in (
            "STATUS_NOW.md",
            "HANDOFF_TRAINE_CDE_STATUS.md",
            "chain_cde_resume_stdout.log",
            "chain_cde_trainE.log",
            "READY_FOR_PRIMARY_SUBMIT.json",
        ):
            fp = REPO / name
            if fp.exists():
                log(f"--- {name} ---")
                log(fp.read_text(encoding="utf-8", errors="replace")[-2500:])

        return 0


if __name__ == "__main__":
    raise SystemExit(main())
