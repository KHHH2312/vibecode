#!/usr/bin/env python3
"""Poll until trainE CDE chain is done (READY file or DE COMPLETE). Exit 0 to wake agent.

Does NOT push or submit. Agent does primary ultimate notebook after wake.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
READY = REPO / "READY_FOR_PRIMARY_SUBMIT.json"
STATUS = REPO / "HANDOFF_TRAINE_CDE_STATUS.md"
LOG = REPO / "wait_cde_then_exit.log"
CFG_E = Path.home() / ".kaggle_trainE"
DE = "ikoooooooooooop/bh-lineage-realde-e1"
INTERVAL = 60


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(m: str) -> None:
    line = f"[{utc()}] {m}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def de_status() -> str:
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(CFG_E)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "status", DE],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
        )
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        return out.splitlines()[-1] if out else "EMPTY"
    except Exception as e:
        return f"POLL_EXC {e}"


def main() -> int:
    log("WAIT until CDE ready for primary (READY file or DE COMPLETE)")
    n = 0
    while True:
        n += 1
        if READY.exists():
            log(f"READY file present — exit wake agent for primary submits")
            return 0
        st = de_status()
        if n == 1 or n % 5 == 0:
            log(f"poll#{n} DE={st} ready={READY.exists()}")
        u = st.upper()
        if "KERNELWORKERSTATUS" in u and "COMPLETE" in u:
            # DE done — chain may still be publishing warm; give it a few minutes
            log("DE COMPLETE — wait up to 15m for READY file")
            for _ in range(15):
                if READY.exists():
                    log("READY after DE — exit")
                    return 0
                time.sleep(60)
            log("DE COMPLETE but no READY — exit anyway so agent analyzes")
            return 0
        if "KERNELWORKERSTATUS" in u and any(x in u for x in ("ERROR", "CANCEL")):
            log(f"DE terminal bad: {st} — exit for agent")
            return 3
        # also stop if status md says failed
        if STATUS.exists():
            t = STATUS.read_text(encoding="utf-8", errors="replace")
            if "stopped_" in t and "utc_end" in t and "running_" not in t.lower():
                if "CDE_complete" in t or "ready_for_primary" in t.lower():
                    log("status complete — exit")
                    return 0
        time.sleep(INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
