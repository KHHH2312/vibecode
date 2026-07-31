#!/usr/bin/env python3
"""Block until a Kaggle kernel is terminal, then exit 0.

Polls every 30s. Prints only on status change + rare heartbeats.
On COMPLETE/ERROR/CANCEL: write WAKE flag, print TERMINAL, exit.
Does NOT push anything. Agent wakes when this process exits.

Usage:
  python wait_kernel_done.py
  python wait_kernel_done.py mimiiiii0/bh-lineage-reala-d1
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
DEFAULT_KERNEL = "mimiiiii0/bh-lineage-reala-d1"
INTERVAL = 30
STATUS_MD = REPO / "STATUS_NOW.md"
LOG = REPO / "wait_kernel_done.log"

# map owner username → kaggle config dir
_CFG_BY_USER = {
    "mimiiiii0": Path.home() / ".kaggle_trainD",
    "ikoooooooooooop": Path.home() / ".kaggle_trainE",
    "mijuuu8": Path.home() / ".kaggle_trainC",
    "kokiiii": Path.home() / ".kaggle_trainB",
    "khalid000000": Path.home() / ".kaggle",
}


def cfg_for_ref(ref: str) -> Path:
    if os.environ.get("KAGGLE_CONFIG_DIR"):
        return Path(os.environ["KAGGLE_CONFIG_DIR"])
    owner = ref.split("/")[0] if "/" in ref else ""
    return _CFG_BY_USER.get(owner, Path.home() / ".kaggle_trainD")


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(msg: str) -> None:
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def kernel_status(ref: str) -> str:
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(cfg_for_ref(ref))
    try:
        r = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "status", ref],
            capture_output=True,
            text=True,
            timeout=90,
            env=env,
        )
        out = ((r.stdout or "") + (r.stderr or "")).strip()
        return out.splitlines()[-1].strip() if out else "EMPTY"
    except Exception as e:
        return f"POLL_EXC {e}"


def write_status(ref: str, st: str, n: int) -> None:
    STATUS_MD.write_text(
        f"""# STATUS_NOW

Updated: **{utc()}**

| Item | Value |
|------|-------|
| Kernel | `{ref}` |
| Status | `{st}` |
| Polls | {n} (every {INTERVAL}s) |

**Mode:** wait-only. When this process exits → agent analyzes → then may push next.
**No auto-push.**
""",
        encoding="utf-8",
    )


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
    ref = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_KERNEL
    cfg = cfg_for_ref(ref)
    if not (cfg / "kaggle.json").exists():
        log(f"missing {cfg}/kaggle.json for {ref}")
        return 2

    slug = ref.split("/")[-1].replace("-", "_").upper()
    wake_flag = REPO / f"WAKE_{slug}.flag"
    wake_json = REPO / f"WAKE_{slug}.json"

    log(f"WAIT {ref} every {INTERVAL}s — exit on terminal (no push)")
    last = ""
    n = 0
    t0 = time.time()
    while True:
        n += 1
        st = kernel_status(ref)
        write_status(ref, st, n)
        elapsed_m = (time.time() - t0) / 60.0
        # Always log each poll so background task output proves 30s checks
        short = st
        if "KernelWorkerStatus." in st:
            short = st.split("KernelWorkerStatus.")[-1].strip().strip('"')
        log(f"poll#{n} +{elapsed_m:.1f}m → {short}")
        if st != last and last:
            log(f"STATUS_CHANGE {last} → {st}")
        last = st

        kind = terminal_kind(st)
        if kind:
            payload = {
                "utc": utc(),
                "kernel": ref,
                "status": st,
                "outcome": kind,
                "action": "AGENT: pull, analyze, then decide next push — no auto chain",
            }
            wake_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            wake_flag.write_text(f"{kind} {utc()}\n{st}\n", encoding="utf-8")
            log(f"TERMINAL {kind} — writing {wake_flag.name} and exiting so agent wakes")
            # exit 0 on COMPLETE, 3 on ERROR, 4 on CANCEL (agent can branch)
            return 0 if kind == "COMPLETE" else (3 if kind == "ERROR" else 4)

        time.sleep(INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
