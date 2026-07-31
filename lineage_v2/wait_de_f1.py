#!/usr/bin/env python3
"""Watch Stage DE on jiiiiiiiim — NO competition submit."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["KAGGLE_CONFIG_DIR"] = str(Path.home() / ".kaggle_trainF")
REPO = Path(__file__).resolve().parent
LOG = REPO / "wait_de_f1.log"
REF = "jiiiiiiiim/bh-lineage-realde-f1"


def log(msg: str) -> None:
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def status() -> str:
    r = subprocess.run(
        [sys.executable, "-m", "kaggle", "kernels", "status", REF],
        capture_output=True,
        text=True,
        timeout=90,
    )
    return ((r.stdout or "") + (r.stderr or "")).strip()


def main() -> None:
    log(f"WATCH {REF} only (no submit)")
    for i in range(220):
        st = status()
        if i % 5 == 0 or any(x in st for x in ("COMPLETE", "ERROR", "CANCEL")):
            log(st)
        if any(x in st for x in ("COMPLETE", "ERROR", "CANCEL")):
            log("TERMINAL — no auto-submit (train worker only)")
            # optional pull on complete for health
            if "COMPLETE" in st:
                out = REPO / "out_stageDE_f1"
                out.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "kaggle",
                        "kernels",
                        "output",
                        REF,
                        "-p",
                        str(out),
                        "--force",
                    ],
                    timeout=600,
                )
                log(f"pulled to {out}")
            return
        time.sleep(60)
    log("TIMEOUT watching DE")


if __name__ == "__main__":
    main()
