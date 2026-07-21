#!/usr/bin/env python3
"""Poll bh-v112-v34 kernel status until COMPLETE/ERROR/CANCELED, then stop."""
import os, re, sys, time, subprocess
from pathlib import Path

SLUG = "khalid000000/bh-v112-v34"
env = {**os.environ}
env.setdefault("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))

t0 = time.time()
TIMEOUT = 4 * 3600
while time.time() - t0 < TIMEOUT:
    r = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "status", SLUG],
                       text=True, capture_output=True, env=env)
    blob = (r.stdout or "") + (r.stderr or "")
    m = re.search(r'status\s+"?([A-Za-z_.]+)"?', blob)
    status = (m.group(1) if m else blob.strip())[:60]
    low = status.lower()
    el = int(time.time() - t0)
    print(f"[{el:5d}s] {status}", flush=True)
    if "complete" in low:
        print("RESULT: COMPLETE", flush=True); sys.exit(0)
    if "error" in low or "cancel" in low:
        print(f"RESULT: {status}", flush=True); sys.exit(2)
    time.sleep(90)
print("RESULT: TIMEOUT", flush=True)
sys.exit(3)
