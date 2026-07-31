import os, subprocess, sys, time, shutil
from pathlib import Path
from datetime import datetime, timezone

os.environ["KAGGLE_CONFIG_DIR"] = str(Path.home() / ".kaggle")
base = Path(r"C:\Users\Khalid\Desktop\New_folder\vibecode\lineage_v2")
LOG_PATH = base / "hybrid_py_wait.log"
out_root = base / "hybrid_submit_out"
out_root.mkdir(parents=True, exist_ok=True)

def log(msg):
    line = f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}] {msg}"
    print(line, flush=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

def run(args, timeout=120):
    r = subprocess.run([sys.executable, "-m", "kaggle"] + args, capture_output=True, text=True, timeout=timeout)
    return (r.stdout or "") + (r.stderr or ""), r.returncode

def kstatus(ref):
    out, _ = run(["kernels", "status", ref])
    return out.strip()

def submissions_head():
    out, _ = run(["competitions", "submissions", "-c", "biohub-cell-tracking-during-development", "-v"])
    return out

log("PY waiter start")
# poll A score briefly then continue to B regardless
for i in range(20):
    sub = submissions_head()
    for ln in sub.splitlines():
        if "55098457" in ln:
            log("A row: " + ln.strip())
            if "SubmissionStatus.COMPLETE" in ln or "COMPLETE" in ln:
                # score may be empty still if CSV parse odd
                parts = ln.split(",")
                log("A parts: " + str(parts[:8]))
                if any(p.strip().startswith("0.") for p in parts):
                    log("A SCORED")
                    i = 999
            break
    else:
        pass
    if i >= 999:
        break
    if i % 3 == 0:
        log("A still scoring...")
    time.sleep(30)

b = "khalid000000/bh-hyb-b-ult1yusuke"
for i in range(120):
    st = kstatus(b)
    if i % 3 == 0:
        log("B: " + st)
    if "COMPLETE" in st:
        log("B COMPLETE")
        od = out_root / "bh-hyb-b-ult1yusuke"
        if od.exists():
            shutil.rmtree(od)
        od.mkdir(parents=True)
        out, code = run(["kernels", "output", b, "-p", str(od), "--force"], timeout=600)
        log("pull code=" + str(code))
        subf = od / "submission.csv"
        if subf.exists() and subf.stat().st_size > 100000:
            log(f"B submission size={subf.stat().st_size}")
            for ver in ("2", "1", "3"):
                out, code = run([
                    "competitions", "submit",
                    "-c", "biohub-cell-tracking-during-development",
                    "-k", b,
                    "-v", ver,
                    "-f", "submission.csv",
                    "-m", f"hyb-b v{ver} ULT1+Yusuke+350ep T4",
                ], timeout=300)
                log(f"submit v{ver} code={code} out={out[-500:]}")
                if code == 0 and "400 Client Error" not in out:
                    break
        else:
            log("B no valid submission.csv")
            # list files
            for p in od.rglob("*"):
                if p.is_file() and p.stat().st_size > 1000:
                    log(f"  file {p.relative_to(od)} {p.stat().st_size}")
        break
    if "ERROR" in st or "CANCEL" in st:
        log("B FAIL " + st)
        break
    time.sleep(60)

log("Final submissions:")
log(submissions_head()[:2000])
log("DONE")
