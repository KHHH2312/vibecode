#!/usr/bin/env python3
"""Handoff after mijuuu8 realB_c1 → continue train on mimiiiii0 (trainD).

Rules (hard):
- Do NOT push anything while B_c1 is RUNNING
- Only after COMPLETE: pull B → health → publish src+warm on trainD → push A_d1
- CANCEL/ERROR → stop, no push
- Invalid dataset sources → ABORT (no cold 6h waste)
- NO competitions submit from trainC or trainD

Accounts:
  trainC mijuuu8   → ~/.kaggle_trainC  (running B_c1)
  trainD mimiiiii0 → ~/.kaggle_trainD  (from Downloads/kaggle (2).json)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent
LOG = REPO / "chain_handoff_trainD.log"
STATUS = REPO / "HANDOFF_TRAIND_STATUS.md"
TERMINAL = REPO / "chain_handoff_trainD_TERMINAL.json"

CFG_C = Path.home() / ".kaggle_trainC"
CFG_D = Path.home() / ".kaggle_trainD"
USER_C = "mijuuu8"
USER_D = "mimiiiii0"

B_SLUG = "bh-lineage-realb-c1"
# first session on trainD = Stage A polish from B warm
A_D_SLUG = "bh-lineage-reala-d1"
A_D_SESSION = "realA_d1"
A_D_STEPS = 70000
A_D_SCRIPT = "train_stage_a_real.py"

POLL = 180
MAX_WAIT_H = 14.0


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def log(m: str) -> None:
    line = f"[{utc()}] {m}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def write_status(d: dict) -> None:
    lines = [
        "# Handoff trainD status",
        "",
        f"Updated: **{utc()}**",
        "",
        f"- trainC (B running): `{USER_C}` — **no submit**",
        f"- trainD (next GPU): `{USER_D}` — **no submit** (kaggle (2).json)",
        "",
        "## State",
        "",
        "```json",
        json.dumps(d, indent=2)[:8000],
        "```",
        "",
        "## Rules",
        f"- Wait B COMPLETE on `{USER_C}` before any push to `{USER_D}`",
        f"- Never push without valid warm dataset on `{USER_D}`",
        f"- Never `competitions submit` from train accounts",
        "",
    ]
    STATUS.write_text("\n".join(lines) + "\n", encoding="utf-8")
    TERMINAL.write_text(json.dumps(d, indent=2) + "\n", encoding="utf-8")


def run(cmd: list[str], cfg: Path, timeout: int = 3600) -> tuple[int, str]:
    env = os.environ.copy()
    env["KAGGLE_CONFIG_DIR"] = str(cfg)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, env=env)
    return r.returncode, (r.stdout or "") + (r.stderr or "")


def kernel_status(user: str, slug: str, cfg: Path) -> str:
    _, out = run([sys.executable, "-m", "kaggle", "kernels", "status", f"{user}/{slug}"], cfg, 120)
    return out.strip().splitlines()[-1] if out.strip() else out


def wait_kernel(user: str, slug: str, cfg: Path, max_h: float, label: str) -> str:
    log(f"WAIT {user}/{slug} ({label}) — NO PUSH while waiting")
    deadline = time.time() + max_h * 3600
    last = ""
    n = 0
    while time.time() < deadline:
        n += 1
        try:
            st = kernel_status(user, slug, cfg)
        except Exception as e:
            st = f"POLL_EXC {e}"
        if st != last:
            log(st)
            last = st
        elif n % 10 == 0:
            log(f"heartbeat still: {st}")
        u = st.upper()
        if "COMPLETE" in u and "KERNELWORKERSTATUS" in u:
            return "COMPLETE"
        if "ERROR" in u and "KERNELWORKERSTATUS" in u:
            return "ERROR"
        if "CANCEL" in u and "KERNELWORKERSTATUS" in u:
            return "CANCEL"
        time.sleep(POLL)
    return "TIMEOUT"


def pull_kernel(user: str, slug: str, out_dir: Path, cfg: Path) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    code, out = run(
        [sys.executable, "-m", "kaggle", "kernels", "output", f"{user}/{slug}", "-p", str(out_dir), "--force"],
        cfg,
        1800,
    )
    log(f"pull {user}/{slug} rc={code} nfiles={len(list(out_dir.rglob('*')))}")
    if code != 0:
        log(f"pull warn: {out[-400:]}")
    return out_dir


def analyze(out_dir: Path, min_step: int = 1000) -> dict:
    log_path = next(iter(sorted(out_dir.rglob("*.log"), key=lambda p: -p.stat().st_size)), None)
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path else ""
    last_pts = list(out_dir.rglob("last.pt"))
    steps = [int(x) for x in re.findall(r"step=(\d+)", text)]
    nonf = len(re.findall(r"nonfinite", text, re.I))
    abort = bool(re.search(r"\bABORT\b", text))
    train_loop = "TRAIN_LOOP_START" in text
    final_step = max(steps) if steps else 0
    sc = list(out_dir.rglob("session_complete.json"))
    if sc:
        try:
            j = json.loads(sc[0].read_text(encoding="utf-8"))
            final_step = max(final_step, int(j.get("steps") or 0))
        except Exception:
            pass
    ok = bool(last_pts) and train_loop and final_step >= min_step and not (abort and final_step < min_step)
    if sc and last_pts and not ok:
        try:
            j = json.loads(sc[0].read_text(encoding="utf-8"))
            if int(j.get("steps") or 0) >= min_step:
                ok = True
                final_step = max(final_step, int(j.get("steps") or 0))
        except Exception:
            pass
    return {
        "ok": ok,
        "final_step": final_step,
        "nonfinite_n": nonf,
        "aborted": abort,
        "train_loop": train_loop,
        "has_last_pt": bool(last_pts),
        "last_pt": str(last_pts[0]) if last_pts else None,
        "log": str(log_path) if log_path else None,
    }


def publish_dataset_d(folder: Path, ds_slug: str, title: str) -> None:
    meta = {"title": title, "id": f"{USER_D}/{ds_slug}", "licenses": [{"name": "CC0-1.0"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    code, out = run(
        [sys.executable, "-m", "kaggle", "datasets", "create", "-p", str(folder), "--dir-mode", "zip"],
        CFG_D,
        3600,
    )
    log(f"dataset create {USER_D}/{ds_slug} rc={code} {out[-300:]}")
    low = out.lower()
    if code != 0 or "already" in low or "exists" in low or "in use" in low:
        code2, out2 = run(
            [
                sys.executable,
                "-m",
                "kaggle",
                "datasets",
                "version",
                "-p",
                str(folder),
                "-m",
                f"handoff {utc()}",
                "--dir-mode",
                "zip",
            ],
            CFG_D,
            3600,
        )
        log(f"dataset version {USER_D}/{ds_slug} rc={code2} {out2[-300:]}")
        if code2 != 0 and "error" in out2.lower() and "403" in out2:
            raise SystemExit(f"ABORT dataset publish {USER_D}/{ds_slug}: {out2[-400:]}")
    # settle so kernel attach can resolve
    time.sleep(45)


def ensure_src_on_d() -> None:
    src = REPO / "dataset_src_trainD"
    if src.exists():
        shutil.rmtree(src)
    base = REPO / "dataset_src_kokiiii"
    if not base.exists():
        base = REPO / "dataset_src_trainC"
    if not base.exists():
        base = REPO / "dataset_src"
    if not base.exists():
        raise SystemExit("ABORT: no dataset_src tree for trainD")
    shutil.copytree(base, src)
    live = REPO / "lineage_v2"
    dest_pkg = src / "lineage_v2"
    if dest_pkg.exists():
        shutil.rmtree(dest_pkg)
    if live.exists():
        shutil.copytree(
            live,
            dest_pkg,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"),
        )
    (src / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": "lineage-v2-src", "id": f"{USER_D}/lineage-v2-src", "licenses": [{"name": "CC0-1.0"}]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log("publishing lineage-v2-src to trainD…")
    publish_dataset_d(src, "lineage-v2-src", "lineage-v2-src")


def ensure_warm_on_d(last_pt: Path, label: str) -> None:
    if not last_pt.exists() or last_pt.stat().st_size < 1_000_000:
        raise SystemExit(f"ABORT warm: bad last.pt {last_pt}")
    wds = REPO / "dataset_warm_trainD"
    if wds.exists():
        shutil.rmtree(wds)
    wds.mkdir(parents=True)
    shutil.copy2(last_pt, wds / "last.pt")
    best = last_pt.parent / "best_exact.pt"
    if best.exists():
        shutil.copy2(best, wds / "best_exact.pt")
    for extra in ("session_complete.json", "train_meta.json"):
        p = last_pt.parent / extra
        if p.exists():
            shutil.copy2(p, wds / extra)
    (wds / "handoff.json").write_text(
        json.dumps(
            {
                "from": label,
                "source": str(last_pt),
                "utc": utc(),
                "no_submit": True,
                "to_account": USER_D,
                "from_account": USER_C,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    sz = (wds / "last.pt").stat().st_size
    log(f"publishing warm weights ({label}) size={sz} to trainD…")
    publish_dataset_d(wds, "lineage-v2-warm", "lineage-v2-warm-weights")
    if not (wds / "last.pt").exists() or (wds / "last.pt").stat().st_size < 1_000_000:
        raise SystemExit("ABORT warm staging last.pt bad after publish")


def make_notebook(script: Path, session_id: str, steps: int) -> str:
    src = script.read_text(encoding="utf-8")
    pre = f"""import os
os.environ.setdefault("BIOHUB_SESSION_ID", "{session_id}")
os.environ.setdefault("BIOHUB_SESSION_STEPS", "{steps}")
os.environ.setdefault("BIOHUB_MAX_STEPS", "{steps}")
os.environ.setdefault("BIOHUB_TIME_BUDGET_H", "11.0")
os.environ.setdefault("BIOHUB_REQUIRE_REAL", "1")
os.environ.setdefault("BIOHUB_SEED", "94017")
os.environ.setdefault("BIOHUB_USE_AMP", "0")
os.environ.setdefault("BIOHUB_LR", "5e-5")
print("TRAIN_D {session_id} steps={steps} AMP=OFF account={USER_D} — NO SUBMIT")
"""
    full = pre + "\n" + src

    def cell(typ: str, s: str) -> dict:
        d = {
            "cell_type": typ,
            "metadata": {},
            "id": uuid.uuid4().hex[:12],
            "source": s.splitlines(keepends=True),
        }
        if typ == "code":
            d["outputs"] = []
            d["execution_count"] = None
        return d

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}
        },
        "cells": [
            cell(
                "markdown",
                f"# {session_id} trainD handoff\n\nAccount `{USER_D}` from kaggle (2).json. Train-only. **NO SUBMIT.**\n",
            ),
            cell("code", full),
        ],
    }
    return json.dumps(nb, indent=1)


def push_kernel_d(slug: str, session_id: str, script_name: str, steps: int, retries: int = 3) -> None:
    kdir = REPO / f"kernel_{session_id}"
    kdir.mkdir(parents=True, exist_ok=True)
    script = REPO / "kernel_real_max" / script_name
    if not script.exists():
        raise SystemExit(f"missing {script}")
    nb_name = f"bh-lineage-{session_id.lower()}.ipynb"
    (kdir / nb_name).write_text(make_notebook(script, session_id, steps), encoding="utf-8")
    meta = {
        "id": f"{USER_D}/{slug}",
        "title": f"bh-lineage-{session_id}",
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [
            f"{USER_D}/lineage-v2-src",
            f"{USER_D}/lineage-v2-warm",
        ],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    last_out = ""
    for attempt in range(1, retries + 1):
        code, out = run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir)], CFG_D, 300)
        last_out = out
        log(f"push {USER_D}/{slug} attempt={attempt} rc={code} {out[-400:]}")
        low = out.lower()
        if "not valid dataset" in low or "could not be added" in low:
            raise SystemExit(
                f"ABORT: invalid dataset sources on push {USER_D}/{slug}. "
                f"Will NOT burn GPU without warm. out={out[-600:]}"
            )
        if code == 0 and ("successfully" in low or "pushed" in low or not out.strip()):
            return
        if code == 0:
            return
        time.sleep(20 * attempt)
    raise SystemExit(f"push failed after retries: {last_out[-800:]}")


def main() -> int:
    if not (CFG_C / "kaggle.json").exists():
        raise SystemExit(f"missing {CFG_C}/kaggle.json")
    if not (CFG_D / "kaggle.json").exists():
        raise SystemExit(f"missing {CFG_D}/kaggle.json — copy Downloads/kaggle (2).json there")

    # verify trainD identity
    dmeta = json.loads((CFG_D / "kaggle.json").read_text(encoding="utf-8"))
    if dmeta.get("username") != USER_D:
        raise SystemExit(f"trainD username mismatch: expected {USER_D}, got {dmeta.get('username')}")

    state: dict = {
        "utc_start": utc(),
        "user_c": USER_C,
        "user_d": USER_D,
        "no_submit": True,
        "phase": "waiting_realB_c1_on_trainC",
        "note": "NO PUSH until B COMPLETE; next notebook on mimiiiii0 only",
        "b_slug": f"{USER_C}/{B_SLUG}",
        "next_slug": f"{USER_D}/{A_D_SLUG}",
        "gate": "COMPLETE + health + valid warm on trainD",
    }
    write_status(state)
    log(f"=== HANDOFF trainD: wait {USER_C}/{B_SLUG} then push {USER_D}/{A_D_SLUG} ===")

    # 1) Wait only — never push here
    outcome = wait_kernel(USER_C, B_SLUG, CFG_C, MAX_WAIT_H, "B_c1 finish only")
    state["realB_c1_outcome"] = outcome
    write_status(state)

    if outcome != "COMPLETE":
        log(f"B_c1={outcome} — NOT pushing trainD notebook")
        try:
            pull_kernel(USER_C, B_SLUG, REPO / "out_realB_c1", CFG_C)
        except Exception as e:
            log(f"pull after {outcome}: {e}")
        state["phase"] = f"stopped_B_{outcome}_no_push"
        state["utc_end"] = utc()
        write_status(state)
        return 0

    # 2) Pull + health on trainC output
    log("B_c1 COMPLETE — pull + health (still no push yet)")
    state["phase"] = "pull_health_B"
    write_status(state)
    out_dir = pull_kernel(USER_C, B_SLUG, REPO / "out_realB_c1", CFG_C)
    h = analyze(out_dir, min_step=3000)
    state["B_health"] = h
    write_status(state)
    log(f"health B: {h}")

    if not h.get("ok") or not h.get("last_pt"):
        state["phase"] = "B_health_fail_no_push"
        state["utc_end"] = utc()
        write_status(state)
        log("ABORT: B health fail — no trainD push")
        return 4

    pt = Path(h["last_pt"])
    if not pt.exists() or pt.stat().st_size < 1_000_000:
        state["phase"] = "B_bad_weights_no_push"
        state["utc_end"] = utc()
        write_status(state)
        log("ABORT: bad last.pt — no trainD push")
        return 4

    # 3) Publish src + warm to trainD (private datasets don't cross accounts)
    state["phase"] = "publish_src_warm_trainD"
    write_status(state)
    ensure_src_on_d()
    ensure_warm_on_d(pt, f"B_c1 step~{h.get('final_step')}")

    # 4) Only now push A on trainD
    state["phase"] = "push_realA_d1"
    write_status(state)
    log(f"gates passed — pushing {USER_D}/{A_D_SLUG}")
    push_kernel_d(A_D_SLUG, A_D_SESSION, A_D_SCRIPT, A_D_STEPS)
    state["pushed"] = f"{USER_D}/{A_D_SLUG}"
    state["phase"] = "running_realA_d1"
    write_status(state)

    # 5) Wait for A_d1
    outcome2 = wait_kernel(USER_D, A_D_SLUG, CFG_D, MAX_WAIT_H, "session realA_d1")
    state["realA_d1_outcome"] = outcome2
    if outcome2 == "COMPLETE":
        pull_kernel(USER_D, A_D_SLUG, REPO / "out_realA_d1", CFG_D)
        h2 = analyze(REPO / "out_realA_d1", min_step=2000)
        state["A_d1_health"] = h2
        if h2.get("ok") and h2.get("last_pt"):
            ensure_warm_on_d(Path(h2["last_pt"]), f"A_d1 step~{h2.get('final_step')}")
            state["phase"] = "trainD_A_d1_complete_warm_ready"
        else:
            state["phase"] = "trainD_A_d1_complete_health_warn"
    else:
        try:
            pull_kernel(USER_D, A_D_SLUG, REPO / "out_realA_d1", CFG_D)
        except Exception as e:
            log(f"pull A_d1: {e}")
        state["phase"] = f"stopped_A_d1_{outcome2}"
    state["utc_end"] = utc()
    state["no_submit"] = True
    write_status(state)
    log(f"=== HANDOFF trainD DONE phase={state.get('phase')} ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log(f"FATAL {e}")
        raise
