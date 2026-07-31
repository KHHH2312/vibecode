#!/usr/bin/env python3
"""After A_d2 COMPLETE → trainE (ikoooooooooooop): Stage C then D+E only.

NO more Stage A/B on trainE.
NO competitions submit from trainE.
After DE COMPLETE: mark READY_FOR_PRIMARY_SUBMIT.json for ultimate notebook phase.
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
LOG = REPO / "chain_cde_trainE.log"
STATUS = REPO / "HANDOFF_TRAINE_CDE_STATUS.md"
TERMINAL = REPO / "chain_cde_trainE_TERMINAL.json"

CFG_D = Path.home() / ".kaggle_trainD"  # mimiiiii0 — A_d2
CFG_E = Path.home() / ".kaggle_trainE"  # ikoooooooooooop — C/DE
USER_D = "mimiiiii0"
USER_E = "ikoooooooooooop"

A_D2_SLUG = "bh-lineage-reala-d2"
# slugs must match notebook titles (Kaggle slugifies title)
C_SLUG = "bh-lineage-realc-e1"
DE_SLUG = "bh-lineage-realde-e1"
C_SCRIPT = REPO / "kernel_train11h" / "train_stage_c.py"
DE_SCRIPT = REPO / "kernel_train11h" / "train_stage_de.py"
# Time-gated dual C: high step cap; real stop is BIOHUB_TIME_BUDGET_H (~11.85h → ~11h45 wall)
C_STEPS = 200000
DE_STEPS = 200000

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
    STATUS.write_text(
        f"# trainE C/DE status\n\nUpdated: **{utc()}**\n\n"
        f"- trainD A_d2: `{USER_D}` (source warm)\n"
        f"- trainE C/DE: `{USER_E}` — **no submit**, **no more A/B**\n"
        f"- primary submit later: `khalid000000` ≤3\n\n"
        f"```json\n{json.dumps(d, indent=2)[:8000]}\n```\n",
        encoding="utf-8",
    )
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
        # wrong slug / not created yet — do not spin 14h
        if "PERMISSION" in u or "DENIED" in u or "CANNOT ACCESS" in u:
            log(f"WARN cannot access kernel (bad slug?): {st}")
            if n >= 5:
                return "MISSING"
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


def analyze(out_dir: Path, min_step: int) -> dict:
    log_path = next(iter(sorted(out_dir.rglob("*.log"), key=lambda p: -p.stat().st_size)), None)
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path else ""
    last_pts = list(out_dir.rglob("last.pt"))
    steps = [int(x) for x in re.findall(r"step=(\d+)", text)]
    final_step = max(steps) if steps else 0
    sc = list(out_dir.rglob("session_complete.json"))
    if sc:
        try:
            j = json.loads(sc[0].read_text(encoding="utf-8"))
            final_step = max(final_step, int(j.get("steps") or 0))
        except Exception:
            pass
    nonf = len(re.findall(r"nonfinite", text, re.I))
    abort = bool(re.search(r"\bABORT\b", text))
    ok = bool(last_pts) and final_step >= min_step and not (abort and final_step < min_step)
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
        "has_last_pt": bool(last_pts),
        "last_pt": str(last_pts[0]) if last_pts else None,
    }


def publish_dataset_e(folder: Path, ds_slug: str, title: str) -> None:
    meta = {"title": title, "id": f"{USER_E}/{ds_slug}", "licenses": [{"name": "CC0-1.0"}]}
    (folder / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    code, out = run(
        [sys.executable, "-m", "kaggle", "datasets", "create", "-p", str(folder), "--dir-mode", "zip"],
        CFG_E,
        3600,
    )
    log(f"dataset create {USER_E}/{ds_slug} rc={code} {out[-300:]}")
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
                f"cde {utc()}",
                "--dir-mode",
                "zip",
            ],
            CFG_E,
            3600,
        )
        log(f"dataset version {USER_E}/{ds_slug} rc={code2} {out2[-300:]}")
        if code2 != 0 and ("403" in out2 or "error" in out2.lower()):
            raise SystemExit(f"ABORT dataset {USER_E}/{ds_slug}: {out2[-400:]}")
    time.sleep(45)


def ensure_src_on_e() -> None:
    src = REPO / "dataset_src_trainE"
    if src.exists():
        shutil.rmtree(src)
    base = REPO / "dataset_src_kokiiii"
    if not base.exists():
        base = REPO / "dataset_src_trainC"
    if not base.exists():
        base = REPO / "dataset_src_trainD"
    if not base.exists():
        base = REPO / "dataset_src"
    if not base.exists():
        raise SystemExit("ABORT: no dataset_src tree")
    shutil.copytree(base, src)
    live = REPO / "lineage_v2"
    dest = src / "lineage_v2"
    if dest.exists():
        shutil.rmtree(dest)
    if live.exists():
        shutil.copytree(live, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache"))
    (src / "dataset-metadata.json").write_text(
        json.dumps(
            {"title": "lineage-v2-src", "id": f"{USER_E}/lineage-v2-src", "licenses": [{"name": "CC0-1.0"}]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    log("publishing lineage-v2-src to trainE…")
    publish_dataset_e(src, "lineage-v2-src", "lineage-v2-src")


def ensure_warm_on_e(last_pt: Path, label: str) -> None:
    if not last_pt.exists() or last_pt.stat().st_size < 1_000_000:
        raise SystemExit(f"ABORT warm bad {last_pt}")
    wds = REPO / "dataset_warm_trainE"
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
        json.dumps({"from": label, "utc": utc(), "no_submit": True, "account": USER_E}, indent=2) + "\n",
        encoding="utf-8",
    )
    log(f"publishing warm ({label}) size={last_pt.stat().st_size} to trainE…")
    publish_dataset_e(wds, "lineage-v2-warm", "lineage-v2-warm-weights")


def make_notebook(script: Path, session_id: str, steps: int, stage_label: str) -> str:
    src = script.read_text(encoding="utf-8")
    pre = f"""import os
os.environ.setdefault("BIOHUB_SESSION_ID", "{session_id}")
os.environ.setdefault("BIOHUB_SESSION_STEPS", "{steps}")
os.environ.setdefault("BIOHUB_MAX_STEPS", "{steps}")
# 12h Kaggle cap → train to ~11h45 (budget 11.85h, stop_remain 0.12h for final save)
os.environ.setdefault("BIOHUB_TIME_BUDGET_H", "11.85")
os.environ.setdefault("BIOHUB_STOP_REMAIN_H", "0.12")
os.environ.setdefault("BIOHUB_SEED", "94017")
os.environ.setdefault("BIOHUB_USE_AMP", "0")
os.environ.setdefault("BIOHUB_DUAL_GPU", "1")
os.environ.setdefault("BIOHUB_BATCH", "4")
print(
    f"TRAIN_E {stage_label} {session_id} dual_gpu batch=4 time_budget=11.85h "
    f"steps={steps} account={USER_E} — NO SUBMIT — max samples under 12h cap"
)
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
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": [
            cell("markdown", f"# {session_id} trainE C/DE\n\nAccount `{USER_E}`. Train-only. **NO SUBMIT.**\n"),
            cell("code", full),
        ],
    }
    return json.dumps(nb, indent=1)


def push_kernel_e(slug: str, session_id: str, script: Path, steps: int, stage_label: str) -> None:
    if not script.exists():
        raise SystemExit(f"missing {script}")
    kdir = REPO / f"kernel_{session_id}"
    kdir.mkdir(parents=True, exist_ok=True)
    nb_name = f"bh-lineage-{session_id.lower()}.ipynb"
    (kdir / nb_name).write_text(make_notebook(script, session_id, steps, stage_label), encoding="utf-8")
    meta = {
        "id": f"{USER_E}/{slug}",
        "title": f"bh-lineage-{session_id}",
        "code_file": nb_name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [f"{USER_E}/lineage-v2-src", f"{USER_E}/lineage-v2-warm"],
        "competition_sources": ["biohub-cell-tracking-during-development"],
        "kernel_sources": [],
        "model_sources": [],
        "machine_shape": "NvidiaTeslaT4",
    }
    (kdir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    for attempt in range(1, 4):
        code, out = run([sys.executable, "-m", "kaggle", "kernels", "push", "-p", str(kdir)], CFG_E, 300)
        log(f"push {USER_E}/{slug} attempt={attempt} rc={code} {out[-400:]}")
        low = out.lower()
        if "not valid dataset" in low or "could not be added" in low:
            raise SystemExit(f"ABORT invalid datasets on push: {out[-600:]}")
        if code == 0 and ("successfully" in low or "pushed" in low or not out.strip()):
            return
        if code == 0:
            return
        time.sleep(20 * attempt)
    raise SystemExit(f"push failed {slug}")


def main() -> int:
    if not (CFG_E / "kaggle.json").exists():
        raise SystemExit(f"missing {CFG_E}/kaggle.json")
    if not (CFG_D / "kaggle.json").exists():
        raise SystemExit(f"missing {CFG_D}/kaggle.json")
    u = json.loads((CFG_E / "kaggle.json").read_text(encoding="utf-8")).get("username")
    if u != USER_E:
        raise SystemExit(f"trainE user mismatch {u} != {USER_E}")

    state: dict = {
        "utc_start": utc(),
        "user_e": USER_E,
        "no_submit": True,
        "no_more_ab": True,
        "phase": "wait_A_d2",
        "plan": ["A_d2 source", "stageC-e1", "stageDE-e1", "primary ultimate submit later"],
    }
    write_status(state)
    log("=== trainE C/DE: wait A_d2 → C → DE (no A/B, no submit) ===")

    # 1) Wait A_d2 on trainD
    o = wait_kernel(USER_D, A_D2_SLUG, CFG_D, MAX_WAIT_H, "A_d2 finish only")
    state["A_d2_outcome"] = o
    write_status(state)
    if o != "COMPLETE":
        state["phase"] = f"stopped_A_d2_{o}_no_cde"
        state["utc_end"] = utc()
        write_status(state)
        log(f"A_d2={o} — NOT starting C/DE")
        return 0

    log("A_d2 COMPLETE — pull + health")
    out_a = pull_kernel(USER_D, A_D2_SLUG, REPO / "out_realA_d2", CFG_D)
    ha = analyze(out_a, min_step=2000)
    state["A_d2_health"] = ha
    write_status(state)
    log(f"health A_d2: {ha}")
    if not ha.get("ok") or not ha.get("last_pt"):
        state["phase"] = "A_d2_health_fail_no_cde"
        state["utc_end"] = utc()
        write_status(state)
        return 4
    pt = Path(ha["last_pt"])

    # 2) Publish on trainE + push C
    state["phase"] = "publish_src_warm_push_C"
    write_status(state)
    ensure_src_on_e()
    ensure_warm_on_e(pt, f"A_d2 step~{ha.get('final_step')}")
    log(f"gates OK — push Stage C {USER_E}/{C_SLUG}")
    push_kernel_e(C_SLUG, "realC_e1", C_SCRIPT, C_STEPS, "StageC")
    state["pushed"] = f"{USER_E}/{C_SLUG}"
    state["phase"] = "running_stageC_e1"
    write_status(state)

    # 3) Wait C
    oc = wait_kernel(USER_E, C_SLUG, CFG_E, MAX_WAIT_H, "Stage C")
    state["C_outcome"] = oc
    write_status(state)
    if oc != "COMPLETE":
        try:
            pull_kernel(USER_E, C_SLUG, REPO / "out_stageC_e1", CFG_E)
        except Exception as e:
            log(f"pull C: {e}")
        state["phase"] = f"stopped_C_{oc}"
        state["utc_end"] = utc()
        write_status(state)
        return 0

    out_c = pull_kernel(USER_E, C_SLUG, REPO / "out_stageC_e1", CFG_E)
    hc = analyze(out_c, min_step=3000)
    state["C_health"] = hc
    write_status(state)
    log(f"health C: {hc}")
    if not hc.get("ok") or not hc.get("last_pt"):
        state["phase"] = "C_health_fail_no_DE"
        state["utc_end"] = utc()
        write_status(state)
        return 4

    # 4) Warm + push DE
    ensure_warm_on_e(Path(hc["last_pt"]), f"C step~{hc.get('final_step')}")
    log(f"gates OK — push Stage DE {USER_E}/{DE_SLUG}")
    push_kernel_e(DE_SLUG, "realDE_e1", DE_SCRIPT, DE_STEPS, "StageDE")
    state["pushed"] = f"{USER_E}/{DE_SLUG}"
    state["phase"] = "running_stageDE_e1"
    write_status(state)

    ode = wait_kernel(USER_E, DE_SLUG, CFG_E, MAX_WAIT_H, "Stage DE")
    state["DE_outcome"] = ode
    if ode == "COMPLETE":
        out_de = pull_kernel(USER_E, DE_SLUG, REPO / "out_stageDE_e1", CFG_E)
        hde = analyze(out_de, min_step=2000)
        state["DE_health"] = hde
        if hde.get("ok") and hde.get("last_pt"):
            ensure_warm_on_e(Path(hde["last_pt"]), f"DE step~{hde.get('final_step')}")
            state["phase"] = "CDE_complete_ready_for_primary"
            ready = {
                "utc": utc(),
                "weights": hde["last_pt"],
                "train_account": USER_E,
                "next": "Build ultimate dual-seed notebook on khalid000000 with these weights; ≤3 submits; no multi-account mention in notebook text",
                "no_submit_from_train": True,
            }
            (REPO / "READY_FOR_PRIMARY_SUBMIT.json").write_text(json.dumps(ready, indent=2) + "\n", encoding="utf-8")
            log("READY_FOR_PRIMARY_SUBMIT written")
        else:
            state["phase"] = "DE_complete_health_warn"
    else:
        try:
            pull_kernel(USER_E, DE_SLUG, REPO / "out_stageDE_e1", CFG_E)
        except Exception as e:
            log(f"pull DE: {e}")
        state["phase"] = f"stopped_DE_{ode}"

    state["utc_end"] = utc()
    state["no_submit"] = True
    write_status(state)
    log(f"=== CDE trainE DONE phase={state.get('phase')} ===")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        log(f"FATAL {e}")
        raise
