#!/usr/bin/env python3
"""
auto_campaign.py — end-to-end Biohub Kaggle campaign automation.

Runs the whole honest loop, encoding the rules from handoff.md:
  poll -> apply decision rule -> (optionally) push -> wait COMPLETE -> submit

RUN THIS LOCALLY (needs network + ~/.kaggle/kaggle.json). It will NOT run
inside a sandbox with no Kaggle egress.

Standing rules baked in (from handoff.md):
  * No exploit. Never push/submit hub/ladder or 5-movie measurement notebooks.
  * T4x2 only, internet OFF, competition + support-pack sources.
  * Exactly the 4 test movies. Code-competition submit path (-k/-v/-f).
  * ~5 submits/day UTC. Submit only clear high-EV deltas.
  * best = max(0.902, any COMPLETE honest score). Regressions => stay on bank.

Usage examples:
  python auto_campaign.py poll
  python auto_campaign.py status
  python auto_campaign.py plan
  python auto_campaign.py run                 # full loop, asks before each submit
  python auto_campaign.py run --yes           # full loop, no prompts (still gated)
  python auto_campaign.py push kernel_v110    # push one kernel + wait COMPLETE
  python auto_campaign.py submit bh-v110-350ep <version>
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

COMPETITION = "biohub-cell-tracking-during-development"
ACCOUNT = "khalid000000"
WORKSPACE = Path(__file__).resolve().parent

# --- Honest baseline and policy constants (handoff §0, §3, §6) ------------
BANK_FLOOR = 0.902           # best COMPLETE honest score to date (ref 54830671)
TARGET = 0.920               # user goal (stretch)
SUBMIT_MIN_DELTA = 0.0005    # only submit if expected/known score beats best by this
DAILY_SUBMIT_CAP = 5

# Kernels we are allowed to touch, in priority order. Anything not listed here
# (v99 exploit, verify/measurement notebooks) is refused by design.
CAMPAIGN_KERNELS = {
    "bh-v100c-submit": {"dir": "kernel_submit", "role": "honest bank 0.900 recipe (re-floor)"},
    "bh-v110-350ep":   {"dir": "kernel_v110",   "role": "bank PP + 350ep weights (push to 0.92)"},
    "bh-v111-gapfn":   {"dir": "kernel_v111",   "role": "v110 + mild GAP2 FN boost"},
}

# Hard blocklist — never push or submit these (handoff §2, §4.2, §4.3).
FORBIDDEN_PATTERNS = [
    "v99", "ultimate", "hack", "fork", "ladder", "hub", "exploit",
    "verify", "clean",  # measurement notebooks emit 5 movies -> blank score
]

REQUIRED_STEMS = {
    "44b6_0113de3b", "44b6_0b24845f", "6bba_05b6850b", "6bba_05db0fb1",
}

STATE_PATH = WORKSPACE / ".auto_campaign_state.json"


# --------------------------------------------------------------------------
# Kaggle CLI plumbing
# --------------------------------------------------------------------------
def kaggle(*args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess:
    """Invoke the Kaggle CLI via `python -m kaggle` (handoff-proven form)."""
    cmd = [sys.executable, "-m", "kaggle", *args]
    env = {**os.environ}
    env.setdefault("KAGGLE_CONFIG_DIR", str(Path.home() / ".kaggle"))
    return subprocess.run(cmd, check=check, text=True,
                          capture_output=capture, env=env)


def preflight() -> None:
    """Fail fast with a clear message if the environment can't reach Kaggle."""
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle")) / "kaggle.json"
    if not cfg.exists():
        sys.exit(f"[FATAL] No kaggle.json at {cfg}. Run this on your local machine "
                 f"where credentials live (chmod 600).")
    import importlib.util
    if importlib.util.find_spec("kaggle") is None:
        sys.exit("[FATAL] kaggle package not importable. `pip install kaggle` first.")
    try:
        r = kaggle("competitions", "list", "-s", COMPETITION, check=False)
        if r.returncode != 0 and "403" in (r.stderr or ""):
            sys.exit("[FATAL] Kaggle egress blocked (403). Run locally, not in a sandbox.")
    except Exception as exc:
        sys.exit(f"[FATAL] Could not reach Kaggle: {exc}")


# --------------------------------------------------------------------------
# Polling submissions
# --------------------------------------------------------------------------
def _raw_submissions_csv() -> str:
    r = kaggle("competitions", "submissions", "-c", COMPETITION, "-v", check=True)
    return r.stdout or ""


def _find_col(fieldnames: list[str], *candidates: str) -> str | None:
    """Case-insensitive, space-insensitive column lookup."""
    norm = {fn.lower().replace(" ", "").replace("_", ""): fn for fn in (fieldnames or [])}
    for cand in candidates:
        key = cand.lower().replace(" ", "").replace("_", "")
        if key in norm:
            return norm[key]
    return None


def poll_submissions() -> list[dict]:
    """Return recent submissions as dicts with ref/date/description/status/score.

    Tolerant of Kaggle CLI variants: column names differ across versions and
    status may be a plain string ('complete') or an enum repr
    ('SubmissionStatus.COMPLETE'). Score column has been publicScore,
    publicScoreFullPrecision, or 'public score' depending on version.
    """
    import csv
    import io
    csv_text = _raw_submissions_csv()
    out: list[dict] = []
    # Find the header LINE (not a mid-line offset). The live CLI emits
    # `ref,fileName,date,description,status,publicScore,privateScore`; using
    # str.find("fileName,") would start parsing mid-header and shift every
    # column by one (status lands in publicScore, the real score drops off).
    lines = csv_text.splitlines()
    header_idx = -1
    for i, line in enumerate(lines):
        low = line.lower()
        if "," in line and "date" in low and ("status" in low or "score" in low):
            header_idx = i
            break
    if header_idx == -1:
        print(csv_text)  # unknown format — show raw so we can adapt
        return out
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_idx:])))
    fns = reader.fieldnames or []
    c_date = _find_col(fns, "date")
    c_desc = _find_col(fns, "description")
    c_status = _find_col(fns, "status")
    c_file = _find_col(fns, "fileName", "ref")
    c_ref = _find_col(fns, "ref")
    c_score = _find_col(fns, "publicScore", "publicScoreFullPrecision",
                        "public score", "publicscorefullprecision")
    for row in reader:
        status_raw = (row.get(c_status, "") if c_status else "") or ""
        # normalize 'SubmissionStatus.COMPLETE' -> 'complete'
        status = status_raw.split(".")[-1].strip().lower()
        out.append({
            "ref": row.get(c_ref, "") if c_ref else "",
            "fileName": row.get(c_file, "") if c_file else "",
            "date": row.get(c_date, "") if c_date else "",
            "description": row.get(c_desc, "") if c_desc else "",
            "status": status,
            "publicScore": (row.get(c_score, "") if c_score else ""),
        })
    return out


def parse_score(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def summarize_poll(subs: list[dict]) -> dict:
    """Compute best honest COMPLETE score and list pending refs."""
    best = BANK_FLOOR
    best_desc = "prior bank lineage (ref 54830671)"
    pending = []
    completed = []
    for s in subs:
        score = parse_score(s["publicScore"])
        status = (s["status"] or "").lower()
        desc = s["description"] or ""
        if "pending" in status or status == "":
            if score is None:
                pending.append(s)
        if score is not None:
            completed.append((score, desc, s["date"]))
            # only count honest scores (exclude exploit-era >0.94)
            if score <= 0.94 and score > best:
                best, best_desc = score, desc
    return {"best": best, "best_desc": best_desc,
            "pending": pending, "completed": completed}


def count_todays_submits(subs: list[dict]) -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    n = 0
    for s in subs:
        if s["date"].startswith(today):
            n += 1
    return n


# --------------------------------------------------------------------------
# Safety checks
# --------------------------------------------------------------------------
def is_forbidden(name: str) -> str | None:
    low = name.lower()
    for pat in FORBIDDEN_PATTERNS:
        if pat in low:
            return pat
    return None


def verify_submission_csv(path: Path) -> None:
    """Guard: exactly the 4 stems, no dangling edges, clean id column."""
    import pandas as pd
    df = pd.read_csv(path)
    stems = set(df["dataset"].unique())
    if stems != REQUIRED_STEMS:
        raise SystemExit(f"[GUARD FAIL] datasets {stems} != required {REQUIRED_STEMS}. "
                         f"Refusing to submit (would blank-score).")
    nodes = df[df.row_type == "node"]
    node_ids = set(zip(nodes.dataset, nodes.node_id.astype(int)))
    edges = df[df.row_type == "edge"]
    dangling = 0
    for ds, grp in edges.groupby("dataset"):
        nid = {n for (d, n) in node_ids if d == ds}
        for s, t in zip(grp.source_id.astype(int), grp.target_id.astype(int)):
            if s not in nid or t not in nid:
                dangling += 1
    if dangling:
        raise SystemExit(f"[GUARD FAIL] {dangling} dangling edges. Refusing to submit.")
    # no exploit sentinels: negative timestamps / far coords
    if (nodes["t"].astype(int) < 0).any():
        raise SystemExit("[GUARD FAIL] negative timestamps present (hub/ladder exploit). Refusing.")
    print(f"[GUARD OK] {len(df)} rows, exactly 4 stems, 0 dangling edges, no exploit sentinels.")


# --------------------------------------------------------------------------
# Push / status / submit
# --------------------------------------------------------------------------
def push_kernel(kernel_dir: str) -> str:
    kdir = WORKSPACE / kernel_dir
    meta = json.loads((kdir / "kernel-metadata.json").read_text())
    kid = meta["id"].split("/")[-1]
    forbidden = is_forbidden(kid) or is_forbidden(meta.get("code_file", ""))
    if forbidden:
        raise SystemExit(f"[REFUSED] '{kid}' matches forbidden pattern '{forbidden}'.")
    # metadata sanity
    assert meta.get("machine_shape") == "NvidiaTeslaT4", "must be T4x2"
    assert meta.get("enable_internet") is False, "internet must be OFF"
    assert COMPETITION in meta.get("competition_sources", []), "missing competition source"
    print(f"[push] {kid}  ({CAMPAIGN_KERNELS.get(kid, {}).get('role', '?')})")
    kaggle("kernels", "push", "-p", str(kdir), capture=False)
    return kid


def wait_complete(kid: str, poll_s: int = 60, timeout_s: int = 4 * 3600) -> str:
    slug = f"{ACCOUNT}/{kid}"
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        r = kaggle("kernels", "status", slug, check=False)
        blob = (r.stdout or "") + (r.stderr or "")
        status = "unknown"
        m = re.search(r'status[\"\s:]+([A-Za-z_]+)', blob)
        if m:
            status = m.group(1).lower()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {kid}: {status}")
        if "complete" in status:
            return "complete"
        if "error" in status or "cancel" in status:
            return status
        time.sleep(poll_s)
    return "timeout"


def download_output(kid: str, dest: Path) -> Path | None:
    dest.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", f"{ACCOUNT}/{kid}", "-p", str(dest), check=False, capture=False)
    csvp = dest / "submission.csv"
    return csvp if csvp.exists() else None


def latest_version(kid: str) -> int | None:
    r = kaggle("kernels", "status", f"{ACCOUNT}/{kid}", check=False)
    m = re.search(r'version\s*[:=]?\s*(\d+)', (r.stdout or ""), re.I)
    return int(m.group(1)) if m else None


def submit(kid: str, version: int, csv_name: str, message: str) -> None:
    """Code-competition submit path (handoff §1.4)."""
    print(f"[submit] {kid} v{version} -f {csv_name}")
    kaggle("competitions", "submit", COMPETITION,
           "-k", f"{ACCOUNT}/{kid}", "-v", str(version),
           "-f", csv_name, "-m", message, capture=False)


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------
def cmd_raw(_args) -> None:
    """Dump the exact CLI output + parsed columns, for debugging the parser."""
    csv_text = _raw_submissions_csv()
    print("=== RAW CLI OUTPUT (competitions submissions -v) ===")
    print(csv_text)
    subs = poll_submissions()
    print("=== PARSED (ref | status | publicScore | date) ===")
    for s in subs[:12]:
        print(f"  {s['ref']:<10} {s['status']:<10} {s['publicScore'] or '-':<8} {s['date'][:19]}")


def cmd_poll(_args) -> None:
    preflight()
    subs = poll_submissions()
    summ = summarize_poll(subs)
    print("\n=== SUBMISSIONS (most recent) ===")
    print(f"  {'ref':<10} {'date':<20} {'status':<10} {'score':<7} description")
    for s in subs[:15]:
        score = s['publicScore'] or '-'
        print(f"  {s['ref']:<10} {s['date'][:19]:<20} {s['status']:<10} {score:<7} {s['description'][:44]}")
    print(f"\nBest honest COMPLETE: {summ['best']:.4f}  ({summ['best_desc'][:50]})")
    print(f"Pending (awaiting score): {len(summ['pending'])}")
    for p in summ["pending"]:
        print(f"    - {p['date']} {p['description'][:50]}")
    print(f"Submits used today (UTC): {count_todays_submits(subs)}/{DAILY_SUBMIT_CAP}")
    _save_state({"best": summ["best"], "polled": datetime.now(timezone.utc).isoformat()})


def cmd_plan(_args) -> None:
    preflight()
    subs = poll_submissions()
    summ = summarize_poll(subs)
    best = summ["best"]
    used = count_todays_submits(subs)
    print(f"\n=== PLAN (best={best:.4f}, target={TARGET}, submits today={used}/{DAILY_SUBMIT_CAP}) ===")
    if summ["pending"]:
        print("There are PENDING submissions. Wait for their scores before spending budget:")
        for p in summ["pending"]:
            print(f"    - {p['date']} {p['description'][:50]}")
        print("Re-run `poll` until they resolve, then `plan` again.")
        return
    print("Decision rule (handoff §0):")
    print(f"  * best = max({BANK_FLOOR}, any COMPLETE honest score) = {best:.4f}")
    print("  * If v110/v111 >= best and climbing -> iterate on that stack.")
    print("  * If they regress vs bank -> re-floor with bh-v100c-submit only.")
    print("\nRecommended next pushes (in order), each gated on beating best:")
    for kid, info in CAMPAIGN_KERNELS.items():
        print(f"    {kid:<16} {info['role']}")
    if used >= DAILY_SUBMIT_CAP:
        print(f"\n[HOLD] Daily cap reached ({used}/{DAILY_SUBMIT_CAP}). Resume after UTC midnight.")


def cmd_push(args) -> None:
    preflight()
    kid = push_kernel(args.kernel_dir)
    status = wait_complete(kid)
    print(f"[done] {kid} -> {status}")


def cmd_submit(args) -> None:
    preflight()
    subs = poll_submissions()
    if count_todays_submits(subs) >= DAILY_SUBMIT_CAP:
        sys.exit(f"[HOLD] daily cap {DAILY_SUBMIT_CAP} reached. Try after UTC midnight.")
    dest = WORKSPACE / f"out_{args.kernel}"
    csvp = download_output(args.kernel, dest)
    if not csvp:
        sys.exit(f"[FATAL] no submission.csv downloaded for {args.kernel}")
    verify_submission_csv(csvp)
    submit(args.kernel, int(args.version), "submission.csv",
           args.message or f"{args.kernel} honest campaign auto-submit")


def cmd_status(_args) -> None:
    preflight()
    for kid in CAMPAIGN_KERNELS:
        r = kaggle("kernels", "status", f"{ACCOUNT}/{kid}", check=False)
        print(f"{kid:<16} {(r.stdout or r.stderr).strip()[:80]}")


def cmd_run(args) -> None:
    """Full loop: poll -> for each campaign kernel, push, wait, guard, submit (gated)."""
    preflight()
    subs = poll_submissions()
    summ = summarize_poll(subs)
    if summ["pending"]:
        print("[HOLD] Pending submissions exist; resolve them before spending budget.")
        cmd_poll(args)
        return
    best = summ["best"]
    used = count_todays_submits(subs)
    print(f"[run] best={best:.4f}  submits today={used}/{DAILY_SUBMIT_CAP}")

    for kid, info in CAMPAIGN_KERNELS.items():
        if used >= DAILY_SUBMIT_CAP:
            print(f"[HOLD] daily cap reached. Stopping."); break
        # v100c is the re-floor / safety net; push the weight-lever kernels for upside.
        print(f"\n--- {kid}: {info['role']} ---")
        push_kernel(info["dir"])
        status = wait_complete(kid)
        if status != "complete":
            print(f"[skip] {kid} ended {status}; not submitting."); continue
        dest = WORKSPACE / f"out_{kid}"
        csvp = download_output(kid, dest)
        if not csvp:
            print(f"[skip] {kid} produced no submission.csv."); continue
        try:
            verify_submission_csv(csvp)
        except SystemExit as e:
            print(e); print(f"[skip] {kid} failed guard."); continue
        ver = latest_version(kid) or 1
        if not args.yes:
            ans = input(f"Submit {kid} v{ver}? [y/N] ").strip().lower()
            if ans != "y":
                print("[skip] user declined."); continue
        submit(kid, ver, "submission.csv", f"{kid} honest campaign")
        used += 1
        print(f"[submitted] {kid}. Re-run `poll` to capture its score.")

    print("\n[run] complete. Poll again once scores resolve, then decide next iteration.")


def _save_state(d: dict) -> None:
    try:
        STATE_PATH.write_text(json.dumps(d, indent=2))
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Biohub honest campaign automation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("poll").set_defaults(func=cmd_poll)
    sub.add_parser("raw").set_defaults(func=cmd_raw)
    sub.add_parser("plan").set_defaults(func=cmd_plan)
    sub.add_parser("status").set_defaults(func=cmd_status)
    p_push = sub.add_parser("push"); p_push.add_argument("kernel_dir"); p_push.set_defaults(func=cmd_push)
    p_sub = sub.add_parser("submit")
    p_sub.add_argument("kernel"); p_sub.add_argument("version")
    p_sub.add_argument("-m", "--message", default=""); p_sub.set_defaults(func=cmd_submit)
    p_run = sub.add_parser("run"); p_run.add_argument("--yes", action="store_true")
    p_run.set_defaults(func=cmd_run)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
