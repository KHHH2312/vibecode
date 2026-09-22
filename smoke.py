"""Refuse to score a build that does not load.

A layer whose module global clashes with one the parent already uses makes the
inner layer call itself, and the agent dies with RecursionError on its first
turn.  The engine scores that as a forfeit, which looks exactly like a real and
catastrophic measurement -- it cost two wrong conclusions before it was caught.

Plays a handful of turns and exits non-zero if the agent logged anything on
stderr.

    smoke.py <agentdir>/main.py
"""
import sys

from kaggle_environments import make


def main():
    path = sys.argv[1]
    env = make("kaggriculture", configuration={"episodeSteps": 40, "seed": 7})
    env.run([path, path])
    for turn in (env.logs or []):
        for entry in turn:
            err = (entry or {}).get("stderr") or ""
            if err.strip():
                tail = [x for x in err.strip().split("\n") if x.strip()][-3:]
                print("SMOKE FAIL %s" % path)
                print("\n".join(tail))
                return 1
    print("smoke ok %s" % path)
    return 0


sys.exit(main())
