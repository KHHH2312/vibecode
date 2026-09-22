"""Hash the actions an agent actually emits, to tell two builds apart.

Five variants of the dump layer -- different price gates, different caps,
different firing periods -- came back from a paired sweep with byte-identical
scores.  Different agents cannot score identically, so either the builds do not
differ in play or the benchmark is not measuring them.  Scores cannot
distinguish those two cases; the action stream can.

Plays one game per build against a fixed opponent and prints, for seat 0, the
number of turns carrying market orders, the number of SELL orders, the total
units offered, and a hash of the whole stream.  Builds that play identically
hash identically.

    probe.py <seed> <build> [build ...]
"""
import hashlib
import json
import os
import sys

os.environ.setdefault("PYTHONHASHSEED", "0")

OPP = "ag61/main.py"


def path_of(name):
    p = name.rstrip("/")
    return p if p.endswith(".py") else os.path.join(p, "main.py")


def run(name, seed):
    from kaggle_environments import make
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([path_of(name), OPP])
    h = hashlib.md5()
    turns = sells = units = 0
    for st in env.steps:
        act = st[0].action
        if not isinstance(act, dict):
            continue
        mkt = act.get("market") or []
        if mkt:
            turns += 1
        h.update(json.dumps(mkt, sort_keys=True, default=str).encode())
        for o in mkt:
            try:
                if o and o[0] == "SELL":
                    sells += 1
                    units += int(o[2])
            except Exception:
                pass
    return turns, sells, units, h.hexdigest()[:12], float(env.steps[-1][0].reward or 0)


def main():
    seed = int(sys.argv[1])
    for name in sys.argv[2:]:
        try:
            turns, sells, units, dig, cash = run(name, seed)
            print("%-8s turns=%3d sells=%4d units=%5d  %s  cash=%.0f"
                  % (name, turns, sells, units, dig, cash))
        except Exception as exc:
            print("%-8s FAILED %r" % (name, exc))
        sys.stdout.flush()


main()
