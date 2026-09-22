"""Absolute end-of-season cash, which is the target we are actually chasing.

Elo is a relative measure and hides the number that matters: what the farm is
worth at the final bell.  This plays an agent against itself over a seed grid
and reports the distribution of final cash, so a build can be checked against a
flat goal (150,000) rather than against an opponent.

Self-play is exactly symmetric in this engine -- both seats tie to the coin --
so one game per seed is enough and the figure is the score the build reaches
when its opponent is equally strong.  Against a weaker opponent it scores more,
against a stronger one less.

    mirror.py <agentdir> [n=10] [base=900000]
"""
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("PYTHONHASHSEED", "0")


def play(args):
    path, seed = args
    from kaggle_environments import make
    try:
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
        env.run([path, path])
        last = env.steps[-1]
        return seed, float(last[0].reward or 0), float(last[1].reward or 0)
    except Exception as exc:
        return seed, None, repr(exc)[:120]


def main():
    path = sys.argv[1].rstrip("/")
    if not path.endswith(".py"):
        path = os.path.join(path, "main.py")
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    base = int(sys.argv[3]) if len(sys.argv) > 3 else 900000
    seeds = [base + i * 7919 for i in range(n)]

    workers = int(os.environ.get("AB_WORKERS") or 3)
    t0 = time.time()
    with Pool(min(len(seeds), workers), maxtasksperchild=1) as p:
        res = p.map(play, [(path, s) for s in seeds])

    vals = []
    for seed, a, b in res:
        if a is None:
            print("seed %d FAILED %s" % (seed, b))
            continue
        vals.append(a)
    if not vals:
        print("no completed games")
        return
    vals.sort()
    mean = sum(vals) / len(vals)
    print("%s  %d games in %.0fs" % (path, len(vals), time.time() - t0))
    print("  mean   %9.0f" % mean)
    print("  median %9.0f" % vals[len(vals) // 2])
    print("  min    %9.0f   max %9.0f" % (vals[0], vals[-1]))
    print("  >=150k %d of %d" % (sum(1 for v in vals if v >= 150000), len(vals)))


main()
