"""Where does a constant route beat the shop->route table, and where does it lose?

Replacing the router's 64-entry non-YARN table with constant route 124 wins about
+1,200 coins overall (v58).  But "overall" hides the thing worth knowing: the
table is per-pair, so 124 may be winning hugely on some layouts and losing on
others, and a table that kept the table's pick where it is right would beat both.

Per-pair evidence is normally unaffordable -- shop layouts are endogenous, and 40
seeds produce 26 distinct pairs, so a pair gets about one game.  But the pair is
recoverable from a FINISHED game: env.steps retains every step, so one ordinary
A/B run can be split by the pair each game turned out to have, at no extra cost.

    routesplit.py <A/main.py> <B/main.py> <seeds> <base> <stride>
"""
import collections
import os
import sys
from multiprocessing import Pool

os.environ.setdefault("PYTHONHASHSEED", "0")

DAY6 = 150


def play(args):
    a, b, seed, swap = args
    from kaggle_environments import make
    def load(path):
        ns = {}
        exec(compile(open(path, encoding="utf-8").read(), path, "exec"), ns)
        return [v for v in ns.values() if callable(v)][-1]
    order = [load(b), load(a)] if swap else [load(a), load(b)]
    try:
        env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
        env.run(order)
        rewards = [float(s.reward if s.reward is not None else -1) for s in env.steps[-1]]
        ra, rb = (rewards[1], rewards[0]) if swap else (rewards[0], rewards[1])
        step = env.steps[min(DAY6, len(env.steps) - 1)]
        shops = tuple((step[0]["observation"].get("town", {}).get("unlocked_shops") or [])[:2])
        return (shops, ra, rb)
    except Exception as exc:
        return None


def main():
    a, b, n, base, stride = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    jobs = []
    for i in range(n):
        seed = base + i * stride
        jobs.append((a, b, seed, False))
        jobs.append((a, b, seed, True))
    with Pool(int(os.environ.get("AB_WORKERS", 3))) as pool:
        results = pool.map(play, jobs)

    per_pair = collections.defaultdict(lambda: [0, 0, 0, 0.0])   # wins, losses, ties, margin
    for r in results:
        if r is None:
            continue
        shops, ra, rb = r
        rec = per_pair[shops]
        if ra > rb:
            rec[0] += 1
        elif ra < rb:
            rec[1] += 1
        else:
            rec[2] += 1
        rec[3] += ra - rb

    total = [0, 0, 0, 0.0]
    print("%-46s %5s %5s %5s %10s" % ("shop pair at day 6", "W", "L", "T", "mean"))
    rows = sorted(per_pair.items(), key=lambda kv: -(kv[1][0] + kv[1][1] + kv[1][2]))
    for shops, (w, l, t, margin) in rows:
        games = w + l + t
        for i in range(3):
            total[i] += (w, l, t)[i]
        total[3] += margin
        label = "/".join(s[:20] for s in shops) or "(none)"
        print("%-46s %5d %5d %5d %+10.0f" % (label[:46], w, l, t, margin / max(games, 1)))
    games = total[0] + total[1] + total[2]
    print("\nTOTAL %d games over %d distinct pairs: %dW-%dL-%dT  mean %+.0f"
          % (games, len(per_pair), total[0], total[1], total[2], total[3] / max(games, 1)))


if __name__ == "__main__":
    main()
