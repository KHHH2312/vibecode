"""Exactly where each player's money comes from, and what it was paid per unit.

Patches the engine's per-unit commit so every SELL and BUY is logged with the
price it actually cleared at.  That turns a final score into a bill of
materials: units sold per good, coins earned, mean price realised versus the
good's base -- which is the only way to see which of the nine markets we are
leaving on the table.

    revenue.py <a/main.py> <b/main.py> [seed]
"""
import collections
import sys

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as K

LOG = []
SEAT = {}          # id(farm dict) -> seat, captured from the live state
_orig = K._commit_unit
_orig_market = K._process_market


def _market(state, env):
    # env.steps stores deep copies, so the live farm dicts have to be
    # identified while the market is running, not afterwards.
    for i, f in enumerate(state[0].observation.farms):
        SEAT[id(f)] = i
    return _orig_market(state, env)


K._process_market = _market


def _patched(op, item, price, farm, private, market, shed_capacity=100):
    before = farm["money"]
    ok = _orig(op, item, price, farm, private, market, shed_capacity)
    if ok:
        LOG.append((SEAT.get(id(farm), -1), op, item, float(price),
                    farm["money"] - before))
    return ok


K._commit_unit = _patched

BASE = {k: v["base"] for k, v in K.MARKET_PARAMS.items()}


def main():
    a, b = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 500000
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([a, b])
    last = env.steps[-1]
    print("seed %d  rewards %s" % (seed, [x.reward for x in last]))

    for seat in (0, 1):
        rows = [r for r in LOG if r[0] == seat]
        sells = collections.defaultdict(lambda: [0, 0.0])
        spend = collections.defaultdict(lambda: [0, 0.0])
        for _, op, item, price, delta in rows:
            if op == "SELL":
                sells[item][0] += 1
                sells[item][1] += price
            else:
                spend[op + " " + item][0] += 1
                spend[op + " " + item][1] += -delta
        print("\nseat %d  revenue" % seat)
        print("  %-11s %6s %10s %8s %6s" % ("good", "units", "coins", "mean", "base"))
        for item in sorted(sells, key=lambda i: -sells[i][1]):
            n, c = sells[item]
            print("  %-11s %6d %10.0f %8.1f %6d" % (item, n, c, c / n, BASE.get(item, 0)))
        print("  TOTAL revenue %.0f" % sum(v[1] for v in sells.values()))
        print("  spend:")
        for k in sorted(spend, key=lambda i: -spend[i][1]):
            n, c = spend[k]
            print("    %-22s %5d %9.0f" % (k, n, c))
        print("  TOTAL spend %.0f" % sum(v[1] for v in spend.values()))


main()
