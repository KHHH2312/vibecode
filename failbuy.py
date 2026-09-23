"""Count the purchases the engine silently drops, and say why.

`_commit_unit` returns False and moves on when the farm cannot afford an order
or the shed is full.  Nothing surfaces that, so a controller can ask for a cow
every day for a week and never notice it did not get one.

    failbuy.py <a/main.py> <b/main.py> [seed]
"""
import collections
import sys

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as K

FAIL = collections.Counter()
OK = collections.Counter()
SEAT = {}
STEP = [0]
_orig = K._commit_unit
_orig_market = K._process_market


def _market(state, env):
    for i, f in enumerate(state[0].observation.farms):
        SEAT[id(f)] = i
    try:
        STEP[0] = int(K.get(state[0].observation, "step", 0))
    except Exception:
        pass
    return _orig_market(state, env)


def _commit(op, item, price, farm, private, market, shed_capacity=100):
    before_money = farm["money"]
    before_shed = sum(private["shed"].values())
    ok = _orig(op, item, price, farm, private, market, shed_capacity)
    seat = SEAT.get(id(farm), -1)
    key = (seat, op, item)
    if ok:
        OK[key] += 1
    elif op != "SELL":
        why = "money" if before_money < price else (
            "shed_full" if before_shed >= shed_capacity else "other")
        FAIL[(seat, op, item, why, STEP[0] // 24)] += 1
    return ok


K._commit_unit = _commit
K._process_market = _market


def main():
    a, b = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 500000
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([a, b])
    print("seed %d  rewards %s" % (seed, [x.reward for x in env.steps[-1]]))
    print("\nseat 0 purchases that went through:")
    for (seat, op, item), n in sorted(OK.items()):
        if seat == 0 and op != "SELL":
            print("  %-12s %-11s %5d" % (op, item, n))
    print("\nseat 0 purchases the engine dropped:")
    rows = [(k, v) for k, v in FAIL.items() if k[0] == 0]
    if not rows:
        print("  none")
    for (seat, op, item, why, day), n in sorted(rows, key=lambda kv: -kv[1])[:25]:
        print("  day %2d  %-12s %-11s %-9s x%d" % (day, op, item, why, n))


main()
