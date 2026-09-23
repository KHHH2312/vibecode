"""What does the agent spend money on late in the game, and can it ever pay off?

Seeds, feed and fertilizer bought after the last day their crop could still
mature and be harvested are money that cannot come back: the reward is final
cash and nothing in the shed or the seed bag is scored.  This logs every
purchase by game day so the dead ones can be counted.

Crop growth days (from the engine CROPS table) decide the last useful planting
day; the season is 30 days, indexed 0..29.

    seedlog.py <a/main.py> <b/main.py> [seed]
"""
import collections
import sys

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as K

LOG = []
SEAT = {}          # id(farm dict) -> seat, captured from the live state
STEP = [0]
_orig = K._commit_unit
_orig_interp = K.interpreter
_orig_market = K._process_market


def _market(state, env):
    for i, f in enumerate(state[0].observation.farms):
        SEAT[id(f)] = i
    return _orig_market(state, env)


K._process_market = _market


def _commit(op, item, price, farm, private, market, shed_capacity=100):
    ok = _orig(op, item, price, farm, private, market, shed_capacity)
    if ok and op != "SELL":
        LOG.append((STEP[0] // 24, SEAT.get(id(farm), -1), op, item, float(price)))
    return ok


def _interp(state, env):
    try:
        STEP[0] = int(K.get(state[0].observation, "step", 0))
    except Exception:
        pass
    return _orig_interp(state, env)


K._commit_unit = _commit
K.interpreter = _interp


def main():
    a, b = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 500000
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([a, b])
    last = env.steps[-1]
    print("seed %d  rewards %s" % (seed, [x.reward for x in last]))
    fid = 0

    grow = {c: K.CROPS[c]["first_yield_day"] for c in K.CROPS}
    print("crop growth days:", grow)

    by_day = collections.defaultdict(lambda: collections.Counter())
    cost = collections.defaultdict(float)
    for day, farm, op, item, price in LOG:
        if farm != fid:
            continue
        by_day[day][op + " " + item] += 1
        cost[(day, op, item)] += price

    print("\nseat 0 purchases from day 18 on:")
    for day in sorted(by_day):
        if day < 18:
            continue
        row = ", ".join("%s x%d" % (k, v) for k, v in sorted(by_day[day].items()))
        spent = sum(v for (d, _, _), v in cost.items() if d == day)
        print("  day %2d  $%7.0f   %s" % (day, spent, row))

    dead = 0.0
    for (day, op, item), c in cost.items():
        if op == "BUY_SEED" and day + grow.get(item, 0) > 29:
            dead += c
    print("\nseeds bought too late to mature: $%.0f" % dead)


main()
