"""Does the shed ever overflow, and how much does that cost?

The shed holds 100 units.  Harvests land in a farmhand's inventory and are
dropped into the shed at the end of the day; anything that does not fit is
destroyed, and a full shed also silently refuses BUY_PRODUCT and BUY_ANIMAL.
Goods destroyed that way never appear in any money curve, so they are exactly
the kind of loss that a score-only post-mortem cannot see.

    overflow.py <a/main.py> <b/main.py> [seed]
"""
import collections
import sys

from kaggle_environments import make
from kaggle_environments.envs.kaggriculture import kaggriculture as K

DROPPED = collections.Counter()
REFUSED = collections.Counter()
PEAK = [0, 0]

_orig_drop = K._drop_inventories_to_shed
_orig_commit = K._commit_unit
_PRIV = {}


def _drop(private, capacity):
    before = sum(private["shed"].values())
    carried = sum(sum(inv.values()) for inv in private.get("inventories", []))
    _orig_drop(private, capacity)
    after = sum(private["shed"].values())
    lost = before + carried - after
    key = _PRIV.setdefault(id(private), len(_PRIV))
    if lost > 0:
        DROPPED[key] += lost
    PEAK[key] = max(PEAK[key], after)


def _commit(op, item, price, farm, private, market, shed_capacity=100):
    ok = _orig_commit(op, item, price, farm, private, market, shed_capacity)
    if not ok and op in ("BUY_PRODUCT", "BUY_ANIMAL"):
        if sum(private["shed"].values()) >= shed_capacity:
            REFUSED[_PRIV.setdefault(id(private), len(_PRIV))] += 1
    return ok


K._drop_inventories_to_shed = _drop
K._commit_unit = _commit


def main():
    a, b = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 500000
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([a, b])
    print("seed %d  rewards %s" % (seed, [x.reward for x in env.steps[-1]]))
    for key in sorted(_PRIV.values()):
        print("  player slot %d: units destroyed at end of day %d, "
              "buys refused for a full shed %d, peak shed %d"
              % (key, DROPPED.get(key, 0), REFUSED.get(key, 0), PEAK[key]))


main()
