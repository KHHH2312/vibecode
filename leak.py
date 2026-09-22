"""Where do the last coins go?

Half of our peer games are decided by under $100 out of $99,000, so the whole
ladder turns on the closing turns rather than on strategy.  This plays one
local game and accounts for the endgame exactly:

  * what is still sitting in the shed at the final bell (scored at nothing),
  * every coin spent after a cut-off step, broken down by order type,
  * the money curve over the last day.

    leak.py <a/main.py> <b/main.py> [seed] [cut=648]
"""
import json
import sys
from collections import defaultdict

from kaggle_environments import make


def obs_of(step, seat):
    return step[seat]["observation"]


def main():
    a, b = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 500000
    cut = int(sys.argv[4]) if len(sys.argv) > 4 else 648

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([a, b])
    steps = env.steps
    last = steps[-1]
    print("seed %d  rewards %s" % (seed, [x.reward for x in last]))

    for seat in (0, 1):
        o = obs_of(last, seat)
        shed = (o.get("private") or {}).get("shed") or {}
        prices = (o.get("market") or {}).get("prices") or {}
        held = {k: v for k, v in shed.items() if isinstance(v, (int, float)) and v}
        worth = sum(float(prices.get(k, 0)) * v for k, v in held.items())
        cash = float(o["farms"][seat]["money"])
        print("\nseat %d  cash %.0f   shed at bell %s   quoted worth %.0f"
              % (seat, cash, held, worth))

        # money flow after the cut
        m0 = float(obs_of(steps[cut], seat)["farms"][seat]["money"])
        print("  money at step %d: %.0f  ->  %.0f   (%+.0f)" % (cut, m0, cash, cash - m0))

        spend = defaultdict(float)
        count = defaultdict(int)
        for t in range(cut, len(steps) - 1):
            act = steps[t][seat].get("action") or {}
            for o2 in (act.get("market") or []):
                if not isinstance(o2, (list, tuple)) or len(o2) < 2:
                    continue
                kind = str(o2[0])
                count[kind] += 1
                if kind != "SELL":
                    spend[kind] += 1
            for key in ("hire", "fire", "assign"):
                if act.get(key):
                    count[key] += 1
        print("  orders after step %d: %s" % (cut, dict(sorted(count.items()))))

    # last-day money curve
    print("\nstep   seat0      seat1")
    for t in range(len(steps) - 30, len(steps)):
        print("%4d %9.0f %9.0f" % (t, float(obs_of(steps[t], 0)["farms"][0]["money"]),
                                   float(obs_of(steps[t], 1)["farms"][1]["money"])))


main()
