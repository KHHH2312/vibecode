"""How much room is there in the ten-order market list?

An appending layer can only act when the parent left a slot free, so the
distribution of orders per turn is the ceiling on anything built that way.
Also reports, per turn band, how much STRAWBERRY and MILK is sitting in the
shed unsold -- that is the stock an appending layer would be able to move.

    slots.py <a/main.py> <b/main.py> [seed]
"""
import collections
import sys

from kaggle_environments import make


def main():
    a, b = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 500000
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([a, b])
    steps = env.steps

    hist = collections.Counter()
    held = collections.Counter()
    turns_with_stock = 0
    for t, step in enumerate(steps[:-1]):
        act = step[0].get("action") or {}
        n = len([o for o in (act.get("market") or []) if o])
        hist[min(n, 10)] += 1
        obs = step[0].get("observation") or {}
        shed = (obs.get("private") or {}).get("shed") or {}
        s = int(shed.get("STRAWBERRY", 0) or 0) + int(shed.get("MILK", 0) or 0)
        if s:
            turns_with_stock += 1
            held[min(n, 10)] += s
    total = sum(hist.values())
    print("orders issued per turn (seat 0), %d turns" % total)
    for k in sorted(hist):
        print("  %2d orders : %4d turns (%4.1f%%)   strawberry+milk idle in shed %d"
              % (k, hist[k], 100.0 * hist[k] / total, held.get(k, 0)))
    print("turns with strawberry or milk sitting in the shed: %d (%.1f%%)"
          % (turns_with_stock, 100.0 * turns_with_stock / total))
    full = sum(v for k, v in hist.items() if k >= 10)
    print("turns with no free slot: %d (%.1f%%)" % (full, 100.0 * full / total))


main()
