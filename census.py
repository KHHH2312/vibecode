"""How much of the farm is actually being used?

Farm hands cost fib(n) coins with the multiplier at 1 -- ten hands in a day
costs 143 -- and nothing in the engine caps how many you may hire.  Land is
three extra quadrants at 1000, 2000 and 4000.  Against a 144,000 score both are
close to free, so if the agent runs a half-empty board with four hands there is
a large unclaimed production margin, and production is what decides the race
for the town's demand.

Reports, per game day: hands hired, quadrants unlocked, tiles by kind, shed
level and cash.

    census.py <a/main.py> <b/main.py> [seed] [seat]
"""
import collections
import sys

from kaggle_environments import make


def main():
    a, b = sys.argv[1], sys.argv[2]
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 500000
    seat = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([a, b])
    steps = env.steps
    print("seed %d  rewards %s" % (seed, [x.reward for x in steps[-1]]))
    print("%4s %6s %6s %5s %6s %7s %8s  %s"
          % ("day", "hands", "peakH", "quads", "shed", "cash", "planted", "tiles"))

    for day in range(30):
        # hour 23 of the day, just before the end-of-day reset
        t = min(day * 24 + 23, len(steps) - 1)
        obs = steps[t][seat]["observation"]
        farm = obs["farms"][seat]
        priv = obs.get("private") or {}
        peak = 0
        for h in range(24):
            tt = day * 24 + h
            if tt < len(steps):
                peak = max(peak, len(steps[tt][seat]["observation"]["farms"][seat]["hands"]))
        kinds = collections.Counter()
        crops = collections.Counter()
        for row in farm["tiles"]:
            for tile in row:
                if tile == "LOCKED":
                    kinds["LOCKED"] += 1
                elif tile is None:
                    kinds["EMPTY"] += 1
                elif isinstance(tile, dict):
                    kinds[tile.get("kind", "?")] += 1
                    if tile.get("kind") == "PLANT":
                        crops[tile.get("crop", "?")] += 1
                else:
                    kinds[str(tile)] += 1
        shed = sum((priv.get("shed") or {}).values())
        print("%4d %6d %6d %5d %6d %7.0f %8d  %s | %s"
              % (day, len(farm["hands"]), peak, len(farm["unlocked_quadrants"]),
                 shed, float(farm["money"]), sum(crops.values()),
                 dict(sorted(kinds.items())), dict(sorted(crops.items()))))


main()
