"""Two seats of one replay, compared by what they sold and when.

In a near-mirror game the winner is usually the seat that got its goods to
market first: the market inventory is shared and the glut curves are steep, so
a unit sold a day earlier is a unit sold at the top of the curve.  This lines
both seats' sells up day by day and per good, and shows what each seat was
still holding in the shed while the other was selling.

    sellcmp.py <episode_id>
"""
import collections
import sys

import topcensus as T


def main():
    ep = sys.argv[1]
    rep = T.fetch(ep)
    steps = rep.get("steps") or []
    names = (rep.get("info") or {}).get("TeamNames") or ["seat0", "seat1"]
    rewards = [x.get("reward") for x in steps[-1]]
    print("episode %s  %s %s vs %s %s" % (ep, names[0], rewards[0], names[1], rewards[1]))

    sold = [collections.Counter(), collections.Counter()]
    per_day = [collections.defaultdict(collections.Counter) for _ in range(2)]
    for t, step in enumerate(steps):
        day = t // 24
        for seat in (0, 1):
            act = step[seat].get("action") or {}
            for o in (act.get("market") or []):
                if isinstance(o, list) and len(o) >= 3 and o[0] == "SELL":
                    try:
                        n = int(o[2])
                    except Exception:
                        continue
                    sold[seat][o[1]] += n
                    per_day[seat][day][o[1]] += n

    goods = sorted(set(sold[0]) | set(sold[1]))
    print("\nunits ordered sold over the game")
    print("  %-12s %8s %8s" % ("good", names[0][:8], names[1][:8]))
    for g in goods:
        print("  %-12s %8d %8d" % (g, sold[0][g], sold[1][g]))

    print("\ncumulative units sold by day (seat0 / seat1)")
    cum = [collections.Counter(), collections.Counter()]
    for day in range(30):
        for seat in (0, 1):
            cum[seat].update(per_day[seat][day])
        if day % 3 or day == 0:
            continue
        row = "  ".join("%s %d/%d" % (g[:4], cum[0][g], cum[1][g]) for g in goods)
        print("  day %2d  %s" % (day, row))

    print("\nshed held at the end of each third day (seat0 | seat1)")
    for day in range(2, 30, 3):
        t = min(day * 24 + 23, len(steps) - 1)
        row = []
        for seat in (0, 1):
            priv = (steps[t][seat].get("observation") or {}).get("private") or {}
            shed = {k: v for k, v in (priv.get("shed") or {}).items() if v}
            row.append(shed)
        print("  day %2d  %s | %s" % (day, row[0], row[1]))


main()
