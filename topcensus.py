"""Read a top agent's game and compare its farm, day by day, with its opponent's.

We converge around 2650 while the board's best sit above 3200, and our peer
games are decided by tens of coins, so the difference is unlikely to be a
tactic -- it is more likely production: how much land is unlocked, how many
hands are hired, how much of the board is planted, and how full the shed gets.
That is all visible in a public replay.

    topcensus.py <submission_id> [n_episodes=2]
"""
import collections
import gzip
import io
import json
import os
import sys
import urllib.request

CDN = "https://www.kaggleusercontent.com/episodes/%s.json"


def fetch(ep):
    req = urllib.request.Request(CDN % ep)
    req.add_header("User-Agent", "Mozilla/5.0")
    req.add_header("Accept-Encoding", "gzip")
    with urllib.request.urlopen(req, timeout=300) as r:
        raw = r.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read()
    return json.loads(raw.decode())


def census(steps, seat, label):
    print("\n%s" % label)
    print("%4s %6s %5s %6s %8s %8s  %s"
          % ("day", "hands", "quad", "shed", "cash", "planted", "tiles"))
    for day in range(0, 30, 2):
        t = min(day * 24 + 23, len(steps) - 1)
        obs = steps[t][seat].get("observation") or {}
        farms = obs.get("farms") or (steps[t][0].get("observation") or {}).get("farms")
        if not farms:
            continue
        farm = farms[seat]
        priv = obs.get("private") or {}
        kinds = collections.Counter()
        crops = collections.Counter()
        for row in farm.get("tiles") or []:
            for tile in row:
                if tile == "LOCKED":
                    kinds["LOCKED"] += 1
                elif tile is None:
                    kinds["EMPTY"] += 1
                elif isinstance(tile, dict):
                    if "animal" in tile:
                        # An occupied structure still reports its structure
                        # kind, so stock and empty pens have to be split out.
                        kinds[tile["animal"]] += 1
                    else:
                        kinds[tile.get("kind", "?")] += 1
                    if tile.get("kind") == "PLANT":
                        crops[tile.get("crop", "?")] += 1
        shed = sum((priv.get("shed") or {}).values())
        print("%4d %6d %5d %6d %8.0f %8d  %s | %s"
              % (day, len(farm.get("hands") or []),
                 len(farm.get("unlocked_quadrants") or []), shed,
                 float(farm.get("money", 0)), sum(crops.values()),
                 dict(sorted(kinds.items())), dict(sorted(crops.items()))))


def main():
    sub = int(sys.argv[1])
    want = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    lad = json.load(open("ladder2.json"))
    eps = lad["episodes"].get(str(sub)) or []
    print("submission %d  rating %.0f  %d known episodes"
          % (sub, lad["scores"].get(str(sub), lad["scores"].get(sub, 0)), len(eps)))
    done = 0
    for ep in eps:
        try:
            rep = fetch(ep)
        except Exception as exc:
            print("  ep %s fetch failed: %s" % (ep, exc))
            continue
        steps = rep.get("steps") or []
        if len(steps) < 700:
            continue
        info = rep.get("info") or {}
        names = info.get("TeamNames") or ["seat0", "seat1"]
        rewards = [x.get("reward") for x in steps[-1]]
        print("\n=== episode %s  %s %s  vs  %s %s"
              % (ep, names[0], rewards[0], names[1], rewards[1]))
        winner = 0 if (rewards[0] or 0) >= (rewards[1] or 0) else 1
        census(steps, winner, "WINNER seat %d (%s)" % (winner, names[winner]))
        census(steps, 1 - winner, "LOSER  seat %d (%s)" % (1 - winner, names[1 - winner]))
        done += 1
        if done >= want:
            break


if __name__ == "__main__":
    main()
