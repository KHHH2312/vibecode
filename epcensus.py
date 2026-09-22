"""Census a single episode by id: both seats, day by day.

    epcensus.py <episode_id>
"""
import sys
import topcensus as T


def main():
    ep = sys.argv[1]
    rep = T.fetch(ep)
    steps = rep.get("steps") or []
    info = rep.get("info") or {}
    names = info.get("TeamNames") or ["seat0", "seat1"]
    rewards = [x.get("reward") for x in steps[-1]]
    print("episode %s  %s %s  vs  %s %s" % (ep, names[0], rewards[0], names[1], rewards[1]))
    for seat in (0, 1):
        T.census(steps, seat, "seat %d (%s) reward %s" % (seat, names[seat], rewards[seat]))


main()
