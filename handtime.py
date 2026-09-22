"""When does the tape hire, and does it account for every hand it has?

A layer can drive farm hands the parent does not know about -- the parent's
action carries exactly as many entries as it believes it has, and the engine
applies entry i to hand i+1 -- but only while OUR hands sit at higher indices
than the parent's.  If the parent hires after we do, its newest hand lands
above ours and its own action list starts addressing our hand instead.

So the question is when in the day the parent's hiring finishes.

    handtime.py <agentdir> [seed]
"""
import sys

from kaggle_environments import make


def main():
    path = sys.argv[1].rstrip("/") + "/main.py"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 900000
    env = make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed})
    env.run([path, path])
    steps = env.steps

    print("day  hires at hour (real hand count after each hire)")
    mismatch = 0
    for day in range(30):
        row = []
        prev = 0
        for hour in range(24):
            t = day * 24 + hour
            if t >= len(steps):
                break
            obs = steps[t][0]["observation"]
            n = len(obs["farms"][0]["hands"])
            act = steps[t][0].get("action") or {}
            declared = len(act.get("hands") or [])
            if n != prev:
                row.append("h%d->%d" % (hour, n))
                prev = n
            if declared != n:
                mismatch += 1
        if day % 5 == 0 or day == 29:
            print("%3d  %s" % (day, " ".join(row)))
    print("\nturns where the parent's hands list length != real hand count: %d" % mismatch)

    # last hour of any hire, across the season
    last = 0
    for day in range(30):
        prev = 0
        for hour in range(24):
            t = day * 24 + hour
            if t >= len(steps):
                break
            n = len(steps[t][0]["observation"]["farms"][0]["hands"])
            if n != prev:
                last = max(last, hour)
                prev = n
    print("latest hour at which a hire ever happens: %d" % last)


main()
