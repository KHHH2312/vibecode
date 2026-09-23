"""How long are the tape's idle runs, and could a hand leave and get back?

The hands are not permanently free -- the tape commands all twelve until the
final hours -- but they PASS in scattered runs.  A hand can only be borrowed if
it can walk to a tile, work it, and return to its square before the tape needs
it there again.  At one tile per turn that needs 2*distance+1 turns, so the run
length is the whole budget.  This measures the runs.
"""
import collections
ns = {}
exec(compile(open("lh44/main.py", encoding="utf-8").read(), "lh44/main.py", "exec"), ns)
routes = ns["_ROUTES"]

def cmd_of(entry, i):
    hands = entry.get("hands") if isinstance(entry, dict) else None
    if not hands or i >= len(hands):
        return None
    return (list(hands[i])[:1] or [None])[0]

hist = collections.Counter()
runs_by_day = collections.Counter()
total = 0
for key in (0, 2, 7):
    tape = routes[key]
    width = max((len((e.get("hands") or [])) for e in tape if isinstance(e, dict)), default=0)
    for i in range(width):
        run = 0
        start = 0
        for step, e in enumerate(tape):
            c = cmd_of(e, i)
            if c == "PASS" or c is None:
                if run == 0:
                    start = step
                run += 1
            else:
                if run:
                    hist[min(run, 40)] += 1
                    if run >= 5:
                        runs_by_day[start // 24] += 1
                    total += 1
                run = 0
print("idle-run length distribution over routes 0/2/7 (%d runs):" % total)
cum = 0
for n in sorted(hist):
    cum += hist[n]
    print("   %2d turns : %4d runs   (%.0f%% of runs are this long or shorter)"
          % (n, hist[n], 100.0 * cum / total))
print("\nruns of >=5 turns, by day:")
for d in sorted(runs_by_day):
    print("   day %-2d : %d" % (d, runs_by_day[d]))
