"""How many hands does the tape stop commanding, and from when?

Every attempt to use our idle hands has failed on the same constraint: moving a
hand desynchronises every later step that commands it.  But the tape is a fixed
list we can read, so the constraint is checkable rather than fatal -- a hand the
tape never commands again is free.

For each route, this reports how many hand slots go permanently quiet (PASS or
absent for the whole remainder) and at which step, so we know how much free
labour exists and how many days it has left to earn anything.
"""
import sys, collections
ns = {}
exec(compile(open("lh44/main.py", encoding="utf-8").read(), "lh44/main.py", "exec"), ns)
routes = ns["_ROUTES"]
print("routes: %d" % len(routes))

def cmd_of(entry, i):
    hands = entry.get("hands") if isinstance(entry, dict) else None
    if not hands or i >= len(hands):
        return None
    c = hands[i]
    return (list(c)[:1] or [None])[0]

for key in sorted(routes)[:6]:
    tape = routes[key]
    width = max((len((e.get("hands") or [])) for e in tape if isinstance(e, dict)), default=0)
    quiet = {}
    for i in range(width):
        last = None
        for step, e in enumerate(tape):
            c = cmd_of(e, i)
            if c is not None and c != "PASS":
                last = step
        quiet[i] = (last + 1) if last is not None else 0
    freed = sorted(quiet.items(), key=lambda kv: kv[1])
    print("\nroute %-4s width %d" % (key, width))
    for i, step in freed:
        if step < len(tape):
            print("   hand %-2d free from step %-4d (day %4.1f, %d steps left)"
                  % (i, step, step / 24.0, len(tape) - step))
