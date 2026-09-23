"""Append a terminal-frugality layer: stop buying once a purchase cannot pay back.

The reward is final cash.  Nothing in the shed, the seed bag, the barn or the
deeds is scored, so every coin spent after the last moment it could turn back
into a sale is simply deleted from the score.  The engine fixes those moments
exactly: a crop first yields `first_yield_day` days after planting, an animal
`first_yield_day` days after purchase, and the last turn that is played is 718.

The parent already drops non-sell orders at step 717.  This moves that cut
earlier and makes it per-item, which is where the dead spending actually is:
a 500-coin sheep on day 24, a 4000-coin quadrant on day 25, strawberry seed
after day 19.

    MK_SRC=<base>/main.py CUT=672 mkfrugal.py <outdir>

CUT is the step from which every non-sell order is dropped.  Set CUT=0 to keep
only the per-item deadlines, which are safe at any point in the season.
"""
import os
import shutil
import sys

SRC = os.environ.get("MK_SRC") or "ag61/main.py"
CUT = int(os.environ.get("CUT") or 0)
DEADLINES = os.environ.get("DEADLINES", "1") == "1"

LAYER = '''

# --------------------------------------------------------------------------- Terminal Frugality Layer
_TF_PARENT = agent
del agent

# Days after planting/purchase before the first yield; a buy later than
# (29 - lead) can never be harvested, and the season is days 0..29.
_TF_LEAD = {
    "WHEAT": 2, "CARROT": 2, "TOMATO": 8, "STRAWBERRY": 10, "MELON": 10,
    "GOOSE": 4, "COW": 8, "SHEEP": 6,
}
_TF_LAST_DAY = 29
_TF_CUT = __CUT__
_TF_DEADLINES = __DEADLINES__


def agent(observation, configuration=None):
    action = _TF_PARENT(observation, configuration)
    try:
        step = int(observation.get("step", 0))
        day = step // 24
        mkt = [list(o) for o in (action.get("market") or []) if o]
        if mkt:
            keep = []
            for o in mkt:
                op = o[0]
                if op == "SELL":
                    keep.append(o)
                    continue
                if _TF_CUT and step >= _TF_CUT:
                    continue
                if _TF_DEADLINES:
                    item = o[1] if len(o) > 1 else ""
                    if op in ("BUY_SEED", "BUY_ANIMAL"):
                        lead = _TF_LEAD.get(item)
                        if lead is not None and day + lead > _TF_LAST_DAY:
                            continue
                    elif op == "BUY_LAND":
                        # New land has to be cleared, planted and grown before
                        # it returns anything; the cheapest crop needs two days
                        # and a field needs more than that to fill.
                        if day + 4 > _TF_LAST_DAY:
                            continue
                keep.append(o)
            if keep != mkt:
                action = dict(action, market=keep)
    except Exception:
        pass
    return action

kaggle_submission_agent = agent
'''



def _unique_prefix(src_text, base):
    """A layer global that appears nowhere in the parent source."""
    cand = base
    n = 1
    while cand in src_text:
        n += 1
        cand = "%s%d" % (base, n)
    return cand


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, "main.py")
    shutil.copy(SRC, dst)
    layer = (LAYER.replace("__CUT__", str(CUT))
                  .replace("__DEADLINES__", "True" if DEADLINES else "False"))
    src_text = open(dst).read()
    tag = _unique_prefix(src_text, "_TF_")
    layer = layer.replace("_TF_", tag)
    assert tag + "PARENT" not in src_text, tag
    with open(dst, "a") as fh:
        fh.write(layer)
    print("built %s from %s (CUT=%d DEADLINES=%s)" % (out, SRC, CUT, DEADLINES))


main()
