"""Append a race-the-opponent dump layer to a built agent.

The market is one shared inventory pool that only drains at the town's
consumption rate, and the high-base goods sit on square or linear glut curves:
WOOL loses its whole value over ~58 units of glut, MELON over ~158.  So the
units either player sells first are worth several times the units sold last,
and goods held in the shed are goods the opponent is free to sell ahead of us.

Earlier work rejected selling surplus early, but it was measured by replaying
one seat against a fixed tape -- which scores our own cash and cannot see that
a sale also takes the top of the curve away from the other player.  This layer
exists to be measured head to head instead, where only the difference counts.

It never displaces an order the parent issued: it fills the free slots below
the parent's orders, and only while the good is still quoted near its base.

    MK_SRC=<base>/main.py FROM=240 FRAC=0.80 ITEMS=WOOL,MELON mkdump.py <outdir>
"""
import os
import shutil
import sys

SRC = os.environ.get("MK_SRC") or "ag61/main.py"
FROM = int(os.environ.get("FROM") or 240)
FRAC = float(os.environ.get("FRAC") or 0.80)
ITEMS = os.environ.get("ITEMS") or "WOOL,MELON,STRAWBERRY,MILK"
CAP = int(os.environ.get("CAP") or 6)
MOD = int(os.environ.get("MOD") or 1)
REM = int(os.environ.get("REM") or 0)

LAYER = '''

# --------------------------------------------------------------------------- Race Dump Layer
_RD_PARENT = agent
del agent

_RD_BASE = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
            "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
_RD_ITEMS = __ITEMS__
_RD_FROM = __FROM__
_RD_FRAC = __FRAC__
_RD_CAP = __CAP__
_RD_MOD = __MOD__
_RD_REM = __REM__


def agent(observation, configuration=None):
    action = _RD_PARENT(observation, configuration)
    try:
        _rd_step = int(observation.get("step", 0))
        if _rd_step >= _RD_FROM and _rd_step % _RD_MOD == _RD_REM:
            mkt = [list(o) for o in (action.get("market") or []) if o]
            if len(mkt) < 10:
                shed = ((observation.get("private") or {}).get("shed") or {})
                prices = ((observation.get("market") or {}).get("prices") or {})
                pending = {}
                for o in mkt:
                    if o[0] == "SELL" and len(o) > 2:
                        try:
                            pending[o[1]] = pending.get(o[1], 0) + int(o[2])
                        except Exception:
                            pass
                extra = []
                for item in _RD_ITEMS:
                    held = int(shed.get(item, 0) or 0) - pending.get(item, 0)
                    if held <= 0:
                        continue
                    base = _RD_BASE.get(item, 0)
                    if base and float(prices.get(item, 0)) < _RD_FRAC * base:
                        continue
                    extra.append(["SELL", item, min(held, _RD_CAP)])
                if extra:
                    new = (mkt + extra)[:10]
                    action = dict(action, market=new)
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
    items = [s.strip() for s in ITEMS.split(",") if s.strip()]
    layer = (LAYER.replace("__ITEMS__", repr(items))
                  .replace("__FROM__", str(FROM))
                  .replace("__FRAC__", repr(FRAC))
                  .replace("__CAP__", str(CAP))
                  .replace("__MOD__", str(MOD))
                  .replace("__REM__", str(REM)))
    src_text = open(dst).read()
    tag = _unique_prefix(src_text, "_RD_")
    layer = layer.replace("_RD_", tag)
    assert tag + "PARENT" not in src_text, tag
    with open(dst, "a") as fh:
        fh.write(layer)
    print("built %s from %s (FROM=%d FRAC=%.2f CAP=%d MOD=%d REM=%d ITEMS=%s)"
          % (out, SRC, FROM, FRAC, CAP, MOD, REM, items))


main()
