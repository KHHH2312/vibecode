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
PRE = os.environ.get("PRE", "0") == "1"
# BASES overrides the reference price per item, e.g. "WOOL=100,MILK=50".
BASES = os.environ.get("BASES") or ""
MINQ = int(os.environ.get("MINQ") or 1)
# kaggle_environments runs the LAST callable in the module, which is not
# always named `agent` -- some public bases end on their own wrapper.
PARENT = os.environ.get("PARENT") or "agent"

LAYER = '''

# --------------------------------------------------------------------------- Race Dump Layer
_RD_PARENT = __PARENT__

_RD_BASE = __BASES__
_RD_ITEMS = __ITEMS__
_RD_FROM = __FROM__
_RD_FRAC = __FRAC__
_RD_CAP = __CAP__
_RD_MOD = __MOD__
_RD_REM = __REM__
_RD_PRE = __PRE__
_RD_MINQ = __MINQ__


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
                    if held < _RD_MINQ:
                        continue
                    base = _RD_BASE.get(item, 0)
                    if base and float(prices.get(item, 0)) < _RD_FRAC * base:
                        continue
                    qty = held if _RD_CAP <= 0 else min(held, _RD_CAP)
                    extra.append(["SELL", item, qty])
                if extra:
                    # Prepending quotes our sells ahead of the parent's
                    # own orders, at the cost of pushing those later.
                    new = (extra + mkt)[:10] if _RD_PRE else (mkt + extra)[:10]
                    action = dict(action, market=new)
    except Exception:
        pass
    return action

kaggle_submission_agent = agent
# kaggle_environments picks the LAST callable inserted into the module
# namespace.  Rebinding a name the base already defined leaves it in its
# original slot, so `def agent` above does not necessarily land last -- while
# `_RD_PARENT` above, being a new name holding a function, does.  A layer that
# only rebinds therefore publishes the base as the entry point and silently
# plays without its own layer.  Publishing under a fresh name settles it.
_RD_ENTRY = agent
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
    bases = {"WHEAT": 25, "CARROT": 35, "TOMATO": 60, "STRAWBERRY": 120,
             "MELON": 250, "EGG": 50, "MILK": 160, "WOOL": 200, "FERTILIZER": 100}
    for part in BASES.split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            bases[k.strip()] = float(v)
    layer = (LAYER.replace("__PARENT__", PARENT)
                  .replace("__BASES__", repr(bases))
                  .replace("__ITEMS__", repr(items))
                  .replace("__FROM__", str(FROM))
                  .replace("__FRAC__", repr(FRAC))
                  .replace("__CAP__", str(CAP))
                  .replace("__MOD__", str(MOD))
                  .replace("__REM__", str(REM))
                  .replace("__PRE__", "True" if PRE else "False")
                  .replace("__MINQ__", str(MINQ)))
    src_text = open(dst).read()
    tag = _unique_prefix(src_text, "_RD_")
    layer = layer.replace("_RD_", tag)
    assert tag + "PARENT" not in src_text, tag
    with open(dst, "a") as fh:
        fh.write(layer)
    _verify(dst, tag)
    print("built %s from %s (FROM=%d FRAC=%.2f CAP=%d MOD=%d REM=%d ITEMS=%s)"
          % (out, SRC, FROM, FRAC, CAP, MOD, REM, items))


def _verify(dst, tag):
    """Fail the build unless the engine would actually run the layer.

    A layer that is present in the file but not published as the entry point
    plays as the bare base, and scores like one -- which reads as a clean
    negative result rather than as a broken build.  Reproducing the engine's
    own selection here turns that silent failure into a loud one.
    """
    ns = {}
    exec(compile(open(dst).read(), dst, "exec"), ns)
    last = [v for v in ns.values() if callable(v)][-1]
    names = getattr(getattr(last, "__code__", None), "co_names", ())
    if tag + "FRAC" not in names:
        raise SystemExit("%s: entry point is %r, not the dump layer"
                         % (dst, getattr(last, "__name__", last)))


main()
