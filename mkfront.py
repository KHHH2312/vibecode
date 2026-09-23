"""Append a sell-priority layer to a built agent.

Orders resolve per slot index against one shared market inventory: both
players' i-th order is quoted against the same pre-commit inventory, so a sell
of a good at a lower index is quoted before the opponent's sell of the same
good at a higher one.  How much that is worth depends on the good -- the price
above I0 is base - (target*base/f(T)) * f(inv-I0), which is steep for WOOL and
MELON (square) and nearly flat for WHEAT and EGG (log).

The layer changes no volume and no timing, only the order of the list: sells
ahead of buys ahead of hires, and within the sells the ones with the most coins
at risk first.  In a field where half our ladder games are decided by under
$100, that is the cheapest edge available.

    MK_SRC=<base>/main.py FROM=1 SORT=1 mkfront.py <outdir>
"""
import os
import shutil
import sys

SRC = os.environ.get("MK_SRC") or "ag61/main.py"
FROM = int(os.environ.get("FROM") or 1)
SORT = os.environ.get("SORT", "1") == "1"

LAYER = '''

# --------------------------------------------------------------------------- Sell Priority Layer
import math as _sp_math

_SP_PARENT = agent
del agent

# base, T, above_func, above_target -- straight from the engine MARKET_PARAMS.
_SP_PARAMS = {
    "WHEAT":      (25,  400, "log",    0.20),
    "CARROT":     (35,  450, "sqrt",   0.70),
    "TOMATO":     (60,  200, "sqrt",   0.60),
    "STRAWBERRY": (120, 100, "linear", 1.60),
    "MELON":      (250, 300, "sq",     3.60),
    "EGG":        (50,  332, "log",    0.20),
    "MILK":       (160, 122, "linear", 1.60),
    "WOOL":       (200, 105, "sq",     3.20),
    "FERTILIZER": (100, 200, "linear", 0.40),
}
_SP_I0 = 10000
_SP_FROM = __FROM__
_SP_SORT = __SORT__


def _sp_shape(func, x):
    x = x if x > 0.0 else 0.0
    if func == "linear":
        return x
    if func == "sq":
        return x * x
    if func == "sqrt":
        return _sp_math.sqrt(x)
    return _sp_math.log(1.0 + x)


def _sp_price(item, inv):
    # The engine quote for one unit at this inventory, glut side only.
    base, T, func, target = _SP_PARAMS[item]
    if inv <= _SP_I0:
        return float(base)
    amp = target * base / _sp_shape(func, T)
    p = base - amp * _sp_shape(func, inv - _SP_I0)
    return p if p > 1.0 else 1.0


def agent(observation, configuration=None):
    action = _SP_PARENT(observation, configuration)
    try:
        if int(observation.get("step", 0)) >= _SP_FROM:
            mkt = [list(o) for o in (action.get("market") or []) if o]
            sells = [o for o in mkt if o[0] == "SELL"]
            hires = [o for o in mkt if o[0] == "HIRE"]
            buys = [o for o in mkt if o[0] not in ("SELL", "HIRE")]
            if sells:
                if _SP_SORT:
                    inv = ((observation.get("market") or {}).get("inventory") or {})

                    def _risk(o):
                        # Coins this order gives up if an equal order lands first.
                        item = o[1] if len(o) > 1 else ""
                        if item not in _SP_PARAMS:
                            return 0.0
                        try:
                            qty = int(o[2])
                        except Exception:
                            qty = 1
                        if qty < 1:
                            qty = 1
                        if qty > 40:
                            qty = 40
                        i0 = float(inv.get(item, _SP_I0))
                        now = 0.0
                        late = 0.0
                        for k in range(qty):
                            now += _sp_price(item, i0 + k)
                            late += _sp_price(item, i0 + qty + k)
                        return now - late

                    sells = sorted(sells, key=_risk, reverse=True)
                new = (sells + buys + hires)[:10]
                if new != mkt:
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
    layer = LAYER.replace("__FROM__", str(FROM)).replace(
        "__SORT__", "True" if SORT else "False")
    src_text = open(dst).read()
    tag = _unique_prefix(src_text, "_SP_")
    layer = layer.replace("_SP_", tag)
    assert tag + "PARENT" not in src_text, tag
    with open(dst, "a") as fh:
        fh.write(layer)
    print("built %s from %s (FROM=%d SORT=%s)" % (out, SRC, FROM, SORT))


main()
