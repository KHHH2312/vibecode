"""Append a layer that keeps room in the shed.

The shed holds 100 units.  At the end of every day each hand's carried
inventory is dropped into it and **anything that does not fit is destroyed**,
and a full shed also silently refuses BUY_PRODUCT and BUY_ANIMAL.  Goods lost
that way never appear in any money curve.

The feed-reserve experiment made the case by accident: holding wheat back from
the market to guarantee feed cost 26,368 coins a game, 0W-40L, because the
withheld wheat filled the shed and the day's harvest was destroyed on top of
it.  The race-dump layer that wins may be winning as much by keeping the shed
empty as by taking the top of the price curve.

So this layer sells into the free order slots only when the shed is filling,
and only goods that are outputs.  WHEAT is animal feed and FERTILIZER is a crop
input -- both are consumed from the shed by the parent's own routines, and
selling either has already been measured as a heavy loss.

    MK_SRC=<base>/main.py ROOM=60 CAP=8 mkroom.py <outdir>
"""
import os
import shutil
import sys

SRC = os.environ.get("MK_SRC") or "ag61/main.py"
ROOM = int(os.environ.get("ROOM") or 60)
CAP = int(os.environ.get("CAP") or 8)
FROM = int(os.environ.get("FROM") or 24)
ITEMS = os.environ.get("ITEMS") or "WOOL,MELON,STRAWBERRY,MILK,EGG,CARROT,TOMATO"

LAYER = '''

# --------------------------------------------------------------------------- Shed Room Layer
_SR_PARENT = agent
del agent

_SR_ITEMS = __ITEMS__
_SR_ROOM = __ROOM__
_SR_CAP = __CAP__
_SR_FROM = __FROM__


def agent(observation, configuration=None):
    action = _SR_PARENT(observation, configuration)
    try:
        if int(observation.get("step", 0)) >= _SR_FROM:
            shed = ((observation.get("private") or {}).get("shed") or {})
            total = 0
            for v in shed.values():
                if isinstance(v, (int, float)):
                    total += v
            if total >= _SR_ROOM:
                mkt = [list(o) for o in (action.get("market") or []) if o]
                if len(mkt) < 10:
                    pending = {}
                    for o in mkt:
                        if o[0] == "SELL" and len(o) > 2:
                            try:
                                pending[o[1]] = pending.get(o[1], 0) + int(o[2])
                            except Exception:
                                pass
                    # Biggest pile first: that is the one crowding the harvest.
                    held = []
                    for item in _SR_ITEMS:
                        n = int(shed.get(item, 0) or 0) - pending.get(item, 0)
                        if n > 0:
                            held.append((n, item))
                    held.sort(reverse=True)
                    extra = [["SELL", item, min(n, _SR_CAP)] for n, item in held]
                    if extra:
                        new = (mkt + extra)[:10]
                        if new != mkt:
                            action = dict(action, market=new)
    except Exception:
        pass
    return action

kaggle_submission_agent = agent
'''


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    dst = os.path.join(out, "main.py")
    shutil.copy(SRC, dst)
    items = [s.strip() for s in ITEMS.split(",") if s.strip()]
    layer = (LAYER.replace("__ITEMS__", repr(items))
                  .replace("__ROOM__", str(ROOM))
                  .replace("__CAP__", str(CAP))
                  .replace("__FROM__", str(FROM)))
    with open(dst, "a") as fh:
        fh.write(layer)
    print("built %s from %s (ROOM=%d CAP=%d FROM=%d ITEMS=%s)"
          % (out, SRC, ROOM, CAP, FROM, items))


main()
