"""Append a feed-reserve layer: stop selling the wheat the animals need.

In episode 111711110 our seat and the opponent's ran the same plan on the same
board -- same tiles, same crops, same hands -- and we lost by 27,867 coins.  The
sell ledger says why: we sold 57 MORE wheat than they did, and produced 99
fewer MILK and 57 fewer WOOL.

Wheat is animal feed.  FEED consumes one wheat per animal per day; an animal
that goes two consecutive days unfed escapes and its structure is left empty,
and the care bonus that doubles a production day is only granted on a day the
animal was both fed and cared for.  Wheat is also the flattest market in the
game -- log curve, 0.20 target -- so it sells for about 22 coins whatever we do,
while the milk and wool it turns into are worth 100 to 200.  Selling feed is
the worst trade on the board, and it is the one our tape makes.

This layer holds back a reserve of wheat sized to the herd, and can top the
reserve up from the market when the farm is short.

    MK_SRC=<base>/main.py DAYS=3 BUY=1 mkfeed.py <outdir>
"""
import os
import shutil
import sys

SRC = os.environ.get("MK_SRC") or "ag61/main.py"
DAYS = float(os.environ.get("DAYS") or 3)
BUY = os.environ.get("BUY", "0") == "1"
BUYCAP = int(os.environ.get("BUYCAP") or 4)
FROM = int(os.environ.get("FROM") or 24)

LAYER = '''

# --------------------------------------------------------------------------- Feed Reserve Layer
_FR_PARENT = agent
del agent

_FR_DAYS = __DAYS__
_FR_BUY = __BUY__
_FR_BUYCAP = __BUYCAP__
_FR_FROM = __FROM__
_FR_LAST_DAY = 29


def _fr_herd(farm):
    n = 0
    for row in farm.get("tiles") or []:
        for tile in row:
            if isinstance(tile, dict) and "animal" in tile:
                n += 1
    return n


def agent(observation, configuration=None):
    action = _FR_PARENT(observation, configuration)
    try:
        step = int(observation.get("step", 0))
        day = step // 24
        if step >= _FR_FROM and day < _FR_LAST_DAY:
            seat = int(observation["player"])
            farm = observation["farms"][seat]
            herd = _fr_herd(farm)
            if herd:
                # One wheat per animal per day, for as many days as we hold
                # cover for -- but never more than the season has left.
                need = int(herd * min(_FR_DAYS, _FR_LAST_DAY - day))
                shed = ((observation.get("private") or {}).get("shed") or {})
                have = int(shed.get("WHEAT", 0) or 0)
                mkt = [list(o) for o in (action.get("market") or []) if o]
                new = []
                for o in mkt:
                    if o[0] == "SELL" and len(o) > 2 and o[1] == "WHEAT":
                        try:
                            qty = int(o[2])
                        except Exception:
                            qty = 0
                        spare = have - need
                        if spare <= 0:
                            have -= 0
                            continue
                        if qty > spare:
                            qty = spare
                        have -= qty
                        new.append(["SELL", "WHEAT", qty])
                        continue
                    new.append(o)
                if _FR_BUY and have < need and len(new) < 10:
                    short = need - have
                    if short > _FR_BUYCAP:
                        short = _FR_BUYCAP
                    new.append(["BUY_PRODUCT", "WHEAT", short])
                new = new[:10]
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
    layer = (LAYER.replace("__DAYS__", repr(DAYS))
                  .replace("__BUY__", "True" if BUY else "False")
                  .replace("__BUYCAP__", str(BUYCAP))
                  .replace("__FROM__", str(FROM)))
    with open(dst, "a") as fh:
        fh.write(layer)
    print("built %s from %s (DAYS=%s BUY=%s BUYCAP=%d FROM=%d)"
          % (out, SRC, DAYS, BUY, BUYCAP, FROM))


main()
