"""Append a layer that funds the turn's big-ticket purchases before they fire.

`_commit_unit` silently drops a BUY_ANIMAL or BUY_LAND the farm cannot afford
at the moment that order's slot comes up -- no retry, no error.  Orders resolve
in slot order, so a SELL sitting below the purchase in the same turn is money
that arrives too late.

This matters because the animals are where the compounding is.  In replay
111711110 we finished with 9 cows against the opponent's 12 and three pens
stood empty all game, and our cash at the moments the controller wanted a
400-coin cow was 203 and then 303.  The whole 99-unit milk shortfall and the
57-unit wheat surplus that went with it follow from those three purchases
failing.

The parent's own precedence layer reorders only when `money > 1500`, which is
exactly when reordering is not needed.  This one reorders when the turn holds a
purchase the current purse cannot cover, and otherwise leaves the order list
alone.

    MK_SRC=<base>/main.py mkprec.py <outdir>
"""
import os
import shutil
import sys

SRC = os.environ.get("MK_SRC") or "ag61/main.py"
ALSO_SEED = os.environ.get("ALSO_SEED", "0") == "1"

LAYER = '''

# --------------------------------------------------------------------------- Purchase Funding Layer
_PF_PARENT = agent
del agent

_PF_COST = {"GOOSE": 300, "COW": 400, "SHEEP": 500}
_PF_LAND = [1000, 2000, 4000]
_PF_SEEDS = __ALSO_SEED__


def agent(observation, configuration=None):
    action = _PF_PARENT(observation, configuration)
    try:
        if int(observation.get("step", 0)) >= 24:
            mkt = [list(o) for o in (action.get("market") or []) if o]
            sells = [o for o in mkt if o[0] == "SELL"]
            if sells and len(sells) != len(mkt):
                seat = int(observation["player"])
                farm = observation["farms"][seat]
                money = float(farm["money"])
                need = 0.0
                for o in mkt:
                    if o[0] == "BUY_ANIMAL" and len(o) > 1:
                        qty = 1
                        if len(o) > 2:
                            try:
                                qty = max(1, int(o[2]))
                            except Exception:
                                qty = 1
                        need += _PF_COST.get(o[1], 0) * qty
                    elif o[0] == "BUY_LAND":
                        extra = len(farm.get("unlocked_quadrants") or [1]) - 1
                        if 0 <= extra < len(_PF_LAND):
                            need += _PF_LAND[extra]
                    elif _PF_SEEDS and o[0] == "BUY_SEED":
                        need += 100
                # Only touch the order list when the purse cannot cover the
                # turn's purchases; otherwise the parent's ordering stands.
                if need > money:
                    rest = [o for o in mkt if o[0] != "SELL"]
                    new = (sells + rest)[:10]
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
    layer = LAYER.replace("__ALSO_SEED__", "True" if ALSO_SEED else "False")
    src_text = open(dst).read()
    tag = _unique_prefix(src_text, "_PF_")
    layer = layer.replace("_PF_", tag)
    assert tag + "PARENT" not in src_text, tag
    with open(dst, "a") as fh:
        fh.write(layer)
    print("built %s from %s (ALSO_SEED=%s)" % (out, SRC, ALSO_SEED))


main()
