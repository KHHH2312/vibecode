"""Append a layer that buys cows where the parent would buy sheep.

Both animals stand on a PASTURE, so they compete for the same tiles, but they
are not worth the same. From the revenue ledger of one V61 game:

    MILK  245 units, mean 188 (base 160), from ~9 cows   -> ~5,100 a head
    WOOL  133 units, mean 104 (base 200), from ~6 sheep  -> ~2,300 a head

A cow yields every 2 days from day 8 and a sheep every 3 from day 6, and milk
has three shops plus the town centre behind it while wool has only YARN_STORE,
which may not unlock until late -- in that game the wool book was in deep glut
and realised half of base. The cow also costs 400 against the sheep's 500.

The risk is placement: the animal lands in the shed and a hand has to PICKUP
that specific item and PLACE it on the pen, so if the parent's routine asks for
a sheep it will find none. That is exactly what the head-to-head is for.

    MK_SRC=<base>/main.py mkswap.py <outdir>
"""
import os
import shutil
import sys

SRC = os.environ.get("MK_SRC") or "ag61/main.py"
FROM_ANIMAL = os.environ.get("FROM_ANIMAL") or "SHEEP"
TO_ANIMAL = os.environ.get("TO_ANIMAL") or "COW"

LAYER = '''

# --------------------------------------------------------------------------- Herd Swap Layer
_HS_PARENT = agent
del agent

_HS_FROM = "__FROM__"
_HS_TO = "__TO__"


def agent(observation, configuration=None):
    action = _HS_PARENT(observation, configuration)
    try:
        mkt = [list(o) for o in (action.get("market") or []) if o]
        new = []
        changed = False
        for o in mkt:
            if o[0] == "BUY_ANIMAL" and len(o) > 1 and o[1] == _HS_FROM:
                o = [o[0], _HS_TO] + list(o[2:])
                changed = True
            new.append(o)
        if changed:
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
    layer = LAYER.replace("__FROM__", FROM_ANIMAL).replace("__TO__", TO_ANIMAL)
    src_text = open(dst).read()
    tag = _unique_prefix(src_text, "_HS_")
    layer = layer.replace("_HS_", tag)
    assert tag + "PARENT" not in src_text, tag
    with open(dst, "a") as fh:
        fh.write(layer)
    print("built %s from %s (%s -> %s)" % (out, SRC, FROM_ANIMAL, TO_ANIMAL))


main()
