"""Override the day-6 route choice so the shop->route table can be tested.

The router picks a tape on day 6 from the first two shops that unlock, via two
hand-built tables: _R108_SHOP_ROUTES for layouts without YARN_STORE and
_R110_OLD_SHOPS for layouts with it.  Both map all 64 pairs, so the fallback is
almost never used and the tables themselves are the thing worth testing -- but
a per-pair search needs seeds that produce each pair, and 40 seeds gave 26
distinct pairs, so per-pair evidence is far too thin.

The cheap decisive test is a global override: send EVERY game down one route and
compare against the table.  If the table encodes real knowledge, every constant
loses to it.  If some constant wins, the table is worse than a fixed guess and
the mapping is worth rebuilding.

    mkroute.py <out_dir> [MK_SRC=lh44/main.py] [RT_NEW=n] [RT_OLD=n]

RT_NEW overrides the non-yarn branch, RT_OLD the yarn branch; leave either unset
to keep that branch's table.
"""
import os
import pathlib
import sys

SRC = os.environ.get("MK_SRC") or "lh44/main.py"
NEW = os.environ.get("RT_NEW")
OLD = os.environ.get("RT_OLD")

ANCHOR = ("        state['route']=_R108_SHOP_ROUTES.get(shops,100) if use_new "
          "else _R110_OLD_SHOPS.get(shops,0)")


def emit(out_dir):
    src = pathlib.Path(SRC).read_text(encoding="utf-8")
    assert src.count(ANCHOR) == 1, "router assignment not found exactly once"
    new_expr = NEW if NEW else "_R108_SHOP_ROUTES.get(shops,100)"
    old_expr = OLD if OLD else "_R110_OLD_SHOPS.get(shops,0)"
    line = "        state['route']=%s if use_new else %s" % (new_expr, old_expr)
    body = src.replace(ANCHOR, line)
    compile(body, "x", "exec")
    d = pathlib.Path(out_dir)
    d.mkdir(exist_ok=True)
    (d / "main.py").write_text(body, encoding="utf-8")
    print("wrote %s/main.py  non-yarn=%s  yarn=%s"
          % (out_dir, NEW or "table", OLD or "table"))


if __name__ == "__main__":
    emit(sys.argv[1])
