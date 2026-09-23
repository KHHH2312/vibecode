"""Append a homestead layer: buy the fourth quadrant and work it with our own crew.

Every agent in the public meta scores the same ~124,000 in self-play, because
they are all the same tape and they all stop at three quadrants, 58 planted
tiles and 17 animals.  Meanwhile STRAWBERRY clears at 216 against a base of 120
and MILK at 188 against 160 -- both permanently scarce, so the town wants more
than anyone supplies.  Production, not price, is the binding constraint, and
the unused asset is the fourth quadrant: 4,000 coins for 25 tiles.

Two engine facts make an independent crew possible.

  * The parent's action carries exactly as many `hands` entries as it believes
    it has, and the engine applies entry i to hand i+1.  Hands at higher
    indices are invisible to it.  Measured over a full season, the parent's
    hiring always finishes by hour 4, so a hand hired from hour 6 onward is
    always above its indices and it never hires past us.
  * LAND_ORDER is NE, SW, SE, so the fourth quadrant is SE -- and the
    shed-access tile (5,5) sits inside it.  A crew spawns on its own land.

Hiring is the n-th hire of the DAY at fib(n) coins, so a crew added after the
parent's eleven costs 144, 233, 377 a head per day.  Three hands for twenty
days is about 15,000 against roughly 40,000 of strawberry at its realised
price, so the crew must stay small and must actually work.

One hard safety rule: the engine drops ALL plant requests for a crop when the
turn asks for more than the seed bag holds, so a careless PLANT from the crew
would silently cancel the parent's planting too.  Every PLANT here is counted
against the parent's own requests first.

    MK_SRC=<base>/main.py CREW=2 LAND_DAY=8 mkhome.py <outdir>
"""
import os
import shutil
import sys

SRC = os.environ.get("MK_SRC") or "ag61/main.py"
CREW = int(os.environ.get("CREW") or 2)
LAND_DAY = int(os.environ.get("LAND_DAY") or 8)
LAND_CASH = int(os.environ.get("LAND_CASH") or 9000)
HIRE_HOUR = int(os.environ.get("HIRE_HOUR") or 6)
CROP = os.environ.get("CROP") or "STRAWBERRY"
SEED_BUF = int(os.environ.get("SEED_BUF") or 4)
CARRY = int(os.environ.get("CARRY") or 8)
BUY_LAND = os.environ.get("BUY_LAND", "1") == "1"
PLOT = int(os.environ.get("PLOT") or 5)   # side of the worked block
# SCOPE=OWNED works the land we already hold, so no 4,000-coin quadrant is
# needed: the parent leaves tiles empty as it harvests, and those are free.
SCOPE = os.environ.get("SCOPE") or "SE"
# OWNED mode shares the board with the parent, so the crew must not touch
# what the parent is managing: it works only the tiles it planted itself.
MINE_ONLY = os.environ.get("MINE_ONLY", "1") == "1"
MIN_LAND = int(os.environ.get("MIN_LAND") or 2)

LAYER = '''

# --------------------------------------------------------------------------- Homestead Layer
_HM_PARENT = agent
del agent

_HM_CREW = __CREW__
_HM_LAND_DAY = __LAND_DAY__
_HM_LAND_CASH = __LAND_CASH__
_HM_HIRE_HOUR = __HIRE_HOUR__
_HM_CROP = "__CROP__"
_HM_SEED_BUF = __SEED_BUF__
_HM_CARRY = __CARRY__
_HM_BUY_LAND = __BUY_LAND__
_HM_PLOT = __PLOT__
_HM_SCOPE = "__SCOPE__"
_HM_MINE_ONLY = __MINE_ONLY__
_HM_OURS = set()
_HM_MIN_LAND = __MIN_LAND__
_HM_SHED = [(4, 4), (5, 4), (4, 5), (5, 5)]
# crop -> (ongoing, max_yield, max_yield_day).  A non-ongoing crop is destroyed
# by HARVEST, and it starts at one unit and gains only from WATER inside the
# window [(max_yield_day+1)//2, max_yield_day], so harvesting it early throws
# the whole tile away for a single unit.
_HM_CROPS = {
    "WHEAT": (False, 6, 4), "CARROT": (False, 4, 3), "MELON": (False, 6, 12),
    "TOMATO": (True, 4, 8), "STRAWBERRY": (True, 4, 10),
}
# Hiring is priced fib(n) on the n-th hire of the day, so the crew size has to
# be counted, not inferred from the index gap: the parent's declared hand count
# moves around during the day and an inferred gap re-hires at 377, 610, 987...
_HM_STATE = {"day": -1, "hired": 0}
_HM_HOME = (5, 5)          # shed access tile that also sits inside SE


def _hm_quadrant(x, y):
    return ("N" if y < 5 else "S") + ("W" if x < 5 else "E")


def _hm_step_toward(pos, target):
    """One orthogonal step; x first so two hands do not tangle."""
    x, y = pos[0], pos[1]
    tx, ty = target
    if x < tx:
        return ["EAST"]
    if x > tx:
        return ["WEST"]
    if y < ty:
        return ["SOUTH"]
    if y > ty:
        return ["NORTH"]
    return None


def _hm_survey(tiles, claimed, day, quads):
    """Work available in SE, best first: harvest, then water, then clear, then plant."""
    urgent, harvest, water, clear, plant = [], [], [], [], []
    if _HM_SCOPE == "OWNED":
        ys, xs = range(0, 10), range(0, 10)
    else:
        ys, xs = range(5, 5 + _HM_PLOT), range(5, 5 + _HM_PLOT)
    for y in ys:
        row = tiles[y] if y < len(tiles) else []
        for x in xs:
            if _HM_SCOPE == "OWNED" and _hm_quadrant(x, y) not in quads:
                continue
            if (x, y) in claimed or x >= len(row):
                continue
            tile = row[x]
            if tile == "LOCKED":
                continue
            if tile is None:
                plant.append((x, y))
                continue
            if not isinstance(tile, dict):
                continue
            kind = tile.get("kind")
            if kind == "PLANT":
                if _HM_MINE_ONLY and _HM_SCOPE == "OWNED" and (x, y) not in _HM_OURS:
                    continue
                thirsty = not tile.get("watered_today")
                dying = thirsty and int(tile.get("consecutive_unwatered", 0) or 0) >= 1
                units = int(tile.get("yield_units", 0) or 0)
                ongoing, cap, ripe_day = _HM_CROPS.get(
                    tile.get("crop"), (True, 4, 10))
                age = day - int(tile.get("planted_day", day) or day)
                if ongoing:
                    ready = units > 0
                else:
                    # Only cut it once it cannot grow further.
                    ready = units >= cap or age >= ripe_day
                in_window = (not ongoing) and ((ripe_day + 1) // 2) <= age <= ripe_day
                if ready:
                    harvest.append((x, y))
                elif dying:
                    urgent.append((x, y))
                elif thirsty:
                    water.append((x, y))
            elif kind == "WEED":
                if not (_HM_MINE_ONLY and _HM_SCOPE == "OWNED") or (x, y) in _HM_OURS:
                    clear.append((x, y))
                    _HM_OURS.discard((x, y))
    return urgent, harvest, water, clear, plant


def agent(observation, configuration=None):
    action = _HM_PARENT(observation, configuration)
    try:
        step = int(observation.get("step", 0))
        day, hour = step // 24, step % 24
        seat = int(observation["player"])
        farm = observation["farms"][seat]
        priv = observation.get("private") or {}
        quads = farm.get("unlocked_quadrants") or []
        money = float(farm.get("money", 0))
        mkt = [list(o) for o in (action.get("market") or []) if o]
        hands_act = [list(h) for h in (action.get("hands") or [])]
        tape_n = len(hands_act)
        real_n = len(farm.get("hands") or [])
        have_se = "SE" in quads

        # ---- market: land, crew, seed -- only ever into free slots ----
        extra = []
        if _HM_BUY_LAND and not have_se and day >= _HM_LAND_DAY and money >= _HM_LAND_CASH:
            extra.append(["BUY_LAND"])
        if _HM_STATE.get("day") != day:
            _HM_STATE["day"] = day
            _HM_STATE["hired"] = 0
            _HM_STATE["base_n"] = None
        if _HM_STATE.get("base_n") is None or hour < _HM_HIRE_HOUR:
            # Before our first hire of the day the parent owns every hand.
            _HM_STATE["base_n"] = real_n
        owned = real_n - int(_HM_STATE.get("base_n") or 0)
        if owned < 0:
            owned = 0
        _HM_STATE["hired"] = owned
        can_work = have_se or _HM_SCOPE == "OWNED"
        idle_land = 0
        if _HM_SCOPE == "OWNED":
            _tiles = farm.get("tiles") or []
            for _y in range(10):
                _row = _tiles[_y] if _y < len(_tiles) else []
                for _x in range(10):
                    if _x < len(_row) and _row[_x] is None and _hm_quadrant(_x, _y) in quads:
                        idle_land += 1
            # Keep paying only while the crew still has somewhere to plant or
            # something of its own still growing.
            if idle_land < _HM_MIN_LAND and not _HM_OURS:
                can_work = False
        if (can_work and hour >= _HM_HIRE_HOUR and owned < _HM_CREW
                and money > 2500 and len(mkt) + len(extra) < 10):
            extra.append(["HIRE"])
        seeds = int((priv.get("seeds") or {}).get(_HM_CROP, 0) or 0)
        if can_work and seeds < _HM_SEED_BUF and money > 4000 and day < 26:
            extra.append(["BUY_SEED", _HM_CROP, _HM_SEED_BUF])
        if extra and len(mkt) < 10:
            room = 10 - len(mkt)
            action = dict(action, market=mkt + extra[:room])

        # ---- drive our own hands ----
        # The parent sizes its hands list to the real hand count, so the crew is
        # not invisible to it after all -- but it has no work for them and
        # issues PASS.  Ours are the last hires of the day, so the trailing
        # entries are replaced rather than appended.
        n_mine = int(_HM_STATE.get("hired", 0))
        if n_mine and real_n and can_work:
            tiles = farm.get("tiles") or []
            invs = priv.get("inventories") or []
            # The engine cancels every PLANT for a crop when the turn asks for
            # more than the bag holds, so count what the parent already wants.
            asked = 0
            for a in [action.get("farmer") or []] + hands_act:
                if isinstance(a, list) and len(a) >= 2 and a[0] == "PLANT" and a[1] == _HM_CROP:
                    asked += 1
            budget = seeds - asked
            claimed = set()
            mine = []
            first_mine = real_n - n_mine
            if first_mine < 0:
                first_mine = 0
            for idx in range(first_mine, real_n):
                pos = farm["hands"][idx]
                inv = invs[idx + 1] if idx + 1 < len(invs) else {}
                carried = 0
                for v in (inv or {}).values():
                    if isinstance(v, (int, float)):
                        carried += v
                here = (pos[0], pos[1])

                urgent, harvest, water, clear, plant = _hm_survey(tiles, claimed, day, quads)

                def _near(cands):
                    best, bd = None, 99
                    for c in cands:
                        d = abs(c[0] - here[0]) + abs(c[1] - here[1])
                        if d < bd:
                            best, bd = c, d
                    return best

                target, job = None, None
                if urgent:
                    target, job = _near(urgent), ["WATER"]
                elif harvest:
                    target, job = _near(harvest), ["HARVEST"]
                elif water:
                    target, job = _near(water), ["WATER"]
                elif plant and budget > 0 and hour <= 20:
                    target, job = _near(plant), ["PLANT", _HM_CROP]
                elif clear:
                    target, job = _near(clear), ["DIG"]

                if target is None:
                    mine.append(["PASS"])
                    continue
                claimed.add(target)
                if here == target:
                    if job[0] == "PLANT":
                        budget -= 1
                        _HM_OURS.add(target)
                    elif job[0] == "HARVEST":
                        _HM_OURS.discard(target)
                    mine.append(job)
                else:
                    mine.append(_hm_step_toward(pos, target) or ["PASS"])

            keep = list(hands_act[:first_mine])
            while len(keep) < first_mine:
                keep.append(["PASS"])
            action = dict(action, hands=keep + mine)
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
    layer = (LAYER.replace("__CREW__", str(CREW))
                  .replace("__LAND_DAY__", str(LAND_DAY))
                  .replace("__LAND_CASH__", str(LAND_CASH))
                  .replace("__HIRE_HOUR__", str(HIRE_HOUR))
                  .replace("__CROP__", CROP)
                  .replace("__SEED_BUF__", str(SEED_BUF))
                  .replace("__CARRY__", str(CARRY))
                  .replace("__BUY_LAND__", "True" if BUY_LAND else "False")
                  .replace("__PLOT__", str(PLOT))
                  .replace("__SCOPE__", SCOPE)
                  .replace("__MINE_ONLY__", "True" if MINE_ONLY else "False")
                  .replace("__MIN_LAND__", str(MIN_LAND)))
    src_text = open(dst).read()
    tag = _unique_prefix(src_text, "_HM_")
    layer = layer.replace("_HM_", tag)
    assert tag + "PARENT" not in src_text, tag
    with open(dst, "a") as fh:
        fh.write(layer)
    print("built %s from %s (CREW=%d LAND_DAY=%d CROP=%s)"
          % (out, SRC, CREW, LAND_DAY, CROP))


main()
