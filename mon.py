"""Ladder monitor: every active submission, its climb, and whether 3100 is reachable.

The ladder seeds at 600 and the K-factor decays, so a rating only means
something read together with how many games produced it and how fast it is
still moving.  Prints one row per submission plus a verdict.
"""

import base64
import json
import os
import sys
import time
import urllib.request

CFG = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
TARGET = 3100.0


def post(url, body, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
            req.add_header("Content-Type", "application/json")
            tok = base64.b64encode(("%s:%s" % (CFG["username"], CFG["key"])).encode()).decode()
            req.add_header("Authorization", "Basic " + tok)
            return json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
        except Exception as e:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def _secs(v):
    """endTime comes back as {"seconds": n} on some routes and ISO text on others."""
    if isinstance(v, dict):
        return float(v.get("seconds") or 0)
    if isinstance(v, str):
        import calendar, datetime
        t = v.replace("Z", "").split(".")[0]
        try:
            return calendar.timegm(datetime.datetime.strptime(t, "%Y-%m-%dT%H:%M:%S").timetuple())
        except Exception:
            return 0.0
    return 0.0


def games_for(sub):
    eps = post("https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes",
               {"submissionId": sub}).get("episodes") or []
    out = []
    for e in eps:
        if e.get("state") != "COMPLETED":
            continue
        me = next((a for a in e["agents"] if a.get("submissionId") == sub), None)
        op = next((a for a in e["agents"] if a.get("submissionId") != sub), None)
        if not me or not op or me.get("updatedScore") is None:
            continue
        out.append({
            "ep": e["id"],
            "end": _secs(e.get("endTime") or e.get("createTime")),
            "pre": me.get("initialScore"),
            "post": me.get("updatedScore"),
            "mine": me.get("reward"),
            "theirs": op.get("reward"),
            "opp": op.get("submissionId"),
        })
    out.sort(key=lambda g: g["end"])
    return out


def main():
    import kaggle
    api = kaggle.KaggleApi(); api.authenticate()
    subs = api.competition_submissions("kaggriculture")
    want = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    print("%-10s %-22s %6s %5s %6s %7s %8s  %s" % (
        "sub", "name", "elo", "n", "win%", "d/last10", "age_h", "verdict"))
    now = time.time()
    for s in subs[:want]:
        sub = int(s.ref)
        try:
            g = games_for(sub)
        except Exception as e:
            print("%-10d  FETCH FAIL %s" % (sub, e))
            continue
        if not g:
            print("%-10d %-22s  no completed games yet" % (sub, str(s.description)[:22]))
            continue
        elo = g[-1]["post"]
        n = len(g)
        w = sum(1 for x in g if (x["mine"] or 0) > (x["theirs"] or 0))
        last = g[-10:]
        d = (last[-1]["post"] - last[0]["pre"]) / max(1, len(last))
        age = (now - g[0]["end"]) / 3600.0
        # projection: rating gain per game is decaying; extrapolate 40 more games
        proj = elo + d * 40
        if elo >= TARGET:
            v = "AT TARGET"
        elif proj >= TARGET:
            v = "maybe (proj %.0f)" % proj
        else:
            v = "NO (proj %.0f)" % proj
        print("%-10d %-22s %6.0f %5d %5.0f%% %7.1f %8.1f  %s" % (
            sub, str(s.description)[:22], elo, n, 100.0 * w / n, d, age, v))
        sys.stdout.flush()


main()
