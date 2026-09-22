"""How much money do stronger opponents actually make against us?

Elo says we lose; it does not say why.  Bucketing every completed game by the
opponent's rating and printing both seats' final cash separates the two
possible stories: either strong opponents out-earn us outright (an economic
gap, worth chasing) or they beat us by a rounding error (a margin game, where
only variance reduction pays).
"""
import base64, json, os, sys, time, urllib.request

CFG = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))


def post(url, body, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", "Basic " + base64.b64encode(
                ("%s:%s" % (CFG["username"], CFG["key"])).encode()).decode())
            return json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def main():
    subs = [int(x) for x in sys.argv[1:]]
    rows = []
    for sub in subs:
        eps = post("https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes",
                   {"submissionId": sub}).get("episodes") or []
        for e in eps:
            if e.get("state") != "COMPLETED":
                continue
            me = next((a for a in e["agents"] if a.get("submissionId") == sub), None)
            op = next((a for a in e["agents"] if a.get("submissionId") != sub), None)
            if not me or not op or me.get("reward") is None or op.get("reward") is None:
                continue
            rows.append((op.get("initialScore") or 0, float(me["reward"]), float(op["reward"]),
                         sub, e["id"]))
    rows.sort()
    print("n games", len(rows))
    buckets = [(0, 2200), (2200, 2500), (2500, 2700), (2700, 2900), (2900, 3300)]
    print("%-12s %5s %7s %10s %10s %9s %7s" % ("opp elo", "n", "win%", "our$", "their$", "margin", "med|m|"))
    for lo, hi in buckets:
        b = [r for r in rows if lo <= r[0] < hi]
        if not b:
            continue
        w = sum(1 for r in b if r[1] > r[2])
        ours = sum(r[1] for r in b) / len(b)
        theirs = sum(r[2] for r in b) / len(b)
        mg = sorted(abs(r[1] - r[2]) for r in b)
        print("%-12s %5d %6.0f%% %10.0f %10.0f %+9.0f %7.0f" % (
            "%d-%d" % (lo, hi), len(b), 100.0 * w / len(b), ours, theirs,
            ours - theirs, mg[len(mg) // 2]))
    # worst losses to strong opponents
    strong = [r for r in rows if r[0] >= 2700 and r[1] < r[2]]
    strong.sort(key=lambda r: r[1] - r[2])
    print("\nworst losses vs >=2700:")
    for r in strong[:12]:
        print("  ep %d sub %d  opp_elo %.0f  us %.0f  them %.0f  (%+.0f)" % (
            r[4], r[3], r[0], r[1], r[2], r[1] - r[2]))
    json.dump([{"opp_elo": r[0], "ours": r[1], "theirs": r[2], "sub": r[3], "ep": r[4]} for r in rows],
              open("gap.json", "w"))


main()
