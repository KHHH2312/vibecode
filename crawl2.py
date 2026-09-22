"""Walk the ladder outward from our own submissions to find the top agents' ids.

ListEpisodes only takes a submissionId, and every episode names both seats, so
the ladder is a graph: each known submission reveals its opponents and their
ratings, and the highest-rated of those become the next frontier.  A few hops
from our own rating band reach the top of the board.
"""
import base64, json, os, sys, time, urllib.request

CFG = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
URL = "https://www.kaggle.com/api/i/competitions.EpisodeService/ListEpisodes"


def post(body, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), method="POST")
            req.add_header("Content-Type", "application/json")
            req.add_header("Authorization", "Basic " + base64.b64encode(
                ("%s:%s" % (CFG["username"], CFG["key"])).encode()).decode())
            return json.loads(urllib.request.urlopen(req, timeout=120).read().decode())
        except Exception:
            if i == tries - 1:
                return {}
            time.sleep(2 ** (i + 1))


def main():
    seeds = [int(x) for x in sys.argv[1:]] or [56446210]
    hops = 4
    scores, eps_of, seen = {}, {}, set()
    frontier = list(seeds)
    for hop in range(hops):
        nxt = []
        for sub in frontier:
            if sub in seen:
                continue
            seen.add(sub)
            eps = post({"submissionId": sub}).get("episodes") or []
            for e in eps:
                if e.get("state") != "COMPLETED":
                    continue
                for a in e["agents"]:
                    sid = a.get("submissionId")
                    sc = a.get("updatedScore")
                    if sid is None or sc is None:
                        continue
                    if sc > scores.get(sid, -1):
                        scores[sid] = sc
                    eps_of.setdefault(sid, []).append(e["id"])
        top = sorted(scores.items(), key=lambda kv: -kv[1])
        nxt = [s for s, _ in top[:12] if s not in seen]
        print("hop %d: know %d subs, best %s" % (hop, len(scores), [(s, round(v)) for s, v in top[:6]]))
        sys.stdout.flush()
        frontier = nxt
        if not frontier:
            break
    json.dump({"scores": scores, "episodes": {str(k): v[:40] for k, v in eps_of.items()}},
              open("ladder2.json", "w"))
    print("wrote ladder2.json")


main()
