"""Download our own submission bundles and unpack each agent source.

Kaggle serves the uploaded tarball back at
/api/v1/competitions/submissions/download/<id>, so the exact code behind every
rating on the board is recoverable -- including the ones submitted from the
other machine.
"""
import base64, json, os, sys, tarfile, io, urllib.request, time

CFG = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
TOK = base64.b64encode(("%s:%s" % (CFG["username"], CFG["key"])).encode()).decode()
OUT = "subs"


def get(url, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(url)
            req.add_header("Authorization", "Basic " + TOK)
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.read()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(2 ** (i + 1))


def main():
    import kaggle
    api = kaggle.KaggleApi(); api.authenticate()
    subs = api.competition_submissions("kaggriculture")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    os.makedirs(OUT, exist_ok=True)
    for s in subs[:n]:
        sid = int(s.ref)
        d = os.path.join(OUT, str(sid))
        if os.path.isdir(d) and os.listdir(d):
            print(sid, "already have")
            continue
        raw = get("https://www.kaggle.com/api/v1/competitions/submissions/download/%d" % sid)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, "_desc.txt"), "w").write(str(s.description))
        try:
            with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
                tf.extractall(d)
            names = sorted(os.listdir(d))
        except tarfile.ReadError:
            open(os.path.join(d, "raw.bin"), "wb").write(raw)
            names = ["raw.bin"]
        print(sid, len(raw), names[:6], "|", str(s.description)[:40])
        sys.stdout.flush()


main()
