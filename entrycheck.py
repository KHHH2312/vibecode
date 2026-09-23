"""Report which global kaggle_environments would actually run, for any build."""
import sys
for path in sys.argv[1:]:
    p = path if path.endswith(".py") else path + "/main.py"
    ns = {}
    try:
        exec(compile(open(p).read(), p, "exec"), ns)
        last = [k for k, v in ns.items() if callable(v)][-1]
        fn = ns[last]
        print("%-10s entry=%-24s defined_as=%s" % (path, last, getattr(fn, "__name__", "?")))
    except Exception as exc:
        print("%-10s FAILED %r" % (path, exc))
