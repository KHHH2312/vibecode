#!/bin/sh
# submit.sh <agentdir> "<message>"
# Kaggle runs the submitted main.py by exec-ing its source, so the tarball holds
# exactly one self-contained file at its root.
set -e
cd "$(dirname "$0")"
D="$1"; MSG="$2"
test -f "$D/main.py"
./venv/bin/python -c "import ast,sys; ast.parse(open('$D/main.py').read()); print('syntax ok')"
# A build that does not load forfeits every game, so never ship one unplayed.
./venv/bin/python smoke.py "$D/main.py"
# A layer can be present in the file and not published as the entry point, in
# which case the engine runs the base and the submission silently spends a
# daily slot on an agent we did not mean to send.  Print what will actually run.
./venv/bin/python entrycheck.py "$D"
T="$(mktemp -d)"
cp "$D/main.py" "$T/main.py"
tar -czf "sub_$(basename $D).tar.gz" -C "$T" main.py
rm -rf "$T"
./venv/bin/kaggle competitions submit -c kaggriculture \
    -f "sub_$(basename $D).tar.gz" -m "$MSG"
