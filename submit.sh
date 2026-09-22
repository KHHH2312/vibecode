#!/bin/sh
# submit.sh <agentdir> "<message>"
# Kaggle runs the submitted main.py by exec-ing its source, so the tarball holds
# exactly one self-contained file at its root.
set -e
cd "$(dirname "$0")"
D="$1"; MSG="$2"
test -f "$D/main.py"
./venv/bin/python -c "import ast,sys; ast.parse(open('$D/main.py').read()); print('syntax ok')"
T="$(mktemp -d)"
cp "$D/main.py" "$T/main.py"
tar -czf "sub_$(basename $D).tar.gz" -C "$T" main.py
rm -rf "$T"
./venv/bin/kaggle competitions submit -c kaggriculture \
    -f "sub_$(basename $D).tar.gz" -m "$MSG"
