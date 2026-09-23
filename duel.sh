#!/bin/sh
# Sequential duels: each arg "A B SEED".  One ab.py at a time (4-core box).
cd "$(dirname "$0")"
P=./venv/bin/python
for spec in "$@"; do
  set -- $spec
  a=$1; b=$2; s=$3
  r=$(AB_WORKERS=3 $P ab.py "$a/main.py" "$b/main.py" 20 "$s" 2>/dev/null | grep games)
  echo "$a vs $b s$s | $r"
done
echo DUEL_DONE
