#!/bin/sh
# Score each variant against tk4, paired, on a grid tk4 was never tuned on.
# A positive mean here is a real improvement; self-play cannot tell.
#
# Before scoring, every variant is checked against the base it was layered on:
# a layer that is present in the file but not published as the module's entry
# point plays as the bare base and scores like one, which reads as a clean
# negative rather than as a broken build.  Identical action hashes mean the
# variant is inert, and an inert build is reported, never scored.
#
# Args: each "NAME BASE"
cd "$(dirname "$0")"
P=./venv/bin/python
SEED=${SEED:-1700000}
probe_hash() { $P probe.py "$SEED" "$1" 2>/dev/null | grep -o 'hash=[0-9a-f]*' | cut -d= -f2; }

for spec in "$@"; do
  set -- $spec
  name=$1; base=$2
  [ -f "$name/main.py" ] || { echo "$name MISSING"; continue; }
  $P smoke.py "$name/main.py" >/dev/null 2>&1 || { echo "$name SMOKE FAIL"; continue; }
  hv=$(probe_hash "$name")
  hb=$(probe_hash "$base")
  if [ -z "$hv" ] || [ -z "$hb" ]; then
    echo "$name PROBE FAIL (hv='$hv' hb='$hb') -- not scored"
    continue
  fi
  if [ "$hv" = "$hb" ]; then
    echo "$name INERT (plays identically to $base, hash $hv) -- not scored"
    continue
  fi
  r=$(AB_WORKERS=3 $P ab.py "$name/main.py" tk4/main.py 20 "$SEED" 2>/dev/null | grep games)
  echo "$name vs tk4 s$SEED | $r"
done
echo VSWEEP_DONE
