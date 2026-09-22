#!/bin/sh
# A single A/B worker draining a priority directory.
#
# One head-to-head at a time: the box has four cores and AB_WORKERS=3 leaves
# one free.  Jobs are files named jobs/<priority>_<name> whose contents are
# "<A>" (played against the V61 base) or "<A> <B>" for an explicit opponent.
# The worker always takes the lexicographically first name, so a new job can
# jump the queue without restarting anything.
#
# Every build is smoke-tested first.  An agent that dies on its first turn
# forfeits, and a forfeit scores like a real, catastrophic result -- that
# already produced two wrong conclusions, so a build that does not load is
# never handed to ab.py.
cd "$(dirname "$0")"
P=./venv/bin/python
mkdir -p jobs jobs_done runs
while true; do
  job=$(ls jobs 2>/dev/null | sort | head -1)
  if [ -z "$job" ]; then
    sleep 20
    continue
  fi
  spec=$(cat "jobs/$job")
  mv "jobs/$job" "jobs_done/$job"
  a=$(echo "$spec" | awk '{print $1}')
  b=$(echo "$spec" | awk '{print $2}')
  [ -z "$b" ] && b=ag61
  if [ ! -f "$a/main.py" ] || [ ! -f "$b/main.py" ]; then
    continue
  fi
  if ! $P smoke.py "$a/main.py" > "runs/smoke_${a}.log" 2>&1; then
    cp "runs/smoke_${a}.log" "runs/ab_${a}_vs_${b}.log"
    continue
  fi
  AB_WORKERS=3 $P ab.py "$a/main.py" "$b/main.py" 20 500000 \
    > "runs/ab_${a}_vs_${b}.log" 2>&1
done
