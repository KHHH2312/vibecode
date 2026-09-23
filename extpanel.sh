#!/bin/bash
# Score a build against EXTERNAL agents -- ones outside our own lineage.
#
# Every tuning decision in this project until 19 Sep was made against our own
# family, which all shares one tape chassis, so a change that exploited that
# shared behaviour looked like a gain and was not.  v58 beat v57 78-85% in
# self-play and scores 67.5% against a real opponent where v57 scores 90%.
#
# These eight are the agents that actually separate our builds: they were
# screened from the 23 (of 49) that load at all, keeping those we do NOT beat
# ~100%.  An opponent we always beat teaches nothing.
set -u
cd "$(dirname "$0")"
A=$1; N=${2:-15}; BASE=${3:-5200000}; STRIDE=${4:-8117}
export AB_WORKERS=${AB_WORKERS:-3}
for B in agents/alperen5252525__turn-one-market-advantage-kaggriculture \
         agents/anhadmahajan06__kaggriculture-autonomous-ai-farming-agent \
         agents/flexonafft__kaggriculture-most-powerfull-route \
         agents/lynnsakurai__farming-score-v2-a-better-approach \
         agents/lynnsakurai__farming-score-v5-timing-optimized \
         agents/nathanjacob__kaggriculture-pipe-5-terminal-boost \
         agents/xuanzhang001__kaggriculture-auto-top1 \
         agents/aurax7__kaggriculture-shop-router-reactive-v7
do
  [ -f "$B/main.py" ] || { echo "### $(basename $B) MISSING"; continue; }
  printf "  %-46s " "$(basename $B | cut -c1-46)"
  ./venv/bin/python ab.py "$A" "$B/main.py" "$N" "$BASE" "$STRIDE" 2>&1 | tail -1
done
echo "EXTPANEL DONE"
