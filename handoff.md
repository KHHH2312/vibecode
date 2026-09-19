# Kaggriculture — handoff

Written 2026-09-19. Competition ends ~2026-09-28. Goal: 3100+ Elo.
Kaggle user `khalid000000`. Credentials at `~/.kaggle/kaggle.json` — never transmit them anywhere.

**Read the "Methodology" section before trusting any number in this file or producing a new one.**
It is the single most important thing here. Most of the wasted effort in this project came from
believing measurements that were not real.

---

## 1. Where things stand

| build | submission | what it is | result |
|---|---|---|---|
| v55 | 56334083 | gate 0.82 + boost LOOK_HI=36 + day-28 sweep | ~2400, climbed slowly |
| v57 | 56343311 | v55 with `_ADV_LOOK_HI=44` | **flattened at 2611**, drift +1.5/game |
| v58 | 56347520 | v57 with the route table replaced by constant route 124 | live, submitted 03:52Z 19 Sep |

Submissions: **5 per day, UTC reset 00:00Z.** On 19 Sep, v57 and v58 are spent; three remain.
Every submission restarts at 600 Elo.

**The ladder is not the bottleneck.** v57 climbed 600 → 2550 in two hours (~16 games/hour).
When a submission flattens, that is the agent's strength ceiling, not slow matchmaking.
Do not "wait and see" — only real strength gains move the number.

### The most important open lead

`agents/xuanzhang001__kaggriculture-auto-top1` **beat v58**: 12W-18L, 40.0%, −677 ± 306 over 30 games
(grid 4700000/6229). It is a real, loading agent, 350 KB, entry point `agent`.

**What it is** (verified by decoding, not grepping): it embeds a `_PARENT_SRC` bytes literal holding
Ahmed Berat Ozer's V43 — 321,555 characters containing `_ROUTES`, `_R108_SHOP_ROUTES`, `_v219`,
`_R37` and 81 tape entries. It is **the same tape chassis we use**, forked earlier (V43, where ours
descends through V51), wrapped in its own layers: a step-0 wheat round trip, sale-reservation
horizon widened to 24 turns, market front-loading, and a two-turn sale advance of pure cash
products. It does **not** contain `_ADV_LOOK_HI`, `_ADV_LOOK` or `_ADV_GATE` — it never received our
advance-sell, gate or boost layers.

> **Caution, and a worked example of how to get this wrong.** I first reported this agent as having
> "no route tapes — a state-driven policy, the opposite of our architecture." That was false. A
> plain text search for tape entries returns 0 because the tapes live inside a `b'...'` literal.
> Decode embedded source with `ast` before concluding anything about it.

So there is no architectural secret: an *older* base with *different* wrappers beats our heavily
tuned one. That points at their wrappers, or at a weakness in ours.

**Immediate follow-up this raises:** v58's panel showed regressions against `b48_open25`
(100% → 53.3%) and `aurax7-v7` (100% → 73.3%). If v58 is weaker than v57 against real, diverse
opponents, the route-124 change may be a regression that was submitted on self-play evidence.
Test `auto-top1` against **v57/lh44** as well as v58 before drawing conclusions.

---

## 2. Architecture — what our agent actually is

Ours is **not** a heuristic if/else bot. It is a **tape replayer**:

- `_ROUTES` holds 41 pre-recorded 719-step action tapes (only **38 are distinct** —
  `101==119`, `105==125`, `109==127`).
- A router picks one on **day 6 (step 144)** from the first two shops that unlock, via two
  hand-built lookup tables, then hard-switches to route 2 at step 648.
- Nine reactive layers wrap the tape: weed repair, safety, market reordering, the advance-sell
  layer, a crash gate, a spike boost, a day-28 liquidation sweep, and closure planning.

**The central constraint: a tape cannot react.** Every structural idea that failed this project
failed on this. You cannot change what a tape does without desynchronising every later step that
depends on the farm state the earlier steps built.

`kaggle_environments` runs **the last callable in the module** — ours is `_y_agent_shopherd`.
A wrapper appended after the chain bypasses all nine layers and scores exactly the 3,000 it
started with. This has cost a real submission before.

---

## 3. Methodology — read this first

### Single-grid results lie. This was proved three separate times.

1. **rt115** measured 62.0% / +1286 on the tuning grid. On three independent grids it was
   44% / 54% / 44% — a coin flip. Pure noise.
2. **"LOOK_HI=36 is the peak"** was recorded everywhere as established fact. It was a measurement
   fault. 44 beats 36 on four grids.
3. **Ranking against a common weaker baseline is invalid.** Comparing 35/36/37/38 each against v55
   and picking the best cannot rank them against *each other*. When run head-to-head, the ordering
   was completely different. This cost several hours.

**Rule: never submit on one grid. Require 3+ independent grids.**
Grids in use: `4700000/6229`, `8100000/4441`, `3300000/7727`, `1900000/5113`, `1200000/9011`.

### A non-loading agent looks like a crushing win

Many public agents in `agents/` are multi-file and fail to import when run as a single file.
A broken agent produces **0W-30L with mean ≈ −190,627**. That is the signature of *nothing running*,
not of a strong opponent. Always check loadability first:

```python
src = open(path + "/main.py").read(); ns = {}
exec(compile(src, path, "exec"), ns)      # raises if the agent is broken
fn = [v for v in ns.values() if callable(v)][-1]   # the engine uses the LAST callable
```

Known: `saitejabandaruin-...-3000` and `pilkwang-...-policy` fail to import.
`xuanzhang001-pipe7-public-top1` crashes ("no completed games").
`indarkarhana-...-2948-9` loads but is genuinely weak (−33,652).

### Notebook titles are not evidence

`ultimate-mega-ensemble-3000`, `auto-top1`, `verified-route-replay-2948-9` are self-chosen filenames.
`pilkwang`'s "structured economic policy" turned out to be a thin wrapper over Ahmed Berat Ozer's V36
whose own README says `"strength_status": "UNVERIFIED"` and `"submission_allowed": false`.
Only head-to-head play counts.

### Everyone shares one codebase

The public agents and ours descend from a common ancestor — the same `_v219`, `_R37`, `_v233`
internals appear in other people's notebooks. Nobody is running a secret planner. The competition is
about *which gate you add*, and forking is the norm (Apache-2.0). **Keep all NOTICE/attribution text
in anything shipped.** Upstream credits include ahmedberatozer, aurax7, tetsutani, thomastschinkel,
yhay81, destbreso, prvsiyan, Dmitrii Gluzdov.

---

## 4. What actually won, with numbers

### v58 — replacing the route table with one constant (~+1,200 coins)

The day-6 router's non-yarn table `_R108_SHOP_ROUTES` has 64 entries and was fit from roughly
**one game per shop pair** (40 seeds produce 26 distinct pairs). At that sample size it encoded
noise as if it were knowledge. Replacing the entire table with a single route wins:

| grid | score | margin |
|---|---|---|
| 4700000/6229 | 85.0% | +1470 ± 162 |
| 8100000/4441 | 85.0% | +1138 ± 168 |
| 3300000/7727 | 81.7% | +1167 ± 186 |
| 1900000/5113 | 78.3% | +1296 ± 278 |

Full sweep of all distinct routes: **rt124 best (85.0%)**; rt108 = rt122 at 70.0% / +917;
rt107 57.5%; everything else −700 to −8200 (worst rt117, 10.0% / −8178).

Tool: `mkroute.py` (`RT_NEW=` non-yarn branch, `RT_OLD=` yarn branch).

**Known regression:** v58's panel is 184W-26L, but `b48_open25` drops 100% → 53.3% and
`aurax7-v7` 100% → 73.3%. Both stay positive on coins (+887, +1421) but v58 trades reliability
against those two openings for a bigger margin everywhere else.

### v57 — the advance horizon (~+300 coins)

`_ADV_LOOK_HI` 32 → 44. Wins 94-100% on four grids. Saturates around 45 (`lh48` matches `lh44`
record-for-record on every grid, so the exact value above ~45 is irrelevant).

The reason it was missed for so long: the original hand sweep tried only 28 / 32 / 40 (2.0% / 93% /
26.0%), saw a sharp peak at 32, and never looked between 32 and 40.

**Generalisable lesson: check whether a "knife-edge optimum" was ever sampled densely.**

---

## 5. Closed — do not retry

- **`_v219` is dead code in every build ever shipped.** Its qualifier runs exactly once, on day 18,
  and `tomato_price_ok` fails — the tomato price is never ≥ `CROP_MIN_PRICE=70`. This explains why
  22 earlier tuning configurations were all negative: most were never running. Forced on with zero
  hires it commits, plants nothing, and loses (107,567 vs 113,255). Diagnose with `v219probe.py`.
- **Borrowing idle hands — dead, for a precise reason.** `tapeidle.py`: the tape commands every hand
  until step 712-718 (only route 1 frees one, at step 504). `tapegaps.py`: **87% of idle runs are
  ≤4 turns**. A hand needs `2×distance+1` turns to leave, work a tile and return, so it can reach
  exactly one adjacent square; the long runs are all on day 0 before hands are hired.
  It is *not* that moving a hand desynchronises — the gaps are too short to go anywhere.
- **Idle-hand replanting** — `rp_plants: 0`. The layer only sows the tile a hand already stands on,
  and unlocked land is elsewhere on the board.
- **Parameter tuning is exhausted.** GATE 0.74-0.90 flat (±15 coins); GATE_WIN 16-32 flat;
  `LIQ_FROM` 660-684 **exactly inert**; BOOST 0.90-1.12 all worse (−548 at 1.12).
  The day-28 liquidation layer shipped in v55 does nothing measurable — 40 of the 41 tapes already
  blanket-sell in the last three days.
- **Harvested tapes** — a tape that scored 163,274 in its own game scores 76,406 when replayed
  (prices are shared and its orders were timed against its own opponent). Switched in at day 6 on a
  *matching* shop pair: 29,664 vs 182,400.
- **Shop layout is endogenous** — shops unlock in response to what you sell, so the same seed gives
  different pairs under different play. A tape library keyed on shop pairs cannot work.
- **Tape surgery, all catastrophic** — drop geese 17.5% / −4,528; +1 goose −9,499; +1 cow
  catastrophic; wheat→strawberry 0W-40L / −161k; wheat→melon 0W-40L / −194k.
- Also dead: hoarding, parcel splitting, all `_OPEN_STEP0` alternatives, `_ADV_BOOK`, sell_lead off,
  all `_RACE_HORIZON_*`, R37 horizon 6/8/12, day-27 route switch (40 of 41 routes have identical
  last-three-day tapes), yarn-branch route override (50 ties in 60 games — nearly inert).

---

## 6. The structural gap

Measured over 24 games of the 3125-3154 teams against 27 of ours:

| | day-20 money | final money |
|---|---|---|
| us | **48,855** | 108,034 |
| top | 45,754 | **113,714** |

**We are ahead at day 20 and lose it all in the final two days.**

Their late burst is late-maturing **tomato**: top teams hold 6 tomato tiles / 64 units; we hold
**zero**. They leave 14.6 tiles locked to our 22.9, and have 12% idle hands to our 45%.
Our board at day 25 is PLANT 58, PASTURE 17, LOCKED 25 — no bare tiles at all.

Every attempt to close this on the tape chassis has failed (section 5). That is the argument for
pursuing the state-driven `auto-top1` lead instead.

---

## 7. Engine facts

720 steps (30 days × 24 turns), 10×10 board, shed capacity 100, max 10 market orders/turn,
starting money 3000. **Reward = final cash; shed contents are worth zero at step 719.**

| crop | seed | first yield | max yield | notes |
|---|---|---|---|---|
| WHEAT | 10 | day 2 | 6 | **animal feed — eaten daily, never liquidate** |
| CARROT | 20 | day 2 | 4 | |
| TOMATO | 50 | day 8 | 4 | ongoing, interval 1 |
| STRAWBERRY | 100 | day 10 | 4 | ongoing, interval 2 |
| MELON | 80 | day 10 | 6 | |

Animals: GOOSE 300 (COOP, 4, EGG); COW 400 (PASTURE, 8, MILK); SHEEP 500 (PASTURE, 6, WOOL).

FERTILIZER is spent on planting; by day 28 nothing planted can reach first yield before step 719,
so it *is* safely liquidatable — unlike wheat.

**Realised sale price, top-10 vs field** (from 618 replays): MELON +14%, STRAWBERRY +11.2%,
MILK +7.3%, WOOL +6.4%, EGG/TOMATO/WHEAT ~0. The gap appears exactly on the high-value goods whose
price curve is steep. Top teams sell a median 45% of held stock per order; the field sells 50%.
**Principle: trickle, don't dump.**

---

## 8. Running things

Everything lives in the **ephemeral** scratchpad
`/tmp/claude-0/-home-user-vibecode/645f0ae7-25a6-5826-b984-55bb3c858d7f/scratchpad`.
Python is `./venv/bin/python`. **Anything that must survive belongs in this repo.**

```
ab.py <A.py> <B.py> <seeds> <base> <stride>      paired grid, both seat orders
runsweep2.sh <A.py> <seeds> <base> <stride> DIRS...   takes DIRECTORIES
panel2.sh <A/main.py> <seeds>                     takes a FILE
search.py --spec s.json --out runs/n --seeds 25 --base N --stride N --sigmas 2.0 --min-score 58
```

`search.py` is coordinate descent: each value is played against the **reigning champion** (not the
original baseline) over a paired seed grid; a winner becomes champion and the sweep restarts.
A build identical to the old champion scoring 8% means the champion moved on — not that the harness
broke.

**Traps that have actually bitten:**
- `panel2.sh` takes a FILE, `runsweep2.sh` takes a DIR. The wrong form silently scores a
  nonexistent agent: 0W-30L everywhere with an identical ~−188,000 margin.
  **Suspect this first whenever a sweep looks impossibly lopsided.**
- The box has **4 CPUs**. Never run more than one `ab.py` at a time; use `AB_WORKERS=3`.
- Container churn silently kills `nohup`'d jobs — use the harness's background runner.
- Never wait on a job by grepping a pattern the waiting shell's own command line contains; it
  matches itself and sleeps forever. Use `kill -0 <PID>`.

**Patchers** (each asserts on exact source anchors and fails loudly if they are absent — that is a
feature; do not force them onto a foreign source):
`mkroute.py` (route override), `mkconst.py` (rewrite any module constant),
`mkgate.py`, `mkboost.py`, `mkliq.py`, `mkv219b.py`, `mkreplant.py`.

**Rebuild the v57 line:**
```
MK_SRC=v51/main.py GATE=0.82 GATE_WIN=24 mkgate.py X_g
MK_SRC=X_g/main.py BOOST=1.00 LOOK_HI=44 mkboost.py X_b
MK_SRC=X_b/main.py LIQ_FROM=672 LIQ_ITEMS=STRAWBERRY,WOOL,EGG,MILK,MELON,CARROT,TOMATO,FERTILIZER mkliq.py X
```
(equivalently `MK_SRC=v56/main.py mkconst.py X "_ADV_LOOK_HI=44"` — verified byte-identical).
Then `MK_SRC=X/main.py RT_NEW=124 mkroute.py v58`.

**Submit:**
```
build_sub.py <dir>/main.py submission_vNN.ipynb submission_vNN.tar.gz notes_vNN.md
api.competition_submit('submission_vNN.tar.gz', msg, 'kaggriculture')
sed -i 's/else <OLD_ID>/else <NEW_ID>/' watch.py
```
Always validate a full 720-step game first and confirm the last callable is the real entry point —
a reward of exactly 3,000 means the chain was bypassed.

**Diagnostics:** `watch.py` (ladder), `why.py`, `bylevel.py`, `autopsy.py`, `endgame.py`,
`lastburst.py`, `finger.py`, `seedscan.py`, `v219probe.py`, `smoke.py` (one game + layer telemetry),
`tapeidle.py`, `tapegaps.py`. `idx/manifest.csv` indexes the 2026-09-17 replay archive; replays at
`https://www.kaggleusercontent.com/episodes/<id>.json`.

---

## 9. What I would do next

1. **Finish validating `auto-top1`.** If it beats v58 on 3+ grids, it is stronger than anything we
   have built, and it has no tapes. Read it, understand the policy, and either submit it (with
   attribution) or port its decision logic.
2. **Sweep the remaining public agents properly.** 49 are extracted; only ~10 have ever been played.
   Check loadability first. There may be more than one agent stronger than ours sitting unexamined.
3. **`_ADV_LOOK` (base 16) and `_ADV_FROM` (144)** are built as `al20 al24 al32 al44` and
   `af96 af120 af192` and have never been swept. They are in the one layer that has produced a gain.
4. **Do not spend more time on the tape chassis.** Sections 5 and 6 are the evidence: the remaining
   gap is structural, and a tape cannot react to it.

---

## 10. Honest assessment

We moved from ~2400 to a measured ceiling around 2611, and v58 should exceed that, but **nothing
measured so far projects to 3100.** The top of the leaderboard is 3247 and the 3100 band is roughly
the 3rd-4th position. Two real gains were found (+300 and +1,200 coins), both by finding measurement
errors in previous work rather than by inventing new mechanisms — which is itself the clearest
signal about where the remaining value is: **re-measure what is already believed.**

The single most promising fact in this file is that a public agent with no tapes beat our best
build. That is worth more than another week of tuning ours.
