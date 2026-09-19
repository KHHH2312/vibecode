# Kaggriculture — handoff

Written overnight 2026-09-19. Competition ends ~2026-09-28. Goal: 3100+ Elo.
Kaggle user `khalid000000`; credentials at `~/.kaggle/kaggle.json` — never transmit them anywhere.

**Read section 3 (Methodology) before trusting any number in this file or producing a new one.**
Most of the wasted effort in this project came from believing measurements that did not predict the
ladder. Section 3 records which measurements were tried, how each one failed, and — in 3.0 — why the
ladder's own ratings are not directly comparable either. Start at 3.0.

---

## 1. Where things stand

| build | submission | change from parent | ladder |
|---|---|---|---|
| v55 | 56334083 | gate 0.82 + LOOK_HI=36 + day-28 sweep | 2417 — 111 games, 93.7% W |
| v57 | 56343311 | v55 + `_ADV_LOOK_HI=44` | **2716** — 93 games, 77.4% W (highest rating) |
| v58 | 56347520 | v57 + non-yarn route table → constant 124 | 2579 — 70 games, 74.3% W |
| v59 | 56351569 | v57 + yarn branch → route 7 | 2424 — 59 games, 81.4% W |
| v60 | 56353015 | v57 + `_ADV_SUBTRACT_DEBTS=True` | 2546 — 39 games, 84.6% W |
| v61 | 56356463 | **CONTROL — byte-identical to v57** | submitted 11:00Z 19 Sep |

**v61 is the most useful submission of the day.** It is the same file as v57 (hash-verified), so the
gap it ends up at from v57 *is the noise floor* for comparing any two of our submissions. Read it
first: if v57 and v61 land within ~50 points, the differences in this table mean something; if they
land 200+ apart, none of tonight's ladder verdicts do, and candidates must be judged another way.
No control like this had ever been run, which is why the noise floor was unknown and the verdicts
were stated with more confidence than they deserved.

**Do not read that column as a ranking.** All five are still winning 74-94% of their games, so none
has converged, and the ratings differ largely by submission age under a decayed K-factor. See 3.0.

**v57 holds the highest rating**, and the leaderboard takes your best submission, so keep it. But it
is *not* established as the strongest build — see 3.0 for why these ratings cannot be compared
across submissions of different ages.

**Submissions: 5 per day, UTC reset 00:00Z.** All five for 19 Sep are spent (v57, v58, v59, v60,
v61); the next five arrive at 00:00Z 20 Sep. Each restarts at 600 Elo, climbs fast for ~40 games, then crawls once K decays.

**The ladder is not rate-limiting** — it serves ~16 games/hour, and v57 climbed 600 → 2550 in two
hours. But **a submission that flattens has not hit a strength ceiling**; it has run out of
K-factor while still winning 74-94% of its games (3.0). Waiting genuinely does not help, but for
the opposite reason to the one I first gave: the rating is stuck, not the agent.

### The prediction I made, and how it went

Before their results were known I predicted v60 (market-behaviour change) would beat v57 and v59
(tape-selection change) would not. v59 landed below v57, v60 also landed below v57. So the
prediction was half right, and section 3.0 explains why neither half is worth much: the comparison
was never sound.

---

## 2. Architecture — what our agent actually is

It is **not** a heuristic if/else bot. It is a **tape replayer**:

- `_ROUTES` holds 41 pre-recorded 719-step action tapes — only **38 distinct**
  (`101==119`, `105==125`, `109==127`).
- A router picks one on **day 6 (step 144)** from the first two shops that unlock, via two
  hand-built lookup tables, then hard-switches to route 2 at step 648.
- Nine reactive layers wrap the tape: weed repair, safety, market reordering, advance-sell, a crash
  gate, a spike boost, a day-28 liquidation sweep, and closure planning.

```python
state['route'] = _R108_SHOP_ROUTES.get(shops,100) if use_new else _R110_OLD_SHOPS.get(shops,0)
#                ^ non-yarn layouts                                ^ layouts containing YARN_STORE
```

The two branches are **mutually exclusive**, so each can be measured and changed independently —
verified empirically, not just by reading (section 5).

**The central constraint: a tape cannot react.** Every structural idea that failed did so because
of this. You cannot change what a tape does without desynchronising every later step that depends
on the farm state the earlier steps built.

`kaggle_environments` runs **the last callable in the module** — ours is `_y_agent_shopherd`. A
wrapper appended after the chain bypasses all nine layers and scores exactly the 3,000 it started
with. This has cost a real submission before.

---

## 3. Methodology — read this first

### 3.0 READ THIS BEFORE 3.1 — ladder ratings are not comparable across submissions

Discovered late (10:38Z) and it undermines the verdicts below. Snapshot:

```
            games   win rate   Elo    drift
v55  111     93.7%   2417     +2.2
v57   93     77.4%   2716     +2.8
v58   70     74.3%   2579     +1.3
v59   59     81.4%   2424     +1.7
v60   39     84.6%   2546     +2.2
```

**Every submission is winning 74-94% of its games.** A rating is at equilibrium when the win rate is
50%, so **none of these has converged** and every rating is a *lower bound* on true strength.

They stop moving because of **K-factor decay, not a strength ceiling.** v60 drifted +82/game at 22
games and +2.2/game at 39. Once K decays, even a 93.7% win rate moves v55 by only +2.2/game. **The
final rating is therefore set largely by how the first ~40 games happened to go**, which is high
variance — and `watch.py` will print "converged" long before the agent is actually at equilibrium.

**What this invalidates (stated plainly, because these claims appear elsewhere in this file and in
the commit history):**
- "v58 is a −137 Elo regression", "v59 is −330", "v60 failed" — **not established.** Those gaps are
  consistent with early-game luck under a decayed K, and all three were still drifting upward.
- "v57's 2716 is this family's ceiling" — **wrong.** That is where K ran out, not where strength ran
  out.
- "The ladder confirmed v57 > v55, so self-play beats the external panel" — **weakened.** v55 wins
  93.7% of its games to v57's 77.4%; its lower rating may simply be a worse early run.

**How to compare builds honestly from here:**
- Win rate is age-independent but confounded by opponent strength (a lower-rated agent draws weaker
  opponents), so it is not a clean ranking either.
- The only sound comparison is **same-age**: compare ratings at equal game counts, or submit
  candidates close together and compare their trajectories over the same window.
- Treat any single-submission Elo difference under ~300 points at unequal game counts as noise.

### 3.0b The same-age comparison — the gaps mostly vanish

Trajectories recorded live on 19 Sep (rating at game count). This is the sound comparison 3.0 asks
for, and it was reconstructible from data already collected:

```
games:        ~12      ~26      ~40      ~55      ~70      ~93
v57          —       2464     ~2580    2611     2660     2716
v58         1811     ~2400     2526    2558     2572     —
v59         1700ish  2217      2386    2424     —        —
v60         —        2435     2546     —        —        —
```

**At ~40 games: v57 ~2580, v60 2546, v58 2526 — a spread of ~55 points.** At final observation the
same three read 2716 / 2546 / 2579, a spread of ~170. Most of the apparent difference is therefore
**age, not strength**, exactly as 3.0 predicts.

v59 is the one that still looks genuinely weaker: ~195 below v57 at equal game count, and behind at
every point on the curve.

**Method to reuse:** log `(games, elo)` for every submission, not just the latest rating, and
compare candidates at equal game counts. `watch.py` already prints both; nothing new is needed
except recording it. Had this been done from the start, none of tonight's reversed verdicts would
have been stated.

### 3.1 The class rule — PROPOSED, NOT ESTABLISHED (see 3.0)

The evidence below is real self-play data, but the *ladder* half of it rests on the unsound
comparisons described in 3.0. Keep the rule as a hypothesis worth testing properly, not a finding.

> **Self-play validates changes to HOW WE SELL. It does not validate changes to WHICH TAPE WE RUN.**

| change | class | self-play said | ladder said |
|---|---|---|---|
| `_ADV_LOOK_HI` 32→44 | market behaviour | 94-100% on 4 grids, +300 coins | **+272 Elo — confirmed** |
| route table → constant 124 (v58) | tape selection | 66W-2L over 30 layouts, +1459 coins | **−137 Elo — regression** |
| yarn branch → route 7 (v59) | tape selection | 73.9% of decisive games, 4 grids | **−330 Elo — regression** |

The v59 prediction was made in advance, written into this file before its result was known. It
appeared to come true — but per 3.0 those ladder numbers compare submissions of different ages under
a decayed K, so the apparent confirmation does not carry the weight I first gave it. **v60, a
market-behaviour change that the rule predicted would beat v57, also landed below it** (2546 vs
2716) — which the rule does not explain and which 3.0 does.

Net: the rule is unproven in both directions. Testing it properly means submitting a
tape-selection and a market-behaviour candidate **at the same time** and comparing their
trajectories over the same window.

**Why.** Shop layout is *endogenous*: shops unlock in response to what gets sold. In self-play both
sides run the same tape, co-adapt, and generate exactly the layouts that tape expects. A per-pair
analysis cannot detect this, because every layout it measures was itself produced by two agents
running the change. Against a ladder opponent the layouts evolve differently and the tape is wrong.
Selling behaviour has no such coupling to the opponent.

**Consequence:** treat any route/tape-selection change as **unvalidatable offline**. It costs a slot
and ~5 hours to test on the ladder. Market-behaviour changes can be trusted from self-play.

### 3.2 Single-grid results lie — demonstrated four times

1. **rt115** measured 62.0% / +1286 on the tuning grid; 44% / 54% / 44% on three others. Noise.
2. **"LOOK_HI=36 is the peak"** was recorded everywhere as established fact. A measurement fault;
   44 beats 36 on four grids.
3. **Ranking against a common weaker baseline is invalid.** Comparing 35/36/37/38 each against v55
   and picking the best cannot rank them against *each other*. Head-to-head gave a different order.
4. **"auto-top1 beats v58"** — 60% on one grid, then 58% / 24% / 42% on three others. It does not.

**Rule: never submit on one grid. Require 3+ independent grids.**
Grids in use: `4700000/6229`, `8100000/4441`, `3300000/7727`, `1900000/5113`, `1200000/9011`.

### 3.3 Judge magnitude, not just win rate

`af192` scores **80% by a +2 coin margin** in a 130,000-coin game — a rounding artifact, not an
edge. The one ladder-confirmed gain was +300 coins for +272 Elo. Anything two orders of magnitude
below that will not move the ladder whatever its win rate.

### 3.4 A non-loading agent looks like a crushing win

**Only 23 of 49 extracted public agents load.** A broken one produces **0W-30L with mean ≈ −190,627**
— the signature of *nothing running*. Always check first:

```python
src = open(path + "/main.py").read(); ns = {}
exec(compile(src, path, "exec"), ns)              # raises if the agent is broken
fn = [v for v in ns.values() if callable(v)][-1]  # the engine uses the LAST callable
```

Known broken: `saitejabandaruin-…-3000`, `pilkwang-…-policy` (missing imports);
`xuanzhang001-pipe7-public-top1` (crashes). `indarkarhana-…-2948-9` loads but is weak (−33,652).

### 3.5 A weak-opponent panel does not predict the ladder either

`extpanel.sh` scores a build against 8 non-lineage agents. It ranked **v55 best (87.1%)** — and v55
is the ladder's **worst** performer at 2417 converged, while lh44/v57 ranked near-last on the panel
and leads the ladder at 2716. We beat every panel member 70-97%, so they all sit far below us;
beating a weak agent by a wider margin does not predict beating a peer, and the ladder only pairs
you with peers. The panel is still useful for catching catastrophic regressions, not for ranking.

**Also deduplicate it.** `auto-top1` and `aurax7-v7` return identical records for every build
(29W-1L/29W-1L for v55; 21W-9L/21W-9L for v58) — near-duplicates that double-count one opponent.
Other exact duplicates: `ahmedberatozer-v45 = aurax7-v6 = reyhanksatria` (+1193);
`anhadmahajan06 = foysalemonshanto` (+1739).

### 3.6 Before trusting any evaluation method, test it against a ladder result you already have

I asserted mid-session that the panel was ground truth and self-play was broken. It was the
reverse, and the check that showed it (v55 vs v57, already resolved on the ladder) was available
the whole time.

### 3.7 Notebook titles are not evidence

`ultimate-mega-ensemble-3000`, `auto-top1`, `verified-route-replay-2948-9` are self-chosen
filenames. `pilkwang`'s "structured economic policy" is a thin wrapper over Ahmed Berat Ozer's V36
whose own config says `"strength_status": "UNVERIFIED"`, `"submission_allowed": false`.

### 3.8 Everyone shares one codebase

The public agents and ours descend from a common ancestor — the same `_v219`, `_R37`, `_v233`
internals appear in other people's notebooks. `auto-top1` embeds a `_PARENT_SRC` bytes literal
containing V43 with `_ROUTES`, `_R108_SHOP_ROUTES` and 81 tape entries: the same chassis, forked
earlier, without our advance-sell layers. **Nobody has a secret architecture.** Forking is the norm
and Apache-2.0 licensed — **keep all NOTICE/attribution text in anything shipped** (upstream:
ahmedberatozer, aurax7, tetsutani, thomastschinkel, yhay81, destbreso, prvsiyan, Dmitrii Gluzdov).

*(A plain text search of `auto-top1` reports zero tapes because they sit inside a `b'...'` literal.
Decode embedded source with `ast` before concluding anything about it — I got this wrong first.)*

---

## 4. What actually won

### v57 — the advance horizon, `_ADV_LOOK_HI` 32 → 44 (+272 Elo, ladder-confirmed)

The only change in this project confirmed by the ladder itself. Wins 94-100% on four grids,
~+300 coins. Saturates around 45 (`lh48` matches `lh44` record-for-record on every grid).

**Why it was missed for so long:** the original hand sweep tried only 28 / 32 / 40
(2.0% / 93% / 26.0%), saw a sharp peak at 32, and never looked between 32 and 40.

> **Generalisable: check whether a "knife-edge optimum" was ever sampled densely.** Both real gains
> this project came from re-measuring something already believed, not from new mechanisms.

### v58 — the route table (looked like the biggest win; was a −144 Elo regression)

Recorded because the evidence was *excellent* and still wrong, which is the lesson.

The non-yarn table has 64 entries fit from roughly one game per shop pair (40 seeds → 26 distinct
pairs), so it encoded noise. Replacing it with constant route 124 won:

```
4700000/6229  85.0% +1470     8100000/4441  85.0% +1138
3300000/7727  81.7% +1167     1900000/5113  78.3% +1296
per-pair (routesplit.py, 90 games, 31 layouts): 66W-2L-22T, mean +1459
  - beats the table's pick on 30 of 31 layouts, margins +958 to +6061
  - the 22 ties are all YARN pairs, which the override does not touch (a correctness check)
  - only losing pair: FARMERS_MARKET/SMOOTHIE_SHOP, 0W-2L, −93 coins
```

Four grids, a per-pair breakdown, and a built-in correctness check — and the ladder still said
−144 Elo, for the endogenous-shop reason in section 3.1.

---

## 5. Closed — do not retry

**Router**
- The **yarn branch is not uniformly wrong** the way the non-yarn one was. Forcing route 0 there is
  a wash (16W-14L-90T, +86) with `ICE_CREAM_SHOP/YARN_STORE` +4108 but `YARN_STORE/BAKERY` −5538.
  Route 7 is the best of the constants tried (73.9% of decisive games over four grids, worst pair
  −286 vs −5500 craters for routes 0 and 2) and is what v59 tests — but it is a tape-selection
  change, so treat its offline numbers with section 3.1 in mind.
- The two branches are independent: a yarn result measured on the route-124 base reproduces on the
  table base (14W-4L vs 22W-4L), as the mutually-exclusive structure predicts.
- Day-27 route switch: 40 of 41 routes have identical last-three-day tapes. Inert.

**Advance-sell layer — now fully explored**
- `_ADV_LOOK` 20/24/32/44 → 62%/44%/44%/42%, margins **+8 to +18 coins**. Closed.
- `_ADV_FROM` 96 and 120 inert (48 ties of 50); 192 scores 80% by **+2 coins**. Closed.
- `_ADV_TO` is already maximal (718).
- `_ADV_GATE_ITEMS` is **already the correct set**. The widened horizon (`_adv_far`) only fires for
  goods in this tuple. Adding EGG (54%), EGG+CARROT+TOMATO (54%) or CARROT+TOMATO (50%) is inert,
  +0 ± 2 coins, 36-48 ties of 50. Retested deliberately because the original dismissal predated the
  horizon win — it still holds, and the replay data says why: the top teams' price edge is +14%
  MELON, +11.2% STRAWBERRY, +7.3% MILK, +6.4% WOOL and **~0% on EGG/TOMATO/WHEAT**. Selling a
  flat-curve good earlier gains nothing. The tuple already holds exactly the four steep-curve goods.
- `_ADV_PROTECT=True` → 4.0% / −325.
- `_ADV_GATE` 0.74-0.90 flat (±15 coins); `GATE_WIN` 16-32 flat; `BOOST` 0.90-1.12 all worse.
- `LIQ_FROM` 660-684 **exactly inert** — the day-28 liquidation layer shipped in v55 does nothing,
  because 40 of the 41 tapes already blanket-sell in the last three days.

**Parcel splitting — actively harmful, monotonically**
Caps of 30/45/60/75% of stock on the steep-curve goods score 0% / 14% / 32% / 36%
(−537 / −348 / −199 / −183). The trend points at *no cap* being optimal.
*Mechanism:* reward is final cash and **shed contents are worth zero at step 719**, while
`_ADV_LOOK_HI=44` already spreads sales over 44 turns. We were already trickling; a cap only delays
sales and risks ending holding worthless stock. Under-selling is punished absolutely; over-selling
only costs price. The replay statistic (top teams sell a median 45% of stock per order vs the
field's 50%) is real but does not transfer to an agent that already has a wide sell horizon.

**Land, labour and tapes**
- **`_v219` is dead code in every build ever shipped.** Its qualifier runs exactly once, on day 18,
  and `tomato_price_ok` fails — the tomato price is never ≥ `CROP_MIN_PRICE=70`. This explains why
  22 earlier tuning configurations were all negative: most were never running. Forced on with zero
  hires it commits, buys the land, plants nothing, and loses (107,567 vs 113,255). Diagnose with
  `v219probe.py`.
- **Borrowing idle hands — dead, for a precise reason.** `tapeidle.py`: the tape commands every hand
  until step 712-718 (only route 1 frees one, at step 504). `tapegaps.py`: **87% of idle runs are
  ≤4 turns**. A hand needs `2×distance+1` turns to leave, work a tile and return, so it reaches
  exactly one adjacent square; the long runs are all on day 0 before hands are hired. It is *not*
  that moving a hand desynchronises — the gaps are too short to go anywhere.
- **Idle-hand replanting** — `rp_plants: 0`. The layer only sows the tile a hand already stands on,
  and unlocked land is elsewhere on the board.
- **Harvested tapes** — a tape that scored 163,274 in its own game scores 76,406 replayed (prices
  are shared and its orders were timed against its own opponent). Switched in at day 6 on a
  *matching* shop pair: 29,664 vs 182,400.
- **Tape surgery, all catastrophic** — drop geese 17.5% / −4,528; +1 goose −9,499; +1 cow
  catastrophic; wheat→strawberry 0W-40L / −161k; wheat→melon 0W-40L / −194k.
- Also dead: hoarding, all `_OPEN_STEP0` alternatives, `_ADV_BOOK`, sell_lead off, all
  `_RACE_HORIZON_*`, R37 horizon 6/8/12.

---

## 6. The structural gap

24 games of the 3125-3154 teams against 27 of ours:

| | day-20 money | final money |
|---|---|---|
| us | **48,855** | 108,034 |
| top | 45,754 | **113,714** |

**We are ahead at day 20 and lose it all in the final two days.** Their late burst is late-maturing
**tomato**: top teams hold 6 tomato tiles / 64 units, we hold **zero**. They leave 14.6 tiles locked
to our 22.9, and have 12% idle hands to our 45%. Our board at day 25 is PLANT 58, PASTURE 17,
LOCKED 25 — no bare tiles at all.

Every attempt to close this on the tape chassis has failed (section 5), each for a specific and
now-understood reason.

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

**Realised sale price, top-10 vs field** (618 replays): MELON +14%, STRAWBERRY +11.2%, MILK +7.3%,
WOOL +6.4%, EGG/TOMATO/WHEAT ~0. The edge appears only on goods whose price curve is steep — which
is why `_ADV_GATE_ITEMS` contains exactly those four.

---

## 8. Running things

Everything lives in the **ephemeral** scratchpad
`/tmp/claude-0/-home-user-vibecode/645f0ae7-25a6-5826-b984-55bb3c858d7f/scratchpad`.
Python is `./venv/bin/python`. **Anything that must survive belongs in this repo.**

```
ab.py <A.py> <B.py> <seeds> <base> <stride>         paired grid, both seat orders
runsweep2.sh <A.py> <seeds> <base> <stride> DIRS…   takes DIRECTORIES
panel2.sh <A/main.py> <seeds>                       takes a FILE — our lineage
extpanel.sh <A/main.py> <seeds>                     takes a FILE — 8 external agents
routesplit.py <A.py> <B.py> <seeds> <base> <stride> A/B split by day-6 shop pair
search.py --spec s.json --out runs/n --seeds 25 --sigmas 2.0 --min-score 58
```

`routesplit.py` recovers the day-6 shop pair from a *finished* game via `env.steps`, so one ordinary
A/B run yields per-pair results at no extra cost. Per-pair evidence is otherwise unaffordable,
since layouts are endogenous and a pair gets about one game per 40 seeds.

`search.py` is coordinate descent: each value plays the **reigning champion** (not the original
baseline) over a paired grid; a winner becomes champion and the sweep restarts. A build identical to
the old champion scoring 8% means the champion moved on — not that the harness broke.

**Traps that have actually bitten:**
- `panel2.sh` takes a FILE, `runsweep2.sh` takes a DIR. The wrong form silently scores a nonexistent
  agent: 0W-30L everywhere with an identical ~−188,000 margin.
  **Suspect this first whenever a sweep looks impossibly lopsided.**
- The box has **4 CPUs**. Never run more than one `ab.py` at a time; use `AB_WORKERS=3`.
- Container churn silently kills `nohup`'d jobs — use the harness's background runner.
- Never wait on a job by grepping a pattern the waiting shell's own command line contains; it
  matches itself. **The same applies to `pkill -f`** — a `pkill -9 -f "ab.py"` in a command that
  also mentions `ab.py` later kills its own shell. Both happened here.

**Patchers** (each asserts on exact source anchors and fails loudly if absent — that is a feature;
do not force them onto a foreign source):
`mkroute.py` (RT_NEW / RT_OLD), `mkconst.py` (rewrite any module constant), `mkgate.py`,
`mkboost.py`, `mkliq.py`, `mklot.py`, `mkv219b.py`, `mkreplant.py`.

**Rebuild v57:**
```
MK_SRC=v51/main.py GATE=0.82 GATE_WIN=24 mkgate.py X_g
MK_SRC=X_g/main.py BOOST=1.00 LOOK_HI=44 mkboost.py X_b
MK_SRC=X_b/main.py LIQ_FROM=672 LIQ_ITEMS=STRAWBERRY,WOOL,EGG,MILK,MELON,CARROT,TOMATO,FERTILIZER mkliq.py X
```
(equivalently `MK_SRC=v56/main.py mkconst.py X "_ADV_LOOK_HI=44"` — verified byte-identical).

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

1. **Fix the measurement before doing anything else.** Every verdict in this project rests on
   comparisons that turned out to be unsound (3.0). Concretely: check how the five live submissions
   compare *at equal game counts* by recording rating-vs-games rather than rating alone, and make
   `watch.py` stop printing "converged" while the win rate is far from 50%. Without this you cannot
   tell a real gain from a lucky first 40 games.
2. **Exploit K-decay rather than fighting it.** Since the final rating is set largely in the first
   ~40 games, and the ladder gives 5 submissions a day, **the same build submitted several times
   will land on several different ratings.** That is worth knowing before concluding a change
   helped: submit a *duplicate of the current best* alongside any candidate, as a control. Nobody
   has done this here, and it would have prevented most of tonight's wrong conclusions cheaply.
3. **Market-behaviour levers are exhausted** — every constant, both flags, both bounds and the item
   set in the advance-sell layer are now swept (section 5). Further tuning of this chassis is very
   unlikely to pay.
4. **The remaining gap is structural** (section 6) and needs a mechanism a tape cannot express.
   The honest options are: accept ~2700, or move to an agent that decides from game state rather
   than replaying a tape. The second is a rewrite, and nine days is tight but not absurd — note that
   `auto-top1`, our nearest public rival, is the *same* chassis, so the field has not solved this
   either.
5. **Offline evaluation remains unsolved** (3.5). A panel of much weaker agents predicts nothing,
   and the only opponents at our strength are our own builds, which is the self-play trap. This is
   the deepest unsolved problem here and everything else depends on it.

---

## 10. Honest assessment

The highest rating we hold is **2716 (v57)**. The leaderboard top is 3247 and the 3100 band is
roughly 3rd-4th place. **Nothing measured projects to 3100**, and — importantly — nothing measured
reliably projects anything at all, for the reasons in 3.0.

Two candidate gains were found, both by **re-measuring something already believed** rather than
inventing a mechanism: the sell horizon (a "knife-edge optimum" that had never been sampled between
32 and 40) and the route table (64 entries fit on roughly one game per cell). The first is the only
change with any ladder support; the second looked far stronger offline and appears to have hurt,
though 3.0 means even that is not firmly established.

**The real output of this session is negative, and it is worth more than a build.** Three
evaluation methods were tried and none of them is trustworthy as used:

- *self-play* — opponents at our strength, but they share our tape chassis, so changes that exploit
  the shared behaviour score as gains;
- *a panel of public agents* — genuinely foreign, but we beat all of them 70-97%, so it cannot rank
  us and in fact ranked our weakest build first;
- *the ladder itself* — the only ground truth, but its ratings are not comparable across
  submissions of different ages, because K decays before an agent reaches equilibrium.

Each of those was discovered by checking a claim against evidence that already existed. That is
cheap and was not being done. Anyone continuing this should assume the same is true of whatever they
currently believe, including everything written above.

A note on this file's history: several sections here were written confidently and later reversed
within the same session — the route-table verdict twice, and the "v58 is a regression" claim. The
reversals are kept visible rather than tidied away, because the pattern (a strong offline result,
confidently shipped, then undone by a measurement nobody had run) is the most reliable thing this
project has taught.
