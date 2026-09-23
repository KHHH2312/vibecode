# v59 -- v57 plus the yarn-branch route fix

Built on v57 (= lh44), the build the LADDER endorses: 2715 Elo at 71W-20L, versus v55's
2417 converged. v58's route change is still unproven on the ladder (2558 and climbing),
so this deliberately does NOT carry it.

The day-6 router has two independent branches:

    state['route'] = _R108_SHOP_ROUTES.get(shops,100) if use_new else _R110_OLD_SHOPS.get(shops,0)

They are mutually exclusive, so each can be measured and changed on its own. v59 replaces
only the YARN branch with route 7 and leaves the non-yarn table untouched.

routesplit.py labels each finished game by the shop pair it actually produced (recovered
from env.steps), so one A/B run yields per-pair results. Yarn layouts are ~22% of games;
the rest tie because the override cannot reach them, so the decisive subset is what counts:

    vs v58 (route-124 base)   grid 4700000/6229   22W-4L    85%   +98
                              grid 8100000/4441   13W-9L    59%   +35
                              grid 3300000/7727   17W-3L    85%  +253
                              grid 1900000/5113   13W-7L    65%   +54
                              pooled              65W-23L   73.9%       (~4.5 sigma)

    vs lh44 (table base)      grid 4700000/6229   14W-4L    77.8%  +45

The last line is the check that the gain is independent of the other branch, as the
mutually-exclusive structure predicts. It is.

Route 7 was chosen over routes 0 and 2, which are coin flips on the same subset (16W-14L
each) and carry craters: YARN_STORE/BAKERY at -5538 for route 0 and -5454 for route 2.
Route 7's worst pair is -286. Uniformly positive with no disasters is the same signature
route 124 showed on the other branch.

Honest magnitude: +35 to +253 coins overall, far smaller than the horizon fix (+300 coins,
worth +272 Elo on the ladder). Expect a modest gain, not a step change.

Unchanged from v57: _ADV_GATE 0.82, _ADV_BOOST 1.00, _ADV_LOOK_HI 44, _LIQ_FROM 672 with
WHEAT excluded and FERTILIZER included, and the full non-yarn shop->route table.
