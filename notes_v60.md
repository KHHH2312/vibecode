# v60 -- v57 with _ADV_SUBTRACT_DEBTS=True

Built on v57 (= lh44), the build the ladder endorses at 2716. It does NOT carry v58's route
change, which the ladder scored at 2572 -- a ~144 Elo regression.

One flag moves. The advance-sell layer plans future SELL orders and pulls them forward; the
sale-reservation layer separately books units against future turns. With the flag off, the
advance layer ignores those bookings and can plan to sell stock that is already committed.
With it on it subtracts them:

    if _ADV_SUBTRACT_DEBTS: q -= debts.get(t,{}).get(o[1],0)

Measured against v57, paired seed grids, both seat orders:

    grid 4700000/6229   27W-13L   64%   -14 +/- 16
    grid 8100000/4441   37W-7L    80%   +22 +/- 95
    grid 3300000/7727   30W-8L    72%   +24 +/- 27
    grid 1900000/5113   38W-8L    80%   +20 +/- 35
    pooled decisive     132W-36L  78.6%

Why this is expected to transfer where v58 did not: it changes HOW WE SELL, not WHICH TAPE WE
RUN. Tape selection interacts with the opponent through the endogenous shop mechanism -- shops
unlock in response to what gets sold, so in self-play both sides co-adapt and generate the very
layouts the chosen tape expects. Route 124 won 66W-2L across 30 layouts in self-play and lost 144
Elo on the ladder. Selling behaviour has no such coupling: _ADV_LOOK_HI 32->44 was validated the
same way and the ladder confirmed +272 Elo.

Honest expectation: the coin margin is about +20, an order of magnitude below the horizon fix.
This should be a small gain, not a step change.

Unchanged from v57: _ADV_GATE 0.82, _ADV_BOOST 1.00, _ADV_LOOK_HI 44, _ADV_GATE_ITEMS the four
steep-curve goods, _LIQ_FROM 672 with WHEAT excluded and FERTILIZER included, and the full
shop->route tables on both router branches.
