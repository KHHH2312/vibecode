# v61 -- a CONTROL. Byte-identical to v57.

This is not a new agent. It is the same file as v57 (56343311), submitted again on purpose.

Why: ladder ratings turned out not to be comparable across submissions. Every live submission is
still winning 74-94% of its games, so none has reached equilibrium; they stop moving because the
K-factor decays after roughly 40 games. The final rating is therefore set largely by how the first
~40 games happen to go, which is high variance.

That means a 100-300 point gap between two of our builds may be nothing but early-game luck, and
several conclusions drawn tonight from exactly such gaps are unsafe:

    v57  2716   93 games   77.4% W      v58  2579   70 games   74.3% W
    v59  2424   59 games   81.4% W      v60  2546   39 games   84.6% W

A duplicate of the current best measures that variance directly. v61 and v57 are the same agent, so
whatever they end up apart by is the noise floor for comparing any two submissions. If they land
within ~50 points, the differences above are probably real; if they land 200+ apart, then none of
tonight's ladder verdicts mean anything and future candidates must be judged some other way.

No control like this had been run in the project, which is why the noise floor was unknown and the
verdicts were stated with more confidence than they deserved.

Identical to v57: _ADV_GATE 0.82, _ADV_BOOST 1.00, _ADV_LOOK_HI 44, _ADV_SUBTRACT_DEBTS False,
_LIQ_FROM 672 with WHEAT excluded and FERTILIZER included, both shop->route tables unmodified.
