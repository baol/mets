# P7: the sweep inserts by GENI, and the extra route goes

Plan item (1) of the external researcher: make the sweep the paper's and
rerun Golden 2, 3, 4, 7, 8 (GENI, seed 1, η = 10⁵, default fleet m =
reference + 1), routes counted. Every saved `.sol` recounted by
`scripts/verify_vrp.py` (0 failures).

## What was different

Our sweep's rule for opening routes was already the paper's: a new route
only when the next customer would violate capacity or duration, while
vehicles remain. The difference was the insertion inside the sweep:
plain best insertion, where the paper uses GENI on every problem without
time windows. Duration is what cuts the routes on these instances, and
GENI's routes are shorter, so they hold more customers. Routes the sweep
opens with an unbounded fleet (5 seeds each, the ignored test
`sweep_route_counts_on_golden` in `examples/cvrp.rs`):

| Golden | reference routes | m | plain insertion | GENI, p = 5 |
|---|---|---|---|---|
| 2 | 10 | 11 | 12 ×5 | **10** ×5 |
| 3 | 10 | 11 | 12 ×5 | **10** ×5 |
| 4 | 10 | 11 | 11 ×5 | **10** ×5 |
| 7 | 8 | 9 | 10–11 | 9 ×5 |
| 8 | 10 | 11 | 12 ×5 | 11 ×5 |

So with plain insertion the sweep needs more routes than m and fills all
of them (the last one overloaded). The search then never empties one: the
Golden 3 watch before the change spent 105 phases with no fewer than 11
routes in use in any iteration. Since `cc45067`, `--insertion geni` also
makes the sweep insert by GENI.

## Rerun (build `cc45067`, `--insertion geni --phase-log`)

| Golden | paper % (rebased) | GENI, old sweep % | routes | **GENI sweep %** | routes | phases with 11 routes at some point |
|---|---|---|---|---|---|---|
| 2 | 0.73 | 0.80 | 10 | **0.80** | 10 | 19 of 212 |
| 3 | 0.59 | 2.57 | 11 | **0.88** | 10 | 15 of 175 |
| 4 | 0.79 | 2.56 | 11 | **1.46** | 10 | 0 of 179 |
| 7 | 1.51 | 1.74 | 9 | **1.66** | 9 | (9 = m throughout) |
| 8 | 1.80 | 2.16 | 11 | **2.24** | 11 | (11 = m throughout) |
| **mean** | **1.08** | 1.97 | | **1.40** | | |

To the paper's own reference: 1.14% (Golden 2: 0.47, 3: 0.53, 4: 1.42,
7: 1.12, 8: 2.16).

## Reading

* **The extra route is gone on Golden 2, 3 and 4**, and Golden 3 and 4
  drop by 1.7 and 1.1 points. On these five the deficit to the paper
  falls from 0.89 to 0.32 points. Golden 3 is now 0.3 above the paper's
  rebased mean, Golden 4 0.7, with one seed each.
* The search can open and then empty the spare route: on Golden 2 and 3
  the eleventh route is in use during 15–19 phases and empty again
  afterwards. Emptying works when the route is small. What failed before
  was a start that filled every route with about 36–40 customers.
* Golden 7 and 8 keep reference + 1 routes because the GENI sweep needs
  9 and 11, which is all of m here. The paper's m there is reference + 2
  (10 and 12), so `FLEET=paper` gives these two a spare as well. Their
  gaps (1.66%, 2.24%) are near the paper's (1.51%, 1.80% rebased).
* Expected effect on the set: the four extra-route instances were 0.22
  of the 0.31-point deficit at 10⁵. With Golden 3 and 4 fixed the set
  mean should land near 1.6%, against 1.50% rebased. By the plan, the
  Golden acceptance follows: `FLEET=paper`, 10 seeds, 10⁵, both gaps.

## `--reinsert geni` (stopped when the plan reordered)

Old sweep, GENI insertion, GENI cluster reinsertion, seed 1, build
`1f71ec1`. Only two of the eight runs finished before the queue was
stopped: Golden 1 0.94% and Golden 2 0.75%, against GENI alone 0.74% and
0.80%. Golden 3, 4, 9 and 10 were stopped mid-run and have no result. (A
first version of this report gave numbers for those four: they were
wrong, written without checking that the runs had finished, and are
withdrawn.) Two runs say nothing either way. `--reinsert geni` stays an
unmeasured flag, and with the GENI sweep it is no longer a suspect for
Golden 3 and 4.
