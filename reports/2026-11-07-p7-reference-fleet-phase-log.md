# P7: Golden 7 and 8 at the reference fleet, phase by phase

The plan asked why Golden 7 and 8 reach no feasible solution with the
reference fleet (no spare) in 10⁵ iterations: log β and the duration
violation per phase. `cvrp --phase-log` (6f9f5cf) prints one line per
phase: feasible iterations, the smallest load and duration excess seen,
the excesses at the phase end, α, β, the largest β, distance, best.
Seed 1, η = 10⁵, `--no-spare`.

| run | vehicles × limit | total length allowed | best known | phases | feasible iterations | smallest duration excess | distance at phase ends |
|---|---|---|---|---|---|---|---|
| Golden 7, plain | 8 × 1300 | 10,400 | 10,102.7 | 214 | **0** | 98.2 | 10,500–10,800 |
| Golden 7, GENI | 8 × 1300 | 10,400 | 10,102.7 | 183 | **0** | 47.3 | 10,560–10,810 |
| Golden 8, plain | 10 × 1200 | 12,000 | 11,635.3 | 195 | **0** | 14.3 | 12,030–12,250 |

(Golden has no service times, so a route's duration is its length.)

* **β is at its clamp (10⁹) at the end of every phase of all three runs**,
  from the first phase on. α reaches the clamp too whenever load is
  violated. That is the self-adjusting rule doing what it says: β is
  multiplied by 1 + δ after every infeasible iteration and is never
  divided, because no iteration is feasible. The paper's rule has no
  clamp. Ours stops at 10⁹ only to keep the costs finite, and it makes
  no difference here.
* With both weights pinned, the search minimises the excess and
  distance hardly counts. The excess still falls over the run (Golden 7
  plain: 482 after phase 1, 100 by phase 61; Golden 8: 276, then 23 in
  the last phases). So the search works toward feasibility, but too slowly
  for 10⁵ iterations.
* **The fleet is tight, as the head of research found.** Feasibility
  needs total length within 2.9% (Golden 7) or 3.1% (Golden 8) of the
  best known, while the routes are still being found. Our one-seed gaps
  with a spare vehicle are 2.55% and 2.85% at 10⁵, so the target sits at
  the edge of what the method reaches in that budget. None of the
  plan's suspects shows: the weight is not too small (it is at the
  clamp), refinement only shortens routes, and the cluster reinsertion
  is by distance as in the paper. GENI halves the smallest excess on
  Golden 7 (98 → 47) but does not reach feasibility.
* This agrees with the correction in `PLAN-problems.md`: the paper's m
  on Golden 7 and 8 is the reference + 2. With the tighter fleet, Golden 8
  needs about 600,000 iterations to turn feasible, and Golden 7 none in 10⁶.

One remark for the method, not a bug: while β stays pinned the weights
never oscillate, and the paper relies on that oscillation to trade
distance against feasibility. A fleet that leaves no slack turns the
search into a pure excess minimiser until the first feasible iteration.
Logging feasible iterations per phase (now in `--phase-log`) shows
whether a run is in that regime.
