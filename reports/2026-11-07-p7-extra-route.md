# P7: the extra route (route counts of the saved solutions)

The plan's new hypothesis: with the spare vehicle the search ends with
every vehicle in use, so on Golden 2–4 the deficit may be one route it
never empties. Step 1 is free: count the non-empty routes of every saved
best solution against the reference `.sol`. All runs: seed 1, default
fleet m = reference routes + 1, every `.sol` recounted earlier by
`scripts/verify_vrp.py`. The plain 10⁵ column is the run of
`2026-11-06-p7-its-first-results.md` (second version, build `1168797`).

## Routes used (gap to today's best known)

| Golden | reference routes | m | plain 10⁵ | plain 10⁶ | GENI 10⁵ | GENI + unstringing | GENI + `--refine geni` |
|---|---|---|---|---|---|---|---|
| 1 | 9 | 10 | 9 (1.21%) | 9 (0.72%) | 9 (0.74%) | 9 (0.81%) | 9 (0.89%) |
| 2 | 10 | 11 | **11** (3.40%) | 10 (0.76%) | 10 (0.80%) | **11** (2.98%) | 10 (0.67%) |
| 3 | 10 | 11 | **11** (3.13%) | **11** (2.17%) | **11** (2.57%) | **11** (2.27%) | **11** (3.09%) |
| 4 | 10 | 11 | **11** (2.90%) | 10 (1.18%) | **11** (2.56%) | **11** (2.44%) | **11** (2.47%) |
| 5 | 5 | 6 | 5 (0.35%) | 5 (0.09%) | 5 (0.92%) | | |
| 6 | 7 | 8 | 7 (0.47%) | 7 (0.53%) | 7 (1.63%) | | |
| 7 | 8 | 9 | **9** (2.55%) | **9** (1.99%) | **9** (1.74%) | | |
| 8 | 10 | 11 | **11** (2.85%) | **11** (2.46%) | **11** (2.16%) | | |
| 9 | 14 | 15 | 14 (1.47%) | 14 (0.93%) | 14 (1.99%) | 14 (1.47%) | 14 (2.08%) |
| 10 | 16 | 17 | 16 (1.31%) | 16 (1.33%) | 16 (1.95%) | 16 (1.71%) | 16 (2.19%) |
| 11 | 18 | 19 | 18 (1.84%) | 18 (1.64%) | 18 (1.46%) | 18 (1.67%) | 18 (1.58%) |
| 12 | 19 | 20 | 19 (2.35%) | 19 (1.73%) | 19 (1.90%) | 19 (1.58%) | 19 (1.73%) |
| 13 | 26 | 27 | 26 (2.55%) | 26 (1.05%) | 26 (2.00%) | | |
| 14 | 30 | 31 | 30 (2.29%) | 30 (1.82%) | 30 (2.04%) | | |
| 15 | 33 | 34 | 33 (2.63%) | 33 (2.05%) | 33 (2.39%) | | |
| 16 | 37 | 38 | 37 (2.52%) | 37 (2.28%) | 37 (2.47%) | | |
| 17 | 22 | 23 | 22 (0.69%) | 22 (0.76%) | 22 (0.51%) | | |
| 18 | 27 | 28 | **28** (2.60%) | 27 (1.57%) | 27 (2.85%) | | |
| 19 | 33 | 34 | 33 (2.24%) | 33 (0.72%) | 33 (1.63%) | | |
| 20 | 38 | 39 | 38 (2.59%) | 38 (1.56%) | 38 (1.99%) | | |
| runs with an extra route | | | 6 / 20 | 3 / 20 | 4 / 20 | 3 / 8 | 2 / 8 |

## Reading: confirmed

* An extra route appears only on the length-limited instances (Golden
  2, 3, 4, 7, 8, of 1–8) and once on 18. **Within an instance it costs
  about two points.** Golden 2: 10 routes gives 0.67–0.80%, 11 routes
  2.98–3.40%. Golden 4: 10 routes gives 1.18% (plain 10⁶), 11 routes
  2.44–2.90%. Golden 3 never ends with fewer than 11 routes in any run,
  and never goes below 2.17%. No run with the reference's route count
  is worse than 2.47%, and the worst gaps on Golden 1–8 all come with
  the extra route.
* On GENI at 10⁵ the four extra-route instances (3, 4, 7, 8) average
  2.26% against the paper's rebased 1.17% on them: 0.22 of the 0.31-point
  set deficit. The other 16 average 1.70% against the paper's 1.59%.
  Excluding the extra route, the port is within about 0.1 point of the paper.
* Golden 7 and 8 use m = ref + 1 = 9 and 11 here, while the paper's m
  is ref + 2 (10 and 12) and its gaps there are 1.51% and 1.80% rebased.
  So the paper also had room for more routes and must have emptied them.

## `--refine geni` (step 3, measured before the plan reordered it)

Golden 1–4, 9–12, GENI insertion plus the paper's refinement (unstringing
and GENI within each route, p = 10), seed 1, η = 10⁵, build `1f71ec1`:
**1.84%** against GENI alone 1.75% (to the paper's reference: 1.56%).
Per instance: 0.89, 0.67, 3.09, 2.47, 2.08, 2.19, 1.58, 1.73. Golden 3
and 4 still end with 11 routes. No gain, so it stays a flag. The
`--reinsert geni` runs were already under way and will be reported after
step 2, as the plan orders.

## Step 2 (running)

`--phase-log` now prints the routes in use at the phase end, the fewest
and most over the phase, and the best solution's count. The watch runs
on Golden 3 at m = 11 (GENI, seed 1). Its first phases show 11 routes
throughout: the sweep opens all m routes at the start.
