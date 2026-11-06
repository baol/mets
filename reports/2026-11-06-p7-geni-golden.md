# P7: GENI insertion on CMT and Golden (η = 10⁵, one seed)

Build `709ed58`, `--insertion geni` (p = 5, types I and II, both
orientations, plain insertion included), removal by reconnection,
2-opt/Or-opt refinement, fleet = reference + 1, seed 1, sandbox with 4
runs at a time. Every saved `.sol` recounted by `scripts/verify_vrp.py`
(0 failures). The plain-insertion column is the same seed and budget
(`2026-11-06-p7-its-first-results.md`, second version).

## Golden 1–20

| Golden | plain % | GENI % | GENI s |
|---|---|---|---|
| 1 | 1.21 | 0.74 | 590 |
| 2 | 3.40 | 0.80 | 814 |
| 3 | 3.13 | 2.57 | 1,035 |
| 4 | 2.90 | 2.56 | 1,272 |
| 5 | 0.35 | 0.92 | 468 |
| 6 | 0.47 | 1.63 | 685 |
| 7 | 2.55 | 1.74 | 913 |
| 8 | 2.85 | 2.16 | 1,144 |
| 9 | 1.47 | 1.99 | 584 |
| 10 | 1.31 | 1.95 | 739 |
| 11 | 1.84 | 1.46 | 950 |
| 12 | 2.35 | 1.90 | 1,216 |
| 13 | 2.55 | 2.00 | 564 |
| 14 | 2.29 | 2.04 | 696 |
| 15 | 2.63 | 2.39 | 916 |
| 16 | 2.52 | 2.47 | 1,143 |
| 17 | 0.69 | 0.51 | 587 |
| 18 | 2.60 | 2.85 | 723 |
| 19 | 2.24 | 1.63 | 914 |
| 20 | 2.59 | 1.99 | 1,082 |
| **mean** | **2.10** | **1.82** | |
| Golden 1–8 (route length limit) | 2.11 | 1.64 | |
| Golden 9–20 | 2.09 | 1.93 | |

Paper (Table 9, ITS, η = 10⁵): 1.30%.

## CMT 1–14

GENI 0.46% (plain, same seed: 0.44%; plain over 10 seeds: 0.50%). Per
instance: 0.00, 0.11, 0.15, 0.20, 2.60, 0.00, 0.00, 0.00, 0.25, 1.59,
0.00, 0.00, 1.60, 0.00. Per run 100–480 s against 1–15 s plain.

## Reading, by `PLAN-problems.md` P7 decision 2

* Golden falls from 2.10% to 1.82%: better than plain, but between 1.5%
  and 2.1%. So GENI does not yet become the default, and the next suspects
  are measured one at a time on Golden 1–4 and 9–12: (1) removal by
  unstringing, (2) GENI with q doubled in the route refinement instead of
  2-opt/Or-opt, (3) GENI in the cluster reinsertion.
* The gain is largest where route length binds (Golden 1–8: 2.11% →
  1.64%), as expected: GENI reshapes the route around the inserted
  customer, so a long route absorbs a customer at a lower length cost.
  Single instances move both ways (5, 6, 9, 10 worse), one seed each.
* On CMT GENI changes nothing measurable at η = 10⁵ (0.46% against 0.44%
  on the same seed).

## Step (1): unstringing, implemented

`e49bbbf`, `--removal unstring` (p = `--geni-p`). The removals are
derived as the inverses of insertion types I and II, and a test checks
that every insertion on tours of 3–8 nodes is undone by an unstringing of
its own type. The other tests are like GENI's: every valid unstringing on
tours of 2–10 nodes, v at every position, both orientations, against
the definition's arc changes and a recount; each offset at its smallest
and largest value; brute force for the choice; relocations out of routes
of 1–5 and 14 customers against full evaluations; the cache against a
fresh neighbourhood along a search, for every combination. Unstringing
adds about 6% to GENI's time per iteration (CMT4, 10⁴ iterations: 36.7 s
against 34.5 s). The measurement on Golden 1–4 and 9–12 (one seed, η = 10⁵,
GENI insertion plus unstringing, against the GENI column above) is
running.
