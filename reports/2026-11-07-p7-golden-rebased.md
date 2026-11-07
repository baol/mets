# P7: the Golden comparison on a common reference

Head of research, from the paper's Table 9 (verified against the PDF) and
the reference `.sol` costs of the CVRPLIB mirror the sandbox uses. No new
runs: a re-reading of `2026-11-06-p7-geni-golden.md` and
`2026-11-06-p7-acceptance-and-diagnostics.md`.

The paper's gaps are to the "previous best known" of 2011 (Table 9, column
"Prev. best"). Ours are to today's CVRPLIB best known. The CMT references
are unchanged (CMT11: 1042.11 then, 1042.12 now), so the CMT acceptance
stands. The Golden references moved by 0.20% on average, up to 0.59%:

| Golden | paper's reference (2011) | today's best known | moved % |
|---|---|---|---|
| 1 | 5626.81 | 5623.47 | 0.06 |
| 2 | 8431.66 | 8404.61 | 0.32 |
| 3 | 11036.22 | 10997.80 | 0.35 |
| 4 | 13592.88 | 13588.60 | 0.03 |
| 5 | 6460.98 | 6460.98 | 0.00 |
| 6 | 8404.26 | 8400.33 | 0.05 |
| 7 | 10156.58 | 10102.70 | 0.53 |
| 8 | 11643.90 | 11635.30 | 0.07 |
| 9 | 580.42 | 579.70 | 0.12 |
| 10 | 738.49 | 735.43 | 0.42 |
| 11 | 914.72 | 911.98 | 0.30 |
| 12 | 1106.76 | 1100.67 | 0.55 |
| 13 | 857.19 | 857.19 | 0.00 |
| 14 | 1080.55 | 1080.55 | 0.00 |
| 15 | 1342.53 | 1337.27 | 0.39 |
| 16 | 1620.85 | 1611.28 | 0.59 |
| 17 | 707.76 | 707.76 | 0.00 |
| 18 | 995.13 | 995.13 | 0.00 |
| 19 | 1365.97 | 1365.60 | 0.03 |
| 20 | 1820.02 | 1817.59 | 0.13 |

Rebasing the paper's per-instance averages (Table 9, ITS/1, 10 runs) onto
today's best known, against our one-seed gaps:

| Golden | paper 10⁵ | ours, GENI 10⁵ | ours, plain 10⁵ | paper 10⁶ | ours, plain 10⁶ |
|---|---|---|---|---|---|
| 1 | 0.80 | 0.74 | 1.21 | 0.70 | 0.72 |
| 2 | 0.73 | 0.80 | 3.40 | 0.59 | 0.76 |
| 3 | 0.59 | **2.57** | 3.13 | 0.46 | **2.17** |
| 4 | 0.79 | **2.56** | 2.90 | 0.81 | 1.18 |
| 5 | 1.02 | 0.92 | 0.35 | 0.27 | 0.09 |
| 6 | 1.40 | 1.63 | 0.47 | 0.68 | 0.53 |
| 7 | 1.51 | 1.74 | 2.55 | 1.26 | 1.99 |
| 8 | 1.80 | 2.16 | 2.85 | 1.27 | **2.46** |
| 9 | 1.18 | 1.99 | 1.47 | 0.83 | 0.93 |
| 10 | 1.69 | 1.95 | 1.31 | 1.06 | 1.33 |
| 11 | 1.79 | 1.46 | 1.84 | 1.15 | 1.64 |
| 12 | 1.90 | 1.90 | 2.35 | 1.30 | 1.73 |
| 13 | 1.79 | 2.00 | 2.55 | 1.30 | 1.05 |
| 14 | 1.87 | 2.04 | 2.29 | 1.07 | 1.82 |
| 15 | 2.57 | 2.39 | 2.63 | 1.70 | 2.05 |
| 16 | 2.59 | 2.47 | 2.52 | 1.72 | 2.28 |
| 17 | 0.53 | 0.51 | 0.69 | 0.27 | 0.76 |
| 18 | 1.71 | **2.85** | 2.60 | 1.17 | 1.57 |
| 19 | 1.33 | 1.63 | 2.24 | 0.77 | 0.72 |
| 20 | 2.46 | 1.99 | 2.59 | 1.34 | 1.56 |
| **mean** | **1.50** | **1.82** | 2.10 | **0.99** | 1.37 |

(All gaps in %, to today's best known. The paper's published means, to
its own references, are 1.30% and 0.79%.)

## Reading

* On a common reference the deficit at 10⁵ is 0.32 points (1.82 against
  1.50), not 0.52; at 10⁶, plain insertion is 0.38 points behind (1.37
  against 0.99). The port is closer to the paper than the published means
  suggested.
* The deficit is concentrated. With GENI at 10⁵, ours is at or below the
  paper's rebased mean on 8 instances (1, 5, 11, 12, 15, 16, 17, 20) and
  within 0.4 points on 7 more. Four instances carry it: Golden 3 (+2.0),
  4 (+1.8), 18 (+1.1), 9 (+0.8). Golden 3 and 4 alone are 0.19 of the
  0.32 points. One seed per instance, so ±0.5 on any one is noise; two
  points is not.
* Golden 3 and 4 are the largest route-length-limited instances (400 and
  480 customers), and they are exactly where the fleet diagnostic
  (`2026-11-06-p7-acceptance-and-diagnostics.md`, plain insertion) gained
  most without the spare vehicle: Golden 2 3.40 → 0.92, Golden 3 3.13 →
  1.63, Golden 4 2.90 → 2.12.
* **Fleet, corrected the same day.** The paper's fleet is the m column of
  Table 9. Against today's reference route counts:

  | Golden | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
  |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
  | paper m | 11 | 11 | 11 | 11 | 6 | 8 | 10 | 12 | 15 | 17 | 19 | 20 | 27 | 31 | 34 | 38 | 23 | 28 | 34 | 39 |
  | reference routes | 9 | 10 | 10 | 10 | 5 | 7 | 8 | 10 | 14 | 16 | 18 | 19 | 26 | 30 | 33 | 37 | 22 | 27 | 33 | 38 |

  So the paper's m is reference + 1 on 17 instances and + 2 on Golden
  1, 7, 8: our default (reference + 1) is the paper's fleet except there,
  and `--no-spare` is a *smaller* fleet than the paper's. On CMT the
  paper's m is reference + 2 on most instances (CMT5: 20 for 16 routes),
  and the port matches the paper with + 1. The infeasibility on Golden 7
  and 8 without the spare is a tight fleet: with plain insertion at
  η = 10⁶ (seed 1), Golden 8 at 10 vehicles turns feasible at about
  600,000 iterations and ends at 1.17%; Golden 7 at 8 vehicles never
  does. Not a discrepancy. The paper reached 0.24% and 0.76% on Golden 3
  and 4 with the m = 11 we use, so the fleet does not explain them.
* The acceptance runs should report both gaps (to today's best known and
  to the paper's reference), so that the comparison with Table 9 is at
  equal footing; `p7-summary.py` needs the paper's reference column
  above.
