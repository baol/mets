# P7: CMT acceptance, Golden diagnostics, GENI

Sandbox, 4 cores, runs with `scripts/p7-cvrp.sh` (build `c2c1e17`, plain
best insertion, 2-opt/Or-opt refinement, fleet = reference routes + 1
unless stated). Every saved `.sol` recounted by `scripts/verify_vrp.py`
through `scripts/p7-summary.py`: 0 failed recounts in every table below.
Gaps are to the reference `.sol` cost; seconds are wall time with 4 runs
sharing the machine.

## CMT 1–14, 10 seeds: the acceptance test passes at both budgets

| instance | η = 10⁵ mean % | worst % | η = 10⁶ mean % | worst % | s/run at 10⁶ |
|---|---|---|---|---|---|
| CMT1 | 0.00 | 0.00 | 0.00 | 0.00 | 8 |
| CMT2 | 0.29 | 0.74 | 0.01 | 0.01 | 21 |
| CMT3 | 0.26 | 0.41 | 0.15 | 0.40 | 29 |
| CMT4 | 0.62 | 1.34 | 0.21 | 0.40 | 66 |
| CMT5 | 1.64 | 2.12 | 1.14 | 1.53 | 165 |
| CMT6 | 0.00 | 0.00 | 0.00 | 0.00 | 16 |
| CMT7 | 0.00 | 0.00 | 0.00 | 0.00 | 38 |
| CMT8 | 0.00 | 0.01 | 0.00 | 0.00 | 50 |
| CMT9 | 0.43 | 0.66 | 0.14 | 0.26 | 113 |
| CMT10 | 1.42 | 2.07 | 0.96 | 1.33 | 158 |
| CMT11 | 0.90 | 3.01 | 0.00 | 0.00 | 38 |
| CMT12 | 0.00 | 0.00 | 0.00 | 0.00 | 33 |
| CMT13 | 1.43 | 2.05 | 0.97 | 1.23 | 49 |
| CMT14 | 0.07 | 0.15 | 0.00 | 0.02 | 37 |
| **mean** | **0.50** | | **0.26** | | |
| paper (Table 9, ITS) | 0.60 | | 0.27 | | 188 s per 10⁵ |

Both at or below Table 9: the port is accepted on CMT. The margin at 10⁶
is 0.01 points, so "matches" is the honest word. The residual is in the
three largest instances (CMT5, 10, 13: 1.0–1.1%), where the paper's
GENI-reshaped routes would matter most.

## Golden: the two diagnostics of the plan (one seed each)

**(a) η = 10⁶.** Mean gap 1.37% against 2.10% at 10⁵: ×0.65. The paper
goes from 1.30% to 0.79%: ×0.61. The same ratio, so the phase dynamics
scale as the paper's do and the deficit is a constant factor of the
neighbourhood: by the plan's rule, GENI.

| Golden | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10⁶ gap % | 0.72 | 0.76 | 2.17 | 1.18 | 0.09 | 0.53 | 1.99 | 2.46 | 0.93 | 1.33 |

| Golden | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10⁶ gap % | 1.64 | 1.73 | 1.05 | 1.82 | 2.05 | 2.28 | 0.76 | 1.57 | 0.72 | 1.56 |

**(b) Fleet without the spare vehicle** (`--no-spare`, η = 10⁵, Golden
1–8):

| Golden | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| reference + 1 | 1.21 | 3.40 | 3.13 | 2.90 | 0.35 | 0.47 | 2.55 | 2.85 |
| reference fleet | 1.36 | 0.92 | 1.63 | 2.12 | 0.26 | 1.81 | infeasible | infeasible |

On 1–6 the reference fleet averages 1.35% against 1.91%, but Golden 7 and 8
find no feasible solution in 10⁵ iterations at all. One seed: the
spare vehicle may cost something on 2–4, and the reference fleet is not
safe as a default. The default stays at reference + 1; a 10-seed
comparison on Golden 1–6 is cheap if it matters later.

## GENI (`709ed58`, `--insertion geni`, p = 5)

Types I and II in both orientations over GENI's p-neighbourhoods, plain
insertion included, in the inter-route relocation; removal still by
reconnection (no unstringing), refinement still 2-opt/Or-opt. Tests
enumerate every valid move on tours of 1–9 nodes (an empty route up to 8
customers) against the definition's arc changes and a recount, the
validity rules against a path enumeration, `best` against brute force,
relocations into routes of 0–5 and 14 customers against full
evaluations, and the per-route cache against a fresh neighbourhood
along a search.

Cost: with the cache (a move recomputes two routes' insertions), CMT1 at
10⁴ iterations takes 10 s against 0.1 s for plain insertion, CMT4 35 s
against 0.7 s (with 4 other runs on the machine). On CMT that is ~1.7×
the paper's 188 s per 10⁵; on Golden, with more routes per instance, the
ratio to plain insertion should be smaller.

Running now: GENI at η = 10⁵, one seed, CMT then Golden (the deciding
number: Golden's mean against 2.10% plain and the paper's 1.30%).
