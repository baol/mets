# P7: first results of the sequential iterated tabu search (CPU)

Build `043efcc`; sandbox, one core per run (3 in parallel), η = 10⁵ tabu
iterations, **one seed per instance** (the acceptance test asks for 10:
these are exploratory). Fleet: the reference solution's routes plus one
spare vehicle. Insertion: best insertion between consecutive customers
(the paper's time-window variant), no GENI; route refinement every 200
iterations by reinsertion within the route (no 2-opt/Or-opt). Every saved
`.sol` verified by `scripts/verify_vrp.py` (CMT 14/14, Golden 20/20).

## CMT 1–14

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gap % | 0.00 | 0.06 | 0.40 | 0.55 | 2.94 | 0.00 | 0.00 | 0.00 | 0.92 | 1.27 | 0.00 | 0.00 | 1.70 | 0.09 |
| s | 0.8 | 2.2 | 3.1 | 7.3 | 13.1 | 1.0 | 2.6 | 3.5 | 8.4 | 14.7 | 3.9 | 3.4 | 5.3 | 4.0 |

**Mean gap 0.57%** (paper, Table 9, ITS average at η = 10⁵: 0.60%); six best
knowns reached (1, 6, 7, 8, 11, 12). 0.8–15 s per run against the paper's
188 s average on a 2.93 GHz Xeon.

## Golden 1–20

| | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| gap % | 1.51 | 3.52 | 3.83 | 4.69 | 3.71 | 2.65 | 4.00 | 3.32 | 2.21 | 1.92 |
| s | 16 | 30 | 49 | 85 | 10 | 19 | 34 | 64 | 19 | 30 |

| | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---|---|---|---|---|---|---|---|---|---|
| gap % | 1.76 | 2.61 | 2.55 | 2.11 | 2.89 | 3.02 | 0.77 | 1.58 | 1.88 | 2.49 |
| s | 51 | 90 | 22 | 36 | 60 | 119 | 20 | 33 | 54 | 80 |

**Mean gap 2.65%** (paper at η = 10⁵: 1.30%). The instances with a route
length limit (1–8) average 3.40%, the others 2.15%.

## Reading

* The CMT mean matches the paper at equal η with one seed, so the
  machinery (penalties and their adjustment, attribute tabu and
  aspiration, frequency score, phases, acceptance, perturbation) behaves
  like the paper's.
* Golden is twice the paper's gap, worst where route length binds. The
  plan's step 2 says GENI is added only if the CMT gaps stay above the
  table; CMT does not, Golden does. The two parts missing against the
  paper are GENI insertion (which inserts between two customers that need
  not be consecutive, reshaping the route around the inserted customer)
  and a stronger intra-route refinement (GENI with q doubled in the paper;
  2-opt/Or-opt in the plan's simpler variant). Both shorten routes, which
  is what the duration-limited Golden 1–8 need most.
* Next: 2-opt and Or-opt in the refinement (cheap, and the plan's own
  first choice), measured on Golden at η = 10⁵; then GENI if the gap stays
  above 1.30%. Ten seeds per instance only once the single-seed numbers
  are in range.

## Second version: 2-opt and Or-opt in the route refinement

Build after `1168797` (this commit): the every-200-iterations refinement is
now 2-opt and Or-opt (segments of 1–3 customers, either orientation,
first improvement until neither helps) instead of reinsertion alone. Same
setting: η = 10⁵, one seed, all solutions verified.

| set | first version | with 2-opt/Or-opt | paper (Table 9, η = 10⁵) |
|---|---|---|---|
| CMT 1–14 | 0.57% | **0.44%** (7 best knowns) | 0.60% |
| Golden 1–20 | 2.65% | **2.10%** | 1.30% |
| Golden 1–8 (route length limit) | 3.40% | 2.11% | |
| Golden 9–20 | 2.15% | 2.09% | |

CMT per instance: 0.00, 0.06, 0.00, 0.20, 2.11, 0.00, 0.00, 0.00, 0.35,
2.07, 0.00, 0.00, 1.24, 0.09. Golden per instance: 1.21, 3.40, 3.13, 2.90,
0.35, 0.47, 2.55, 2.85, 1.47, 1.31, 1.84, 2.35, 2.55, 2.29, 2.63, 2.52, 0.69,
2.60, 2.24, 2.59. One seed per instance: single instances move both ways
(Golden_18 from 1.58% to 2.60%); the means are the signal.

Golden stays above the paper by 0.8 points at equal η, so by the plan GENI
is next.
