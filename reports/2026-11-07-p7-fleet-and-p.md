# P7: the fleet as a knob (GENI without the spare vehicle) and GENI's p on Golden 3 and 4

Date 2026-11-07. Sandbox, 4 cores, at most 4 runs at a time. Seed 1,
η = 10⁵, one run per configuration (exploratory, `PLAN-problems.md` §6:
±0.5 points on one instance and one seed is noise). Gaps are to today's
CVRPLIB best known (the reference `.sol`) unless stated; every saved `.sol`
was recounted by `scripts/verify_vrp.py` (none of the solver's code).

## 1. GENI with and without the spare vehicle

**What this measures.** `--no-spare` runs with as many vehicles as the
reference solution has routes. That is a *smaller* fleet than the paper's
(Cordeau & Maischberger 2012, Table 9, column m), which is the reference's
route count + 1 on these instances (+ 2 on Golden 1). Our default,
reference + 1, is the paper's m on all of them except Golden 1 (10 against
11). So this is the fleet measured as a knob, not a reproduction of the
paper's setting.

Builds: job runs at `71f6bb7` (Golden 1–6) and `814d5a2` (Golden 9–12, 18;
the runner rebuilt the binary when another run started). The two commits
differ only in `PLAN-problems.md` and a report, so the solver is the same.
Between `709ed58` (the spare column) and these builds the solver gained
unstringing (`e018752`, off unless `--removal unstring`); read from the
diff, the default path is unchanged (not checked by a rerun).
Command:

```
SET=Golden INSTANCES="Golden_1 Golden_2 Golden_3 Golden_4 Golden_5 Golden_6 Golden_9 Golden_10 Golden_11 Golden_12 Golden_18" \
  SEEDS=1 ETA=100000 TAG=Golden-geni-nospare EXTRA="--insertion geni --no-spare" VRP=<vrpdata> JOBS=3 scripts/p7-cvrp.sh
```

The "with spare" column is `2026-11-06-p7-geni-golden.md` (build `709ed58`,
`--insertion geni`, p = 5, default fleet, same seed and budget).

| Golden | vehicles, spare / no spare (paper m) | with spare % | s | no spare % | s | no spare − spare | no spare, gap to paper's ref % |
|---|---|---|---|---|---|---|---|
| 1 | 10 / 9 (11) | 0.74 | 590 | 0.82 | 581 | +0.08 | 0.76 |
| 2 | 11 / 10 (11) | 0.80 | 814 | 1.16 | 800 | +0.36 | 0.84 |
| 3 | 11 / 10 (11) | 2.57 | 1,035 | 1.20 | 1,020 | −1.37 | 0.85 |
| 4 | 11 / 10 (11) | 2.56 | 1,272 | 1.90 | 1,248 | −0.66 | 1.87 |
| 5 | 6 / 5 (6) | 0.92 | 468 | 0.69 | 451 | −0.23 | 0.69 |
| 6 | 8 / 7 (8) | 1.63 | 685 | 0.16 | 663 | −1.47 | 0.11 |
| 9 | 15 / 14 (15) | 1.99 | 584 | 1.07 | 594 | −0.92 | 0.94 |
| 10 | 17 / 16 (17) | 1.95 | 739 | 1.91 | 771 | −0.04 | 1.49 |
| 11 | 19 / 18 (19) | 1.46 | 950 | 2.25 | 959 | +0.79 | 1.95 |
| 12 | 20 / 19 (20) | 1.90 | 1,216 | 2.94 | 1,204 | +1.04 | 2.37 |
| 18 | 28 / 27 (28) | 2.85 | 723 | **13.14** | 742 | **+10.29** | 13.14 |
| **mean (11)** | | **1.76** | | **2.48** | | +0.72 | 2.27 |
| mean without Golden 18 (10) | | 1.65 | | 1.41 | | −0.24 | |
| Golden 1–6 (route length limit) | | 1.54 | | 0.99 | | −0.55 | |
| Golden 9–12 | | 1.83 | | 2.04 | | +0.22 | |

Every no-spare run used all its vehicles (routes = vehicles). Recount:
11 of 11 `.sol` OK, costs agree with the solver's to the printed
precision (`scripts/p7-summary.py results/p7/Golden-geni-nospare`: 0 failed
recounts).

Golden 18 (300 customers, capacity 200, no route limit, 27 vehicles) is
feasible but converges late: its best was 26.9% above the best known at
λ ≈ 41,000, 18.4% at λ ≈ 90,000 and 13.4% at λ ≈ 99,000, still falling
steeply when the budget ran out. At 27 vehicles the capacity is nearly
exhausted, and most of the run is spent in or near the infeasible region.

## 2. Plain insertion, no spare, η = 10⁶: Golden 7 and 8

Build `71f6bb7`, `--no-spare` with plain (best) insertion, seed 1,
η = 10⁶ (`results/p7/Golden-plain-nospare-1e6/`).

| Golden | vehicles (paper m) | result | s |
|---|---|---|---|
| 7 | 8 (10) | no feasible solution in 10⁶ iterations | 347 |
| 8 | 10 (12) | 1.17% (10 routes), first feasible between λ = 540,765 and 642,592 (log lines at phases 200 and 250; 2.14% at the first) | 658 |

Recount: Golden 8's `.sol` OK (11771.0749 recomputed, 11771.07 stated);
Golden 7 saved none. The paper's m is two vehicles more than these on both
instances, so this is a tight fleet, not a discrepancy.
