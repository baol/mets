# P7: unstringing removal on Golden 1–4 and 9–12

`--insertion geni --removal unstring` (GENI's unstringing, types I and II,
p = 5, the inverse of insertion types I and II; `e018752`, run on build
`71f6bb7`), against `--insertion geni` alone (removal by reconnection,
`2026-11-06-p7-geni-golden.md`). Seed 1, η = 10⁵, default fleet
(reference + 1), 2-opt/Or-opt refinement, 4 runs at a time on the
sandbox. Every saved `.sol` recounted by `scripts/verify_vrp.py` through
`scripts/p7-summary.py` (0 failures). Paper column: Table 9's 10⁵ mean
rebased to today's best known (`2026-11-07-p7-golden-rebased.md`).

| Golden | paper % | GENI % | GENI + unstringing % | change | s (unstringing run) |
|---|---|---|---|---|---|
| 1 | 0.80 | 0.74 | 0.81 | +0.07 | 652 |
| 2 | 0.73 | 0.80 | 2.98 | +2.18 | 890 |
| 3 | 0.59 | 2.57 | 2.27 | −0.30 | 1,136 |
| 4 | 0.79 | 2.56 | 2.44 | −0.12 | 1,313 |
| 9 | 1.18 | 1.99 | 1.47 | −0.52 | 605 |
| 10 | 1.69 | 1.95 | 1.71 | −0.24 | 676 |
| 11 | 1.79 | 1.46 | 1.67 | +0.21 | 814 |
| 12 | 1.90 | 1.90 | 1.58 | −0.32 | 1,030 |
| **mean** | **1.18** | **1.75** | **1.87** | +0.12 | |

To the paper's own reference: GENI + unstringing 1.59%.

## Reading

* No gain on the mean (1.87% against 1.75%). Six of eight instances move
  down by 0.1–0.5 points, which is within one seed's noise, and Golden 2
  moves up by 2.2 points, which is not: the run ended in a different
  region (11 routes, as with GENI alone). One seed cannot say whether that
  is unstringing or chance.
* **Golden 3 and 4 are not closed**: 2.27% and 2.44% against the paper's
  0.59% and 0.79%. Unstringing takes 0.1–0.3 points off GENI there, but
  leaves almost all of the 1.7–2.0 point gap.
* Cost: about 10% more time per run than GENI alone (Golden 3: 1,136 s
  against 1,035 s).
* By the plan's order, unstringing alone does not explain the deficit.
  The next suspects, `--refine geni` (the paper's route refinement:
  unstringing and GENI within each route, p doubled) and `--reinsert
  geni` (GENI in the cluster reinsertion), are measured the same way
  next. Each runs alone, on top of GENI insertion with reconnection.
