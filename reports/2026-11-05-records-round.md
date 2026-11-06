# Records round: results

Sandbox: 4 cores (Xeon 2.3 GHz), CPU only. Owner's machine: NVIDIA GeForce
RTX 3060 (28 SMs). "Best known" is the value in our QAPLIB `.sln` copy or the
DIMACS table unless stated; a tie is a value equal to it. No value below a
best known was found anywhere.

## QAP

### tho150 on the GPU: tie, verified (RTX 3060)

* Build `70d0fcf`. `qap_gpu qapdata/tho150.dat qapdata/tho150.sln --crossover
  cohesive --population 112 --iters 200 --run-to-end --max-seconds 14400
  --trace qap-tho150.trace`.
* Start line: delta matrix in shared memory (93 KB per block), symmetric
  formulas, 1 block per SM × 28 SMs; 30,000 robust tabu iterations per child.
* Timeline: 8,133,676 at 1,053 s (generation 101); **8,133,398 at 2,930 s**
  (generation 278); nothing lower until the cap (14,401 s, 1,295
  generations, about 11.3 s per generation, ~300,000 tabu it/s).
* Trace: `mismatches=0` on all batches.
* Verification (sandbox, independent script): the cost recomputed from
  `tho150.dat` is 8,133,398; the permutation is exactly the inverse of the
  one in `tho150.sln` (QAPLIB lists facility → location), i.e. the published
  assignment itself.
* k-opt polish of it (`--polish-from tho150.best`, k = 9, 1,024 groups of 9
  close positions per round, 20 rounds without improvement): no improvement;
  `tho150.kopt.best` is identical to `tho150.best`.
* Caveat (from `PLAN-problems.md` §0): this one run combined the cohesive
  crossover with the symmetric kernel and the shared-memory delta matrix; it
  does not show that the cohesive crossover helped.

### sko100a on the GPU (RTX 3060)

Build `70d0fcf`, `qap_gpu` defaults (population 224, 100n iterations,
uniform crossover): best known 152,002 at generation 9, **21.6 s**, about
2.4 s per batch of 224 × 10,000 iterations. One run.

### QAP GPU kernel speed (RTX 3060, tai100a, 224 × 10,000 iterations per batch)

| build | change | per batch |
|---|---|---|
| `1d3c936` | first kernel | 22.2 s |
| `f25b85a` | pairs touching the swap recomputed one per warp, from transposes | 6.8 s (same costs, generation by generation) |

The 21.6 s sko100a run above adds the delta matrix in shared memory and the
symmetric formulas (different instance, so not a clean comparison).

### Sandbox QAP results

* tai50a, `qap_advanced --method memetic` (4 runs, 4 threads): best known
  4,938,796 in 756 s with random perturbation; 4,941,410 after 900 s with
  directed perturbation. tai60a: 7,217,850 (0.16%) after 900 s with either.
  One run per arm: exploratory.
* tho150, `qap_gpu --cpu` (population 4, uniform crossover), 1 hour:
  8,141,304 (0.097%). Worse than `ma`'s earlier 8,133,702.
* tai50a, `qap_gpu --cpu` (population 4), 30 min: stalled at 4,941,410 from
  about 1 minute; population 16 with 50n iterations, 15 min: 4,958,170;
  `--stagnation 20` restarts, 15 min: 4,941,410 again.
* tai100a, `qap_advanced --method memetic`, 50 min: 21,118,120 (0.31%).
* Cohesive vs uniform with `ma` on sko100a, 5 seeds each, time to best
  known: uniform 45.4, 19.0, 99.1, 7.9, 46.8 s (mean 43.6); cohesive 16.2,
  92.7, 53.2, 29.9, 97.6 s (mean 57.9). sko64: both about 1 s.
* CPU 7-opt polish of the 8,141,304 tho150 permutation (64 close groups per
  round, 10 rounds): no improvement.
* Robust tabu speed against Taillard's algorithm in C (one core, 10,000
  iterations): tai50a 84 k vs 64 k it/s, tai100a 21 k vs 16 k, tai150b
  6.6–7.2 k vs 7.6 k, tai256c 1.9 k vs 1.9 k.

## Graph colouring

### DSJC500.5 at 47 colours on the GPU (RTX 3060), first run

* Build `32415bd`, `coloring_gpu --k 47 --pairs 112 --iters 60 --threads 256
  --max-seconds 14400` (no polish).
* 2 conflicts by about 1,200 s; **1 conflict from 1,460 s** (generation
  ~1,590) and still 1 at 3,527 s (generation 4,061), when it was stopped to
  restart with polishing. 7.7 M TabuCol it/s.
* Second run with `--polish-every 200 --vss-rounds 4`, build `26c75d3`,
  started (2 blocks per SM × 28, 21 MB of device memory); result pending.

### TabuCol GPU kernel speed (RTX 3060, DSJC500.5, k = 48, 128 × 5,000 iterations per batch)

| kernel | per batch | it/s |
|---|---|---|
| tables in global memory | 0.55 s | 1.2 M |
| 16-bit tables in shared memory | 0.155 s | 4.1 M |
| conflicting vertices listed, moves shared by all threads (`c3895f0`) | 0.06 s | 10–11 M |

Legal 48-colouring in 12.9 s with the last kernel; `mismatches=0` in every run.

### Sandbox colouring results

* Variable Space Search (TabuCol + PartialCol) from 8 stuck TabuCol
  solutions on DSJC500.5 at 48 (after 1 M iterations): 2 M more TabuCol
  iterations end at 8, 10, 9, 7, 5, 8, 8, 10 conflicts (mean 8.1); 20 VSS
  rounds of 50,000 iterations per space end at 7, 8, 7, 5, 5, 8, 8, 4
  (mean 6.5), never worse, in 40–60% less time. One seed per start:
  exploratory by the protocol of `PLAN-problems.md` §6.
* Three one-hour HEAD-style runs at 47 on DSJC500.5 (4 cores): no legal
  colouring.
* TabuCol speed, one core, DSJC500.5 at 48: 210 k it/s before the
  conflicting-vertex list and lower bounds, 337 k after.
