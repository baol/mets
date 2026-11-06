# Q6 on tai50a: aspiration 40n² and off (RTX 3060)

Data from the owner (`run-data.tgz`, `results/q6/tai50a/B-asp40` and
`B-asp1000`); recounts in the sandbox. Same setting as the previous Q6
reports: arm B (independent trajectories), population 112, 200n = 10,000
robust tabu iterations per batch, tenure (0.9n, 1.1n), 30 minutes, seeds
1–5. Build `e597c1b`; `examples/qap_gpu.rs` is unchanged since `5fafddd`
(the build of B-asp20), so the arms compare. Every trace clean (no recount
mismatch, no invalid permutation); all 10 `.best` files recount to
4,938,796 with `scripts/verify_qap.py`. "Off" is 1000n², which no batch of
10,000 iterations can reach.

## Results (best known 4,938,796)

| aspiration | reached | time to it, by seed 1–5 (s) | median, max |
|---|---|---|---|
| 2n² | 3/5 | 493, 658, —, —, 246 | 493, 658 |
| 5n² | 4/5 | 418, —, 389, 59, 575 | 403, 575 |
| 10n² | 5/5 | 1,549, 235, 355, 484, 105 | 355, 1,549 |
| **20n²** | **5/5** | **12, 35, 25, 266, 1,027** | **35, 1,027** |
| 40n² | 5/5 | 143, 503, 42, 1,300, 578 | 503, 1,300 |
| off (1000n²) | 5/5 | 140, 466, 39, 1,303, 598 | 466, 1,303 |

(Rows 2n²–20n² from `2026-11-06-q6-tai50a-arms.md`.)

## Reading

* The sweep peaks at 20n². By the Q6 rule ("extend while the median
  falls") the sweep stops: 40n² has a median 14× that of 20n².
* 40n² and off are the same run within a few seconds per seed, with
  identical starting populations: at 40n² = 100,000 iterations the
  aspiration almost never fires within the 10,000-iteration batches. From
  here on, the aspiration term has effect only at or below about 20n² on this
  batch length.
* Five seeds, wide spreads: 20n²'s advantage rests on three seeds under
  40 s. It is the setting to carry forward, not a proven optimum.
* Decision for the queue: B and C on tai60a (and then tai80a) at 20n².

## Also in the archive: Q10, partial

`results/q10/memetic-asp5`: seed 2 tied the best known 4,938,796 in
1,547 s (recounted OK, build `e6ceac1`); seeds 1, 3 and 4 have empty logs,
still running when the archive was made. Q10 is read when it finishes.
