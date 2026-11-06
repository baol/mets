# RTX 3060 runs: Q6 aspiration sweep on tai50a, and DSJC500.5 at 47 (A3)

Data received from the owner as `run-data.tgz` (logs, traces, `.best`
files); summaries and recounts made in the sandbox.

## Q6: aspiration sweep, tai50a, arm B (independent trajectories)

* Build `2a5f085` (includes the stop-at-target fix `beb9bb6` and the CPU
  resume `1cd68a8`). `SWEEP=1 MINUTES=30 scripts/q6-grid.sh`: `qap_gpu
  --independent --aspiration X`, X = 2, 5, 10 (in n²), population 112,
  200n = 10,000 robust tabu iterations per batch, tenure (0.9n, 1.1n),
  seeds 1–5, 30 minutes per run, one run at a time.
* Start line: delta matrix in shared memory (12 KB per block), symmetric
  formulas, 2 blocks per SM × 28 SMs; `mode=independent`.
* Every trace clean (`mismatches=0`, `bad_permutations=0` on all batches);
  all 15 `.best` files recount correctly with `scripts/verify_qap.py`.

| aspiration | reached 4,938,796 | time to it, by seed (s) | median, max |
|---|---|---|---|
| 2n² | 3/5 | 493, 658, —, —, 246 | 493, 658 |
| 5n² | 4/5 | 418, —, 389, 59, 575 | 403, 575 |
| 10n² | **5/5** | 1,549, 235, 355, 484, 105 | 355, 1,549 |

(— : ended at 4,941,410 after 1,800 s.) Median best value by wall time, from
`scripts/q6-summary.py`: at 5 min, 4,953,480 (2n²), 4,941,410 (5n², 10n²);
at 15 min, 4,938,796 for all three. The summary's "60 min" column is the
final value of these 30-minute runs.

Reading, by the protocol of `PLAN-problems.md` §6 (5 seeds per arm): 10n² is
the only setting that reached the best known on every seed, and it is also
the one Taillard used for tai-a; 2n² is the worst. The differences in time
to target among the seeds that reached are within the seed spread. By the
staging in Q6 the next step is arms A, B and C on tai50a at aspiration 10n².

For scale: every run here that reached the best known did so in under 26
minutes on the GPU, against one 4-core `memetic` run in the sandbox (756 s)
and none of the earlier `qap_gpu --cpu` runs.

The saved permutation of asp10 seed 5 is the `.sln` assignment itself.

## A3: DSJC500.5 at 47 colours with GPU polish and VSS

* Build `26c75d3`. `coloring_gpu DSJC500.5.col --k 47 --pairs 112 --iters 60
  --threads 256 --polish-every 200 --vss-rounds 4 --max-seconds 14400`.
  Start line: tables in 48 KB of shared memory, 2 blocks per SM × 28 SMs,
  21 MB of device memory.
* Trace: 12,415 batches, mean 1.10 s each; no mismatches or bad colourings.
  (The trace file also holds an earlier short run on build `83fd0d4`,
  because traces are appended; the numbers here are from the second run
  only.)
* Best: 2 conflicts at 424 s, **1 conflict at 1,888 s** (generation 2,141),
  still 1 at the cap (14,401 s, 12,411 generations). No legal 47-colouring.
* Polish (GPU batch of copies of the best for 500,000 iterations each, then
  4 rounds of VSS on the CPU threads): 60 polishes, 8 from a 2-conflict and
  52 from a 1-conflict best, 9.7 s on average (584 s in all, 4% of the run);
  none removed the last conflict.
* Last batch: population minimum 4 conflicts, median 21.

Compared with the first run without polish (build `32415bd`, 1 conflict at
1,460 s, stopped at 3,527 s), polish and VSS did not change the outcome: the
search reaches 1 conflict in about half an hour and stays there. 47 on
DSJC500.5 is believed optimal; this was a calibration run (A3), and the
remaining colouring effort is better spent on instances whose best known is
not believed optimal (flat1000_76_0, DSJC1000.5, C2000.5; A2/A5).
