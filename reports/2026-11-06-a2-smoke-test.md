# A2 smoke test: DSJC1000.5 at 82 on the RTX 3060 (global-memory tables)

Build `e6ceac1`. `coloring_gpu DSJC1000.5.col --k 82 --pairs 112
--max-seconds 600 --trace a2-smoke.trace` (defaults: 128 threads per block,
60n = 60,000 TabuCol iterations per child). Data from the owner.

* Start line: tables in global memory, 9 blocks per SM × 28 SMs, 109 MB of
  device memory (as the A2 preparation predicted).
* 53 batches, every one with `mismatches=0 bad_colourings=0`.
* Batch time 10.7–12.2 s for 224 runs × 60,000 iterations: **about 1.2 M
  TabuCol iterations per second**, the rate of the first (global-memory)
  DSJC500.5 kernel, against 10–11 M with shared tables on DSJC500.5.
* Best: 85 conflicts after the first batch, 61 at the 600 s cap; the
  population median went from 105 to 87.

Decision by `PLAN-problems.md` A2: under 3 M it/s, so the 4-hour runs wait
for A2.6, the 8-bit conflict table (implemented in the next commit: the
table is n·k = 82 KB, and with the colouring and the conflict list it fits
the 99 KB opt-in at one block per SM). The comparison to run: the same
smoke test with `--table 8` against this one.
