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

## Saturation fails the run (per the plan's review)

Since the commit after `276c30b`, a batch in which any block's 8-bit
counter saturated aborts `coloring_gpu` with "the 8-bit conflict table
saturated in N blocks this batch; rerun with --table 16", instead of a
warning. After saturation a block's table is wrong (a 256 counted as 255
decrements to 254), so no result may come from it. On these graphs γ is
about 6, so the check is not expected to fire.

## The 8-bit table run (owner, build `53e563e`, same command plus `--table 8`)

Start line: "tables in 84 KB of shared memory, 8-bit conflict table, 1
blocks per SM × 28 SMs, 73 MB of device memory". 13 batches, every one
`mismatches=0 bad_colourings=0`, no saturation (the run would have
failed). Best 72 conflicts at 635 s (global tables: 61).

| table | blocks resident | batch (s) | TabuCol it/s |
|---|---|---|---|
| 16-bit, global memory | 9 per SM × 28 = 252 | ~11.3 | ~1.2 M |
| 8-bit, shared memory | 1 per SM × 28 = 28 | ~47 | ~0.27 M |

The shared table is 4× *slower*: the 84 KB table leaves room for one
block per SM, so 28 of the 224 individuals run at a time with 4 warps per
SM, against 252 blocks of 4 warps on global memory. Shared-memory access
does not pay for the lost concurrency at 128 threads per block. The one
variable left that can change this is the block width: with one block per
SM, 512 or 1,024 threads give 16–32 warps to the scan over conflicting
vertices × colours. A 10-minute run at `--threads 512` decides whether
A2.6 is closed; if it is not faster than 1.2 M it/s, the 4-hour runs go
ahead on global 16-bit tables (`--table 16`), and `--table auto`, which
today prefers the 8-bit shared table whenever the 16-bit one does not
fit, is changed to prefer global 16-bit.

## Block width: `--table 8 --threads 512` (owner, build `ad135ed`)

Same command, 512 threads per block, still 1 block per SM × 28. 68
batches in 606 s, every one `mismatches=0 bad_colourings=0`, no
saturation. Best 57 conflicts (global tables: 61; 8-bit at 128 threads:
72).

| table, threads per block | blocks resident | batch (s) | TabuCol it/s (log's bound) | best at 600 s |
|---|---|---|---|---|
| 16-bit global, 128 | 252 | ~11.3 | ~1.2 M | 61 |
| 8-bit shared, 128 | 28 | ~47 | 0.27 M | 72 |
| **8-bit shared, 512** | 28 | **8.1–9.6 (8.7 median)** | **1.27 → 1.50 M** | **57** |

At 512 threads the shared 8-bit table passes the global tables: 1.3×
faster per batch, 4 fewer conflicts at equal time (one seed). The batch
time falls during the run (10.6 → 8.1 s) as the conflict lists shrink,
as on the global run. So the block width was the missing variable: one
block per SM needs 16 warps to keep the SM busy. Still to run: `--threads
1024` (the plan's second run), then the choice between 512 and 1024 for
the 4-hour runs. `--table auto` is already right for this graph (the
16-bit table does not fit, so it picks 8-bit), but the default 128
threads is not: the 4-hour runs must pass `--threads` explicitly.
