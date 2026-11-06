# A2 preparation: DSJC1000.5 and flat1000_76_0 on the CPU host loop

Sandbox queue item 2 of `PLAN-problems.md`. Build `069d0eb`, `coloring_gpu
--cpu` pinned to 2 of the 4 cores (the Q9 runs held the other two),
`--pairs 2 --iters 10 --max-seconds 60 --trace` (the same checks at 120 s on
an earlier build gave the same picture).

| instance | n | edges | k | batches | trace | best after 60 s |
|---|---|---|---|---|---|---|
| DSJC1000.5 | 1,000 | 249,826 | 82 | 138 | all clean | 54 conflicts |
| flat1000_76_0 | 1,000 | 246,708 | 81 | 141 | all clean | 73 conflicts |

(The DSJC1000.5 file's `p` line says 499,652 edges: it lists each edge
twice; the reader keeps one of each.)

What the GPU path will do on these graphs, from `coloring_gpu`'s formulas
(the CPU path cannot print the GPU start line):

* Shared tables need 2nk + 3n bytes: 163 KB (DSJC1000.5) and 161 KB
  (flat1000_76_0), above the RTX 3060's 99 KB opt-in, so the kernel takes
  its **global-memory path**. That path has never run on the GPU: the only
  measurement of global tables is the first kernel on DSJC500.5 (0.55 s per
  batch against 0.06 s for the shared-memory kernel, `2026-11-05-records-round.md`).
  Expect a much lower TabuCol rate than on DSJC500.5, and check the first
  trace's `mismatches=0` before trusting a long run.
* Device memory at 112 pairs: about 109 MB and 107 MB.
* n·k = 82,000 and 81,000, well under the 24-bit move index (16.7 M); the
  new key layout is in use, though these sizes would also have fit the old
  20-bit one.

**k values, confirmed by the owner (2026-11-06):** DSJC1000.5 k = 82 and
flat1000_76_0 k = 81, the best-known upper bounds as listed in Moalic &
Gondran, "Variations on Memetic Algorithms for Graph Coloring Problems"
(the HEAD paper, arXiv:1401.2184, <https://arxiv.org/abs/1401.2184>), for
the DIMACS graphs of <https://mat.tepper.cmu.edu/COLOR/instances.html>. The
plan's working values were right; the A2 runs use them.
