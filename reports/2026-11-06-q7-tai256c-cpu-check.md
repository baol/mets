# Q7 step 1: tai256c on the CPU host loop

Build `f630430` (sandbox, 4 cores). Command:
`qap_gpu tai256c.dat tai256c.sln --cpu --crossover cohesive --iters 20
--max-seconds 120 --trace tai256c-cpu.trace`.

## Host loop and memory

* Runs: 57 batches, every one `mismatches=0 bad_permutations=0`.
* Best after 121 s (population 4, 20n = 5,120 iterations per child):
  44,867,992, 0.243% above the best known 44,759,294. Not a measurement of
  the method, only of the loop.
* Device memory on the GPU path (from the formulas in `qap_gpu`): the delta
  triangle at n = 256 is 32,640 entries × 8 bytes = 261 KB per member (i64),
  which does not fit shared memory, so the kernel takes its global-memory
  path; with the tabu table (4n² = 262 KB) that is about 0.5 MB per member,
  56 MB for 112 members.

## Distance detection: cohesive crossover does not apply as implemented

`Instance::distance` accepts a matrix as a distance if it is symmetric with a
zero diagonal and the triangle inequality holds on at least 99% of 1,000
sampled triples. On tai256c:

| matrix | zero diagonal | symmetric | triangle inequality | distinct values | max |
|---|---|---|---|---|---|
| first (a) | no | yes | 91.2% | 2 | 1 |
| second (b) | yes | yes | 76.7% | 42 | 100,000 |

Neither passes, so `--crossover cohesive` falls back to uniform on tai256c
(the run printed the note). The first matrix is the 0/1 grey-density flow
pattern; the second is symmetric with a zero diagonal but is not a metric
(more than a fifth of the sampled triples violate the triangle inequality).
`PLAN-problems.md` Q7 expects cohesive crossover to apply here; it would need
a proximity that is not a metric, for example ranking positions by the
second matrix directly without the triangle test. Not changed: a decision
for the plan's owner.
