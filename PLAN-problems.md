# Research directions: QAP, colouring, TSP and other notable problems

Review of the `records` round and the directions for the next one. The
implementation specs of the finished round are in `PLAN-records.md`; its
ground rules (branch, checks, trace discipline, "a beat is strictly below
the best known, verified independently") stay in force. Items here are
numbered to continue that file (Q5…, A3…), plus new series T (TSP) and P
(other problems).

## 0. Review of the records round

What the evidence supports:

* Fourteen open QAPLIB instances tied exactly (`QAPLIB.md`), including the
  first tie on the random "a" family (tai50a) and tho150 on the GPU. The
  tho150 permutation is the published assignment itself, so the structured
  families (sko, wil, tho, tai-b) are, for us, solved to the known values.
  No beat anywhere.
* The GPU kernels pay off: 10 M TabuCol it/s and a 21.6 s sko100a are
  30–60 sandbox cores' worth. `mismatches=0` on every run so far.
* Two clean negative results worth keeping: exhaustive 9-opt polish of a
  tabu local optimum finds nothing (robust tabu optima are deep; polish is
  not where the gain is), and large neighbourhood search loses to swaps on
  the QAP.

What it does not support yet:

* **Cohesive crossover has no controlled measurement.** It lost to uniform
  with `ma` on sko100a (5 seeds), and the tho150 success was one GPU run
  that also carried the symmetric kernel and the shared-memory triangle.
  The honest statement today is "tho150 was tied with cohesive crossover",
  not "cohesive crossover helped". Q5 fixes this.
* **Directed perturbation: one run per arm** (Q2). Fifteen-minute single
  runs on a stochastic search with a 756 s time-to-target cannot separate
  arms. Either rerun to protocol (§6) or drop the claim.
* **Island results flip sign between instances** (2× faster on sko100a,
  worse on sko90 and tai80b). That is the signature of too few seeds, not
  of an effect. Leave islands off and stop spending on them.
* **Colouring**: everything beyond DSJC250.5 is a tie attempt pending GPU
  time (A1.8, A2). No colouring result since the 48 on DSJC500.5 changes
  the position.

Rule going forward: a method earns a default only after a §6 measurement
on at least two instances from different families. Otherwise it stays a
flag, documented as "available, not better".

## 1. QAP

The structured families are tied; what is left is the random "a" family
(0.05–0.9%), tai256c (0.13%) and lipa90a (solved by `restarts` already).
Records on the QAP moved, in the last fifteen years, only on tai-a,
tai256c and on instances outside QAPLIB. Directions in priority order.

### Q5. Cohesive vs uniform on the GPU (ablation, cheap)

`qap_gpu` on sko100a (21.6 s to best known) and sko100f, 10 seeds per arm,
same `--population`, `--iters`, `--max-seconds 600`. Report reached/10,
median and max time to the best known. Then tho150, 5 seeds per arm,
`--max-seconds 3600`, best value at 10, 30, 60 min. If cohesive is not
faster on both sko instances, it is not the reason tho150 was tied, and the
default stays uniform. Cost: ~1 GPU-hour for sko, ~10 for tho150.

### Q6. tai-a: long single trajectories against populations

Hypothesis: tai-a instances have no shared structure for a child to
inherit, so the population buys nothing, and what the record setters
(Taillard's RoTS at 10⁷–10⁸ iterations; James, Rego & Glover's cooperative
parallel tabu; Misevičius's iterated tabu) really spend is iterations on a
few trajectories with long-term aspiration.

Experiment, `qap_gpu`, tai50a / tai60a / tai80a:
* arm A: as now (population 112, 100n–200n iterations per child, crossover);
* arm B: the same 112 members as *independent* robust tabu runs with no
  crossover, batches of 200n iterations, continuing each member's own
  trajectory (needs a `--no-crossover` or `--independent` flag: children
  are copies of their member; the kernel is unchanged);
* arm C: B plus a random-swap perturbation of k ∈ [n/10, n/4] every batch
  without improvement (an iterated RoTS per member).
Measure best value against wall time at 5, 15, 30, 60 min, 5 seeds. Targets
are the best knowns (4,938,796; 7,205,962; 13,499,184). Only if an arm
reaches tai80a's value inside an hour is tai100a worth hours. tai100a's
literature value is below the `.sln`'s 21,052,466 (verify the current one
and cite it before any run; the DESIGN note already flags this).

RoTS parameters matter more on tai-a than anywhere: tenure in
[0.9n, 1.1n] changed every 2.2n iterations, long-term aspiration at
2n²–5n². Expose both on `qap_gpu` and sweep the aspiration (2n², 5n², 10n²)
on tai50a first, 5 seeds each, since Taillard's own tai-a results use the
large values.

### Q7. tai256c on the GPU

n = 256: the delta triangle (131 KB in i32) does not fit shared memory, so
this is the first real use of the kernel's global-memory path. The flow
matrix is a grey-density pattern and the distance matrix a grid, so
cohesive crossover applies (`proximity_matrix` should accept it; check the
triangle-inequality sample passes). Best known 44,759,294 has been improved
several times since 2000, each by a few hundred units; it is the one
QAPLIB instance where a beat is realistic for a method of this class.

Steps: (1) `--cpu` run of the host loop on tai256c to confirm memory and
`proximity_matrix`; (2) GPU run at k = `--iters 100`, `--run-to-end`,
`--max-seconds 14400`, trace on; (3) if it ties, `--kopt 9` on the result
(cheap, and the only place polish has a chance, since the instance has
many zero flows). Verify any value below 44,759,294 by recount from the
`.dat` file and by a second, independent cost program before saying so.

### Q8. Beyond QAPLIB: where QAP records actually move

QAPLIB has been static for a decade. The instance sets where new values
are still published are Drezner's `dre` instances (hard for tabu by
construction, optimum known: a search-power test, not a record), Taillard's
`tai*e` instances and the microarray-placement instances of de Carvalho &
Rahmann (n up to ~1,000, best knowns set by memetic/tabu codes and still
moving). Add a reader for them to `examples/common/qap.rs` only after Q6
settles the tai-a method; the point is a frontier with real room, and
n ≈ 1,000 needs the global-memory kernel of Q7.

### Q9. Breakout local search as itself

Q2 tested directed perturbation *inside* `memetic`. Benlic & Hao's result
is the whole loop: descent, directed perturbation with a tabu on used
swaps, adaptive strength tied to the number of returns, and acceptance of
the perturbed optimum. Run it as `IteratedSearch(SwapDescent, Directed-
Perturbation)` on tai50a/tai60a (`qap_advanced --method restarts` is not
it; add `--method bls`). If it does not reach tai50a's value in 1 hour,
5 seeds, retire the direction. BLS claims the best knowns on tai50a–tai80a
in minutes to hours of 2013 CPU; a faithful port should at least tie
tai50a.

## 2. Graph colouring

### A3. Finish the pending GPU runs (calibration, not records)

A1.8 (DSJC500.5 at 47 with `--polish-every 200 --vss-rounds 4`) and A2 at
k = best known on DSJC1000.5, flat1000_76_0, C2000.5. Verify each k against
a current source (Hao's group's tables or the HEAD paper) before the run,
not from memory. A tie on DSJC1000.5 and C2000.5 within hours would show
the GPU population is at HEAD's level; nothing is attempted at k − 1 before
that.

### A4. 16-bit colours

k < 256 excludes C4000.5 (best known 259, from HEAD; verify), the instance
with the fewest serious attempts and so the best odds at k − 1. In the
global-memory path a 16-bit colour costs n·p·2 bytes more; the 24-bit move
index already allows n·k = 1.04 M. Do it after A3 shows a tie on C2000.5.

### A5. Choosing targets for k − 1

A k − 1 attempt is only sensible where the upper bound was set by a method
we replicate (HEAD, quantum annealing of Titiloye & Crispin) and the lower
bound leaves room. Candidates to check: DSJC1000.9 (222), latin_square_10
(97, lower bound 90), flat1000_76_0 (81; its planted colouring has 76), and
DSJC1000.5 (82). Rank them by (compute others spent) × (gap to the lower
bound). Record the ranking and the sources in DESIGN before starting.

### A6. Adjacent colouring problems (high plausibility, low cost)

The TabuCol/PartialCol/GPX stack carries over to problems whose best
knowns moved recently and that have fewer competitors than k-colouring:
* **Weighted vertex colouring** (minimise Σ over colours of the max vertex
  weight): instances from the pxx/rxx/DIMACS-W sets; best knowns moved in
  2021–2023 (Hao's group, and Goudet et al.'s learning-guided tabu). Cost
  = a sum over colour classes, move = recolour; the tabu machinery is the
  same, the objective is a different `Model`.
* **Minimum sum colouring** (Σ colours over vertices): many DIMACS values
  moved 2012–2020 (Jin & Hao; Sun et al.).
* **Bandwidth colouring** (GEOM instances, Lai & Hao).
Pick WVCP first: it is the one with the most recent movement and a direct
GPU batch (the kernel's conflict table becomes a per-class max-weight
table). Spec it in the style of A1 before coding.

## 3. TSP

Position: the library has no tour moves (the `invert_subsequence` problem of
METSlib, DESIGN §1 and §6). The TSP is the reference problem for
neighbourhood search and a crate for metaheuristics without 2-opt is
incomplete. But records are not available here: TSPLIB is solved to
optimality by Concorde, the DIMACS challenge values come from LKH and tour
merging, and the open instances (Bosch's Art TSP, mona-lisa100K at 0.002%
from its bound) belong to EAX/LKH-class codes with ejection chains and
genetic edge assembly. Direction T is library completeness and credibility,
with a capped budget, not a record attempt.

### T1. Tour representation and moves (library)

`src/tour.rs` (or `perm::tour`):
* `trait Tour: Model { fn as_slice; fn pos(city) -> usize; fn reverse(i, j); fn dist(a, b) -> Cost }`,
  array representation with a position index; `reverse` flips the shorter
  side so the move is O(min(len, n − len)).
* Moves `TwoOpt { i, j }` and `OrOpt { seg_start, len (1..=3), insert_after, reversed }`
  with deltas from four/six distances; `Move::apply` through `reverse`.
* `CandidateNeighborhood`: for each city its k nearest (k = 8–10, or
  quadrant neighbours), with don't-look bits kept in the neighbourhood
  (`Neighborhood::start` resets, and the search's committed move clears
  the bits of its endpoints; this needs a hook from `TabuSearch`/
  `LocalSearch` to the neighbourhood after a commit, which is a small,
  general addition: `Neighborhood::committed(&mut self, s, &m)`).
* `DoubleBridge` perturbation (segment-based, O(n) on the array).
* Tests: `check::deltas` on random Euclidean instances; a 2-opt descent
  from a random tour reaches the Johnson–McGeoch 4.9% ± over the Held–Karp
  bound on random uniform instances; `reverse` keeps `pos` consistent.

### T2. `examples/tsp.rs` on TSPLIB

EUC_2D and explicit-matrix instances, nearest-neighbour start, ILS with
2-opt + Or-opt and double-bridge (Martin, Otto & Felten), plus tabu on
2-opt for comparison. Targets, 5 seeds, 60 s per instance on one core:
within 1% of optimum on att532, pr1002, rl1889, pcb3038; within 2% on
fl3795 and fnl4461. Published ILS results (Stützle & Hoos) are the
reference; LKH is not a fair comparison and is not the goal. Stop here
unless a downstream problem (P1, P2) needs Or-opt at scale.

## 4. Other notable problems

Ranked by fit with the library (permutations and swap/flip neighbourhoods,
memetic, GPU batches), by whether records still move, and by how easily a
result is verified. "Fit" is weighted most: the library is the product and
the records are the test of it.

| | problem | benchmark | fit | records move? | verification |
|---|---|---|---|---|---|
| P1 | permutation flowshop, makespan | Taillard ta001–ta120 | permutation; insertion moves | yes, 200×20 and 500×20 upper bounds improved through the 2020s | trivial (recompute Cmax) |
| P2 | max-cut / QUBO | G-set (G1–G81), BQP | binary flip, O(1) deltas | yes, by single units on G55–G81 (BLS 2013, later tabu/memetic codes) | trivial |
| P3 | linear ordering | LOLIB, XLOLIB | permutation; insertion moves, O(n) delta | mostly settled by exact methods; XLOLIB n = 250 partly open | trivial |
| P4 | weighted / sum colouring | see A6 | existing colouring stack | yes, recent | trivial |
| P5 | graph partitioning | Walshaw archive | vertex moves with balance | yes; a curated public archive that accepts submissions | by the archive |
| P6 | job shop | Taillard ta41–ta50 open | needs disjunctive graph, N5/N6 moves | yes (2010s–2020s) | moderate |
| P7 | CVRP | CVRPLIB X instances | routes, many move kinds | yes, constantly, but HGS dominates | by CVRPLIB |
| P8 | max clique / vertex cover | DIMACS, BHOSLIB | binary | settled | trivial |

Recommendation: **P1 first, P2 second**, P4 alongside A6, P5 as a later
project, P6/P7 not now, P8 never.

### P1. Permutation flowshop (Taillard instances)

Why first: it is a `Permutation` (job order), so `Memetic`, `IteratedSearch`,
`RobustTabuSearch`'s structure and the GPU driver pattern all carry over;
the state of the art (iterated greedy, Ruiz & Stützle; and its descendants)
is simple; the 200×20 and 500×20 best knowns still move by a few units.

Build: `Instance` reader for Taillard's format; `Model` with makespan;
insertion neighbourhood with Taillard's acceleration (all n insertion
positions of one job in O(n·m) using forward and backward completion
tables), exposed as a `Neighborhood` that visits `Insert { job, to }`
with exact deltas; iterated greedy (destruct d = 4 jobs, NEH reinsertion,
local search, constant-temperature acceptance) as an `IteratedSearch`
whose perturbation is the destruct–reinsert; `examples/flowshop.rs`.
GPU later: one block per candidate sequence, threads over insertion
positions.

Measure: ta051–ta090 (50–100 jobs) relative percentage deviation from the
best knowns at 60 s, 5 seeds, against the published iterated-greedy
numbers; then ta091–ta120 for hours with `--run-to-end`. Any value below a
best known is recomputed from the instance by a separate script.

### P2. Max-cut on the G-set

A binary model (`Vec<bool>` with a gain vector, O(1) flip delta, O(deg)
update), tabu search with the standard tenure n/10 + random, and a memetic
layer with uniform crossover and a distance for diversity; everything
except the model exists. Instances n ≤ 20,000 fit a block's shared memory
for the gain vector (80 KB in i32), so the GPU runs one tabu trajectory per
block as `qap_gpu` does. Verify every value: a cut is a partition, the
recount is a sum over edges.

### P5. Graph partitioning (Walshaw's archive)

The one benchmark with a maintained public record table that accepts
improvements. State of the art is multilevel (coarsen, partition, refine
with tabu or evolutionary refinement). The refinement half fits the
library; the multilevel half is new infrastructure. Keep for after P1/P2
show the GPU memetic pattern transfers to a non-QAP problem.

## 5. Library work the experiments need

* `Neighborhood::committed(&mut self, s: &S, m: &M)` hook (for don't-look
  bits, T1; also lets P1 cache completion tables).
* Insertion moves for permutations (`perm::Insert`, O(n) apply through
  `rearrange` or a direct rotate), with a problem-supplied delta (P1, P3,
  Or-opt).
* A binary model helper (`bits::Flip`, `FlipNeighborhood`, gain-vector
  support) for P2.
* `qap_gpu --independent` (no crossover) and RoTS tenure/aspiration flags
  (Q6).
* A `results/` convention: every reported number has a log, a trace and a
  solution file under `results/<problem>/<instance>/`, and a `verify`
  example per problem that recomputes a cost from instance and solution
  with none of the solver's code.

## 6. Experiment protocol

Applies to every measurement from now on; a number that does not follow it
is reported as "exploratory" and does not change a default.

1. **Arms**: a baseline arm (current default) always runs beside the new
   one, on the same build, same budget, same hardware.
2. **Seeds**: at least 5 per arm (10 when the time to target is under a
   minute). Report reached count, median and max time to target, and the
   best value at fixed wall times for runs that do not reach it.
3. **Budget**: wall time per run, stated; iteration counts stated too, so
   GPU and CPU results are comparable in work as well as time.
4. **Targets**: the current literature best known with a citation, checked
   at the time of the run; the `.sln` or archive value is only a fallback.
5. **Decision rule stated before the run**: what result adopts the new
   default, what retires the direction (as in Q6, Q9).
6. **Artifacts**: log, trace (`build=`, `mismatches=0`), solution file,
   and the independent recount. A beat is announced only with all four and
   the citation of the value it beats.
7. **Negative results are written up** in DESIGN with the same care as
   positive ones (the k-opt polish and LNS notes are the model).

## 7. Order of work

1. Q5 (GPU ablation of cohesive crossover) and A3 (pending GPU runs):
   both are mostly GPU time, and both settle claims already in DESIGN.
2. Q7 tai256c: first use of the global-memory kernel, realistic beat.
3. Q6 tai-a arms on the GPU, with the RoTS parameter sweep first.
4. P1 flowshop: new problem class, maximal reuse, moving records.
5. T1/T2 TSP primitives and example, capped at the T2 targets.
6. A6/P4 weighted vertex colouring on the colouring stack.
7. P2 max-cut, then A4/A5 large-graph k − 1 attempts, then P5.
