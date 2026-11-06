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

Caveat on every time-to-target measured before build beb9bb6: a best
reached by a perturbation or a crossover was recorded only when the inner
scope closed and without an event, so the stop at the target could fire
late (found while fixing Q9: BLS reached tai30a's optimum at 3.5 s and
reported 11.3 s). Random perturbations make this rare, so the ties stand
and the earlier times are upper bounds, not re-measured. Time-to-target
comparisons from now on use builds at or after beb9bb6.

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

*Status.* Not started; needs the GPU. Nothing in the sandbox substitutes
for it (the `ma` comparison is a different improvement operator).

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

*Status (build 2720163, `reports/2026-11-06-q6-q9.md`).* Flags done:
`--independent`, `--perturb-stalled`, `--tenure LO,HI`, `--aspiration X`;
the kernel resumes each member with its tabu table and absolute iteration
stamps, the batch seed changes every launch, and the `--cpu` checks on
tai30a reach the optimum with clean recounts. GPU run pending. Review:
1. The CPU path ignores `resume` (a fresh tabu memory every batch), so
   arms B and C on `--cpu` are a different algorithm from the same arms on
   the GPU, and the trace recount cannot see it (it checks costs, not
   trajectories). Keep one `RobustTabuSearch` per member across batches
   on the CPU path, as the kernel does, before the GPU run: `--cpu` is the
   reference path by the ground rules.
2. The kernel's aspiration test `ti < it − asp` relies on the staggered
   "never" stamps `−q`; with `resume` and a long run, `it` keeps growing
   and that stays right. Add one `--cpu` check of 50 resumed batches on
   tai30a with `--trace` to confirm nothing drifts (costs recounted, best
   never worse than the previous batch).
3. `scripts/q6-grid.sh` runs the spec (3 instances × 3 arms × 5 seeds ×
   1 h = 45 GPU-hours on one card). Stage it: the aspiration sweep on
   tai50a first, then tai50a with all three arms at the winning
   aspiration, then tai60a and tai80a only with the arms that reached
   tai50a's best known on at least 3 of 5 seeds. An arm that cannot tie
   tai50a has nothing to say about tai80a. Results under `results/q6/`
   stay out of git (logs and traces); the summary
   (`scripts/q6-summary.py`) goes in a report.
4. Review items 1 and 2 done (build 1cd68a8: the CPU path resumes each
   member's tabu state; the loop asserts no member's best worsens across a
   batch; 2,546 resumed batches on tai30a clean).

*Sweep result (RTX 3060, build 2a5f085, tai50a, arm B, 112 trajectories,
200n per batch, 5 seeds, 30 minutes each; every trace clean, all 15
`.best` files recounted; `reports/2026-11-06-gpu-q6-sweep-and-a3.md`).*

| aspiration | best at 5 min (median) | reached | time to best known by seed (s) | median, max |
|---|---|---|---|---|
| 2n² | 4,953,480 | 3/5 | 493, 658, —, —, 246 | 493 s, 658 s |
| 5n² | 4,941,410 | 4/5 | 418, —, 389, 59, 575 | 403 s, 575 s |
| 10n² | 4,941,410 | 5/5 | 1,549, 235, 355, 484, 105 | 355 s, 1,549 s |

(—: ended at 4,941,410 after 1,800 s.) Every run that reached the best
known did so in under 26 minutes on the GPU, against one 4-core `memetic`
tie in the sandbox (756 s) and none of the `qap_gpu --cpu` runs.

Reading: the medians of the time to target are over the seeds that
reached it, so 493 s for 2n² is not "faster than 10n²"; the reach count
is the result, and it is monotone in the aspiration age, which is what
Taillard reports for tai-a. Three levels of five seeds each is thin for
any single pair, the trend is the evidence. All three arms sit at
4,941,410 at 5 minutes, the same deep attractor the CPU memetic stalled
on for 30 minutes: on tai50a the whole question is leaving that one
optimum, and independent long trajectories do it in a median of 6
minutes, 5 of 5.

*Second stage (build 5fafddd, same setup, 30 minutes, 5 seeds;
`reports/2026-11-06-q6-tai50a-arms.md`; all 30 tai50a `.best` files
recounted).*

| arm | aspiration | reached | time to best known by seed (s) | median, max |
|---|---|---|---|---|
| B, independent | 20n² | 5/5 | 12, 35, 25, 266, 1,027 | 35 s, 1,027 s |
| C, independent + perturbation of stalled members | 10n² | 5/5 | 71, 353, 402, 438, 185 | 353 s, 438 s |
| A, memetic (crossover, one-to-one) | 10n² | 4/5 | 291, 456, —, 384, 290 | 337 s, 456 s |

Reading. (1) The aspiration age is the strongest knob found so far on
tai-a: 20n² puts the median best at 5 minutes at the best known, and
three seeds tie in under 40 s. The long-term aspiration is a forced
diversification (put back an assignment unseen for the age); with 112
independent trajectories the population already diversifies, so forcing
less of it helps. If that explanation is right, the aspiration can go
much higher, or off, without loss, which is the next level. (2) A at
10n² is the only arm that lost a seed to the 4,941,410 attractor; among
the seeds that reached, its times are B's and C's. By the stated rule the
hypothesis holds at n = 50: B and C go on, A does not. (3) B and C at
10n² are not separable with five seeds (C's max 438 s against B's 1,549 s
is one seed).

Decisions:
* Aspiration 20n² for every tai-a arm from here; the GPU default stays
  Taillard's 5n² until a second instance agrees (§0 rule).
* Next level on tai50a, arm B, same budget: 40n² and effectively off
  (`ASPIRATIONS="40 1000"`: 1000n² is 2.5·10⁹ iterations, beyond a
  30-minute run). If off is as good as 20n², the aspiration is redundant
  with the population on tai-a and the arms on tai60a/tai80a run at off;
  if off loses seeds, the sweep has found its optimum between 20n² and
  40n² and the arms run at the better of the two.
* tai60a: B and C at the settled aspiration, 5 seeds, 30 minutes; a reach
  count under 3/5 for both repeats at 60 minutes before any conclusion.
  tai80a: 60 minutes from the start. Arm A is not run on either.
* One cheap check for the §0 default rule after tai60a: B at 20n² on
  sko100a, 5 seeds, 10 minutes, against the 21.6 s memetic tie. It says
  whether independence also wins where crossover was believed to help,
  which decides the `qap_gpu` default and bears on Q5.
* (Done, see the second stage above.) The 20n² level and arms A, C ran
  at 10n² on tai50a; A reached 4/5, so B and C go on.
* Each tie is checked with `scripts/verify_qap.py` before it is counted;
  the report carries the recounts, not only the summary table.

### Q7. tai256c on the GPU

n = 256: the delta triangle (261 KB in i64) does not fit shared memory, so
this is the first real use of the kernel's global-memory path (step 1
done: `reports/2026-11-06-q7-tai256c-cpu-check.md`, host loop clean,
0.5 MB per member on the device). Best known 44,759,294 has been improved
several times since 2000, each by a few hundred units; it is the one
QAPLIB instance where a beat is realistic for a method of this class.

Decision on the report's finding: the second matrix of tai256c is a
*similarity* (Taillard's grey density: entries ∝ 1/d² between cells of a
torus, larger meaning closer), so it fails the metric test by design and
`--crossover cohesive` was silently uniform. The cohesive crossover only
needs a ranking by closeness, so `Instance::proximity` now also accepts a
symmetric, zero-diagonal, positive matrix whose `1/√s` satisfies the
triangle inequality on the sample (true for s = 1/dᵖ, p ≤ 2, with d a
metric; collinear triples are ties, compared with a 10⁻⁶ tolerance), and
ranks by decreasing s, over elements when it is the second matrix. Both
examples print which matrix and transform the crossover uses; a run whose
log does not say "cohesive crossover on the second matrix, a similarity"
is a uniform run. Unit-tested on a synthetic 8×8 grey-density instance.
`distance()` is unchanged, so the k-opt groups stay random on tai256c.

Steps: (1) done; rerun `--cpu` once to see the new crossover line in the
log; (2) GPU run at `--iters 100`, `--run-to-end`,
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

*Status (builds 839cc59, beb9bb6; `reports/2026-11-06-q6-q9.md`).*
`--method bls` done; the run was restarted on beb9bb6 (stop-at-target
bug, see §0) with 5 seeds on both arms, single-threaded, 1 hour each;
BLS prints cycles and cycles per second (8,700–8,900/s on tai30a); the Q2 perturbation cycled under `Always` acceptance (strength and
directed share only moved on an exact return to the same cost) and was
rewritten to the BLS rule: directed probability max(e^(−ω/T), P₀), ω the
perturbations since the last new best, strength from L₀ = 0.15n, +1 per
return, up to n/2, a strong random perturbation of n/2 swaps after T
without a new best. Run in progress (5 seeds, single-threaded, 1 hour).
Consequences and review:
1. The Q2 numbers in `reports/2026-11-05-records-round.md` describe the
   old rule and are void; Q9 replaces Q2 entirely.
2. The baseline arm (`memetic`, single-threaded, same budget) has 5
   seeds since the restart. The 756 s tai50a tie is a 4-thread result and
   is not the baseline for a single-threaded comparison.
3. T = 2,500, P₀ = 0.75, L₀ = 0.15n, L_max = n/2 match my own recollection
   of Benlic & Hao (C&OR 40, 2013, 192–200); the perturbation tenure (here
   n) I am not sure of. Two known deviations: the local search is
   `SwapDescent` (first improvement, don't-look bits) where BLS uses a
   best-improvement descent, and the directed move's aspiration is the
   best cost of the current perturbation where BLS uses the best found so
   far. If the run is marginal (tai50a reached on some seeds, not all),
   confirm the parameters against the paper before deciding; if it misses
   on every seed, the deviations do not explain a miss of that size and
   the direction is retired as planned.
4. `--steps` defaults to 10⁸ cycles for `bls`, so the stop is the wall
   clock: report cycles per second beside the result, as the paper's
   budgets are in cycles.

*Result: retired (`reports/2026-11-06-q9-bls-tai50a.md`).* BLS on tai50a,
single-threaded, 1 hour, 5 seeds: 0/5, gaps 0.24–0.47% (median 0.45%),
about 5.1 M cycles per run. The baseline `memetic` under the same budget
ties on 2 of 4 seeds so far (840 s, 3,065 s) and its misses sit at
4,941,410 (0.05%). A miss of that size on every seed is not explained by
the two known deviations or by the recalled parameters, so the direction
is closed without checking them; `--method bls` and the BLS-rule
perturbation stay in the code as documented negatives. What the two
measurements say together: on tai-a the robust tabu search with a long
aspiration is the engine, and the perturbation style around it matters
little (Q2, Q9) against the aspiration age (Q6).

### Q10. The aspiration effect on the CPU: a test of the explanation

The Q6 reading says the long aspiration helps *because* a population of
112 trajectories already diversifies. That predicts a smaller gain on a
single trajectory or a population of 10. The alternative is simpler:
Taillard's 5n² is just too short for tai-a at any population size. The
two are separated on the CPU, without the GPU:
* `qap_advanced`: expose `--tenure LO,HI` and `--aspiration X` for the
  methods that use `RobustTabuSearch` (`memetic`, `restarts`); the fields
  are public already. Also make it save its best permutation as
  `NAME.best` through `common::fresh_path`, as `qap_gpu` does: the Q9
  baseline ties could not be recounted, which §6.6 requires.
* Run: `memetic`, single-threaded, 1 hour, seeds 1–5, aspiration 5n²
  (the Q9 baseline, 2/4 so far; its seed 5 completes it) against 20n²
  and off (1000). The flags, the saved `.best` and the runner
  (`scripts/q10-cpu.sh`, `q10-summary.py`) are done (50efeb5, 1168797);
  the runner targets the owner's machine with one core left for the GPU
  host, which is fine: the arms compare within one machine. Then `restarts` (one trajectory at a time) at 5n² and
  20n², same budget.
* Decision: if 20n² lifts the single-threaded memetic from 2/5-ish to
  5/5 with a median under 600 s, the explanation is wrong and the longer
  aspiration becomes the default of both examples for all instances after
  one structured instance agrees (sko100a, §0 rule); if the CPU gain is
  small while the GPU's was 10×, the population is doing the
  diversifying and the aspiration should scale with the population, not
  with n alone.

## 2. Graph colouring

### A3. Finish the pending GPU runs (calibration, not records)

A1.8 (DSJC500.5 at 47 with `--polish-every 200 --vss-rounds 4`) and A2 at
k = best known on DSJC1000.5, flat1000_76_0, C2000.5. Verify each k against
a current source (Hao's group's tables or the HEAD paper) before the run,
not from memory. A tie on DSJC1000.5 and C2000.5 within hours would show
the GPU population is at HEAD's level; nothing is attempted at k − 1 before
that.

*A1.8 done, negative (RTX 3060, build 26c75d3, 4 hours, 12,415 clean
batches; `reports/2026-11-06-gpu-q6-sweep-and-a3.md`).* 1 conflict at
1,888 s and still 1 at the cap; the first run without polish had reached
1 conflict at 1,460 s. Sixty polishes (500,000 GPU iterations on copies of
the best, then 4 VSS rounds on the CPU), 52 of them from the 1-conflict
best, removed nothing, at 4% of the run time. The last batch's population
had a minimum of 4 conflicts and a median of 21: the population is
diverse, and the 1-conflict solution is not where it lives.

Decisions:
* DSJC500.5 at 47 is retired. Two runs reach the last conflict in about
  half an hour and never clear it; 47 is believed optimal, so a tie was
  the ceiling, and 8 GPU-hours bought the same plateau twice.
* A1's hypothesis (PartialCol leaves the plateau TabuCol is stuck on) is
  not supported at this scale: VSS helped from 8 conflicts on the CPU and
  did nothing from 1 on the GPU. VSS and polish stay available and off;
  the DESIGN entry says so.
* The calibration moves to A2: DSJC1000.5 and flat1000_76_0 at k = best
  known, defaults (no polish, no VSS), `--max-seconds 14400`, one run
  each, traces on. Verify both k before the run. A tie there is the
  signal that the population is at HEAD's level; two misses at 1–2
  conflicts would say the limit is the method (HEAD's tie-breaking,
  iteration budget per child) and not the hardware, and A4/A5 wait.

*A2 preparation done (`reports/2026-11-06-a2-preparation.md`).* Host loop
clean on both graphs at k = 82 and 81; shared tables would need 163 KB,
so the kernel takes its global-memory path, which has never run on the
GPU and was 9× slower per batch on DSJC500.5. Directions:
1. k values confirmed by the owner from the HEAD paper (Moalic &
   Gondran, "Variations on Memetic Algorithms for Graph Coloring
   Problems", arXiv:1401.2184, and the DIMACS instance table at
   mat.tepper.cmu.edu/COLOR/instances.html): DSJC1000.5 at k = 82,
   flat1000_76_0 at k = 81. These are the upper bounds HEAD reached; a
   legal colouring at these k is a tie, at k − 1 a beat (flat1000_76_0's
   planted colouring has 76, so 80 would be a tie with nothing known and
   still above the construction).
2. First GPU run is a 10-minute smoke test on DSJC1000.5 at 82: the
   start line (global tables, occupancy, device memory ≈ 109 MB),
   `mismatches=0` on every batch, and the batch time. Only then the two
   4-hour runs. If the rate is near the 1.2 M it/s of the first
   DSJC500.5 kernel, 4 hours is a few hundred generations and the runs
   are a floor, not a verdict.
3. **A2.6, kernel: 8-bit conflict table when 16-bit tables overflow
   shared memory.** γ[v][c] counts v's neighbours of colour c; on these
   graphs it is around 6 and never near 255 in a near-legal colouring.
   An 8-bit table is n·k = 82 KB, and with the 8-bit colouring and the
   conflict list it fits the 99 KB opt-in at one block per SM. Saturate
   at 255 and set a per-block overflow flag that the host reads with the
   batch (the recount already catches any wrong cost). Measure global
   against 8-bit shared on DSJC1000.5 at 82 (batch time, it/s) before
   the 4-hour runs if the smoke test shows the global path under 3 M
   it/s. This is the same move that took DSJC500.5 from 1.2 to 4.1 M.

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
| P7 | CVRP and variants | CMT, Golden, CVRPLIB X and XL | routes; relocation moves, attribute tabu | yes: X values last moved 2021, the XL set (1,000–10,000 customers, Jan 2026) moves now | by CVRPLIB |
| P8 | max clique / vertex cover | DIMACS, BHOSLIB | binary | settled | trivial |

Recommendation: **P7 (CVRP) and P1 first**, P2 second, P4 alongside A6,
P5 as a later project, P6 not now, P8 never. P7 is in because the library
descends from it: METSlib was written for the Cordeau & Maischberger
parallel iterated tabu search, so the port is both a validation of the
crate and the owner's own method brought to current hardware.

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

### P7. CVRP: the Cordeau & Maischberger parallel iterated tabu search

**Source.** Cordeau & Maischberger, "A parallel iterated tabu search
heuristic for vehicle routing problems", Computers & Operations Research
39(9), 2012, 2033–2050. It was coded on COIN-OR METSlib, so this is the
crate's ancestor brought back: the port is the validation of `mets` on
the problem it was designed for. The method below is taken from the
paper's §2–3; nothing is left to check.

**The method (sequential, §2).**
* *Routes and insertion.* Problems without time windows use GENI
  insertion and removal (Gendreau, Hertz & Laporte) with a neighbourhood
  of size q around the customer; with time windows, plain best insertion
  between two consecutive customers and removal by reconnection, with the
  route's duration minimised by forward time slack.
* *Initial solution.* Sweep: customers sorted by angle around the depot,
  inserted one by one into the current route, a new route opened when
  capacity or duration would be violated while vehicles remain; so the
  first m − 1 routes are feasible.
* *Objective.* f(s) = c(s) + α·q(s) + β·d(s) + γ·t(s): cost plus capacity,
  duration and time-window violations. α, β, γ start at 1; at the end of
  every tabu iteration each is multiplied by 1 + δ if its constraint is
  violated, divided by 1 + δ otherwise. δ is drawn uniformly in [0, 1] at
  the start of each improvement phase.
* *Attributes and moves.* A solution is its attribute set B(s) of triples
  (i, k, l): customer i served by vehicle k on day l. A move (R, R′)
  replaces attributes R by R′: either relocate a customer to another route
  on the same day (GENI insertion at least cost in g), or change its visit
  combination (remove it from the days dropped, insert it on the days
  added). Both neighbourhoods are explored in full; the best non-tabu move
  by g(s′) is taken.
* *Tabu and aspiration.* After (R, R′), the attributes in R \ R′ (what the
  customer left) are tabu for τ iterations; τ is drawn uniformly from the
  integers [0, ⌈√(n·m·t)⌉] at the start of each phase. Aspiration: a tabu
  move is allowed if f(s′) is below the best cost ever seen for at least
  one attribute of B(s′).
* *Diversification.* At a local minimum, non-improving moves are scored
  g(s′) = f(s′)·(1 + ζ·Σ_{(i,k,l)∈R′\R} r_ikl / λ), with r_ikl the number
  of times the attribute has been added since the start, λ the current
  iteration number, ζ uniform in [0, 1] per phase.
* *Route refinement.* Every 200 tabu iterations, each route is rebuilt by
  removing and reinserting each of its customers in turn (GENI with q
  doubled; with time windows, in random customer order).
* *Tabu stop.* A phase ends when s* has not improved for
  μ = √((η − λ)·π) consecutive iterations: long phases early, short ones
  late, longer after a big perturbation. η is the total tabu-iteration
  budget of the whole run (10⁵ or 10⁶ in the paper), π the current
  perturbation size.
* *Perturbation* (cluster removal, after Pisinger & Ropke): a random seed
  customer and its π closest customers, π uniform in [0, ⌈√n⌉], are
  removed; each gets a random visit combination from C_i; they are
  reinserted one by one in random order at least cost c(s).
* *Acceptance.* Any improvement replaces s*. The phase's final working
  solution s̃ (the last visited, not the best) is kept with probability
  1 − (λ/η)², otherwise the search returns to s*: broad early, converging
  late.
* *Stop.* η tabu iterations in total, whatever the number of phases.

**The parallel scheme (§3; pC/KS/MPDS in Crainic's taxonomy).** N
processes, each with its own seed, initial solution and its own δ, ζ, τ;
π and μ are synchronised so phases end together. At the end of a phase
process p keeps s̃_p with probability 1 − (λ/η)², else restarts from the
j-th best s* over all processes with j = ⌊√p⌋ (also tried: p − 1, p/2,
√N − √p; √p balanced breadth and depth best). Before perturbing, with
probability 0.1 a process applies a crossover: the cluster removal with
the removed customers' visit combinations copied from another process's
solution chosen uniformly, cluster size in [0, 4⌈√n⌉]. Boost::MPI; 8 and
64 processes at 125,000 iterations each.

**Results to reproduce (Table 1, Table 9; one process unless stated).**

| set | ITS avg, η = 10⁵ | ITS avg, η = 10⁶ | ITS best of 10, η = 10⁶ | NB09 avg / best | time ITS 10⁵ / 10⁶ |
|---|---|---|---|---|---|
| CMT (14) | 0.60% | 0.27% | 0.07% | 0.03% / 0.00% | 188 s / 1,877 s |
| Golden (20) | 1.30% | 0.79% | 0.41% | 0.12% / 0.02% | (same runs) |

Times on a 2.93 GHz Xeon X7350, 10 runs per instance. On the variants
the parallel version found new best knowns (PVRP 3 new + 24 ties of 42;
SDVRP 16 new + 17 ties of 35; several on PVRPTW, MDVRPTW, SDVRPTW) and
the gap on pr05 falls from 0.74% at N = 8 to ≈ 0.05% at N = 80 at a
fixed 125,000 iterations per process (Fig. 2), with time growing only
about 25% per tripling of N (Fig. 3). That curve is the research
argument of this direction: the paper stopped at 80 processes for lack
of hardware, and a GPU offers a thousand.

**Mapping to `mets`.**
1. `examples/common/vrp.rs`: `.vrp` reader (CVRPLIB/TSPLIB format: the
   X, XL, Golden and CMT sets all come in it, with `.sol`), with the
   instance's distance convention (CMT/Golden real-valued Euclidean, X
   rounded to integers: get this right first). `Routes` model: per-route
   sequence, load, length; a position index; α, β (γ later) *inside the
   model* so `evaluate()` is f(s); `Cost = f64`.
2. Insertion. Start with the paper's time-window mechanism (best insertion
   between consecutive customers, removal by reconnection) on all
   problems, and the route refinement as an Or-opt/2-opt pass (T1 moves):
   it is the paper's own simpler variant, and it decides whether GENI is
   needed. GENI (types I and II, neighbourhood q) is the second step, done
   only if the CMT gaps stay above the table at equal η.
3. `Relocate { customer, route, position }` as a `Move` with an O(1) delta
   (three arcs plus the two routes' load and duration terms);
   `RelocateNeighborhood` visits, per customer, the best position in every
   other route; a granular option (k nearest neighbours) for n ≥ 1,000.
4. Tabu status: `RecencyTabu` keyed by `TabuAttr::key = (i, old route)`
   (the attribute in R \ R′) is exactly the paper's rule. The tenure
   redraw per phase goes in `TabuList::start`.
5. Aspiration: a custom `Aspiration` holding `best_with[(i, k)]`, the
   best f over visited solutions containing the attribute; `allows` tests
   `new_cost < best_with` over the attributes of s′ (the moved customer's
   new attribute suffices, the others are unchanged by the move and
   already in B(s)). The table is updated after each move by the per-step
   hook below.
6. **Library addition: `TabuSearch::with_score(|st, m, delta| score)`.**
   g(s′) changes which move is chosen, not the cost that is applied;
   today a neighbourhood can only report the true delta or corrupt the
   cache. The closure sees f(s′) = cost + delta and keeps r_ikl and λ in
   its state. General: frequency diversification and guided local search.
7. **Library addition: `Adjusting<I, F>`**, a `Search` that runs
   `inner.step()` then `f(&mut st)`: the α/β update every iteration (then
   `Evaluated::resync`), the aspiration table update, the route
   refinement every 200 iterations, and the frequency counts. It also
   closes DESIGN §6's reactive-tabu point.
8. The ILS loop is written in the example, not with `IteratedSearch`: its
   acceptance, stop rule and parallel synchronisation all read λ, η and π.
   Per phase: draw δ, ζ, τ; `TabuSearch` runner with `stop_when(no
   improvement for μ)`; acceptance by 1 − (λ/η)²; cluster perturbation
   (`Perturbation` impl, reusable). Parallel: `std::thread::scope`, one
   process per thread, a barrier at phase end, the sorted s* list and the
   ⌊√p⌋ rule, crossover with probability 0.1.
9. `examples/cvrp.rs` flags: `--eta`, `--jobs N`, `--insertion
   best|geni`, `--granular K`, `--max-seconds`; writes `NAME.sol` in
   CVRPLIB's format. `examples/verify_vrp.rs` recomputes cost and
   feasibility of a `.sol` from the `.vrp` with none of the solver's code.

**Measure.** (1) CMT 1–14 and Golden 1–20, η = 10⁵ then 10⁶, 10 runs: the
port is accepted when its average gaps are at or below Table 9's
(0.60/1.30% and 0.27/0.79%) at equal η; iteration counts make this
machine-independent, and wall time per 10⁵ iterations is reported beside
the paper's 188 s. (2) The parallel scheme at N = 4 (sandbox) and the
pr05 curve of Fig. 2 at N = 8 on the user's machine. (3) X instances,
n ≤ 500, against the CVRPLIB best known and the published HGS gaps:
expect to lose; record by how much and where.

*Status (builds 5abacba–e6ceac1; `reports/2026-11-06-p7-steps-1-7.md`,
`reports/2026-11-06-p7-its-first-results.md`).* Steps 1–8 done except
GENI: reader, `Routes` with penalties inside, `Relocate`, attribute tabu,
attribute aspiration, frequency score through `TabuSearch::with_score`,
the per-iteration bookkeeping through `Adjusting`, the phase loop, the
cluster perturbation, the 1 − (λ/η)² acceptance, `NAME.sol` output, and
`scripts/verify_vrp.py` in Python (a second language, independent by
construction, which is better than the plan's Rust verifier). All 212
received solutions and the 34 CMT/Golden reference solutions recount in
both programs; CMT and Golden were fetched from GitHub mirrors, so the
owner need not supply them. The parallel scheme (step 8's threads,
barrier, ⌊√p⌋ rule, crossover) is not built yet.

First results, η = 10⁵, one seed per instance (exploratory by §6), route
refinement by 2-opt and Or-opt:

| set | ours | paper, Table 9, ITS/1 at η = 10⁵ |
|---|---|---|
| CMT 1–14 | 0.44% (7 best knowns) | 0.60% |
| Golden 1–20 | 2.10% | 1.30% |

Our run is 20× faster per iteration than the paper's (no GENI), so time
is not comparable; iterations are the unit, as planned. Decisions:
1. CMT: lock the acceptance now with 10 seeds at η = 10⁵ (about 15 CPU
   minutes) and 10 seeds at 10⁶; passes if the means are at or below
   0.60% and 0.27%. Then the DESIGN entry for the CVRP and the two
   library additions (§3 for the API, §4 for the numbers).
2. Golden, before GENI, two cheap diagnostics: (a) η = 10⁶, one seed
   (the paper's mean falls from 1.30% to 0.79%; if ours falls by a
   similar ratio, the deficit is a constant factor of the neighbourhood,
   which is GENI's; if it does not fall, the phase dynamics are wrong
   somewhere and GENI would hide it); (b) the fleet: the default is the
   reference solution's routes plus one spare vehicle, a choice the paper
   does not state in the pages read; run Golden 1–8 with `--vehicles` at
   the reference count, since a spare vehicle changes what the duration
   limit costs. Both together are under 3 CPU-hours.
3. GENI (types I and II, neighbourhood q as the paper) only if (2a) shows
   the constant-factor pattern; then Golden with 10 seeds at 10⁵ is the
   acceptance run (at or below 1.30%).
4. The parallel scheme follows the acceptance, and is measured on pr05 as
   the paper does (Fig. 2) before any GPU work.

**Research direction: Fig. 2 at N = 1,000.** The parallel scheme is the
`qap_gpu` pattern exactly: one block per process, a phase of tabu
iterations as the batch, acceptance, ranking and crossover on the host.
Costs per block: the routes (n bytes), loads and lengths, a tabu table
over (i, k) (n·m entries of 32 bits: 2 MB for n = 1,000, m = 100 in
global memory), and the relocation scan parallel over customers with
granular neighbour lists in shared memory. Measure on pr05 (the paper's
curve), then on the X set, then on the XL set (1,000–10,000 customers,
January 2026), the one CVRP benchmark where best knowns still move and
where a thousand cooperating processes is a new regime, not a rerun.
Submit any new value through CVRPLIB, which verifies independently.

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
* `TabuSearch::with_score` (selection score distinct from the applied
  delta: frequency diversification, guided local search) and `Adjusting`
  (a per-step hook with `&mut State`: self-adjusting penalties, reactive
  tenure), both from P7 and both general. Done (4df6a55): a `Score`
  trait whose `IS_DELTA` turns off neighbourhood pruning when a score is
  in use, the real delta still applied; `Adjusting { inner, adjust }`
  with the caveat that bests recorded under old penalty weights keep
  their old costs, so the example tracks its best feasible solution
  itself. Reviewed; nothing to change.
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

GPU queue (one run at a time on the RTX 3060): Q6 tai50a, B at 40n² and
off (5 h) → A2 smoke test on DSJC1000.5 (10 min), then the 8-bit table
comparison if needed → Q6 tai60a, B and C (5 h), then the sko100a check
(1 h) → A2 calibration runs, 4 h each → Q7 tai256c, cohesive, 4 h → Q6
tai80a, B and C at 60 minutes (10 h) → Q5.

Sandbox queue while the GPU runs, in order:
1. Q9 done (retired); the baseline's seed 5 completes the 5n² arm of Q10.
   Q10: `qap_advanced --tenure/--aspiration`, solutions saved and
   recounted, the memetic arms at 20n² and off, then `restarts`.
2. A2 preparation (done, report in); A2.6 (8-bit conflict table) only if
   the GPU smoke test asks for it.
   A2 preparation, as it was: `coloring_gpu --cpu` host-loop checks on DSJC1000.5 at
   k = 82 and flat1000_76_0 at k = 81 (memory and occupancy in the start
   line, the 24-bit move index path, a few generations with `--trace`),
   and the two k values confirmed from a source the owner provides (the
   sandbox cannot reach the literature). No GPU run before both are done.
3. P7: steps 1–8 and the library additions are done. Next, in order:
   the CMT acceptance with 10 seeds at 10⁵ and 10⁶; the two Golden
   diagnostics (η = 10⁶, fleet without the spare vehicle); GENI if the
   diagnostics say so; the DESIGN entry.
4. Q10 runs on the owner's machine; its report is read like the others.

1. Q5 (GPU ablation of cohesive crossover) and A3 (pending GPU runs):
   both are mostly GPU time, and both settle claims already in DESIGN.
2. Q7 tai256c: first use of the global-memory kernel, realistic beat.
3. Q6 tai-a arms on the GPU, with the RoTS parameter sweep first.
4. P7 CVRP: port of the Cordeau & Maischberger heuristic (best insertion
   first, GENI if needed), with the two library additions it needs;
   acceptance test = Table 9's CMT and Golden gaps at equal η.
5. P1 flowshop: new problem class, maximal reuse, moving records.
6. T1/T2 TSP primitives and example, capped at the T2 targets (Or-opt
   also serves P7's intra-route improvement).
7. A6/P4 weighted vertex colouring on the colouring stack.
8. P2 max-cut, then A4/A5 large-graph k − 1 attempts, P7 on the GPU for
   the XL set, then P5.
