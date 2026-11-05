# Plan: record attempts on QAP and graph colouring

Implementation spec for the next round of work on the `records` branch. Each
item says why, what to build (files, types, algorithms), how to test it, and
how to measure it. Items are independent unless stated; the suggested order is
A1, Q1, Q3, Q2, A2, Q4.

## 0. Ground rules

* Work on branch `records`. Commits are authored by Mirko Maischberger
  (`user.name`/`user.email` as configured); the tool may be the committer
  only (`GIT_COMMITTER_NAME=Claude GIT_COMMITTER_EMAIL=noreply@anthropic.com`).
  No `Co-Authored-By` lines.
* Before every push: `cargo fmt`, `cargo clippy --all-targets`,
  `cargo clippy --all-targets --features gpu`, `cargo test`. The GPU kernels
  cannot run in the sandbox (no GPU): compile them with `--features gpu`,
  test the host loops with `--cpu`, and keep the `--trace` recounts, which are
  how kernel bugs show up on the user's machine (`mismatches` must stay 0).
* The GPU examples log the build (`METS_GIT_HASH`, from `build.rs`) and, in
  the trace, the batch time (`improve_s`), cost spread and recounts. Keep
  that: it is how results from the user's RTX 3060 are read.
* Library first, examples thin: anything generic (a search space, a
  crossover, a perturbation) goes in `src/`, with a unit test; the examples
  wire it up behind a flag.
* A "beat" is a value strictly below the best known. QAPLIB's `.sln` files
  are outdated (tai100a's published record is below the file's 21,052,466),
  so before claiming anything, check the current literature value and keep
  the solution file (`NAME.K.sol`, `NAME.best`) for independent verification.

Current position (sandbox, 4 cores, plus the user's RTX 3060):

| problem | instance | ours | best known | notes |
|---|---|---|---|---|
| QAP | tho150 | 8,133,702 | 8,133,398 | 0.004%; structured (grid distances) |
| QAP | tai100a | 21,118,120 | 21,052,466 (file; literature lower) | random "a" family |
| QAP | tai50a / tai80a | 4,941,410 / — | 4,938,796 / 13,499,184 | 0.05% / 0.9% |
| QAP | 12 open instances | tie | tie | tai80b, sko81–100f, wil50/100, tai150b |
| colouring | DSJC500.5 | 48 (47 reached 1 conflict on GPU) | 47 | 47 is believed optimal; 46 is not a target |
| colouring | large DIMACS | not attempted | moved in the last decade | the realistic colouring frontier |

## A1. Colouring: PartialCol search space and Variable Space Search (done)

Implemented as specified (see DESIGN.md, "Variable Space Search"): at
equal iterations VSS beat TabuCol alone on 5 of 8 stuck DSJC500.5 k = 48
solutions and tied on 3. Remaining: the GPU run of A1.8.


**Why.** The 47-colour run on DSJC500.5 (GPU, 112 HEAD pairs, 7.7 M TabuCol
it/s) reached 1 conflict after 25 minutes and stayed there for 35 more.
TabuCol's space (all vertices coloured, minimise conflicting edges) has
plateaus where the last conflict is only shuffled around. PartialCol
(Blöchliger & Zufferey 2008) searches legal *partial* colourings
(minimise uncoloured vertices): a different landscape with different local
optima. Hertz, Plumettaz & Zufferey's Variable Space Search (2008) alternates
the two spaces and was one of the methods that reached the hard DIMACS
values. A 1-conflict TabuCol solution converts to PartialCol for free.

**Where.** `examples/common/coloring.rs` (shared by `coloring` and
`coloring_gpu`); drivers in both examples. Nothing in `src/` is needed: the
library's `TabuSearch`, `DynamicTabu`, `Visitor::can_skip` and
`State::replace_current` already fit.

### A1.1 The model `PartialColoring`

```rust
pub const UNCOLORED: usize = usize::MAX;

#[derive(Clone)]
pub struct PartialColoring<'g> {
    pub g: &'g Graph,
    pub k: usize,
    pub color: Vec<usize>,      // UNCOLORED or 0..k; coloured vertices never conflict
    pub gamma: Vec<u32>,        // gamma[v*k + c] = coloured neighbours of v with colour c
    pub low: Vec<u32>,          // min over c of gamma[v][c], and how many colours reach it
    pub at_low: Vec<u32>,
    pub list: Vec<u32>,         // uncoloured vertices, any order, plus pos[] as in Coloring
    pub pos: Vec<u32>,
}
impl Model for PartialColoring<'_> { type Cost = i64; fn evaluate(&self) -> i64 { uncoloured count } }
```

Invariant: every coloured vertex has `gamma[v][color[v]] == 0` (the
colouring is legal). `rebuild()` recomputes everything from `color` (same
pattern as `Coloring::rebuild`, but only coloured neighbours count).

### A1.2 The move `Assign { v, c }`

Colour the uncoloured vertex `v` with `c`, uncolouring every neighbour `u`
with `color[u] == c`. Delta (in uncoloured vertices) = `gamma[v][c] − 1`.

`apply`:
1. for each neighbour `u` of `v` with `color[u] == c`: set `color[u] = UNCOLORED`,
   add `u` to the list, and for each neighbour `w` of `u`: `gamma[w][c] -= 1`
   (maintain `low/at_low` as `Recolor::apply` does: a decrement may create a
   new minimum);
2. `color[v] = c`, remove `v` from the list, and for each neighbour `w` of `v`:
   `gamma[w][c] += 1` (an increment may leave the minimum: rescan if
   `at_low` hits 0).

`Move::delta` returns `Some(gamma(v, c) as i64 − 1)`.

### A1.3 Tabu status: a custom `TabuList`

The paper's rule: after `Assign{v, c}`, every vertex it uncoloured may not
be given colour `c` again for the tenure. That is several keys per move, so
`TabuAttr` (one key) does not fit; implement `TabuList` directly:

```rust
pub struct PartialTabu { until: Vec<u64>, rng: DefaultRng }   // until[v*k + c]
impl<'g> TabuList<PartialColoring<'g>, Assign> for PartialTabu {
    fn is_tabu(&self, s, m, it) -> bool { self.until[m.v * s.k + m.c] > it }
    fn make_tabu(&mut self, before, m, it) {
        let tenure = rng.random_range(0..10) + (0.6 * before.list.len() as f64) as u64;
        for u in neighbours of m.v with before.color[u] == m.c { until[u*k + m.c] = it + tenure }
    }
    fn start(&mut self, s, rng) { resize/clear until; reseed rng from the run's rng }
}
```

Start from the TabuCol tenure formula (random 0..10 + 0.6 × uncoloured);
the paper's reactive tenure is a later tuning knob. Aspiration: the default
`BestEver` of `TabuSearch` (allow a tabu move that reaches a new scope best).

### A1.4 The neighbourhood `UncoloredVertices`

Like `ConflictingVertices`: iterate `s.list`; per vertex the lower bound on
its deltas is `low[v] as i64 − 1`, so `if vis.can_skip(low[v] − 1) { continue }`;
visit `Assign{v, c}` with delta `gamma[v][c] − 1` for every `c` not skipped by
`can_skip(delta)`. Use `.random_ties()` on the `TabuSearch`, as `tabucol()` does.

Builder next to `tabucol()`:

```rust
pub fn partialcol<'g>() -> impl Search<PartialColoring<'g>> {
    TabuSearch::new(UncoloredVertices, PartialTabu::new()).random_ties()
}
```

### A1.5 Conversions

* `PartialColoring::from_coloring(c: &Coloring) -> Self`: while a conflict
  remains, uncolour the vertex with the most conflicting neighbours (ties at
  random, pass an RNG). A 1-conflict colouring yields exactly one uncoloured
  vertex.
* `Coloring::from_partial(p: &PartialColoring, rng) -> Coloring`: give each
  uncoloured vertex (random order) the colour with the smallest `gamma`, ties
  at random; then `rebuild()`. From one uncoloured vertex this gives a
  colouring with `low[v]` conflicts.

### A1.6 Variable Space Search driver

In `examples/common/coloring.rs`:

```rust
/// Alternate TabuCol and PartialCol from `start`, `rounds` times, each space for
/// `iters` iterations (stopping early at a legal colouring). Returns the best
/// legal-or-not colouring found in TabuCol space (cost 0 if legal).
pub fn vss(g: &Graph, k: usize, start: &Coloring, iters: u64, rounds: usize, seed: u64) -> Evaluated<Coloring>
```

Per round: `State::new(coloring, seed)` → `tabucol()` for `iters`
(`MaxIterations.or(Target::new(0))`) → take `global_best()`; if cost 0 return;
convert to partial → `State::new(partial, seed')` → `partialcol()` for `iters`
(`Target::new(0)` means all coloured, i.e. legal) → take `global_best()`;
convert back; continue. Track the best TabuCol-space solution across rounds.

Wire it in:
* `coloring_gpu`: in the polish phase, after the GPU polish batch, run `vss`
  on CPU threads from the 8 best copies (`--vss-rounds R`, default 0 = off;
  `--vss-iters L` in units of n, default 100). PartialCol stays on the CPU;
  at 337 k TabuCol it/s per core, 100n = 50,000 iterations is 0.15 s.
* `coloring`: `--vss` to run `vss` on the best colouring of each failed k
  (CPU-only experiments and the regression check below).

### A1.7 Tests (in `examples/coloring.rs` `mod tests`, which already has
`random_graph`)

1. Incremental tables match `rebuild()` after 300 random `Assign` moves on a
   random graph (compare `gamma`, `low`, `at_low`, sorted `list`, invariant
   "coloured vertices have gamma 0 in their colour", and `cost() == evaluate()`).
2. `check::deltas(&mut UncoloredVertices, &s)` is empty.
3. Round trip: `from_coloring` of a legal colouring has no uncoloured vertex;
   `from_partial` of an all-coloured partial equals it; a 1-conflict colouring
   gives exactly one uncoloured vertex.
4. `vss` colours a random `G(50, 0.5)` with the k that `hea` colours (existing
   test's setup).

### A1.8 Measure

* CPU sanity: `coloring DSJC125.5.col --k 17 --vss` and `DSJC250.5 --k 28 --vss`
  must still succeed, within 2× of the current times (0.1 s, 1.4 s).
* The real test, on the user's GPU: DSJC500.5 at 47 with
  `--polish-every 200 --vss-rounds 4`. Success = a legal 47-colouring
  (`DSJC500.5.47.sol`); that is a tie with the best known, and the signal
  that the machinery is ready for A2.

## A2. Colouring: large DIMACS graphs on the GPU

**Why.** 46 on DSJC500.5 would contradict a value believed optimal. The
colouring values that moved in the last decade are on large graphs, and
they moved because of population-scale compute (HEAD; Goudet, Grelier & Hao's
GPU memetic, 2022). That is what `coloring_gpu` does. Candidates (verify the
current best known before choosing k; these are from memory): DSJC1000.5
(82–83), DSJC1000.9 (222–223), flat1000_76_0 (81–82), C2000.5 (145–146),
latin_square_10 (97–98), r1000.5 (234). Try k = best known first (a tie
proves the setup), then k − 1.

**What to change in `examples/coloring_gpu.rs`.**
1. The kernel's global-memory path (`use_shared = 0`) exists but has never
   run: the 16-bit tables (`gamma`, `clist`) and 8-bit colours are allocated
   with `global(len)`; check the host passes the right lengths and that the
   `start` line reports "tables in global memory" and the occupancy.
2. The move key packs `v*k + c` in 20 bits: C4000.5 at k ≈ 260 needs
   1,040,000 > 2²⁰. Repack as 24 bits index | 16 bits random | 24 bits
   (delta + 2²³) and change the `assert!(n * k < 1 << 20)` accordingly.
3. Memory: tabu table `4·p·n·k` bytes (C2000.5, p = 224: 261 MB), gamma
   `2·p·n·k`, fine on a 12 GB card; print the total in the `start` line.
4. Defaults: with global tables the occupancy limit is threads, not shared
   memory, so more blocks fit per SM; set `--pairs` from the reported
   occupancy (blocks per SM × SMs / 2). Start with `--iters 20` (each
   iteration costs more on dense large graphs and there are more
   conflicting vertices), then 60.
5. GPX on the host is O(n·k) per child and runs on one thread: for n = 2000
   and p = 448 children it is ~0.1 s per generation, comparable to a batch;
   parallelise the host crossover loop with `std::thread::scope` over
   chunks (deterministic per chunk).

**Measure.** Time to a legal colouring at k = best known on DSJC1000.5 and
C2000.5, and the polish/VSS behaviour at k − 1. Any legal k − 1 colouring is
saved and must be verified independently (`is_legal` already checks; also
re-check the saved file with a separate script).

## Q1. QAP: cohesive crossover (Drezner) for structured instances (implemented)

Implemented as specified. First measurement, sko100a with `ma`, 5 seeds:
uniform 43.6 s mean to the best known, cohesive 57.9 s: no gain with a
descent as the improvement. Still to test: with tabu improvement
(`qap_advanced --method memetic --crossover cohesive`, `qap_gpu
--crossover cohesive`) on tho150.


**Why.** tho150 is 0.004% from its best known; sko/wil/tho have grid distance
matrices. The record-setting method on these families is Drezner's hybrid
genetic algorithm, whose key idea is a *cohesive merging* crossover: the
child inherits, from one parent, the facilities of a set of locations that
are close to each other under the distance matrix, and the rest from the
other parent. Uniform crossover ignores that structure.

**Where.** `src/perm.rs`, next to `UniformCrossover`.

```rust
/// Drezner's cohesive crossover: the child takes parent `a`'s elements on a
/// set of mutually close positions and parent `b`'s elsewhere.
pub struct CohesiveCrossover {
    /// order[l] = all positions sorted by increasing proximity to l (l first).
    pub order: Vec<Vec<usize>>,
    /// The cohesive set's size is uniform in [min, max] × n.
    pub fraction: (f64, f64),   // default (0.25, 0.75)
}
impl CohesiveCrossover {
    pub fn new(n: usize, proximity: impl Fn(usize, usize) -> f64) -> Self  // sorts each row
}
impl<S: Permutation> Crossover<S> for CohesiveCrossover {
    fn cross(&mut self, a: &S, b: &S, child: &mut S, rng: &mut dyn RngCore) {
        // 1. pivot l0 = random position; m = random in [min·n, max·n]; set S = order[l0][..m]
        // 2. target[i] = a[i] for i in S; mark those elements used
        // 3. for i not in S, in increasing i: target[i] = b[i] if unused, else EMPTY
        // 4. fill EMPTY positions, in increasing i, with the unused elements in the
        //    order they appear in b
        // 5. rearrange(child, &target)   // child holds a copy of a, as Memetic guarantees
    }
}
```

`rearrange` (already in `perm.rs`) turns `child` into `target` by swaps, so
delta structures (`QapTaillard`) stay consistent.

**Which matrix is "distance"?** `examples/common/qap.rs` has `Instance { a, b, .. }`
with cost Σ a[i][j]·b[p[i]][p[j]], positions indexed in `a`. Add
`Instance::proximity_matrix(&self) -> Option<&[C]>`: return `a` if it looks
metric (symmetric, zero diagonal, triangle inequality holds on a sample of
1,000 random triples with ≥ 95% success), else `None` (tai-a: random, no
geometry; the flag then falls back to uniform crossover with a printed note).

**Wire in.** `qap_advanced`: `--crossover cohesive` for `memetic` and `ma`.
`qap_gpu`: `--crossover uniform|cohesive` (host-side crossover; the kernel
is unchanged).

**Tests (`src/perm.rs` or `tests/advanced.rs`).** Child is a permutation;
child equals `a` on the cohesive set (expose the last set for the test or
check with a fixed RNG); identical parents give the parent; `fraction`
bounds respected.

**Measure.** On the sandbox (4 cores, `qap_advanced --method memetic`),
compare uniform vs cohesive on sko100a and tho150: time to the best-known
value on sko100a (uniform: ~25 min for the 4-run long run), and the best
value on tho150 in 1 hour. Then on the GPU (`qap_gpu --crossover cohesive`,
after Q3), tho150 for several hours with `--run-to-end`.

## Q2. QAP: directed perturbation (breakout local search)

**Why.** On the random tai-a family (0.05–0.9% off), Benlic & Hao's BLS
(2013) matched the best knowns with far less compute than plain robust
tabu by replacing random perturbation swaps with *directed* ones. Our
`adaptive_perturbation` adapts the number of swaps but picks them at random.

**Where.** `src/perm.rs`:

```rust
/// BLS-style perturbation: each of the `strength` swaps is the best (smallest
/// delta) swap not recently used, with aspiration for a new best; a swap used
/// in the last `tenure` perturbation moves is excluded.
pub struct DirectedPerturbation<C> { pub tenure: usize, min: usize, step: usize, max: usize,
                                     strength: usize, last: Option<C>, returns: u32,
                                     last_used: Vec<u64>, moves: u64, p_random: f64 }
pub fn directed_perturbation<S: Permutation>(n: usize) -> DirectedPerturbation<S::Cost>
impl<S: Permutation> Perturbation<S> for DirectedPerturbation<S::Cost>
```

Behaviour per `perturb(s, rng)`:
1. Strength as `Adaptive`: reset to `min` when the incumbent cost changed,
   else `+= step` up to `max`; also count `returns` (consecutive calls with
   the same cost).
2. With probability `p = max(0.2, exp(-returns / 3))` do a directed move,
   else a random swap (BLS mixes directed and random perturbation; the
   random share grows as the search keeps returning to the same optimum).
3. Directed move: scan all swaps (i < j) using `s.swap_delta(i, j)` (O(1) on
   `QapTaillard`); skip swaps with `last_used[i*n+j] + tenure > moves`
   unless `cost + delta < best seen in this perturbation sequence`; take the
   minimum delta; apply with `s.modify(|p| p.swap(i, j))` (or a single
   `modify` around all `strength` swaps); record `last_used`.
4. `tenure` default `n / 2` perturbation moves.

Cost: `strength ≈ n/10` scans of n²/2 deltas per perturbation, against a
tabu run of ~2n iterations each O(n²): about 5% overhead.

**Wire in.** `qap_advanced::improvement()`: `--perturbation random|directed`
(the `IteratedSearch` gets the chosen `Perturbation`). `qap_gpu` is
unaffected (its improvement is a single tabu run).

**Tests.** Deterministic with a seeded RNG; each perturbation applies
exactly `strength` swaps; a swap just used is not reused within `tenure`
moves (unless aspired); result remains a permutation.

**Measure.** tai50a/tai60a/tai80a, `qap_advanced --method memetic`, 4 runs,
1 hour each, random vs directed: best value and time to within 0.1%. The
200-restart table values (0.66%, 0.59%, 0.87%) are the baseline to beat
first; the best knowns (4,938,796; 7,205,962; 13,499,184) the target.

## Q3. QAP GPU kernel: symmetric fast path

**Why.** sko/tho/wil (the Q1 targets) are symmetric with zero diagonal; the
CPU model halves the delta work there, the kernel uses the general formula.

**Where.** `examples/qap_gpu.rs`, kernel `rots`: add an `int symmetric`
argument (from `inst.symmetric`).
* `warp_delta`: if symmetric, `d = 2 · Σ_k (a[r][k] − a[s][k]) · (b[ps][pk] − b[pr][pk])`
  (no transposed terms, no diagonal terms).
* Update of disjoint pairs: `D += 2 · (ca[i] − ca[j]) · (cb[j] − cb[i])`
  (skip `ra`, `rb`; do not compute them).
* Host: pass the flag; the trace's recounts (`mismatches=0`) are the test.
  Also run the `--cpu` path on sko42 to confirm the host loop is untouched.

**Measure.** `improve_s` on sko100a before/after; expect ~1.6–2×.

## Q4. QAP GPU: exhaustive k-opt polish (optional)

**Why.** The GPU is best at breadth without dependency chains. For a
near-record permutation, enumerate all `k!` rearrangements of `k` random
positions per launch (k = 9: 362,880; k = 10: 3.6 M) across thousands of
random subsets: a "k-opt optimality" check far beyond what a CPU affords
(`perm::BestRearrangement` does k ≤ 8 on the CPU).

**Sketch.** Kernel `kopt`: block = one subset of k positions (passed in);
threads enumerate rearrangements by index (factorial number system →
permutation), each computing the cost difference with the O(k²) partial
formula (only pairs within the subset change: Σ over i, j in the subset of
a[i][j]·(b[q[i]][q[j]] − b[p[i]][p[j]]) plus cross terms with positions
outside, which need the row sums r_i(x) = Σ_{j∉S} a[i][j]·b[x][p[j]] for each
subset position i and candidate element x: precompute k×k in shared memory
per block, O(k²·n) per block). Block minimum → host applies the best
improving rearrangement, repeats. Add as `qap_gpu --polish-k K` on the best
member every G generations, like the colouring polish.

Only worth doing after Q1–Q3 show a near-record on tho150.

## Run plan on the user's GPU (RTX 3060, 28 SMs)

Order: A1 (DSJC500.5 at 47 with VSS polish) → Q3 + Q1 (tho150 cohesive, hours,
`--run-to-end`) → Q2 on the tai-a family (CPU sandbox first) → A2 (large
graphs, k = best known, then k − 1).

Every run: `--trace FILE`, check `build=` and `mismatches=0` first, keep the
log, the trace and any `.sol`/`.best` file. Report ties as ties. A beat is
claimed only after re-verifying the saved solution with an independent
check and the current literature value for the instance.
