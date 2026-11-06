# METSlib in Rust — API proposal

This repository holds the design of `mets`, a Rust equivalent of
[METSlib](https://github.com/baol/metslib), and the crate that implements it (about 2,200 lines including docs, no `unsafe`,
no clippy warnings). It has ports of the examples in
[coin-or/metslib-examples](https://github.com/coin-or/metslib-examples), and it
was benchmarked against the C++ library.

> **Status.** The design went through two reviews, and the crate implements
> it. A few details changed during implementation; §5 lists them and why.
> The first-version prototype has been removed.

```
src/model.rs          Cost, Model, Move, Evaluated (the cost cache)
src/neighborhood.rs   Visitor, Neighborhood, MoveSource, moves_fn, WithDeltas,
                      FullEvaluation, Sampler, Sampled, BestMove, collect_moves
src/state.rs          State, Scope, Event, Observer + observer helpers
src/search.rs         Search, Step, Stop, Outcome, Runner (SearchExt::runner)
src/termination.rs    MaxIterations, NoImprovement, Target, TimeLimit, stop_when, or/and
src/local_search.rs   best / first improvement descent
src/tabu.rs           TabuSearch, TabuAttr, DenseAttr, RecencyTabu, DenseTabu, aspiration
src/stochastic.rs     StochasticSearch + SA, LAHC, RRT, TA, Great Deluge, …
src/ils.rs            IteratedSearch (ILS, iterated tabu search, random restarts), Adaptive
src/lns.rs            LargeNeighborhoodSearch, Rebuild (destroy + repair)
src/population.rs     Memetic (population + crossover), Crossover
src/parallel.rs       run_many (independent runs on threads), Cancel
src/perm.rs           permutation problems: Swap, SwapNeighborhood, perturb, shuffle,
                      RobustTabuSearch, BestRearrangement, UniformCrossover, rearrange
src/check.rs          check::deltas, for testing delta formulas
examples/subset_sum   tutorial port (tabu search)
examples/qap          qap/main_ts.cc + qap/main.cc ports, plus SA/LAHC/RRT/ILS
examples/qap_advanced memetic / restarts / LNS solvers assembled from library pieces
examples/common/qap   QAP instance, models and QAPLIB sweep shared by both QAP examples
examples/vcp          vcp/vcp.cc port (TabuCol-like iterated tabu search)
examples/coloring     DIMACS graph colouring: TabuCol and Galinier & Hao's hybrid
                      evolutionary algorithm (Memetic + GPX crossover)
tests/                every algorithm, full-evaluation models, delta checks,
                      nesting and scopes, dyn Search, a user-defined VND,
                      LNS, memetic, robust tabu, parallel runs
```

```bash
cargo test && cargo run --release --example qap
cargo run --release --example qap -- tai60a.dat tai60a.sln   # a QAPLIB instance (not included)
```

---

## 1. What METSlib offers today

References below are to the [METSlib](https://github.com/baol/metslib)
sources (`metslib/*.hh`) and to the
[metslib-examples](https://github.com/coin-or/metslib-examples).

| Concept | C++ (METSlib) | Role |
|---|---|---|
| Solution | `feasible_solution` → `evaluable_solution` (`cost_function()`, `copy_from()`) | the state being optimised |
| Move | `move` (`evaluate`, `apply`) → `mana_move` (+ `clone`, `hash`, `==`, `opposite_of`) | neighbor transformation; `mana_` variant is needed by tabu lists |
| Neighborhood | `move_manager`: a `std::deque<const move*>` + `refresh(sol)` | the list of moves explored each iteration |
| Recorder | `solution_recorder` / `best_ever_solution` | keeps the best solution (`copy_from`) |
| Algorithms | `local_search`, `tabu_search`, `simulated_annealing` (templated on the move manager) | each has a monolithic `search()` |
| Tabu machinery | `tabu_list_chain` / `simple_tabu_list`, `aspiration_criteria_chain` / `best_ever_criteria` | chain of responsibility via raw pointers |
| Termination | `termination_criteria_chain`: iteration, no-improve, threshold, forever | chain, stateful `operator()` |
| SA cooling | `abstract_cooling_schedule`: exponential, linear | |
| Observers | `subject`/`observer`, `search_listener`, `step()` codes (`MOVE_MADE`, `IMPROVEMENT_MADE`, …) | progress/logging hooks |
| Permutation helpers | `permutation_problem` (`evaluate_swap`, `apply_swap`), `swap_elements`, `invert_subsequence`, `swap_neighborhood` (k random), `swap_full_neighborhood`, `invert_full_neighborhood`, `random_shuffle`, `perturbate` | ready-made model pieces |

**What the examples show.** Every non-trivial example (QAP `main.cc`, VCP, ATSP)
builds an *iterated* search by hand. The pattern is three nested loops with
three `best_ever_solution` recorders (global, per restart, per run). Each run
re-randomises the tabu tenure, calls `copy_from` and `perturbate`, and uses a
`noimprove_termination_criteria` that is called by hand on the outer loop. ATSP
also cycles through two neighborhoods (a VND). These are the "ILS",
"random-restart ILS/ILD" and "iterated tabu search" patterns. They have no
first-class support, so the Rust API makes them composable (§3.9).

### Issues found in the C++ code (they motivate several design choices)

1. **Two meanings of `move::evaluate`.** `swap_elements::evaluate` returns the
   *cost after the move*. `invert_subsequence::evaluate` returns a *delta*.
   `local_search`, `tabu_search` and `simulated_annealing` all treat the value
   as an absolute cost, so the invert neighborhoods choose wrong moves. The
   invert delta is also the sum of the swap deltas, each computed on the
   *unmodified* solution. That is only correct when the swaps do not interact.
   → In Rust, the full evaluation is the reference. A move may also supply a
   delta, which has one documented meaning, and debug builds check it against
   the full evaluation (§3.2).
2. **The delta is computed twice.** `permutation_problem::apply_swap` calls
   `evaluate_swap` again (O(n) for the QAP), even though the algorithm has just
   evaluated that same move. → In Rust, the library keeps the cached cost and
   reuses the delta it already computed. Models never update the cost
   themselves.
3. **Neighborhoods are materialised as heap objects.** `swap_full_neighborhood`
   allocates n(n-1)/2 moves, `invert_full_neighborhood` n², and the ATSP 3-opt
   manager n³. VCP `refresh` calls `delete`/`new` on every move each iteration.
   Each evaluation costs two virtual calls and a `static_cast`/`dynamic_cast`.
   → In Rust, neighborhoods are generators of small `Copy` moves, with static
   dispatch and no allocation.
4. **Tabu bookkeeping cost.** `tabu_search` calls `is_tabu` (hash plus virtual
   `==` with `dynamic_cast`) for *every* move, including moves that cannot be
   selected anyway. Each iteration also allocates the opposite move
   (`opposite_of` → `clone`). → In Rust, tabu status is checked only for moves
   that would beat the current candidate. Lists store expiry stamps per key.
5. **Recording the best copies the instance.** `qap_model::copy_from` copies
   both n×n matrices on every improvement. → In Rust, a solution borrows its
   instance (`&'a Instance`), and `Clone::clone_from` reuses buffers.
6. **Termination criteria have side effects.** Calling `operator()` twice
   counts twice. The examples rely on this by calling a criterion manually on
   the outer loop. → In Rust, criteria are called once per iteration by the
   runner, and their state is reset at the start of each run.
7. **SA's RNG can't be seeded or shared.** `simulated_annealing` owns a
   default-seeded `mt19937`. → In Rust, one seeded RNG with portable output
   lives in `State`, so runs are reproducible.
8. **Maintenance.** The code uses dynamic exception specifications, `std::tr1`,
   and `std::random_shuffle`, which are all removed in C++17. There is a typo,
   `abstract_search::operator==`, where `operator=` was meant. The tutorial
   and ATSP examples target a different API version (`copyable_solution`,
   `complex_mana_move`, `best_ever()`, `<metslib/mets.h>`, non-template
   `search_listener`), so they do not build against the current METSlib sources.

---

## 2. Design principles

1. **Full evaluation first, deltas optional.** A model must have a full
   evaluation, its objective function from scratch. With that and `apply`,
   every algorithm works. A delta formula is an optional speed-up, added per
   move type when you have one. Nothing else changes when you add it.
2. **The library owns the cost cache.** Your solution is wrapped in
   `Evaluated<S>`. The wrapper evaluates once, then keeps the cached cost
   correct by adding known deltas or re-evaluating. You can't forget to update
   it, because the only ways to change the solution go through the wrapper.
3. **Invariants are enforced by types, not by documentation.** `State` has
   private fields. An algorithm can only apply a move through `commit`, which
   updates the cost, records the best and notifies observers. The runner, not
   the algorithm, advances the iteration counter.
4. **Generic on the hot path, `dyn`-compatible elsewhere.** Code that runs once
   per *move* is generic and monomorphised: move generation, delta
   evaluation, the visitor, and the tabu check. Code that runs once per
   *iteration* is object-safe: `Search::step` and observers. Algorithms can
   then be chosen at runtime (`Box<dyn Search<S>>`), at the cost of one
   indirect call per iteration.
5. **Neighborhoods are generators.** They are never stored as lists. They push
   `(move, delta)` pairs into a visitor. The visitor can stop them early and
   tells them which deltas still matter (`bound()`, `can_skip(lower)`), so
   they can prune. Plain
   iterators are accepted too, for simple cases (§3.4).
6. **Algorithms are small steppable state machines.** A new metaheuristic is
   one `step` function. Algorithms nest, so ILS, iterated tabu search and
   restarts are compositions, not new code.
7. **Model state is separate from instance data.** A solution borrows the
   instance it solves, so cloning a best solution copies only the state.
8. **Reproducible by default.** One seeded RNG is passed everywhere. It is
   generic, and the default type gives the same sequence on every platform.
9. **One verb for "a run begins".** Every component that has per-run state
   (`Search`, `Neighborhood`, `TabuList`, `Acceptance`, `Termination`) has a
   `start` method, called by the runner at the beginning of each run.

---

## 3. The API

### 3.1 Getting started: evaluate first, add deltas later

The problem: pick a subset of `set` whose sum is as close as possible to
`target`, with overshooting penalised 100×.

**Step 1: only the objective.** This is enough to run every algorithm.
`evaluate` recomputes the sum from the decision variables; nothing is cached.

```rust
#[derive(Clone)]
struct Subset<'a> { set: &'a [i64], target: i64, picked: Vec<bool> }

impl Model for Subset<'_> {
    type Cost = i64;
    fn evaluate(&self) -> i64 {                       // full evaluation, from scratch
        let sum: i64 = (0..self.set.len()).filter(|&i| self.picked[i]).map(|i| self.set[i]).sum();
        penalty(self.target - sum)
    }
}
fn penalty(diff: i64) -> i64 { if diff < 0 { -100 * diff } else { diff } }

#[derive(Clone, Copy)]
struct Toggle(usize);

impl<'a> Move<Subset<'a>> for Toggle {
    fn apply(&self, s: &mut Subset<'a>) { s.picked[self.0] ^= true; }
}

/// What the tabu list remembers: the index (dense, so DenseTabu can be used).
impl<'a> TabuAttr<Subset<'a>> for Toggle {
    type Key = usize;
    fn key(&self, _: &Subset<'a>) -> usize { self.0 }
}
impl<'a> DenseAttr<Subset<'a>> for Toggle {
    fn key_count(s: &Subset<'a>) -> usize { s.set.len() }
}

struct AllToggles;
impl<'a> Neighborhood<Subset<'a>> for AllToggles {
    type Move = Toggle;
    fn explore<V: Visitor<Toggle, i64>>(&mut self, s: &Evaluated<Subset<'a>>, v: &mut V) -> ControlFlow<()> {
        for i in 0..s.set.len() {
            let m = Toggle(i);
            v.visit(m, s.delta_of(&m))?;              // delta_of: full evaluation here
        }
        ControlFlow::Continue(())
    }
}
```

**Step 2: a delta, if you know one.** Keep a running `sum` in the solution so
the delta is O(1), and tell the library how to compute it. `evaluate` stays
as it is: it is the reference the delta is checked against.

```rust
#[derive(Clone)]
struct Subset<'a> { set: &'a [i64], target: i64, picked: Vec<bool>, sum: i64 }

impl<'a> Move<Subset<'a>> for Toggle {
    fn apply(&self, s: &mut Subset<'a>) {
        let v = s.set[self.0];
        s.sum += if s.picked[self.0] { -v } else { v };
        s.picked[self.0] ^= true;
    }
    fn delta(&self, s: &Evaluated<Subset<'a>>) -> Option<i64> {
        let v = s.set[self.0];                        // Deref: your fields
        let sum = if s.picked[self.0] { s.sum - v } else { s.sum + v };
        Some(penalty(s.target - sum) - s.cost())      // s.cost(): cached, O(1)
    }
}
```

**Run it.** (`runner`, `until`, `or` and `on_move` are introduced in §3.6.
The complete program is `examples/subset_sum.rs`.)

```rust
let mut st = State::new(Subset::new(&numbers, 109), 42);      // evaluates once
let mut ts = TabuSearch::new(AllToggles, DenseTabu::new(Tenure::Fixed(7)));
let outcome = ts.runner(&mut st)
    .until(Target::new(0).or(NoImprovement::new(200)))
    .on_move(|st| eprintln!("{}: {}", st.iteration(), st.current().cost()))
    .run();
println!("{} {:?}", st.best().cost(), outcome.stop);
```

### 3.2 Model, moves and the cost cache

```rust
pub trait Model: Clone {
    type Cost: Cost;
    /// The objective, computed from scratch. Required: it is the reference
    /// every delta is checked against, and gives the starting cost.
    fn evaluate(&self) -> Self::Cost;
}

pub trait Move<S: Model>: Clone {
    fn apply(&self, s: &mut S);
    /// `cost(after) - cost(before)` without applying, if you have a formula.
    /// `None` (the default) means "evaluate the neighbor in full".
    fn delta(&self, s: &Evaluated<S>) -> Option<S::Cost> { None }
    /// A move that undoes `self`, computed on the solution *before* `self` is
    /// applied (a swap is its own inverse). Optional. It lets full-evaluation
    /// neighborhoods use apply/evaluate/undo instead of cloning (§3.3).
    fn inverse(&self, before: &S) -> Option<Self> { None }
}
```

`Evaluated<S>` holds your solution and its cached cost:

```rust
pub struct Evaluated<S: Model> { inner: S, cost: S::Cost }   // private fields

impl<S: Model> Evaluated<S> {
    pub fn new(s: S) -> Self;                       // calls evaluate() once
    pub fn cost(&self) -> S::Cost;                  // O(1)
    /// The move's delta, or `evaluate(after) - cost()` (clone + apply) if it has none.
    pub fn delta_of<M: Move<S>>(&self, m: &M) -> S::Cost;
    /// Apply one move. Costs `delta_of(m)`: for a move without a formula that
    /// is a clone and a full evaluation, so apply *sequences* of such moves
    /// through `modify` (one evaluation in total).
    pub fn apply<M: Move<S>>(&mut self, m: &M);
    /// Arbitrary change (re-initialisation, repair, several moves at once…).
    /// Re-evaluates once afterwards.
    pub fn modify<T>(&mut self, f: impl FnOnce(&mut S) -> T) -> T;
    /// Re-evaluate and reset the cache (float drift, see below). Returns the
    /// drift it corrected (`cached - evaluated`).
    pub fn resync(&mut self) -> S::Cost;
    pub fn into_inner(self) -> S;
}
impl<S: Model> Deref for Evaluated<S> { type Target = S; }   // read access to your fields
// No DerefMut: changes only go through apply / modify, so the cache is always valid.
// Clone is implemented by hand so `clone_from` reuses the inner buffers.
```

* **`Deref`, deliberately.** The API guidelines reserve `Deref` for
  smart-pointer-like types. `Evaluated<S>` is one: it owns one value and
  guards access to it, like `ManuallyDrop` or a lock guard. Reading your own
  fields as `s.picked[i]` rather than `s.get().picked[i]` is the point of the
  wrapper, so this is a conscious exception, and it is the only `Deref` in
  the crate.
* **Delta checking.** In debug builds (`cfg(debug_assertions)`), every
  committed move is checked against two full evaluations: `delta == evaluate(after) - evaluate(before)`. This uses no
  cached value, so it is independent of both float drift and the search
  tolerance (§3.5). The comparison is exact for integer costs and within a
  relative 1e-9 for floats. A wrong delta formula fails at the move that
  broke it, instead of quietly misleading the search. For tests,
  `mets::check::deltas(&mut neighborhood, &s)` compares every move of a
  neighborhood with full evaluations, without applying anything permanently.
* **Floating-point drift.** With float costs, the cached `cost + delta` can
  drift from `evaluate()` by a few ulps per move. `resync()` re-evaluates;
  `State::resync_every(n)` does it for both `current` and the best
  solutions every n iterations, and in debug builds asserts that the drift it
  corrected stayed within the tolerance (when one is set). Integer costs are exact and
  never need it.
* **Mixed models.** `delta` returns an `Option` per call, so one move type can
  give a formula for some cases and fall back to full evaluation for others.
  An example is a move with no incremental form for one constraint.

### 3.3 When only full evaluation exists

There are three ways to evaluate neighbors, from simplest to fastest:

1. **Default.** `delta_of` clones the solution, applies the move and
   evaluates. It is correct, and fine for small solutions and first versions.
2. **`FullEvaluation` adapter.** It wraps a `MoveSource` (§3.4: moves only, no
   deltas) and owns a scratch copy, synchronised with `clone_from` once per
   iteration. For each move it does apply → `evaluate()` → apply the
   `inverse`, so each move costs one full evaluation and no clone. That is as
   fast as it gets without writing any incremental code. Moves without an
   inverse fall back to option 1. Parallel exploration (§6) gives each thread
   its own scratch copy.
   ```rust
   let neigh = FullEvaluation::new(moves_fn(|s: &Sched| all_swaps(s.len())));
   ```
3. **Evaluate fewer neighbors.** When a full evaluation is expensive, use
   sampling methods (SA, LAHC, RRT: one evaluation per iteration), or the
   `Sampled::new(sampler, k)` adapter, so tabu search pays for k evaluations
   per iteration instead of the whole neighborhood.

Perturbations on such models should apply all their moves inside one
`modify` call (§3.9), not through repeated `Evaluated::apply`.

### 3.4 Neighborhoods: exhaustive, pruned, sampled, or plain iterators

```rust
pub trait Visitor<M, C> {
    fn visit(&mut self, m: M, delta: C) -> ControlFlow<()>;   // Break = stop
    fn bound(&self) -> Option<C> { None }                     // deltas >= bound are useless
}
// any FnMut(M, C) -> ControlFlow<()> is a Visitor

pub trait Neighborhood<S: Model> {
    type Move: Move<S>;
    fn explore<V: Visitor<Self::Move, S::Cost>>(&mut self, s: &Evaluated<S>, v: &mut V)
        -> ControlFlow<()>;
    /// Called when a run starts: build candidate lists, size buffers, seed an
    /// internal RNG from the run's RNG…
    fn start<R: Rng + ?Sized>(&mut self, s: &Evaluated<S>, rng: &mut R) {}
}

/// Moves only, no deltas: the simple way to describe a neighborhood when the
/// deltas come from `delta_of` anyway.
pub trait MoveSource<S: Model> {
    type Move: Move<S>;
    type Iter<'a>: Iterator<Item = Self::Move> where Self: 'a, S: 'a;
    fn moves<'a>(&'a mut self, s: &'a S) -> Self::Iter<'a>;
}
/// A closure returning an iterator of moves: both a MoveSource and a Neighborhood.
pub fn moves_fn<S, I, F>(f: F) -> MovesFn<F, I>
where F: FnMut(&S) -> I, I: Iterator, I::Item: Move<S>;

/// Any MoveSource as a Neighborhood, with deltas from `delta_of`.
pub struct WithDeltas<MS>(pub MS);
/// Any MoveSource as a Neighborhood, with apply/evaluate/undo (§3.3).
pub struct FullEvaluation<MS, S> { .. }   // FullEvaluation::new(source)

pub trait Sampler<S: Model> {                 // one random move, for huge neighborhoods
    type Move: Move<S>;
    fn sample<R: Rng + ?Sized>(&mut self, s: &Evaluated<S>, rng: &mut R) -> Option<Self::Move>;
}
```

A hand-written neighborhood is a pair of nested loops. `?` propagates early
stops (first improvement), and `delta_of` works whether or not the move has a
formula:

```rust
impl<S: Permutation> Neighborhood<S> for SwapNeighborhood {
    type Move = Swap;
    fn explore<V: Visitor<Swap, S::Cost>>(&mut self, s: &Evaluated<S>, v: &mut V) -> ControlFlow<()> {
        for i in 0..s.size() { for j in i + 1..s.size() {
            let m = Swap { i, j };
            v.visit(m, s.delta_of(&m))?;
        }}
        ControlFlow::Continue(())
    }
}
```

**Iterator convenience.** Many Rust users would expect a neighborhood to be an
`Iterator`. `moves_fn` accepts exactly that, and the result can be used
directly wherever a neighborhood is expected:

```rust
let neigh = moves_fn(|s: &Tour| {
    let n = s.len();
    (0..n).flat_map(move |i| (i + 2..n).map(move |j| TwoOpt { i, j }))
});
```

For debugging and tests, `collect_moves(&mut n, &s) -> Vec<(M, C)>` returns
the moves of any `Neighborhood` with their deltas. The visitor remains the
primary interface for two reasons. Nested loops are easier to write than
nested iterator adapters once there are conditions. And only the visitor
exposes `bound()` and `can_skip()`, which an iterator cannot see.

The neighborhood, not the algorithm, produces the delta. That is what makes the
large-neighborhood tricks expressible:

* **Incremental delta structures** live in the solution and are read in O(1).
  `examples/qap.rs` keeps Taillard's delta matrix, so a full O(n²) scan costs
  O(n²) instead of O(n³). `examples/vcp.rs` keeps the TabuCol γ table.
* **Candidate lists / restricted neighborhoods.** VCP only generates recolourings
  of conflicting vertices.
* **Pruning with `can_skip(lower)`.** A neighborhood that knows a lower
  bound on the deltas of a group of moves asks the visitor whether the
  group can still matter. VCP skips a vertex when its best possible delta
  (−γ[v][colour(v)]) cannot beat the best candidate so far; `coloring.rs`
  keeps each vertex's smallest γ, a much tighter bound. The question, not a
  raw `bound()`, is what lets a visitor with random tie-breaking (which may
  still take a move *equal* to the best) prune too: it answers `lower > best`
  instead of `lower >= best`. Ignoring both is always correct.
* **Sampling.** `Sampler` gives O(1) random moves for SA/LAHC/RRT. The
  `Sampled::new(sampler, k)` adapter turns any sampler into a "k random moves"
  neighborhood (METSlib's `swap_neighborhood(rng, k)`), so tabu search can run
  on neighborhoods too large to scan. It seeds its own RNG from the run's RNG
  in `start`, so it is covered by the run's seed.

### 3.5 Costs and tolerance

```rust
/// Blanket-implemented for every type with these num-traits bounds:
/// i32, i64, i128, isize, f32, f64, and fixed-point / decimal / user types
/// that implement them.
pub trait Cost: Copy + PartialOrd + Debug + num_traits::Signed + num_traits::Bounded
    + num_traits::ToPrimitive {}
impl<T> Cost for T where T: Copy + PartialOrd + Debug + Signed + Bounded + ToPrimitive {}
```

The improvement tolerance is not part of the cost type. It is a setting on the
`State`, which every component uses through `st.is_better(a, b)`
(`a < b - tolerance`): recording the best, local search, aspiration and
`NoImprovement`. So one setting covers the whole run, and different components
can't disagree.

```rust
let st = State::new(model, seed).with_tolerance(1e-9);   // default: zero
```

Integer costs (`i64`) are encouraged: comparisons are exact and no tolerance
is needed. For floats, debug builds also reject NaN costs. SA-style rules
convert to `f64` with `ToPrimitive`.

### 3.6 State, observers, termination, running

```rust
pub type DefaultRng = rand_xoshiro::Xoshiro256PlusPlus;

pub struct State<S: Model, R = DefaultRng> { /* private */ }

impl<S: Model> State<S> {
    pub fn new(initial: S, seed: u64) -> Self;          // DefaultRng; wraps in Evaluated (one evaluation)
}
impl<S: Model, R> State<S, R> {
    pub fn with_rng(initial: S, rng: R) -> Self;        // any other generator
    pub fn with_tolerance(self, tol: S::Cost) -> Self;
    pub fn resync_every(self, n: u64) -> Self;

    // read access
    pub fn current(&self) -> &Evaluated<S>;
    pub fn best(&self) -> &Evaluated<S>;                // best of the current scope, see 3.9
    pub fn global_best(&self) -> &Evaluated<S>;         // best of the whole State, any scope
    pub fn iteration(&self) -> u64;                     // advanced by the runner only
    pub fn last_improvement(&self) -> u64;              // iteration when global_best last changed
    pub fn elapsed(&self) -> Duration;
    pub fn tolerance(&self) -> S::Cost;
    pub fn is_better(&self, a: S::Cost, b: S::Cost) -> bool;

    // the only ways to change the state
    pub fn rng(&mut self) -> &mut R;
    pub fn current_and_rng(&mut self) -> (&Evaluated<S>, &mut R);   // e.g. to sample a move
    pub fn commit<M: Move<S>>(&mut self, m: &M, delta: S::Cost, obs: &mut dyn Observer<S, R>) -> Step;
    pub fn modify_current<T>(&mut self, f: impl FnOnce(&mut Evaluated<S>, &mut R) -> T) -> T;
    pub fn scope(&mut self) -> Scope<'_, S, R>;         // nested runs, see 3.9
}
```

`commit` applies the move through `Evaluated::apply` (reusing the delta the
algorithm already computed), records the scoped best and, when it is also a
global improvement, the global best, then notifies the observer. It is the
only way an algorithm changes `current`, so the "update the cost, record the
best, notify" sequence can't be forgotten. `modify_current` is for
perturbations and restarts; it records nothing and notifies no one.

**Reproducibility.** `Xoshiro256PlusPlus` gives the same stream on every
platform. `State::new(_, seed)` derives the generator state through
`rand_core`'s `seed_from_u64`, which has been stable across releases but is
not formally guaranteed; use `with_rng` with an explicitly built generator
when bit-for-bit reproduction across `rand` upgrades matters. `State::new` is
defined for the default generator only, because a type-parameter default
does not drive inference in expressions.

```rust
#[non_exhaustive]
pub enum Event<C> {
    Moved { delta: C },
    /// The best of the current scope improved (inside an inner run: "best of this run").
    Improved,
    /// `global_best()` improved. Always preceded by `Improved` in the same iteration.
    GlobalImproved,
    Aspiration,
    /// A nested search begins a new inner run.
    Restart,
    Stuck,
}

pub trait Observer<S: Model, R = DefaultRng> {
    fn notify(&mut self, e: Event<S::Cost>, st: &State<S, R>);
}
// implemented for (), for &mut O where O: Observer + ?Sized, for tuples (A, B),
// for log_improvements(writer), and for the closure wrappers returned by
// on_move, on_improvement, on_global_improvement and on_event (see below).

pub trait Termination<S: Model, R = DefaultRng> {
    fn start(&mut self, st: &State<S, R>) {}
    fn stop(&mut self, st: &State<S, R>) -> bool;
}
// implemented for &mut T where T: Termination + ?Sized, so one criterion can be
// reused across the inner runs of a nested search.
// Unlimited (the default), MaxIterations::new(n), NoImprovement::new(n),
// Target::new(c), TimeLimit::new(d), stop_when(f), and a.or(b) / a.and(b)
```

`MaxIterations` counts the iterations of the search it is attached to, so on
an ILS it counts perturbations, not inner moves. `NoImprovement` and `Target`
watch the scoped best (§3.9): attached to an inner run they mean "this run",
attached to the outer search they mean the outer search. Use
`stop_when(|st| st.global_best().cost() <= c)` for a target on the global
best from anywhere.

**One way to run.** `runner` returns a builder. `#[must_use]` warns if it is
never run.

```rust
let outcome: Outcome = search.runner(&mut st)
    .until(NoImprovement::new(1000))          // default: Unlimited (stops when Stuck/Exhausted)
    .stop_when(|st| st.elapsed() > limit)     // OR-ed with `until`
    .on_improvement(|st| println!("{}", st.best().cost()))
    .on_event(|e, st| log(e, st))             // closure observers add up
    .run();                                    // calls search.start(st) first
// .resume() instead of .run() continues without calling start()
// .observe(o) sets an observer value (or &mut reference), replacing earlier ones

pub struct Outcome { pub stop: Stop, pub iterations: u64, pub elapsed: Duration }
#[non_exhaustive]
pub enum Stop {
    /// The termination criterion fired.
    Terminated,
    /// No admissible move exists (local optimum, every move tabu).
    Stuck,
    /// The algorithm finished on its own (SA frozen, deluge level reached…).
    Exhausted,
}
```

Closures are given to the runner's methods (`on_move`, `on_improvement`,
`on_global_improvement`, `on_event`, `stop_when`) because there `S` and `R`
are already known, so the closure needs no type annotation. The free functions
of the same names work anywhere else (for example as an inner criterion of a
nested search) but need an annotated argument (`|st: &State<Qap>| ..`).
`until` and `observe` accept owned values or `&mut` references, thanks to the
`&mut T` / `&mut O` impls above. Starting is explicit: `run()` begins a fresh
run (tabu list cleared, tenure re-drawn, SA reheated), and `resume()`
continues where the last run stopped. That fixes the surprise of an implicit
reset.

### 3.7 Algorithms

```rust
pub trait Search<S: Model, R = DefaultRng> {               // object-safe
    fn step(&mut self, st: &mut State<S, R>, obs: &mut dyn Observer<S, R>) -> Step;
    fn start(&mut self, st: &mut State<S, R>) {}
}
#[non_exhaustive]
pub enum Step { Moved, Improved, Rejected, Stuck, Exhausted }

// algorithms can be chosen at runtime
let mut alg: Box<dyn Search<Qap>> = match args.algorithm {
    Algo::Tabu => Box::new(TabuSearch::new(SwapNeighborhood, DenseTabu::new(Tenure::Fixed(n as u64)))),
    Algo::Sa   => Box::new(StochasticSearch::new(SwapNeighborhood, SimulatedAnnealing::new(t0, 1e-3, 0.95, 1000))),
};
alg.runner(&mut st).until(TimeLimit::new(Duration::from_secs(10))).run();
```

`runner` is a provided method of an extension trait implemented for every
`Q: Search + ?Sized`, so it works on `Box<dyn Search>` too. Only the per-
iteration `step` and observer calls are indirect. The per-move work inside
`step` stays fully generic.

| Metaheuristic | Rust |
|---|---|
| steepest descent / first improvement | `LocalSearch::best_improvement(n)` / `::first_improvement(n)` |
| tabu search | `TabuSearch::new(n, tabu_list)` (+ `.with_aspiration(a)`) |
| simulated annealing | `StochasticSearch::new(sampler, SimulatedAnnealing::new(t0, t_end, alpha, plateau))` |
| late acceptance hill climbing | `StochasticSearch::new(sampler, LateAcceptance::new(len))` |
| record-to-record travel | `StochasticSearch::new(sampler, RecordToRecord::new(deviation))` |
| threshold accepting / great deluge | `ThresholdAccepting::new(t, decay)` / `GreatDeluge::new(rain)` |
| randomised descent / plateau walk | `Improving` / `NonWorsening` acceptance |
| ILS (iterated local search / descent) | `IteratedSearch::new(LocalSearch…, inner_stop, perturb, Improving)` |
| iterated tabu search | `IteratedSearch::new(TabuSearch…, NoImprovement::new(k), perturb, Improving)` |
| random restarts (RR-…) | `IteratedSearch::new(inner, stop, re_initialise, Always)` |
| LSMC / SA-ILS | `IteratedSearch` with `SimulatedAnnealing` as acceptance |
| robust tabu search (Taillard) | `perm::RobustTabuSearch::new()` (§3.13) |
| large neighborhood search | `LargeNeighborhoodSearch::new(rebuild, acceptance)` (§3.11) |
| memetic algorithm | `Memetic::new(inner, inner_stop, init, crossover, size)` (§3.12) |
| VND, or anything new | implement `Search::step` (§3.10) |

The `Acceptance` trait is shared by the stochastic searches and the ILS
acceptance step, so every rule works in both places:

```rust
pub trait Acceptance<C: Cost> {
    fn start(&mut self, initial: C) {}
    fn accept<R: Rng + ?Sized>(&mut self, current: C, candidate: C, best: C, rng: &mut R) -> bool;
    fn update(&mut self, current: C) {}     // cooling, history…
    fn finished(&self) -> bool { false }    // e.g. frozen → Step::Exhausted
}
```

### 3.8 Tabu lists

```rust
pub trait TabuAttr<S: Model> {                   // replaces clone/hash/==/opposite_of
    type Key: Hash + Eq + Clone;
    fn key(&self, s: &S) -> Self::Key;           // attribute of a candidate move
    fn reverse_key(&self, before: &S) -> Self::Key { self.key(before) } // what to forbid
}

/// Attributes that are small integers, for array-backed lists.
pub trait DenseAttr<S: Model>: TabuAttr<S, Key = usize> {
    fn key_count(s: &S) -> usize;                // every key is < key_count(s)
}

pub trait TabuList<S: Model, M> {
    fn is_tabu(&self, s: &S, m: &M, iteration: u64) -> bool;
    fn make_tabu(&mut self, before: &S, m: &M, iteration: u64);
    fn start<R: Rng + ?Sized>(&mut self, s: &S, rng: &mut R) {}
}
// RecencyTabu<K>: HashMap<K, tabu-until>; K is inferred from the move's TabuAttr
//   at the point of use (write RecencyTabu::<usize>::new(..) if it is not).
// DenseTabu: Vec<u64>, sized from M::key_count(s) in start(); no size argument.
// (A, B): tabu if either is (the old chain)    Tenure::Fixed(t) | Tenure::Random(a..=b)
```

`DenseTabu::new(tenure)` takes no size. Its `start` sizes the array from the
move's `key_count` on the actual solution, so the key formula and the array
size can't disagree. `Swap` implements `DenseAttr` with `key = i*n + j` and
`key_count = n*n`. With the first version's `DenseTabu::new(n * n, …)`, the
two had to be kept consistent by hand, and a mismatch panicked.

`reverse_key` receives the solution *before* the move. That makes the
classic "v may not go back to its old colour" attribute a one-liner.
Previously it needed a custom `opposite_of()` move. `Tenure::Random` is
re-drawn at each `start`, which is what the QAP and VCP examples do by hand.
Aspiration is a trait with `BestEver` (default) and `NoAspiration`:

```rust
pub trait Aspiration<S: Model, M> {
    fn allows<R>(&self, st: &State<S, R>, m: &M, new_cost: S::Cost) -> bool;
}
```

### 3.9 Nesting: ILS, iterated tabu, restarts

The best solution is *scoped*. `IteratedSearch` opens a scope for each inner
run, so inside the run `best()` means "best of this run"; `global_best()` is
always the best over the whole `State`. When the scope ends, the run-best goes
to the acceptance rule and is merged into the enclosing scope. This replaces
the three hand-managed recorders of the C++ examples.

```rust
pub struct Scope<'a, S: Model, R: Rng> { /* holds &'a mut State<S, R> */ }
impl<'a, S: Model, R: Rng> Scope<'a, S, R> {
    /// Access to the state while the scope is open (`best()` is the run's best).
    pub fn state(&mut self) -> &mut State<S, R>;
    /// Close the scope. Returns the run's best and whether it improved on the
    /// enclosing scope's best. The state's `best()` is the enclosing best again.
    pub fn finish(self) -> (Evaluated<S>, bool);
}
// Drop closes the scope without reporting (used only on unwinding).
```

This is `IteratedSearch::step`, in full. It is also the reference for how a
nested search reuses its termination criterion and forwards the observer:

```rust
fn step(&mut self, st: &mut State<S, R>, obs: &mut dyn Observer<S, R>) -> Step {
    if let Some(inc) = &self.incumbent {                       // 1. perturb the incumbent
        let p = &mut self.perturbation;
        st.modify_current(|cur, rng| { cur.clone_from(inc); p.perturb(cur, rng) });
    }
    obs.notify(Event::Restart, st);

    let mut scope = st.scope();                                 // 2. inner run, own best
    self.inner.runner(scope.state()).until(&mut self.inner_stop).observe(&mut *obs).run();
    let (run_best, improved_outer) = scope.finish();

    let best = st.best().cost();                                // 3. accept as new incumbent?
    let accept = match &self.incumbent {
        None => true,
        Some(inc) => self.acceptance.accept(inc.cost(), run_best.cost(), best, st.rng()),
    };
    if accept { self.incumbent = Some(run_best); }
    if let Some(inc) = &self.incumbent { self.acceptance.update(inc.cost()); }
    if improved_outer { Step::Improved } else { Step::Moved }
}
```

`improved_outer` refers to the enclosing scope, so the step result keeps the
same meaning at every nesting level. `Event::Improved` / `GlobalImproved` are
emitted by the inner `commit`s as they happen.

The three nested loops of `qap/main.cc` become:

```rust
let ts = TabuSearch::new(SwapNeighborhood, DenseTabu::new(Tenure::Random(7..=7 * n as u64)));
let its = IteratedSearch::new(ts, NoImprovement::new(200),
    perturb_with(|s: &mut Evaluated<Qap>, rng| perm::perturb(s, rng.random_range(7..=n / 2), rng)),
    Improving);
let mut restarts = IteratedSearch::new(its, NoImprovement::new(100),
    perturb_with(|s: &mut Evaluated<Qap>, rng| perm::shuffle(s, rng)),
    Always);

let mut st = State::new(Qap::new(&inst), seed);
restarts.runner(&mut st).until(MaxIterations::new((n as f64).sqrt() as u64)).run();
println!("{}", st.best().cost());                 // == st.global_best() at top level
```

`perturb_with` fixes the closure's argument types, so only the solution type
needs writing. A perturbation receives `&mut Evaluated<S>` and
`&mut dyn RngCore`: a concrete RNG type is what lets the closure call
`rng.random_range(..)` without annotations, and the indirect call happens once
per perturbation, not per move. For models with delta formulas, applying moves
one by one (`s.apply(&m)`) keeps the cost exact at O(1) each. For models
without, apply them all inside one `s.modify(|s| …)`, which re-evaluates
once; `perm::perturb` and `perm::shuffle` do that internally. Either way the cache stays valid.

### 3.10 Writing a new metaheuristic

This is a complete VND over two neighborhoods with different move types
(`tests/custom_search.rs`). It needs no framework changes:

```rust
struct Vnd<N1, N2> { n1: N1, n2: N2 }

impl<S: Model, R: Rng, N1: Neighborhood<S>, N2: Neighborhood<S>> Search<S, R> for Vnd<N1, N2> {
    fn step(&mut self, st: &mut State<S, R>, obs: &mut dyn Observer<S, R>) -> Step {
        let mut v = BestMove::improving(st);          // uses the state's tolerance
        let _ = self.n1.explore(st.current(), &mut v);
        if let Some((m, d)) = v.best { return st.commit(&m, d, obs); }
        let mut v = BestMove::improving(st);
        let _ = self.n2.explore(st.current(), &mut v);
        match v.best { Some((m, d)) => st.commit(&m, d, obs), None => Step::Stuck }
    }
    fn start(&mut self, st: &mut State<S, R>) {
        let (cur, rng) = st.current_and_rng();
        self.n1.start(cur, rng);
        self.n2.start(cur, rng);
    }
}
```

The algorithm doesn't touch the iteration counter or the best solution, and it
can't bypass the cost cache. `commit` and the runner handle all three.

### 3.11 Large neighborhood search

A large neighborhood is explored by an operator, not enumerated: destroy part
of a copy of the solution and repair it. `LargeNeighborhoodSearch` does one
rebuild per iteration and lets any `Acceptance` rule (§3.7) decide:

```rust
pub trait Rebuild<S: Model> {
    fn rebuild(&mut self, s: &mut Evaluated<S>, rng: &mut dyn RngCore);   // in place
}
// rebuild_with(|s: &mut Evaluated<P>, rng| ..) for closures

LargeNeighborhoodSearch::new(perm::BestRearrangement::new(5), LateAcceptance::new(10 * n))
    .runner(&mut st).until(MaxIterations::new(1_000_000)).run();
```

The rebuilt candidate is a whole solution, not a `Move`, so it enters the
state through `State::replace_current(&candidate, obs)`, which records the
best and notifies like `commit` (in debug builds it checks the candidate's
cached cost against a full evaluation). The search keeps one candidate buffer
and reuses it with `clone_from`.

### 3.12 Memetic search: crossover instead of restarts

`Memetic` replaces the random restarts of an outer `IteratedSearch` with
recombination. It is a steady-state population of `size` solutions:

1. A starting point: while the population is filling, `init` (e.g. a
   shuffle); then a `Crossover` of two random members.
2. The inner search improves it in its own best-scope, exactly as an
   `IteratedSearch` run (§3.9).
3. The result enters the population by one of three rules:
   * default: it replaces the worst member if it is better and its cost is
     not already present (a cheap duplicate test, which needs no
     `PartialEq` on the model);
   * `with_distance(d, min)`: a result closer than `min` to some member
     competes with that member only (it replaces it if better); a result far
     from every member replaces the worst if better. Each region of the
     search space keeps its own best, which is how strong memetic methods
     for the QAP and for colouring keep diversity;
   * `replace_parent()`: differential evolution's one-to-one selection: the
     result replaces its first parent if better, and nothing else. The
     population fills unconditionally (equal costs can't stall it). With two
     members this is the scheme of HEAD, the simple record-holding colouring
     method (Moalic & Gondran 2018).
4. After `renew_after(k)` steps without an insertion, every member but the
   best is replaced by fresh solutions.

```rust
pub trait Crossover<S: Model> {
    /// `child` holds a copy of `a`; the library re-evaluates it afterwards.
    fn cross(&mut self, a: &S, b: &S, child: &mut S, rng: &mut dyn RngCore);
}
// crossover_with(|a: &P, b, child, rng| ..) for closures

Memetic::new(iterated_tabu, NoImprovement::new(20), perturb_with(|s: &mut Evaluated<P>, rng| shuffle(s, rng)),
             perm::UniformCrossover, 10)
    .renew_after(20)
    .runner(&mut st).until(MaxIterations::new(50)).run();
```

`Memetic` is a `Search`, so it nests and runs like any other.

### 3.13 Permutation problems

`perm::Permutation` exposes the permutation itself (`as_slice`, required;
`size` is provided), so generic components can read which element sits
where. Changes still go through `swap`, which keeps a model's incremental
structures (a delta matrix) valid.

| component | what it is |
|---|---|
| `RobustTabuSearch` | Taillard's robust tabu search on swaps: tabu attributes on assignments (element at position), tenure re-drawn in [0.9n, 1.1n], a long-term aspiration that *forces* moves re-creating assignments unseen for 5n² iterations. Its memory persists across runs, so as the inner search of an iterated or memetic search it remembers the whole history. |
| `BestRearrangement` | A very large neighborhood for LNS: k random positions, all k! − 1 *other* arrangements of their elements tried by Heap's algorithm (one swap per neighbor), move to the best even if worse. C(n, k)·k! neighbors in all. |
| `UniformCrossover` | Merz & Freisleben's crossover: common assignments kept, other positions take a parent's element at random when free, the rest filled at random. |
| `adaptive_perturbation(n)` | `perturb` with `ils::Adaptive` strength: from n/10 swaps, growing while the incumbent is unchanged, reset on change. `Adaptive` itself takes any `(s, strength, rng)` operation. |
| `PathRelinking { fraction }` | Differential evolution's `a + F·(b − a)` for permutations: the child starts as `a` and takes `b`'s element at a random `fraction` of the positions where they differ. |
| `hamming(a, b)` | Positions where two permutations differ: a distance for `Memetic::with_distance`. |
| `rearrange(s, target)` | Move to any target permutation with at most n − 1 swaps (used by the crossovers). |

`RobustTabuSearch` is its own `Search`, not a `TabuList` for `TabuSearch`:
a tabu list can only make moves admissible or not, while RoTS's long-term
aspiration forces a worsening move. `BestRearrangement` never keeps the
current arrangement: if it could, a rebuild could never worsen the solution,
every acceptance rule would degenerate to a descent, and the search would stop
at the first local optimum of the neighborhood (measured: 2.35% on `tai60a`,
unchanged from 20,000 to 100,000 steps).

### 3.14 Dynamic tabu tenure

`DynamicTabu::new(|s: &S, rng| tenure)` is a `DenseTabu` whose tenure is
computed at every move from the solution before the move. TabuCol's rule,
`random(0..10) + 0.6 · conflicting vertices`, is one line with it
(`examples/coloring.rs`). `Tenure::Fixed`/`Random` remain for the common case.

### 3.15 Parallel runs

```rust
let results = parallel::run_many(runs, jobs, |run, cancel| {
    let mut st = State::new(model(run), run);               // the run index seeds everything
    search.runner(&mut st).until(MaxIterations::new(50).or(cancel)).run();
    st.global_best().cost()
});                                                         // Vec<(run, result)>, in run order
```

Each run owns its state and seed, so the results don't depend on the number of
threads. `&Cancel` is a termination criterion: one run calling
`cancel.cancel()` (e.g. on reaching a target cost) stops the others, inner
searches included if they observe it too, and runs not started yet are
skipped.

### 3.16 A worked example: the CVRP iterated tabu search

`examples/cvrp.rs` ports Cordeau & Maischberger's (2012) iterated tabu
search, which was first written on the C++ METSlib. It is the example
that uses most of the tabu machinery at once, and the reason for two
additions:

```rust
// a score that ranks moves without changing the cost that is applied
TabuSearch::new(InterRoute::new(geni_p), DenseTabu::new(Tenure::Random(0..=tau)))
    .with_aspiration(AttributeAspiration(mem.clone()))
    .with_score(move |s: &Evaluated<Routes>, mv: &Relocate, delta: f64| {
        if delta < 0.0 { return delta; }                     // improving: unpenalised
        let r = freq[mv.c * s.m + mv.to] as f64;             // times (c, route) was added
        (s.cost() + delta) * (1.0 + zeta * r / lambda) - s.cost()
    });
// per-iteration bookkeeping that may change how the model evaluates
adjusting(tabu, |st: &mut State<Routes>, _: Step| {
    /* frequencies, aspiration table, best feasible, then: */
    st.modify_current(|e, _| e.modify(|r| { r.penalties.capacity *= 1.0 + delta; /* or ÷ */ }));
});
```

* **`with_score`** (`tabu::Score`, default `NoScore`). The frequency
  penalty g(s′) changes which move is chosen, not the cost the move
  produces. Before, a neighbourhood could only report the true delta or
  corrupt the cost cache. A score that is not the delta switches off
  bound-based pruning (`Score::IS_DELTA`). The same hook serves guided
  local search and any long-term frequency diversification.
* **`Adjusting`** (`search::adjusting(inner, |st, step| ..)`) runs one
  inner step and then the closure with `&mut State`. Here it multiplies
  or divides the capacity and duration weights by 1 + δ every
  iteration. The weights live inside the model, so `evaluate()` is f(s),
  and `modify_current` re-evaluates. Every 200 iterations it also runs
  the route refinement. It closes the reactive tabu point of §6: a hook
  with `&mut` access to the state.
* **Attribute tabu and aspiration**, with no new library code. The
  attribute is (customer, route), as `TabuAttr::key = c·m + to` with
  `reverse_key` the route the customer leaves, so `DenseTabu` forbids
  putting it back. The aspiration is a user `Aspiration` that allows a
  tabu move when f(s′) beats the best f seen with the move's new
  attribute. The table it reads is updated in the `Adjusting` hook.
* **GENI** (`examples/common/geni.rs`, `--insertion geni`), Gendreau, Hertz
  & Laporte's insertion types I and II, is a pure function of one route's
  sequence. So the neighbourhood caches each customer's best insertion
  per route under a per-route version number, and a move recomputes two
  routes, not m.

The phase loop (stop after μ = √((η − λ)·π) iterations without
improvement, acceptance with probability 1 − (λ/η)², cluster
perturbation) is written in the example rather than with
`IteratedSearch`: all three rules read the global iteration count λ and
budget η.

---

## 4. Performance

Benchmark: tabu search on the full swap neighborhood of a random asymmetric
QAP, tenure n, best-ever aspiration, starting from the identity
(`examples/qap.rs --bench`). The C++ side is METSlib's
`tabu_search<swap_full_neighborhood>` with `simple_tabu_list` and a QAP model
using **the same O(n) swap-delta formula**. It was compiled with
`g++ 13 -O3 -march=native`, and the Rust side with `-C target-cpu=native`,
LTO. Times are µs per iteration on a shared cloud VM, two runs each.

| n | C++ METSlib | Rust, same model (`DenseTabu`) | Rust, Taillard delta matrix |
|---|---|---|---|
| 12  | 7.2–8.4   | **3.0–3.2** | 1.4–1.8 |
| 30  | 96–103    | **47–52**   | 16–17 |
| 60  | 383–423   | 380–441     | **39–40** |
| 100 | 1541–1883 | 1804–1865   | **107–108** |

* **The machine is noisy.** The same C++ binary measured 64–70 µs at n = 30
  in an earlier session and 96–103 µs in this one. Compare within a row, not
  across sessions.
* **The revised API costs nothing measurable.** The first-version prototype
  measured 2.7–3.0 / 47–48 / 397–429 / 1800–1920 µs for the same model, and
  reached the same best costs at every size: the search trajectories are
  identical.
* Both implementations follow **the same trajectory** as C++ for n = 12, 30
  and 60 (same best cost). At n = 100 they diverge slightly, because
  `simple_tabu_list` bounds the number of *distinct* tabu moves while `mets`
  uses expiry stamps.
* At small n, where the framework dominates, `mets` is 2.3–2.8× faster: no
  virtual calls, no heap moves, lazy tabu checks, no double delta
  evaluation. At larger n the user's O(n) delta loop dominates and the two are
  within noise; earlier sessions measured the Rust model 5–15% slower there,
  because of bounds checks in that loop (with unchecked indexing, tried as an
  experiment, it matched C++).
* Moving the delta trick into the model is worth far more: the Taillard model
  is 14–17× faster at n = 100, with no special support from the framework.
* Other figures (release builds): the VCP port finds a 17-colouring of
  G(125, 0.5) in 0.03 s and runs about 1.5 M tabu iterations/s with 16
  colours; the stochastic searches run about 3–4 M sampled moves/s on the
  n = 60 QAP.
* **Debug builds are much slower**, by design: every committed move is
  checked with two full evaluations. VCP with n = 60 runs about 18 k
  iterations/s in debug and 1.5 M/s in release.

### Solution quality on QAPLIB

`qap --qaplib DATA SOLN --restarts 200` runs the iterated tabu search of
`qap/main.cc` on each of the 136 QAPLIB instances: 200 random restarts, each
an iterated tabu search that stops after 20 perturbations without
improvement, whose tabu runs (Taillard model, random tenure) stop after 200
iterations without improvement. There is no time limit, so the results
depend only on the seed and are reproducible. The per-instance table is in
[`QAPLIB.md`](QAPLIB.md); the summary over the 128 instances with a
published solution:

| family | instances | best known reached | mean gap | max gap |
|---|---|---|---|---|
| bur, chr, els, esc, had, kra, nug, rou, scr, ste | 68 | 68 | 0.00% | 0.00% |
| lipa | 16 | 15 | 0.03% | 0.43% |
| sko | 13 | 4 | 0.04% | 0.10% |
| tai | 26 | 17 | 0.13% | 0.90% |
| tho | 3 | 2 | 0.03% | 0.10% |
| wil | 2 | 0 | 0.05% | 0.08% |
| **all** | **128** | **106** | | |

* The remaining gaps are on the hard classes: the uniform random `tai*a`
  instances (up to 0.90% on `tai100a`) and the large `sko`, `tai*b`, `tho`
  and `wil` instances (all under 0.3%).
* The run took 3.6 CPU-hours on a shared VM, from under a second for small
  instances to 43 minutes for `tai256c`.
* With a 1 s budget per instance instead, the same search reaches the
  best-known value on 88 instances, and the deceptive `lipa*b` instances can
  stay ~20% above their optimum.

* On `tai60a` with the same 1 s budget and 3 seeds (`qap tai60a.dat
  tai60a.sln`), mean gaps are: restarts(ITS) 1.33%, full-neighborhood tabu
  1.63%, simulated annealing 1.89%, ILS 2.30%, late acceptance 2.67%,
  record-to-record travel 2.68%, sampled tabu (`main_ts.cc`) 2.96%. Tabu
  search evaluates about 4·10⁷ moves/s with the Taillard model; the
  sampling methods about 3–4·10⁶ with the O(n) delta.

The published solutions need care. Most `.sln` files list the permutation in
the direction of the cost formula, but 8 list its inverse (`esc128`,
`kra30a`, `kra30b`, `ste36c`, `tai60a`, `tai80a`, `tho150`, `tho30`).
`kra32`'s header says 88,900 while its own permutation costs 88,700, and
`tai40a`'s permutation is invalid (position 40 is missing). The example
checks both directions, uses the lower of the header and the verified cost,
and prints a note when they disagree.

### Advanced QAP solvers

`examples/qap_advanced.rs` assembles three solvers from §3.11–3.14, on the
symmetric fast path of the shared QAP models (2 products per delta term
instead of 4 for symmetric, zero-diagonal instances: 20.9 instead of 39 µs per
full-neighborhood tabu iteration on `tai60a`). Each run below is 4
independent runs on 4 threads, best of them; times are wall-clock.

| instance | restarts of ITS(RoTS) | memetic, population 10 | memetic, population 5 |
|---|---|---|---|
| sko72 | 0.01% (36.8 s) | **0.00%** (33.2 s) | **0.00%** (25.4 s) |
| sko81 | 0.01% (53.1 s) | 0.01% (44.7 s) | 0.01% (37.3 s) |
| tho40 | 0.00% (10.4 s) | 0.00% (8.2 s) | — |
| lipa90a | 0.00% (89.1 s) | 0.00% (80.2 s) | — |
| tai50a | **0.57%** (12.5 s) | 0.69% (12.0 s) | 0.61% (11.7 s) |
| tai60a | **0.62%** (18.1 s) | 0.77% (17.2 s) | — |

* Same budget for both: 50 inner improvement runs per run (restarts: 50
  restarts; memetic: 10 random starts, then 40 children).
* Crossover helps on the structured `sko` instances, reaching `sko72`'s
  best-known value, and is a little faster. Restarts stay better on the
  uniform random `tai*a` instances, which have little common structure for a
  child to inherit.
* Large neighborhood search with `BestRearrangement` is correct but not
  competitive on the QAP: on `tai60a` the best setting found (k = 5, late
  acceptance over 10n iterations, 1.5 million rebuilds per run) reaches 1.68%
  in 17 s, against 0.62% for tabu restarts in 18 s. Larger k (6, 7) and
  shorter acceptance histories were worse at the same time. This matches the
  literature, where tabu search on swaps is the strongest local method for
  the QAP.
* Long runs to the end (4 runs on 4 threads; `memetic` with 2,000 children
  per run, or `ma` with 10⁶) reach the best-known value **exactly** on 12
  open instances that the 200-restart table missed, and never go below it
  (tai150b: `ma`, two jobs sharing 4 cores, stopped by `--max-seconds 3300`):

  | instance | best known | method, wall time |
  |---|---|---|
  | tai80b | 818,415,043 | memetic, 24 min |
  | sko81 | 90,998 | memetic, 16 min |
  | sko90 | 115,534 | memetic, 18 min |
  | wil50 | 48,816 | memetic, 5 min |
  | sko100a | 152,002 | memetic, 25 min |
  | sko100b | 153,890 | memetic, 25 min |
  | sko100c | 147,862 | memetic, 25 min |
  | sko100d | 149,576 | memetic, 23 min |
  | sko100e | 149,150 | memetic, 25 min |
  | sko100f | 149,036 | ma, 28 min |
  | wil100 | 273,038 | ma, 27 min |
  | tai150b | 498,896,643 | ma, ≤ 55 min |

  Ties, not records: these values have stood for decades. tho150 under the
  same 55-minute budget ended at 8,133,702, 0.004% above its 8,133,398.
* Speed against Taillard's robust tabu search in C (`gcc -O3
  -march=native`, same parameters and update), 10,000 iterations, one core:

  | instance | n | C, it/s | `RobustTabuSearch` + `QapTaillard`, it/s |
  |---|---|---|---|
  | tai50a | 50 | 64,000 | 84,000 |
  | tai100a | 100 | 16,000 | 21,000 |
  | tai150b (asymmetric) | 150 | 7,600 | 6,600–7,200 |
  | tai256c (asymmetric) | 256 | 1,900 | 1,900 |

  The library's tabu search costs nothing over hand-written C. The model's
  update reads only rows: Taillard's O(1) term factors into per-position
  differences computed once per swap, and columns are read from stored
  transposes. A 50-minute `memetic` run (4 runs, 4 threads) on tai100a ends
  at 21,118,120, 0.31% above the best known: the "a" instances, where
  records still move, need a stronger search, not a faster one.
* Cohesive crossover (`perm::CohesiveCrossover`, Drezner; `--crossover
  cohesive`): on sko100a with `ma` (4 runs, 4 threads), time to the best
  known over 5 seeds, uniform 43.6 s mean (45.4 median), cohesive 57.9 s
  (53.2). No gain with `ma`'s first-improvement descent; Drezner pairs it
  with a tabu search, which `memetic` and `qap_gpu` use.
* tai50a reaches its best-known value, 4,938,796, with `memetic` (iterated
  robust tabu, population 10, 4 runs on 4 threads) in 756 s: the first tie
  on the random "a" family. Directed perturbation (`perm::DirectedPerturbation`,
  Benlic & Hao's breakout local search, `--perturbation directed`) did not
  help in this budget: tai50a 4,941,410 in 15 minutes; tai60a 7,217,850
  (0.16%) with either perturbation. One run per arm.
* tho150 reaches its best-known value, 8,133,398, with `qap_gpu`
  (population 112, 200n robust tabu iterations per child, cohesive
  crossover, symmetric kernel, delta matrix in shared memory: 93 KB per
  block, 1 block per SM, 28 at a time; build 70d0fcf) on an RTX 3060 in 49
  minutes. The best CPU result before was 8,133,702.
* Exhaustive k-opt polish (`qap_gpu --kopt K`, `--polish-from`): on a
  tho150 permutation at 8,141,304, 640 groups of 7 close positions held no
  improving rearrangement (CPU). Tabu solutions are 7-opt optimal on such
  groups; the GPU at k = 9 searches far more.
* Islands (`--migrate-every G`, `parallel::islands`): 4 islands against 4
  independent `ma` runs, wall time to the best-known value, 5 trials each:

  | instance | independent | migrate every 50 | migrate every 500 |
  |---|---|---|---|
  | tai100b, avg (median) | 9.4 s (8.5) | 14.0 s (11.7) | 8.9 s (8.4) |
  | sko100a, avg (median), reached | 43 s (45), 5/5 | 116 s (69), 5/5 | 37 s (16), 4/5 |

  Frequent migration hurts: the best member spreads to every island and the
  synchronised exchange makes fast islands wait for slow ones. Rare migration
  is on par with independent runs within this noise.
* Migrants as partners instead of replacements (`--migration cross|part`,
  `Memetic::receive`): the migrant is the second parent of the next child.
  With `part` the child takes only 20% of the migrant's differing assignments
  (path relinking), a guided perturbation. Same 5 trials:

  | configuration | sko100a: reached, avg (median, max) | tai100b: avg (median) |
  |---|---|---|
  | independent runs | 5/5, 43 s (45, 98) | 9.4 s (8.5) |
  | cross, every 50 | 5/5, 96 s (61, 219) | 10.5 s (12.1) |
  | cross, every 50, async | 4/5, 52 s (74, 94) | 10.3 s (9.1) |
  | part, every 50 | 5/5, 26 s (24, 41) | 9.6 s (8.9) |
  | part, every 50, async | 5/5, 60 s (63, 76) | 8.2 s (8.1) |
  | part, every 500 | 5/5, 18 s (17, 34) | 23.5 s (17.9) |

  Taking only part of another island's best is about 2× faster and much more
  consistent on `sko100a`; copying a migrant whole (replace, cross) hurts.
  `tai100b` is solved fast by everything and doesn't separate the variants.
  Confirmation on 10 new seeds per configuration:

  | instance | independent | part, every 50 | part, every 500 |
  |---|---|---|---|
  | sko100a | 10/10, 37 s (26) | 9/10, 64 s (35) | 10/10, 21 s (15) |
  | sko90 | 10/10, 31 s (31) | 8/10, 28 s (31) | 8/10, 50 s (59) |
  | tai80b | 10/10, 2.2 s (2.4) | 10/10, 3.0 s (3.2) | 10/10, 3.0 s (2.2) |

  Partial migration is the best way to migrate, but not better than
  independent runs in general: it is about 2× faster on `sko100a` (15 trials
  in all), and worse on `sko90` (2 misses in 10) and `tai80b`. It stays
  available (`--migration part --migrate-every 500`) and off by default.
* Time to the best-known value, one core per run, 5 seeds, memetic search
  stopping at the target (`--runs 1 --jobs 1 --first-seed s`), against
  Merz & Freisleben's memetic algorithm (CEC 1999, Table 2: one core of a
  300 MHz Pentium II, 30 runs):

  | instance | ours: reached, avg (min–max) | theirs: reached, avg (min) |
  |---|---|---|
  | bur26a | 5/5, 0.04 s (0–0.1) | 30/30, 1.0 s (0.2) |
  | chr25a | 5/5, 0.3 s (0.1–0.5) | 29/30, 2.6 s (0.3) |
  | kra30a | 5/5, 0.06 s | 30/30, 2.7 s |
  | nug30 | 5/5, 0.1 s | 27/30, 7.1 s |
  | ste36a | 5/5, 0.08 s | 17/30, 36.7 s |
  | tai60b | 5/5, 6.5 s (1.7–12.8) | 30/30, 23.2 s (7.5) |
  | tai80b | 5/5, 52 s (31–85) | 20/30, 258 s (55) |
  | tai100b | 5/5, 95 s (33–282) | 14/30, 629 s (58) |
  | sko100a | 5/5, 233 s (101–384) | 0/30 in 161 s |

  Our timer has 0.1 s resolution. A Pentium II at 300 MHz is very roughly
  30–50× slower per core than this VM, so after normalising this robust-tabu
  memetic search is comparable or faster on the small instances, more
  reliable everywhere, but *slower per unit of hardware* on the large `tai*b`
  instances (3.5–6.6× raw). Their algorithm is cheaper per child.
* `--method ma` reproduces it from library pieces: `Memetic` (population 40,
  uniform crossover, Hamming distance, renewal) around `perm::SwapDescent`
  (first-improvement swaps with don't-look bits) on the plain model. Same
  measurement (5 seeds, one core):

  | instance | `ma`: reached, avg (min–max) |
  |---|---|
  | tai60b | 5/5, 0.9 s (0.3–2.4) |
  | tai80b | 5/5, 9.0 s (5.1–15.7) |
  | tai100b | 5/5, 36 s (8.7–71) |
  | sko100a | 4/5, 96 s (45–164); one run ended its 100,000 children 34 above |
  | bur26a, chr25a, kra30a, nug30, ste36a | 5/5, ≤ 0.3 s |

  That is 3–7× faster than the robust-tabu memetic search on `tai*b`, and
  18–29× faster than the 1999 results in raw time: about on par per unit of
  hardware, as a faithful reimplementation should be, and more reliable.
* Compared with the `QAPLIB.md` setup (`qap`, 200 restarts of position-pair
  tabu search), restarts of robust tabu search reach better costs on 9 of 12
  hard instances at about the same CPU time, including `tai100b`'s
  best-known value.

### Graph colouring (DIMACS)

`examples/coloring.rs` reads DIMACS `.col` files, starts from a DSATUR
colouring, and runs Galinier & Hao's hybrid evolutionary algorithm for
decreasing k: `Memetic` (population 10, GPX crossover, a partition
distance for diversity) around TabuCol (`TabuSearch` on recolourings of
conflicting vertices with `DynamicTabu`). 4 independent runs on 4 threads;
the first legal colouring cancels the others. Every colouring reported is
re-checked for legality.

| instance | best known | found | time |
|---|---|---|---|
| DSJC125.5 | 17 | **17** | 0.1 s |
| DSJC250.5 | 28 | **28** | 1.4 s |
| DSJC500.5 | 47 | 49 | 5 s (48 not found in 23 min) |
| flat300_28_0 | 28 | 31 | 1.5 s (30 not found in 7.5 min) |
| le450_25c | 25 | 26 | 2.9 s (25 not found in 10 min) |

With HEAD's scheme (`--head`: population 2, children always replace their
parent, elites, random tie-breaking in TabuCol via `TabuSearch::random_ties`,
60n iterations per child, as in the authors' code), `DSJC500.5` reaches **48
colours in 11.6 s**, which the 1999 setting did not find in 23 minutes. 47
(the record, from HEAD itself and quantum annealing) was not found in two
one-hour runs. On the queen graphs, `queen15_15` reaches 16 and
`queen16_16` 17 colours in 1.4 s, one above their known chromatic numbers
(15 and 16, settled by SAT up to n = 26).

No record has been beaten. The best-known values come from much longer runs
(hours to weeks per instance) of methods of the same family. A third
one-hour `--head` run at k = 47, after the TabuCol speed-up below, did not
find 47 either.

**TabuCol speed.** The model keeps the list of conflicting vertices and
each vertex's smallest γ (a lower bound on its recolouring deltas), and
`Visitor::can_skip` lets the tabu search prune with random ties: 210,000 →
337,000 iterations/s on DSJC500.5 at k = 48 (`coloring --bench`, one core).

**Variable Space Search.** `examples/common/coloring.rs` adds PartialCol
(Blöchliger & Zufferey): legal partial colourings, cost = uncoloured
vertices, move = colour an uncoloured vertex and uncolour its neighbours of
that colour, tabu on giving those neighbours the colour back. It is a
`Model`, a `Move`, a custom `TabuList` (one move makes several keys tabu)
and a `Neighborhood` using `can_skip`, run by the library's `TabuSearch`.
`vss()` alternates it with TabuCol (Hertz, Plumettaz & Zufferey). From the
same stuck TabuCol solutions (DSJC500.5, k = 48, 1 M iterations from 8
random starts), 2 M more iterations of TabuCol end at 8.1 conflicts on
average; 20 VSS rounds of 50,000 iterations per space end at 6.5, never
worse on any seed, in 40–60% less time (PartialCol iterations are cheaper).
`coloring --vss R` applies it to the runs of a failed k; `coloring_gpu
--vss-rounds R` after each polish, on the CPU threads. On the GPU it did
not help where it was meant to: DSJC500.5 at 47 reaches 1 conflict in
about half an hour with or without polish and VSS, and 52 VSS rounds from
that 1-conflict solution over a 4-hour run removed nothing
(`reports/2026-11-06-gpu-q6-sweep-and-a3.md`). Both stay off by default.

**GPU (`examples/coloring_gpu.rs`, `--features gpu`).** HEAD runs as many
independent pairs; each generation's children are improved in one batch by
a CUDA kernel (one block per individual, compiled at start-up by NVRTC
through `cudarc`) or, with `--cpu`, by the library's `TabuSearch` on the
CPU threads. `--trace` logs each batch with a CPU recount of returned
colourings, and the build's git commit. On an RTX 3060 (28 SMs), DSJC500.5,
k = 48, a batch of 128 runs × 5,000 iterations:

| kernel | per batch | TabuCol it/s |
|---|---|---|
| tables in global memory | 0.55 s | 1.2 M |
| 16-bit conflict table and colouring in shared memory (2 blocks per SM) | 0.155 s | 4.1 M |
| + conflicting vertices listed, their moves shared by all threads | 0.06 s | 10–11 M |

10 M iterations/s is about 30 of the 4-core sandbox's cores; a legal
48-colouring took 12.9 s. Every recount matched across all test runs.

On DSJC1000.5 at k = 82 the 16-bit table (163 KB) does not fit the 99 KB
of shared memory a block may use. The 8-bit table (84 KB, counters
saturating at 255, which fails the run) fits, but leaves room for one
block per SM (`reports/2026-11-06-a2-smoke-test.md`, 10 minutes each):

| tables, threads per block | blocks resident | per batch of 224 × 60,000 iterations | TabuCol it/s |
|---|---|---|---|
| 16-bit in global memory, 128 | 9 per SM | 11.3 s | 1.2 M |
| 8-bit in shared memory, 128 | 1 per SM | 47 s | 0.27 M |
| 8-bit in shared memory, 512 | 1 per SM | 10.6 → 8.1 s | 1.27 → 1.50 M |
| 8-bit in shared memory, 1,024 | 1 per SM | 7.8 → 5.6 s | 1.73 → 2.19 M |

With one block per SM, the block must be as wide as the SM holds. A
shared table pays only with enough warps behind it, not merely while it
leaves two blocks per SM. So `--threads` now follows the table: 1,024
when shared memory leaves room for one block per SM, 128 otherwise.
(Batches get faster during a run as the conflict lists shrink, so the
rates are bounds.)

### Vehicle routing (CMT and Golden)

`examples/cvrp.rs` (§3.16) on the 14 CMT and 20 Golden instances, with
real-valued Euclidean distances as in the paper. Gaps are to the CVRPLIB
best known, and every saved solution is recounted by
`scripts/verify_vrp.py`, which shares no code with the solver. The
fleet is the reference solution's routes plus one spare vehicle.
Insertion is plain best insertion, and route refinement is 2-opt and
Or-opt. Iterations are the unit, because the paper's machine differs.

| set, η tabu iterations | ours | paper (Table 9, ITS) |
|---|---|---|
| CMT 1–14, 10⁵, 10 seeds | **0.50%** | 0.60% |
| CMT 1–14, 10⁶, 10 seeds | **0.26%** | 0.27% |
| Golden 1–20, 10⁵, 1 seed | 2.10% | 1.30% |
| Golden 1–20, 10⁶, 1 seed | 1.37% | 0.79% |

* **CMT:** the port matches the paper at both budgets, so it is
  accepted there. The residual is CMT5, 10 and 13, at about 1% each.
* **Golden:** the gap falls by ×0.65 from 10⁵ to 10⁶, against the
  paper's ×0.61. The search dynamics scale like the paper's, so the
  deficit points to the neighbourhood: the paper uses GENI insertion.
* **Speed:** plain insertion runs 10⁵ iterations in 1–15 s on CMT on one
  sandbox core; the paper reports 188 s on average. GENI costs about
  100× per iteration on CMT, which is the paper's cost scale.

The Golden comparison with GENI is in
`reports/2026-11-06-p7-acceptance-and-diagnostics.md` and the reports
after it.

The C++ benchmark source is not committed. It is a 40-line program that
includes `metslib/mets.hh`, and can be added to METSlib's `test/` if useful.

---

## 5. Changes made during implementation

The reviewed design was implemented as written, except for these points,
which the compiler or ergonomics forced:

| Reviewed design | Implemented | Why |
|---|---|---|
| Every `MoveSource` is a `Neighborhood` (blanket impl) | `WithDeltas(source)` adapter; `moves_fn` closures are neighborhoods directly | The blanket impl conflicts with the crate's own neighborhoods (`SwapNeighborhood`, `Sampled`, `FullEvaluation`): a downstream crate could implement `MoveSource` for them. |
| `Observer` implemented for closures | closure observers via the runner's `on_move`, `on_improvement`, `on_global_improvement`, `on_event`; free functions of the same names | A closure impl overlaps with `impl Observer for &mut O` (`&mut F` is also a closure). Closures passed to free functions also can't be type-checked before `S`/`R` are known; on the runner they are. |
| `stop_when(\|st\| ..)` as the closure criterion | also `runner.stop_when(..)`, OR-ed with `until` | Same inference reason. |
| `State::new(s, seed)` for any seedable `R` | `State::new` for `DefaultRng`; `State::with_rng` for others | A type-parameter default does not drive inference, so `State::new(s, 42)` would need annotations. |
| Perturbation closures get `&mut R` | they get `&mut dyn RngCore` | Lets `rng.random_range(..)` type-check without annotations; once per perturbation, so the indirect call is irrelevant. |
| Aspiration implemented for closures | `BestEver`, `NoAspiration`, or implement the trait | `allows` is generic over the RNG type, which a closure can't be. |
| `st.current()` and `st.rng()` used together | `st.current_and_rng()` | Borrowing rules: an immutable and a mutable borrow of the state at once. |
| `Neighborhood` implemented for an RNG-owning `Sampled { .. }` | `Sampled::new(sampler, k)`, seeded in `Neighborhood::start` | Constructor style and reproducibility. |
| `Permutation` with `size()` and `swap()` | `as_slice()` required, `size()` provided | Robust tabu search and the crossover need to read which element is at each position. |
| A rearrangement neighborhood that keeps the best arrangement, current included | `BestRearrangement` always moves to the best *other* arrangement | Otherwise a rebuild never worsens, every acceptance rule becomes a descent, and LNS stops at the first local optimum (§3.13). |
| The duplicate test as the memetic diversity rule | also `with_distance` and `replace_parent` | With replace-parent and the duplicate test together, a population of equal-cost members never filled, so no crossover ever happened (found in testing). |
| Robust tabu search as a `TabuList` for `TabuSearch` | its own `Search` (`perm::RobustTabuSearch`) | Its long-term aspiration forces worsening moves, which a tabu list cannot express. |

---

## 6. Open points / next steps

* **Parallel neighborhood evaluation** (feature `rayon`). Add a
  `ParNeighborhood` with `explore_par(&self, s, chunk) -> Option<(M, C)>`.
  Best-move selection is a reduction, and the tabu check is read-only, so
  `TabuSearch`/`LocalSearch` can use it when `S: Sync`. `FullEvaluation`
  would keep one scratch copy per thread.
* **Neighborhood combinators**: `Union<A, B>` with an `Either` move, a
  `Shuffled` adapter (random order + first improvement), and a built-in `Vnd`.
* **More move kinds** in `perm`: insertion, 2-opt/or-opt with *problem-supplied*
  deltas or full evaluation, which avoids the `invert_subsequence` problem
  from §1.
* **Lexicographic costs** `(hard, soft)`. A pair type can implement the
  `num-traits` bounds, with a lossy `to_f64` (e.g. `hard * BIG + soft`), so
  the blanket `Cost` impl doesn't block it. The open question is whether
  `Mul`/`Div`, which `Signed` requires, should exist for such a type at all;
  only the SA-style acceptance rules would use the `f64` projection. Decide
  between shipping a `Lexicographic<H, S>` type and narrowing `Cost` to
  `Add + Sub + Neg + Zero + Bounded + ToPrimitive`.
* **Hybrid uses of large neighborhoods.** On the QAP, `BestRearrangement`
  in an LNS loses to tabu search (§4). It may do better as an intensification
  step (applied to the best solution of each inner run) or as the
  perturbation of an iterated search; neither is measured yet.
* **Reactive tabu search**: observers get `&State` only. `Adjusting`
  (§3.16) gives a per-step hook `&mut State`, which covers penalty weights
  inside the model. A tenure that reacts still needs `&mut` access to the
  tabu list, which the hook does not have.
* **Naming / packaging**: crate name (`mets`? `metslib`?) and whether to
  publish on crates.io. The license follows the C++ library (GPL-3.0-or-later
  or CPL-1.0, see `COPYING` and `COPYING-CPL`).
