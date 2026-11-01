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
src/ils.rs            IteratedSearch (ILS, iterated tabu search, random restarts)
src/perm.rs           permutation problems: Swap, SwapNeighborhood, perturb, shuffle
src/check.rs          check::deltas, for testing delta formulas
examples/subset_sum   tutorial port (tabu search)
examples/qap          qap/main_ts.cc + qap/main.cc ports, plus SA/LAHC/RRT/ILS
examples/vcp          vcp/vcp.cc port (TabuCol-like iterated tabu search)
tests/                every algorithm, full-evaluation models, delta checks,
                      nesting and scopes, dyn Search, a user-defined VND
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
   tells them which deltas still matter (`bound()`), so they can prune. Plain
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
exposes `bound()`, which an iterator cannot see.

The neighborhood, not the algorithm, produces the delta. That is what makes the
large-neighborhood tricks expressible:

* **Incremental delta structures** live in the solution and are read in O(1).
  `examples/qap.rs` keeps Taillard's delta matrix, so a full O(n²) scan costs
  O(n²) instead of O(n³). `examples/vcp.rs` keeps the TabuCol γ table.
* **Candidate lists / restricted neighborhoods.** VCP only generates recolourings
  of conflicting vertices.
* **Pruning with `bound()`.** VCP skips a vertex entirely when its best
  possible delta (−γ[v][colour(v)]) cannot beat the best candidate so far.
  Partial evaluations or lower bounds plug in the same way. Ignoring `bound()`
  is always correct.
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
* **Reactive tabu search**: observers get `&State` only. Changing the tenure
  from an observer would need `&mut` access to the algorithm. A user loop
  that calls `step` directly already allows it.
* **Naming / packaging**: crate name (`mets`? `metslib`?) and whether to
  publish on crates.io. The license follows the C++ library (GPL-3.0-or-later
  or CPL-1.0, see `COPYING` and `COPYING-CPL`).
