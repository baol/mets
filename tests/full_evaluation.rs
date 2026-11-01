//! Models without delta formulas, and delta checking.
mod common;

use common::*;
use mets::perm::{Swap, SwapNeighborhood};
use mets::*;

fn all_swaps(n: usize) -> impl Iterator<Item = Swap> {
    (0..n).flat_map(move |i| (i + 1..n).map(move |j| Swap { i, j }))
}

/// Best-improvement descent with a given neighborhood: the result must not
/// depend on how the deltas are obtained.
fn descend<S: Model, N: Neighborhood<S>>(s: S, n: N) -> (S::Cost, u64) {
    let mut st = State::new(s, 1);
    LocalSearch::best_improvement(n).runner(&mut st).run();
    (st.best().cost(), st.iteration())
}

#[test]
fn every_evaluation_path_gives_the_same_descent() {
    let reference = descend(Sort(reversed(25)), SwapNeighborhood);
    // full evaluation only, clone fallback
    assert_eq!(descend(SortFull(reversed(25)), SwapNeighborhood), reference);
    // iterator neighborhood
    assert_eq!(
        descend(
            SortFull(reversed(25)),
            moves_fn(|s: &SortFull| all_swaps(s.0.len()))
        ),
        reference
    );
    // MoveSource adapters
    assert_eq!(
        descend(
            SortFull(reversed(25)),
            WithDeltas(moves_fn(|s: &SortFull| all_swaps(s.0.len())))
        ),
        reference
    );
    assert_eq!(
        descend(
            SortFull(reversed(25)),
            FullEvaluation::new(moves_fn(|s: &SortFull| all_swaps(s.0.len())))
        ),
        reference
    );
    // FullEvaluation also uses delta formulas when moves have them
    assert_eq!(
        descend(
            Sort(reversed(25)),
            FullEvaluation::new(moves_fn(|s: &Sort| all_swaps(s.0.len())))
        ),
        reference
    );
}

#[test]
fn full_evaluation_models_work_with_every_algorithm() {
    let mut st = State::new(SortFull(reversed(20)), 1);
    TabuSearch::new(SwapNeighborhood, DenseTabu::new(Tenure::Fixed(7)))
        .runner(&mut st)
        .until(Target::new(0).or(MaxIterations::new(2000)))
        .run();
    assert_eq!(st.best().cost(), 0);

    let mut st = State::new(SortFull(reversed(20)), 1);
    StochasticSearch::new(SwapNeighborhood, LateAcceptance::new(20))
        .runner(&mut st)
        .until(MaxIterations::new(50_000))
        .run();
    assert!(st.best().cost() < SortFull(reversed(20)).evaluate() / 4);
    assert_eq!(st.current().cost(), st.current().evaluate());
}

#[test]
fn evaluated_apply_and_modify_keep_the_cache_valid() {
    let mut s = Evaluated::new(SortFull(reversed(8)));
    s.apply(&Swap::new(0, 7)); // no formula: re-evaluates
    assert_eq!(s.cost(), s.evaluate());
    let mut t = Evaluated::new(Sort(reversed(8)));
    t.apply(&Swap::new(0, 7)); // formula: cost += delta
    assert_eq!(t.cost(), s.cost());
    t.modify(|p| p.0.reverse());
    assert_eq!(t.cost(), t.evaluate());
    assert_eq!(t.resync(), 0);
}

#[test]
fn check_deltas_finds_wrong_formulas() {
    assert!(check::deltas(&mut SwapNeighborhood, &Evaluated::new(Sort(reversed(12)))).is_empty());
    assert!(check::deltas(
        &mut SwapNeighborhood,
        &Evaluated::new(SortFull(reversed(12)))
    )
    .is_empty());
    let bad = check::deltas(
        &mut SwapNeighborhood,
        &Evaluated::new(SortBuggy(reversed(12))),
    );
    assert!(!bad.is_empty());
    assert_ne!(bad[0].reported, bad[0].evaluated);
}

#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "wrong delta")]
fn debug_builds_reject_a_wrong_delta_at_commit() {
    let mut st = State::new(SortBuggy(reversed(12)), 1);
    LocalSearch::best_improvement(SwapNeighborhood)
        .runner(&mut st)
        .run();
}

/// Float costs: the cached cost drifts, `resync_every` corrects it.
#[derive(Clone)]
struct Noisy(Vec<f64>);
impl Model for Noisy {
    type Cost = f64;
    fn evaluate(&self) -> f64 {
        self.0.iter().map(|x| x * x * 0.1).sum()
    }
}
#[derive(Clone, Copy)]
struct Nudge(usize, f64);
impl Move<Noisy> for Nudge {
    fn apply(&self, s: &mut Noisy) {
        s.0[self.0] += self.1;
    }
    fn delta(&self, s: &Evaluated<Noisy>) -> Option<f64> {
        let x = s.0[self.0];
        Some(((x + self.1) * (x + self.1) - x * x) * 0.1)
    }
}
struct Nudges;
impl Sampler<Noisy> for Nudges {
    type Move = Nudge;
    fn sample<R: rand::Rng + ?Sized>(
        &mut self,
        s: &Evaluated<Noisy>,
        rng: &mut R,
    ) -> Option<Nudge> {
        Some(Nudge(
            rng.random_range(0..s.0.len()),
            rng.random_range(-0.3..0.3),
        ))
    }
}

#[test]
fn float_costs_with_tolerance_and_resync() {
    let mut st = State::new(Noisy(vec![3.7; 50]), 1)
        .with_tolerance(1e-9)
        .resync_every(1000);
    StochasticSearch::new(Nudges, SimulatedAnnealing::new(1.0, 1e-4, 0.9, 500))
        .runner(&mut st)
        .run();
    let c = st.current();
    assert!((c.cost() - c.evaluate()).abs() < 1e-6);
    assert!(st.best().cost() < Noisy(vec![3.7; 50]).evaluate() / 10.0);
}
