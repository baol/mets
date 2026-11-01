//! Writing a new metaheuristic outside the crate: Variable Neighborhood
//! Descent over two neighborhoods with different move types.
mod common;

use common::*;
use mets::perm::{Permutation, Swap, SwapNeighborhood};
use mets::rand::Rng;
use mets::*;
use std::ops::ControlFlow;

/// The whole algorithm.
struct Vnd<N1, N2> {
    n1: N1,
    n2: N2,
}

impl<S: Model, R: Rng, N1: Neighborhood<S>, N2: Neighborhood<S>> Search<S, R> for Vnd<N1, N2> {
    fn step(&mut self, st: &mut State<S, R>, obs: &mut dyn Observer<S, R>) -> Step {
        let mut v = BestMove::improving(st);
        let _ = self.n1.explore(st.current(), &mut v);
        if let Some((m, d)) = v.best {
            return st.commit(&m, d, obs);
        }
        let mut v = BestMove::improving(st);
        let _ = self.n2.explore(st.current(), &mut v);
        match v.best {
            Some((m, d)) => st.commit(&m, d, obs),
            None => Step::Stuck,
        }
    }
    fn start(&mut self, st: &mut State<S, R>) {
        let (cur, rng) = st.current_and_rng();
        self.n1.start(cur, rng);
        self.n2.start(cur, rng);
    }
}

/// Rotate a block one step left: a second move type.
#[derive(Clone, Copy)]
struct Rotate {
    from: usize,
    to: usize,
}
impl Move<Sort> for Rotate {
    fn apply(&self, s: &mut Sort) {
        s.0[self.from..=self.to].rotate_left(1);
    }
}
struct Rotations;
impl Neighborhood<Sort> for Rotations {
    type Move = Rotate;
    fn explore<V: Visitor<Rotate, i64>>(
        &mut self,
        s: &Evaluated<Sort>,
        v: &mut V,
    ) -> ControlFlow<()> {
        for from in 0..s.size() {
            for to in from + 2..s.size() {
                let m = Rotate { from, to };
                v.visit(m, s.delta_of(&m))?; // no formula: full evaluation
            }
        }
        ControlFlow::Continue(())
    }
}

/// Adjacent swaps only: a cheap first neighborhood, written as an iterator.
fn adjacent(s: &Sort) -> impl Iterator<Item = Swap> {
    (0..s.size() - 1).map(|i| Swap { i, j: i + 1 })
}

#[test]
fn vnd_descends_to_a_local_optimum_of_both_neighborhoods() {
    let mut st = State::new(Sort(reversed(30)), 1);
    let mut vnd = Vnd {
        n1: moves_fn(adjacent),
        n2: Rotations,
    };
    assert_eq!(vnd.runner(&mut st).run().stop, Stop::Stuck);
    for (_, d) in collect_moves(&mut moves_fn(adjacent), st.current()) {
        assert!(d >= 0);
    }
    for (_, d) in collect_moves(&mut Rotations, st.current()) {
        assert!(d >= 0);
    }
    // a VND is a Search like any other: it nests in an iterated search
    let mut ils = IteratedSearch::new(
        Vnd {
            n1: moves_fn(adjacent),
            n2: SwapNeighborhood,
        },
        Unlimited,
        perturb_with(|s: &mut Evaluated<Sort>, rng| perm::perturb(s, 4, rng)),
        Improving,
    );
    ils.runner(&mut st).until(MaxIterations::new(20)).run();
    assert_eq!(st.best().cost(), st.best().evaluate());
}
