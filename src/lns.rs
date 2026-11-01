//! Large neighborhood search (Shaw 1998; Pisinger & Ropke): at each iteration,
//! destroy part of a copy of the current solution and repair it, then accept
//! or reject the result with any [`Acceptance`] rule.
//!
//! The neighborhood is never enumerated: one [`Rebuild`] call explores it
//! (exhaustively over the destroyed part, by a heuristic, or by an exact
//! sub-solver). `perm::BestRearrangement` is a ready-made one for
//! permutations.

use crate::{Acceptance, Evaluated, Model, Observer, Search, State, Step};
use rand::{Rng, RngCore};
use std::marker::PhantomData;

/// Destroys part of a solution and repairs it, in place. The cached cost is
/// kept valid by going through `Evaluated::apply` / `Evaluated::modify`.
pub trait Rebuild<S: Model> {
    fn rebuild(&mut self, s: &mut Evaluated<S>, rng: &mut dyn RngCore);
}

/// A closure as a rebuild operator: `rebuild_with(|s: &mut Evaluated<P>, rng| ..)`.
pub struct RebuildWith<F, S>(F, PhantomData<fn(&mut S)>);
pub fn rebuild_with<S: Model, F: FnMut(&mut Evaluated<S>, &mut dyn RngCore)>(
    f: F,
) -> RebuildWith<F, S> {
    RebuildWith(f, PhantomData)
}
impl<S: Model, F: FnMut(&mut Evaluated<S>, &mut dyn RngCore)> Rebuild<S> for RebuildWith<F, S> {
    fn rebuild(&mut self, s: &mut Evaluated<S>, rng: &mut dyn RngCore) {
        (self.0)(s, rng)
    }
}

/// One rebuild of a copy of the current solution per iteration, accepted
/// according to `A` (`Improving` for a descent, `SimulatedAnnealing`,
/// `LateAcceptance`… for the classic LNS acceptance rules).
pub struct LargeNeighborhoodSearch<S: Model, O, A> {
    pub operator: O,
    pub acceptance: A,
    candidate: Option<Evaluated<S>>,
}

impl<S: Model, O, A> LargeNeighborhoodSearch<S, O, A> {
    pub fn new(operator: O, acceptance: A) -> Self {
        LargeNeighborhoodSearch {
            operator,
            acceptance,
            candidate: None,
        }
    }
}

impl<S, R, O, A> Search<S, R> for LargeNeighborhoodSearch<S, O, A>
where
    S: Model,
    R: Rng,
    O: Rebuild<S>,
    A: Acceptance<S::Cost>,
{
    fn step(&mut self, st: &mut State<S, R>, obs: &mut dyn Observer<S, R>) -> Step {
        if self.acceptance.finished() {
            return Step::Exhausted;
        }
        let best = st.best().cost();
        let (cur, rng) = st.current_and_rng();
        let candidate = match &mut self.candidate {
            Some(c) => {
                c.clone_from(cur);
                c
            }
            None => self.candidate.insert(cur.clone()),
        };
        self.operator.rebuild(candidate, rng);
        let c = cur.cost();
        let step = if self.acceptance.accept(c, candidate.cost(), best, rng) {
            st.replace_current(candidate, obs)
        } else {
            Step::Rejected
        };
        self.acceptance.update(st.current().cost());
        step
    }

    fn start(&mut self, st: &mut State<S, R>) {
        self.acceptance.start(st.current().cost());
    }
}
