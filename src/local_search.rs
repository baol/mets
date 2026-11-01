//! Steepest descent / first-improvement local search.

use crate::{BestMove, Model, Neighborhood, Observer, Search, State, Step};
use rand::Rng;

/// Hill climbing on a [`Neighborhood`]. The run ends with `Stop::Stuck` at a
/// local optimum.
pub struct LocalSearch<N> {
    pub neighborhood: N,
    /// Take the first improving move instead of the best one.
    pub first_improvement: bool,
}

impl<N> LocalSearch<N> {
    pub fn best_improvement(neighborhood: N) -> Self {
        LocalSearch {
            neighborhood,
            first_improvement: false,
        }
    }
    pub fn first_improvement(neighborhood: N) -> Self {
        LocalSearch {
            neighborhood,
            first_improvement: true,
        }
    }
}

impl<S: Model, R: Rng, N: Neighborhood<S>> Search<S, R> for LocalSearch<N> {
    fn step(&mut self, st: &mut State<S, R>, obs: &mut dyn Observer<S, R>) -> Step {
        let mut v = BestMove::improving(st);
        v.first = self.first_improvement;
        let _ = self.neighborhood.explore(st.current(), &mut v);
        match v.best {
            Some((m, d)) => st.commit(&m, d, obs),
            None => Step::Stuck,
        }
    }
    fn start(&mut self, st: &mut State<S, R>) {
        let (cur, rng) = st.current_and_rng();
        self.neighborhood.start(cur, rng);
    }
}
