//! Iterated local search, iterated tabu search, random restarts.
//!
//! [`IteratedSearch`] repeats: perturb the incumbent → run an inner search in
//! its own best-scope → accept the run's best as new incumbent (or not):
//!
//! | algorithm            | inner         | perturbation   | acceptance           |
//! |----------------------|---------------|----------------|----------------------|
//! | ILS                  | `LocalSearch` | k random moves | `Improving`          |
//! | iterated tabu search | `TabuSearch`  | k random moves | `Improving`          |
//! | random restarts      | anything      | re-initialise  | `Always`             |
//! | LSMC / SA-ILS        | anything      | k random moves | `SimulatedAnnealing` |
//!
//! It is itself a `Search`, so it nests: `IteratedSearch<IteratedSearch<TabuSearch<..>>>`.

use crate::{
    Acceptance, Evaluated, Event, Model, Observer, Search, SearchExt, State, Step, Termination,
};
use rand::{Rng, RngCore};
use std::marker::PhantomData;

/// Changes a solution in place. Use [`perturb_with`] for closures.
pub trait Perturbation<S: Model> {
    fn perturb(&mut self, s: &mut Evaluated<S>, rng: &mut dyn RngCore);
}

/// A closure as a perturbation: `perturb_with(|s: &mut Evaluated<P>, rng| ..)`.
/// Only the solution type needs writing.
pub struct PerturbWith<F, S>(F, PhantomData<fn(&mut S)>);
pub fn perturb_with<S: Model, F: FnMut(&mut Evaluated<S>, &mut dyn RngCore)>(
    f: F,
) -> PerturbWith<F, S> {
    PerturbWith(f, PhantomData)
}
impl<S: Model, F: FnMut(&mut Evaluated<S>, &mut dyn RngCore)> Perturbation<S>
    for PerturbWith<F, S>
{
    fn perturb(&mut self, s: &mut Evaluated<S>, rng: &mut dyn RngCore) {
        (self.0)(s, rng)
    }
}

/// A perturbation whose strength adapts: it starts at `min`, grows by `step`
/// (up to `max`) each time it is applied to an unchanged incumbent, and goes
/// back to `min` when the incumbent changes. The idea of breakout local
/// search: shake harder while stuck, gently after progress.
///
/// `apply(s, strength, rng)` does the perturbing, e.g. `strength` random
/// moves (see `perm::adaptive_perturbation`).
pub struct Adaptive<F, C> {
    pub min: usize,
    pub step: usize,
    pub max: usize,
    strength: usize,
    last: Option<C>,
    apply: F,
}

impl<F, C> Adaptive<F, C> {
    pub fn new(min: usize, step: usize, max: usize, apply: F) -> Self {
        Adaptive {
            min,
            step,
            max: max.max(min),
            strength: min,
            last: None,
            apply,
        }
    }
    /// The strength used by the last perturbation.
    pub fn strength(&self) -> usize {
        self.strength
    }
}

impl<S, F> Perturbation<S> for Adaptive<F, S::Cost>
where
    S: Model,
    F: FnMut(&mut Evaluated<S>, usize, &mut dyn RngCore),
{
    fn perturb(&mut self, s: &mut Evaluated<S>, rng: &mut dyn RngCore) {
        let cost = s.cost();
        self.strength = match self.last {
            Some(c) if c == cost => (self.strength + self.step).min(self.max),
            _ => self.min,
        };
        self.last = Some(cost);
        (self.apply)(s, self.strength, rng)
    }
}

pub struct IteratedSearch<S: Model, I, T, P, A> {
    pub inner: I,
    /// Termination criterion of each inner run (restarted for every run).
    pub inner_stop: T,
    pub perturbation: P,
    pub acceptance: A,
    incumbent: Option<Evaluated<S>>,
}

impl<S: Model, I, T, P, A> IteratedSearch<S, I, T, P, A> {
    pub fn new(inner: I, inner_stop: T, perturbation: P, acceptance: A) -> Self {
        IteratedSearch {
            inner,
            inner_stop,
            perturbation,
            acceptance,
            incumbent: None,
        }
    }
    /// The solution the next perturbation starts from.
    pub fn incumbent(&self) -> Option<&Evaluated<S>> {
        self.incumbent.as_ref()
    }
}

impl<S, R, I, T, P, A> Search<S, R> for IteratedSearch<S, I, T, P, A>
where
    S: Model,
    R: Rng,
    I: Search<S, R>,
    T: Termination<S, R>,
    P: Perturbation<S>,
    A: Acceptance<S::Cost>,
{
    fn step(&mut self, st: &mut State<S, R>, obs: &mut dyn Observer<S, R>) -> Step {
        // 1. perturb the incumbent (the first run starts from the initial solution)
        if let Some(inc) = &self.incumbent {
            let p = &mut self.perturbation;
            st.modify_current(|cur, rng| {
                cur.clone_from(inc);
                p.perturb(cur, rng)
            });
        }
        obs.notify(Event::Restart, st);

        // 2. inner run with its own best
        let mut scope = st.scope();
        self.inner
            .runner(scope.state())
            .until(&mut self.inner_stop)
            .observe(&mut *obs)
            .run();
        let (run_best, improved_outer) = scope.finish();

        // 3. acceptance of the run's best as the new incumbent
        let best = st.best().cost();
        let accept = match &self.incumbent {
            None => true,
            Some(inc) => self
                .acceptance
                .accept(inc.cost(), run_best.cost(), best, st.rng()),
        };
        if accept {
            self.incumbent = Some(run_best);
        }
        if let Some(inc) = &self.incumbent {
            self.acceptance.update(inc.cost());
        }
        if improved_outer {
            Step::Improved
        } else {
            Step::Moved
        }
    }

    fn start(&mut self, st: &mut State<S, R>) {
        self.incumbent = None;
        self.acceptance.start(st.current().cost());
    }
}
