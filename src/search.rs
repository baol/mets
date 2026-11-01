//! The `Search` trait and the runner.

use crate::{
    on_event, on_global_improvement, on_improvement, on_move, stop_when, DefaultRng, Event, Model,
    Observer, OnEvent, OnGlobalImprovement, OnImprovement, OnMove, Or, State, StopWhen,
    Termination, TerminationExt, Unlimited,
};
use std::time::{Duration, Instant};

/// Result of one iteration.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    /// A move was applied.
    Moved,
    /// A move was applied and the scoped best improved.
    Improved,
    /// Nothing was applied, but the search can continue (e.g. SA rejection).
    Rejected,
    /// No admissible move exists (local optimum, every move tabu).
    Stuck,
    /// The algorithm finished on its own (SA frozen…).
    Exhausted,
}

/// Why a run ended.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The termination criterion fired.
    Terminated,
    /// No admissible move exists.
    Stuck,
    /// The algorithm finished on its own.
    Exhausted,
}

/// Summary of a run.
#[derive(Clone, Copy, Debug)]
pub struct Outcome {
    pub stop: Stop,
    /// Iterations performed by this run (not counting nested runs).
    pub iterations: u64,
    pub elapsed: Duration,
}

/// A neighborhood-based search algorithm, one iteration at a time.
/// Object-safe: `Box<dyn Search<S>>` works.
pub trait Search<S: Model, R = DefaultRng> {
    /// One iteration. Change the current solution only through
    /// [`State::commit`] / [`State::modify_current`].
    fn step(&mut self, st: &mut State<S, R>, obs: &mut dyn Observer<S, R>) -> Step;
    /// Called by [`Runner::run`] at the beginning of each run.
    fn start(&mut self, _st: &mut State<S, R>) {}
}

/// Gives every search (including `dyn Search`) the [`runner`](SearchExt::runner) builder.
pub trait SearchExt<S: Model, R> {
    fn runner<'a>(&'a mut self, st: &'a mut State<S, R>) -> Runner<'a, Self, S, R, Unlimited, ()>;
}

impl<S: Model, R, Q: Search<S, R> + ?Sized> SearchExt<S, R> for Q {
    fn runner<'a>(&'a mut self, st: &'a mut State<S, R>) -> Runner<'a, Self, S, R, Unlimited, ()> {
        Runner {
            search: self,
            st,
            stop: Unlimited,
            obs: (),
        }
    }
}

/// Configures and executes a run: `search.runner(&mut st).until(..).observe(..).run()`.
#[must_use = "a runner does nothing until `.run()` or `.resume()` is called"]
pub struct Runner<'a, Q: ?Sized, S: Model, R, T, O> {
    search: &'a mut Q,
    st: &'a mut State<S, R>,
    stop: T,
    obs: O,
}

// The builder's return types spell out the composed observer/criterion types.
#[allow(clippy::type_complexity)]
impl<'a, Q, S, R, T, O> Runner<'a, Q, S, R, T, O>
where
    Q: Search<S, R> + ?Sized,
    S: Model,
    T: Termination<S, R>,
    O: Observer<S, R>,
{
    /// Termination criterion (default: none, the run ends when stuck or exhausted).
    /// Accepts a value or a `&mut` reference.
    pub fn until<T2: Termination<S, R>>(self, stop: T2) -> Runner<'a, Q, S, R, T2, O> {
        Runner {
            search: self.search,
            st: self.st,
            stop,
            obs: self.obs,
        }
    }

    /// Observer (default: none), replacing any set so far. Accepts a value or
    /// a `&mut` reference. See also `on_move`, `on_improvement`… below.
    pub fn observe<O2: Observer<S, R>>(self, obs: O2) -> Runner<'a, Q, S, R, T, O2> {
        Runner {
            search: self.search,
            st: self.st,
            stop: self.stop,
            obs,
        }
    }

    /// Also stop when the closure returns true (combined with `until` by OR).
    pub fn stop_when<F: FnMut(&State<S, R>) -> bool>(
        self,
        f: F,
    ) -> Runner<'a, Q, S, R, Or<T, StopWhen<F, S, R>>, O> {
        Runner {
            search: self.search,
            st: self.st,
            stop: self.stop.or(stop_when(f)),
            obs: self.obs,
        }
    }

    /// Also call `f` after every applied move. The closure's argument type is
    /// known here, so it needs no annotation.
    pub fn on_move<F: FnMut(&State<S, R>)>(
        self,
        f: F,
    ) -> Runner<'a, Q, S, R, T, (O, OnMove<F, S, R>)> {
        Runner {
            search: self.search,
            st: self.st,
            stop: self.stop,
            obs: (self.obs, on_move(f)),
        }
    }

    /// Also call `f` when the best of the current scope improves.
    pub fn on_improvement<F: FnMut(&State<S, R>)>(
        self,
        f: F,
    ) -> Runner<'a, Q, S, R, T, (O, OnImprovement<F, S, R>)> {
        Runner {
            search: self.search,
            st: self.st,
            stop: self.stop,
            obs: (self.obs, on_improvement(f)),
        }
    }

    /// Also call `f` when the global best improves.
    pub fn on_global_improvement<F: FnMut(&State<S, R>)>(
        self,
        f: F,
    ) -> Runner<'a, Q, S, R, T, (O, OnGlobalImprovement<F, S, R>)> {
        Runner {
            search: self.search,
            st: self.st,
            stop: self.stop,
            obs: (self.obs, on_global_improvement(f)),
        }
    }

    /// Also call `f` for every event.
    pub fn on_event<F: FnMut(Event<S::Cost>, &State<S, R>)>(
        self,
        f: F,
    ) -> Runner<'a, Q, S, R, T, (O, OnEvent<F, S, R>)> {
        Runner {
            search: self.search,
            st: self.st,
            stop: self.stop,
            obs: (self.obs, on_event(f)),
        }
    }

    /// Start a fresh run (`Search::start`: tabu list cleared, SA reheated…).
    pub fn run(self) -> Outcome {
        self.search.start(self.st);
        self.go()
    }

    /// Continue where the previous run stopped, without `Search::start`.
    pub fn resume(self) -> Outcome {
        self.go()
    }

    fn go(mut self) -> Outcome {
        let t0 = Instant::now();
        let mut iterations = 0;
        self.stop.start(self.st);
        let stop = loop {
            if self.stop.stop(self.st) {
                break Stop::Terminated;
            }
            match self.search.step(self.st, &mut self.obs) {
                Step::Stuck => {
                    self.obs.notify(Event::Stuck, self.st);
                    break Stop::Stuck;
                }
                Step::Exhausted => break Stop::Exhausted,
                _ => {}
            }
            self.st.iteration += 1;
            iterations += 1;
            self.st.resync_if_due();
        };
        Outcome {
            stop,
            iterations,
            elapsed: t0.elapsed(),
        }
    }
}
