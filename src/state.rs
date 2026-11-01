//! The search state, scopes for nested searches, events and observers.

#[cfg(debug_assertions)]
use crate::model::approx_eq;
use crate::{DefaultRng, Evaluated, Model, Move, Step};
use num_traits::Zero;
use rand::SeedableRng;
use std::marker::PhantomData;
use std::time::{Duration, Instant};

/// Everything a search shares with the code that drives it.
///
/// Fields are private: algorithms change the current solution only through
/// [`commit`](Self::commit) (moves) and [`modify_current`](Self::modify_current)
/// (perturbations), and the runner advances the iteration counter.
pub struct State<S: Model, R = DefaultRng> {
    current: Evaluated<S>,
    /// Best of the current scope.
    best: Evaluated<S>,
    /// Best over the whole state. Only used while a scope is open: at depth 0
    /// it is `best`.
    global: Option<Evaluated<S>>,
    depth: usize,
    rng: R,
    tolerance: S::Cost,
    resync_every: u64,
    pub(crate) iteration: u64,
    last_improvement: u64,
    started: Instant,
}

impl<S: Model> State<S> {
    /// Start from `initial` (evaluated once), with the [`DefaultRng`] seeded
    /// from `seed`. Use [`with_rng`](State::with_rng) for another generator.
    pub fn new(initial: S, seed: u64) -> Self {
        Self::with_rng(initial, DefaultRng::seed_from_u64(seed))
    }
}

impl<S: Model, R> State<S, R> {
    /// Start from `initial` (evaluated once) with the given RNG.
    pub fn with_rng(initial: S, rng: R) -> Self {
        let current = Evaluated::new(initial);
        State {
            best: current.clone(),
            current,
            global: None,
            depth: 0,
            rng,
            tolerance: S::Cost::zero(),
            resync_every: 0,
            iteration: 0,
            last_improvement: 0,
            started: Instant::now(),
        }
    }

    /// Smallest change counted as an improvement (default zero).
    pub fn with_tolerance(mut self, tolerance: S::Cost) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Re-evaluate the current and best solutions every `n` iterations, to
    /// correct floating-point drift of the cached costs (0 = never).
    pub fn resync_every(mut self, n: u64) -> Self {
        self.resync_every = n;
        self
    }

    #[inline]
    pub fn current(&self) -> &Evaluated<S> {
        &self.current
    }
    /// Best of the current scope: inside an inner run of a nested search, the
    /// best of that run.
    #[inline]
    pub fn best(&self) -> &Evaluated<S> {
        &self.best
    }
    /// Best over the whole state, whatever the scope.
    #[inline]
    pub fn global_best(&self) -> &Evaluated<S> {
        match (&self.global, self.depth) {
            (Some(g), d) if d > 0 => g,
            _ => &self.best,
        }
    }
    /// Iterations performed so far, by every (nested) search on this state.
    #[inline]
    pub fn iteration(&self) -> u64 {
        self.iteration
    }
    /// Iteration at which the global best last improved.
    #[inline]
    pub fn last_improvement(&self) -> u64 {
        self.last_improvement
    }
    pub fn elapsed(&self) -> Duration {
        self.started.elapsed()
    }
    #[inline]
    pub fn tolerance(&self) -> S::Cost {
        self.tolerance
    }
    /// `a < b - tolerance`.
    #[inline]
    pub fn is_better(&self, a: S::Cost, b: S::Cost) -> bool {
        a < b - self.tolerance
    }

    #[inline]
    pub fn rng(&mut self) -> &mut R {
        &mut self.rng
    }
    /// The current solution and the RNG together (e.g. to sample a move).
    #[inline]
    pub fn current_and_rng(&mut self) -> (&Evaluated<S>, &mut R) {
        (&self.current, &mut self.rng)
    }

    /// Apply a move whose delta is known, record the best solutions and
    /// notify the observer. This is how an algorithm changes the current
    /// solution once it has chosen a move.
    ///
    /// In debug builds the delta is checked against two full evaluations.
    pub fn commit<M: Move<S>>(
        &mut self,
        m: &M,
        delta: S::Cost,
        obs: &mut dyn Observer<S, R>,
    ) -> Step {
        debug_assert!(delta.partial_cmp(&delta).is_some(), "NaN delta");
        #[cfg(debug_assertions)]
        let before = self.current.evaluate();
        self.current.apply_known(m, delta);
        #[cfg(debug_assertions)]
        {
            let after = self.current.evaluate();
            assert!(
                approx_eq(after - before, delta, before),
                "wrong delta: the move reported {delta:?}, full evaluations give {:?} ({before:?} -> {after:?})",
                after - before
            );
        }

        obs.notify(Event::Moved { delta }, self);
        if !self.is_better(self.current.cost(), self.best.cost()) {
            return Step::Moved;
        }
        self.best.clone_from(&self.current);
        let global = match (&mut self.global, self.depth) {
            (Some(g), d) if d > 0 => {
                let better = self.current.cost() < g.cost() - self.tolerance;
                if better {
                    g.clone_from(&self.current);
                }
                better
            }
            _ => true,
        };
        obs.notify(Event::Improved, self);
        if global {
            self.last_improvement = self.iteration;
            obs.notify(Event::GlobalImproved, self);
        }
        Step::Improved
    }

    /// Change the current solution arbitrarily (perturbation, restart). The
    /// closure gets the RNG too. Records nothing and notifies no one.
    pub fn modify_current<T>(&mut self, f: impl FnOnce(&mut Evaluated<S>, &mut R) -> T) -> T {
        f(&mut self.current, &mut self.rng)
    }

    /// Open a scope for a nested run: inside it, `best()` is the best of the run.
    pub fn scope(&mut self) -> Scope<'_, S, R> {
        if self.depth == 0 {
            match &mut self.global {
                Some(g) => g.clone_from(&self.best),
                None => self.global = Some(self.best.clone()),
            }
        }
        self.depth += 1;
        let fresh = self.current.clone();
        let outer = std::mem::replace(&mut self.best, fresh);
        // the starting point of the run may itself be a new global best
        let tol = self.tolerance;
        if let Some(g) = &mut self.global {
            if self.current.cost() < g.cost() - tol {
                g.clone_from(&self.current);
                self.last_improvement = self.iteration;
            }
        }
        Scope {
            st: self,
            outer: Some(outer),
        }
    }

    pub(crate) fn resync_if_due(&mut self) {
        if self.resync_every > 0 && self.iteration.is_multiple_of(self.resync_every) {
            let drift = self.current.resync();
            debug_assert!(
                !(drift > self.tolerance || -drift > self.tolerance) || self.tolerance.is_zero(),
                "cached cost drifted by {drift:?}, more than the tolerance"
            );
            self.best.resync();
            if let Some(g) = &mut self.global {
                g.resync();
            }
        }
    }
}

/// A nested-run scope (see [`State::scope`]).
pub struct Scope<'a, S: Model, R> {
    st: &'a mut State<S, R>,
    outer: Option<Evaluated<S>>,
}

impl<S: Model, R> Scope<'_, S, R> {
    /// The state while the scope is open (`best()` is the run's best).
    pub fn state(&mut self) -> &mut State<S, R> {
        self.st
    }

    /// Close the scope. Returns the run's best, and whether it improved on the
    /// enclosing scope's best; `best()` is the enclosing best again (updated
    /// if the run improved it).
    pub fn finish(mut self) -> (Evaluated<S>, bool) {
        self.close()
    }

    fn close(&mut self) -> (Evaluated<S>, bool) {
        let outer = self.outer.take().expect("scope already closed");
        let st = &mut *self.st;
        let run_best = std::mem::replace(&mut st.best, outer);
        let improved = st.is_better(run_best.cost(), st.best.cost());
        if improved {
            st.best.clone_from(&run_best);
        }
        st.depth -= 1;
        (run_best, improved)
    }
}

impl<S: Model, R> Drop for Scope<'_, S, R> {
    fn drop(&mut self) {
        if self.outer.is_some() {
            self.close();
        }
    }
}

// ---------------------------------------------------------------------------
// Events and observers
// ---------------------------------------------------------------------------

/// What happened during an iteration.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Event<C> {
    /// A move was applied.
    Moved { delta: C },
    /// The best of the current scope improved (in an inner run: "best of this run").
    Improved,
    /// The global best improved. Always preceded by `Improved`.
    GlobalImproved,
    /// A tabu move was taken thanks to the aspiration criterion.
    Aspiration,
    /// A nested search begins a new inner run.
    Restart,
    /// No admissible move was found.
    Stuck,
}

/// Search listener. Use `()` for none, [`log_improvements`], a tuple `(a, b)`
/// for several, or closures through the runner's `on_move`, `on_improvement`,
/// `on_global_improvement` and `on_event` methods (which infer the closure's
/// types). The free functions of the same names need an annotated argument.
pub trait Observer<S: Model, R = DefaultRng> {
    fn notify(&mut self, e: Event<S::Cost>, st: &State<S, R>);
}

impl<S: Model, R> Observer<S, R> for () {
    #[inline(always)]
    fn notify(&mut self, _: Event<S::Cost>, _: &State<S, R>) {}
}

impl<S: Model, R, O: Observer<S, R> + ?Sized> Observer<S, R> for &mut O {
    #[inline]
    fn notify(&mut self, e: Event<S::Cost>, st: &State<S, R>) {
        (**self).notify(e, st)
    }
}

impl<S: Model, R, A: Observer<S, R>, B: Observer<S, R>> Observer<S, R> for (A, B) {
    fn notify(&mut self, e: Event<S::Cost>, st: &State<S, R>) {
        self.0.notify(e, st);
        self.1.notify(e, st);
    }
}

macro_rules! typed_observer {
    ($(#[$doc:meta])* $fn:ident, $ty:ident, |$e:ident| $cond:expr) => {
        $(#[$doc])*
        pub struct $ty<F, S, R>(F, PhantomData<fn(&S, &R)>);
        $(#[$doc])*
        pub fn $fn<S: Model, R, F: FnMut(&State<S, R>)>(f: F) -> $ty<F, S, R> {
            $ty(f, PhantomData)
        }
        impl<S: Model, R, F: FnMut(&State<S, R>)> Observer<S, R> for $ty<F, S, R> {
            #[inline]
            fn notify(&mut self, $e: Event<S::Cost>, st: &State<S, R>) {
                if $cond {
                    (self.0)(st)
                }
            }
        }
    };
}

typed_observer!(
    /// Called after every applied move.
    on_move, OnMove, |e| matches!(e, Event::Moved { .. }));
typed_observer!(
    /// Called when the best of the current scope improves.
    on_improvement, OnImprovement, |e| e == Event::Improved);
typed_observer!(
    /// Called when the global best improves.
    on_global_improvement, OnGlobalImprovement, |e| e == Event::GlobalImproved);

/// Called for every event.
pub struct OnEvent<F, S, R>(F, PhantomData<fn(&S, &R)>);
/// Called for every event.
pub fn on_event<S: Model, R, F: FnMut(Event<S::Cost>, &State<S, R>)>(f: F) -> OnEvent<F, S, R> {
    OnEvent(f, PhantomData)
}
impl<S: Model, R, F: FnMut(Event<S::Cost>, &State<S, R>)> Observer<S, R> for OnEvent<F, S, R> {
    #[inline]
    fn notify(&mut self, e: Event<S::Cost>, st: &State<S, R>) {
        (self.0)(e, st)
    }
}

/// Writes `iteration<TAB>cost` on each global improvement.
pub struct LogImprovements<W>(W);
pub fn log_improvements<W: std::io::Write>(writer: W) -> LogImprovements<W> {
    LogImprovements(writer)
}
impl<S: Model, R, W: std::io::Write> Observer<S, R> for LogImprovements<W> {
    fn notify(&mut self, e: Event<S::Cost>, st: &State<S, R>) {
        if e == Event::GlobalImproved {
            let _ = writeln!(self.0, "{}\t{:?}", st.iteration(), st.global_best().cost());
        }
    }
}
