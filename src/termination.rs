//! Termination criteria. The runner calls `start` once per run and `stop`
//! once per iteration, so criteria are plain predicates over the [`State`].

use crate::{DefaultRng, Model, State};
use std::marker::PhantomData;
use std::time::{Duration, Instant};

pub trait Termination<S: Model, R = DefaultRng> {
    /// Called by the runner before the first iteration of each run.
    fn start(&mut self, _st: &State<S, R>) {}
    /// Called before each iteration.
    fn stop(&mut self, st: &State<S, R>) -> bool;
}

/// A criterion can be lent to several runs (the inner runs of a nested search).
impl<S: Model, R, T: Termination<S, R> + ?Sized> Termination<S, R> for &mut T {
    fn start(&mut self, st: &State<S, R>) {
        (**self).start(st)
    }
    fn stop(&mut self, st: &State<S, R>) -> bool {
        (**self).stop(st)
    }
}

/// Never stops: the run ends when the search is stuck or exhausted.
pub struct Unlimited;
impl<S: Model, R> Termination<S, R> for Unlimited {
    fn stop(&mut self, _: &State<S, R>) -> bool {
        false
    }
}

/// Stop after `max` iterations of the search it is attached to. It counts its
/// own calls, so on an iterated search it counts perturbations, not inner moves.
pub struct MaxIterations {
    max: u64,
    done: u64,
}
impl MaxIterations {
    pub fn new(max: u64) -> Self {
        MaxIterations { max, done: 0 }
    }
}
impl<S: Model, R> Termination<S, R> for MaxIterations {
    fn start(&mut self, _: &State<S, R>) {
        self.done = 0;
    }
    fn stop(&mut self, _: &State<S, R>) -> bool {
        if self.done >= self.max {
            return true;
        }
        self.done += 1;
        false
    }
}

/// Stop after `max` consecutive iterations without improving the scoped best
/// (METSlib's `noimprove_termination_criteria`).
pub struct NoImprovement<C> {
    max: u64,
    left: u64,
    best: Option<C>,
}
impl<C> NoImprovement<C> {
    pub fn new(max: u64) -> Self {
        NoImprovement {
            max,
            left: max,
            best: None,
        }
    }
}
impl<S: Model, R> Termination<S, R> for NoImprovement<S::Cost> {
    fn start(&mut self, _: &State<S, R>) {
        self.left = self.max;
        self.best = None;
    }
    fn stop(&mut self, st: &State<S, R>) -> bool {
        let c = st.best().cost();
        if self.best.is_none_or(|b| st.is_better(c, b)) {
            self.best = Some(c);
            self.left = self.max;
        }
        if self.left == 0 {
            return true;
        }
        self.left -= 1;
        false
    }
}

/// Stop when the scoped best reaches `target` (use [`stop_when`] with
/// `st.global_best()` for a target on the global best).
pub struct Target<C>(C);
impl<C> Target<C> {
    pub fn new(target: C) -> Self {
        Target(target)
    }
}
impl<S: Model, R> Termination<S, R> for Target<S::Cost> {
    fn stop(&mut self, st: &State<S, R>) -> bool {
        st.best().cost() <= self.0
    }
}

/// Wall-clock limit for the run.
pub struct TimeLimit {
    limit: Duration,
    since: Instant,
}
impl TimeLimit {
    pub fn new(limit: Duration) -> Self {
        TimeLimit {
            limit,
            since: Instant::now(),
        }
    }
}
impl<S: Model, R> Termination<S, R> for TimeLimit {
    fn start(&mut self, _: &State<S, R>) {
        self.since = Instant::now();
    }
    fn stop(&mut self, _: &State<S, R>) -> bool {
        self.since.elapsed() >= self.limit
    }
}

/// A closure as a criterion; the argument type is fixed, so no annotation is needed.
pub struct StopWhen<F, S, R>(F, PhantomData<fn(&S, &R)>);
pub fn stop_when<S: Model, R, F: FnMut(&State<S, R>) -> bool>(f: F) -> StopWhen<F, S, R> {
    StopWhen(f, PhantomData)
}
impl<S: Model, R, F: FnMut(&State<S, R>) -> bool> Termination<S, R> for StopWhen<F, S, R> {
    fn stop(&mut self, st: &State<S, R>) -> bool {
        (self.0)(st)
    }
}

/// Either criterion. Both are always evaluated, so both see every iteration.
pub struct Or<A, B>(A, B);
/// Both criteria. Both are always evaluated.
pub struct And<A, B>(A, B);

impl<S: Model, R, A: Termination<S, R>, B: Termination<S, R>> Termination<S, R> for Or<A, B> {
    fn start(&mut self, st: &State<S, R>) {
        self.0.start(st);
        self.1.start(st);
    }
    fn stop(&mut self, st: &State<S, R>) -> bool {
        let a = self.0.stop(st);
        let b = self.1.stop(st);
        a || b
    }
}
impl<S: Model, R, A: Termination<S, R>, B: Termination<S, R>> Termination<S, R> for And<A, B> {
    fn start(&mut self, st: &State<S, R>) {
        self.0.start(st);
        self.1.start(st);
    }
    fn stop(&mut self, st: &State<S, R>) -> bool {
        let a = self.0.stop(st);
        let b = self.1.stop(st);
        a && b
    }
}

/// `a.or(b)`, `a.and(b)`.
pub trait TerminationExt: Sized {
    fn or<B>(self, other: B) -> Or<Self, B> {
        Or(self, other)
    }
    fn and<B>(self, other: B) -> And<Self, B> {
        And(self, other)
    }
}
impl<T> TerminationExt for T {}
