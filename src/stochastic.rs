//! Sample-and-accept metaheuristics. They differ only by the [`Acceptance`]
//! rule plugged into [`StochasticSearch`]: one random move per iteration from
//! a [`Sampler`], accepted or rejected. The same rules serve as ILS acceptance
//! criteria.

use crate::model::to_f64;
use crate::{Cost, Model, Observer, Sampler, Search, State, Step};
use rand::Rng;

/// Decides whether to move from a solution of cost `current` to one of cost
/// `candidate`.
pub trait Acceptance<C: Cost> {
    /// Called at the start of each run with the starting cost.
    fn start(&mut self, _initial: C) {}
    fn accept<R: Rng + ?Sized>(&mut self, current: C, candidate: C, best: C, rng: &mut R) -> bool;
    /// Called at the end of each iteration (cooling, history update…).
    fn update(&mut self, _current: C) {}
    /// The rule can't accept anything anymore (frozen): the run ends with
    /// `Stop::Exhausted`.
    fn finished(&self) -> bool {
        false
    }
}

/// Strict improvement (randomised descent).
pub struct Improving;
impl<C: Cost> Acceptance<C> for Improving {
    #[inline]
    fn accept<R: Rng + ?Sized>(&mut self, cur: C, cand: C, _: C, _: &mut R) -> bool {
        cand < cur
    }
}

/// Sideways moves too (plateau walk).
pub struct NonWorsening;
impl<C: Cost> Acceptance<C> for NonWorsening {
    #[inline]
    fn accept<R: Rng + ?Sized>(&mut self, cur: C, cand: C, _: C, _: &mut R) -> bool {
        cand <= cur
    }
}

/// Accept everything (random walk; for ILS: restart from each new run).
pub struct Always;
impl<C: Cost> Acceptance<C> for Always {
    #[inline]
    fn accept<R: Rng + ?Sized>(&mut self, _: C, _: C, _: C, _: &mut R) -> bool {
        true
    }
}

/// Metropolis criterion with geometric cooling every `plateau` iterations.
/// Exhausted when the temperature falls below `final_temp`.
pub struct SimulatedAnnealing {
    initial_temp: f64,
    final_temp: f64,
    alpha: f64,
    plateau: u64,
    temp: f64,
    count: u64,
}

impl SimulatedAnnealing {
    /// `alpha` in (0, 1) multiplies the temperature every `plateau` iterations.
    pub fn new(initial_temp: f64, final_temp: f64, alpha: f64, plateau: u64) -> Self {
        assert!(alpha > 0.0 && alpha < 1.0, "alpha must be in (0, 1)");
        assert!(plateau > 0, "plateau must be > 0");
        SimulatedAnnealing {
            initial_temp,
            final_temp,
            alpha,
            plateau,
            temp: initial_temp,
            count: 0,
        }
    }
    pub fn temperature(&self) -> f64 {
        self.temp
    }
}

impl<C: Cost> Acceptance<C> for SimulatedAnnealing {
    fn start(&mut self, _: C) {
        self.temp = self.initial_temp;
        self.count = 0;
    }
    #[inline]
    fn accept<R: Rng + ?Sized>(&mut self, cur: C, cand: C, _: C, rng: &mut R) -> bool {
        let d = to_f64(cand - cur);
        d <= 0.0 || rng.random::<f64>() < (-d / self.temp).exp()
    }
    #[inline]
    fn update(&mut self, _: C) {
        self.count += 1;
        if self.count >= self.plateau {
            self.count = 0;
            self.temp *= self.alpha;
        }
    }
    fn finished(&self) -> bool {
        self.temp < self.final_temp
    }
}

/// Late Acceptance Hill Climbing (Burke & Bykov): compare with the cost of
/// `length` iterations ago.
pub struct LateAcceptance<C> {
    length: usize,
    history: Vec<C>,
    i: usize,
}

impl<C: Cost> LateAcceptance<C> {
    pub fn new(length: usize) -> Self {
        assert!(length > 0, "length must be > 0");
        LateAcceptance {
            length,
            history: Vec::new(),
            i: 0,
        }
    }
}

impl<C: Cost> Acceptance<C> for LateAcceptance<C> {
    fn start(&mut self, initial: C) {
        self.history.clear();
        self.history.resize(self.length, initial);
        self.i = 0;
    }
    #[inline]
    fn accept<R: Rng + ?Sized>(&mut self, cur: C, cand: C, _: C, _: &mut R) -> bool {
        cand <= cur || cand <= self.history[self.i]
    }
    #[inline]
    fn update(&mut self, current: C) {
        self.history[self.i] = current;
        self.i = (self.i + 1) % self.length;
    }
}

/// Record-to-Record Travel (Dueck): accept within `deviation` of the record.
pub struct RecordToRecord {
    deviation: f64,
}
impl RecordToRecord {
    pub fn new(deviation: f64) -> Self {
        RecordToRecord { deviation }
    }
}
impl<C: Cost> Acceptance<C> for RecordToRecord {
    #[inline]
    fn accept<R: Rng + ?Sized>(&mut self, _: C, cand: C, best: C, _: &mut R) -> bool {
        to_f64(cand) <= to_f64(best) + self.deviation
    }
}

/// Threshold Accepting: accept a worsening up to `threshold`, which decays
/// geometrically every iteration.
pub struct ThresholdAccepting {
    initial: f64,
    decay: f64,
    threshold: f64,
}
impl ThresholdAccepting {
    pub fn new(initial: f64, decay: f64) -> Self {
        ThresholdAccepting {
            initial,
            decay,
            threshold: initial,
        }
    }
}
impl<C: Cost> Acceptance<C> for ThresholdAccepting {
    fn start(&mut self, _: C) {
        self.threshold = self.initial;
    }
    #[inline]
    fn accept<R: Rng + ?Sized>(&mut self, cur: C, cand: C, _: C, _: &mut R) -> bool {
        to_f64(cand - cur) <= self.threshold
    }
    #[inline]
    fn update(&mut self, _: C) {
        self.threshold *= self.decay;
    }
}

/// Great Deluge: accept anything below a water level that drops by `rain`
/// every iteration, starting from the initial cost.
pub struct GreatDeluge {
    rain: f64,
    level: f64,
}
impl GreatDeluge {
    pub fn new(rain: f64) -> Self {
        GreatDeluge {
            rain,
            level: f64::INFINITY,
        }
    }
}
impl<C: Cost> Acceptance<C> for GreatDeluge {
    fn start(&mut self, initial: C) {
        self.level = to_f64(initial);
    }
    #[inline]
    fn accept<R: Rng + ?Sized>(&mut self, _: C, cand: C, _: C, _: &mut R) -> bool {
        to_f64(cand) <= self.level
    }
    #[inline]
    fn update(&mut self, _: C) {
        self.level -= self.rain;
    }
}

/// One random move per iteration, accepted according to `A`.
pub struct StochasticSearch<N, A> {
    pub sampler: N,
    pub acceptance: A,
}

impl<N, A> StochasticSearch<N, A> {
    pub fn new(sampler: N, acceptance: A) -> Self {
        StochasticSearch {
            sampler,
            acceptance,
        }
    }
}

impl<S, R, N, A> Search<S, R> for StochasticSearch<N, A>
where
    S: Model,
    R: Rng,
    N: Sampler<S>,
    A: Acceptance<S::Cost>,
{
    fn step(&mut self, st: &mut State<S, R>, obs: &mut dyn Observer<S, R>) -> Step {
        if self.acceptance.finished() {
            return Step::Exhausted;
        }
        let best = st.best().cost();
        let (cur, rng) = st.current_and_rng();
        let Some(m) = self.sampler.sample(cur, rng) else {
            return Step::Stuck;
        };
        let delta = cur.delta_of(&m);
        let c = cur.cost();
        let step = if self.acceptance.accept(c, c + delta, best, rng) {
            st.commit(&m, delta, obs)
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
