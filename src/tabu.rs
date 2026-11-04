//! Tabu search, tabu lists and aspiration criteria.

use crate::{
    DefaultRng, Evaluated, Event, Model, Neighborhood, Observer, Search, State, Step, Visitor,
};
use num_traits::Bounded;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;
use std::hash::Hash;
use std::ops::{ControlFlow, RangeInclusive};

/// What a tabu list memorises about a move (replaces METSlib's
/// `clone`/`hash`/`==`/`opposite_of`).
pub trait TabuAttr<S: Model> {
    type Key: Hash + Eq + Clone;
    /// Attribute of a candidate move.
    fn key(&self, s: &S) -> Self::Key;
    /// Attribute to forbid once the move is made, computed on the solution
    /// *before* the move ("v may not go back to its old colour"). Defaults to
    /// `key`, right for symmetric moves such as swaps.
    fn reverse_key(&self, before: &S) -> Self::Key {
        self.key(before)
    }
}

/// Attributes that are small integers, for the array-backed [`DenseTabu`].
pub trait DenseAttr<S: Model>: TabuAttr<S, Key = usize> {
    /// Every key is `< key_count(s)`.
    fn key_count(s: &S) -> usize;
}

/// How long a move stays tabu. `Random` is re-drawn at each `start`.
#[derive(Clone, Debug)]
pub enum Tenure {
    Fixed(u64),
    Random(RangeInclusive<u64>),
}

impl Tenure {
    fn draw<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {
        match self {
            Tenure::Fixed(t) => *t,
            Tenure::Random(r) => rng.random_range(r.clone()),
        }
    }
}

/// Short-term memory. Lists get the iteration number so they can store expiry
/// stamps instead of maintaining a queue.
pub trait TabuList<S: Model, M> {
    fn is_tabu(&self, s: &S, m: &M, iteration: u64) -> bool;
    /// Called just before `m` is applied to `before`.
    fn make_tabu(&mut self, before: &S, m: &M, iteration: u64);
    /// Called at the start of each run: clear, re-draw the tenure, size buffers.
    fn start<R: Rng + ?Sized>(&mut self, _s: &S, _rng: &mut R) {}
}

/// Recency list on any hashable attribute: `key -> tabu until`.
/// `K` is inferred from the move's `TabuAttr` at the point of use.
pub struct RecencyTabu<K> {
    pub tenure: Tenure,
    current: u64,
    until: HashMap<K, u64>,
}

impl<K> RecencyTabu<K> {
    pub fn new(tenure: Tenure) -> Self {
        RecencyTabu {
            tenure,
            current: 0,
            until: HashMap::new(),
        }
    }
    /// The tenure in use for the current run.
    pub fn current_tenure(&self) -> u64 {
        self.current
    }
}

impl<S: Model, M: TabuAttr<S>> TabuList<S, M> for RecencyTabu<M::Key> {
    #[inline]
    fn is_tabu(&self, s: &S, m: &M, it: u64) -> bool {
        self.until.get(&m.key(s)).is_some_and(|&t| t > it)
    }
    fn make_tabu(&mut self, s: &S, m: &M, it: u64) {
        if self.until.len() > 2 * self.current as usize + 64 {
            self.until.retain(|_, t| *t > it); // drop expired entries
        }
        self.until.insert(m.reverse_key(s), it + 1 + self.current);
    }
    fn start<R: Rng + ?Sized>(&mut self, _s: &S, rng: &mut R) {
        self.current = self.tenure.draw(rng);
        self.until.clear();
    }
}

/// Recency list on a dense integer attribute: a flat array of expiry stamps,
/// sized from [`DenseAttr::key_count`] at the start of each run.
pub struct DenseTabu {
    pub tenure: Tenure,
    current: u64,
    until: Vec<u64>,
}

impl DenseTabu {
    pub fn new(tenure: Tenure) -> Self {
        DenseTabu {
            tenure,
            current: 0,
            until: Vec::new(),
        }
    }
    pub fn current_tenure(&self) -> u64 {
        self.current
    }
}

impl<S: Model, M: DenseAttr<S>> TabuList<S, M> for DenseTabu {
    #[inline]
    fn is_tabu(&self, s: &S, m: &M, it: u64) -> bool {
        self.until[m.key(s)] > it
    }
    #[inline]
    fn make_tabu(&mut self, s: &S, m: &M, it: u64) {
        self.until[m.reverse_key(s)] = it + 1 + self.current;
    }
    fn start<R: Rng + ?Sized>(&mut self, s: &S, rng: &mut R) {
        self.current = self.tenure.draw(rng);
        self.until.clear();
        self.until.resize(M::key_count(s), 0);
    }
}

/// Like [`DenseTabu`], but the tenure is computed at every move from the
/// solution before the move, e.g. TabuCol's `random(0..10) + 0.6 · conflicting
/// vertices` (Galinier & Hao): `DynamicTabu::new(|s: &Coloring, rng| ..)`.
/// The RNG is seeded from the run's RNG at `start`.
pub struct DynamicTabu<F> {
    tenure: F,
    until: Vec<u64>,
    rng: DefaultRng,
}

impl<F> DynamicTabu<F> {
    pub fn new<S: Model>(tenure: F) -> Self
    where
        F: FnMut(&S, &mut DefaultRng) -> u64,
    {
        DynamicTabu {
            tenure,
            until: Vec::new(),
            rng: DefaultRng::seed_from_u64(0),
        }
    }
}

impl<S: Model, M: DenseAttr<S>, F: FnMut(&S, &mut DefaultRng) -> u64> TabuList<S, M>
    for DynamicTabu<F>
{
    #[inline]
    fn is_tabu(&self, s: &S, m: &M, it: u64) -> bool {
        self.until[m.key(s)] > it
    }
    #[inline]
    fn make_tabu(&mut self, s: &S, m: &M, it: u64) {
        let t = (self.tenure)(s, &mut self.rng);
        self.until[m.reverse_key(s)] = it + 1 + t;
    }
    fn start<R: Rng + ?Sized>(&mut self, s: &S, rng: &mut R) {
        self.rng = DefaultRng::seed_from_u64(rng.next_u64());
        self.until.clear();
        self.until.resize(M::key_count(s), 0);
    }
}

/// Two lists: a move is tabu if either says so (METSlib's chain).
impl<S: Model, M, A: TabuList<S, M>, B: TabuList<S, M>> TabuList<S, M> for (A, B) {
    fn is_tabu(&self, s: &S, m: &M, it: u64) -> bool {
        self.0.is_tabu(s, m, it) || self.1.is_tabu(s, m, it)
    }
    fn make_tabu(&mut self, s: &S, m: &M, it: u64) {
        self.0.make_tabu(s, m, it);
        self.1.make_tabu(s, m, it);
    }
    fn start<R: Rng + ?Sized>(&mut self, s: &S, rng: &mut R) {
        self.0.start(s, rng);
        self.1.start(s, rng);
    }
}

/// Decides whether a tabu move may be taken anyway.
pub trait Aspiration<S: Model, M> {
    fn allows<R>(&self, st: &State<S, R>, m: &M, new_cost: S::Cost) -> bool;
}

/// Allow a tabu move that leads to a new best of the current scope.
pub struct BestEver;
impl<S: Model, M> Aspiration<S, M> for BestEver {
    #[inline]
    fn allows<R>(&self, st: &State<S, R>, _: &M, new_cost: S::Cost) -> bool {
        st.is_better(new_cost, st.best().cost())
    }
}

/// Never allow a tabu move.
pub struct NoAspiration;
impl<S: Model, M> Aspiration<S, M> for NoAspiration {
    #[inline]
    fn allows<R>(&self, _: &State<S, R>, _: &M, _: S::Cost) -> bool {
        false
    }
}

/// Best admissible move at each iteration.
pub struct TabuSearch<N, T, A = BestEver> {
    pub neighborhood: N,
    pub tabu: T,
    pub aspiration: A,
}

impl<N, T> TabuSearch<N, T, BestEver> {
    pub fn new(neighborhood: N, tabu: T) -> Self {
        TabuSearch {
            neighborhood,
            tabu,
            aspiration: BestEver,
        }
    }
}

impl<N, T, A> TabuSearch<N, T, A> {
    pub fn with_aspiration<A2>(self, aspiration: A2) -> TabuSearch<N, T, A2> {
        TabuSearch {
            neighborhood: self.neighborhood,
            tabu: self.tabu,
            aspiration,
        }
    }
}

struct Admissible<'a, S: Model, R, M, T, A> {
    st: &'a State<S, R>,
    cur: &'a Evaluated<S>,
    tabu: &'a T,
    asp: &'a A,
    best: Option<(M, S::Cost, bool)>,
    bound: S::Cost,
}

impl<S: Model, R, M, T: TabuList<S, M>, A: Aspiration<S, M>> Visitor<M, S::Cost>
    for Admissible<'_, S, R, M, T, A>
{
    #[inline]
    fn visit(&mut self, m: M, delta: S::Cost) -> ControlFlow<()> {
        // tabu status is only checked for moves that would be selected
        if delta < self.bound {
            let tabu = self.tabu.is_tabu(self.cur, &m, self.st.iteration());
            let aspired = tabu && self.asp.allows(self.st, &m, self.cur.cost() + delta);
            if !tabu || aspired {
                self.bound = delta;
                self.best = Some((m, delta, aspired));
            }
        }
        ControlFlow::Continue(())
    }
    #[inline]
    fn bound(&self) -> Option<S::Cost> {
        Some(self.bound)
    }
}

impl<S, R, N, T, A> Search<S, R> for TabuSearch<N, T, A>
where
    S: Model,
    R: Rng,
    N: Neighborhood<S>,
    T: TabuList<S, N::Move>,
    A: Aspiration<S, N::Move>,
{
    fn step(&mut self, st: &mut State<S, R>, obs: &mut dyn Observer<S, R>) -> Step {
        let mut v = Admissible {
            st,
            cur: st.current(),
            tabu: &self.tabu,
            asp: &self.aspiration,
            best: None,
            bound: S::Cost::max_value(),
        };
        let _ = self.neighborhood.explore(st.current(), &mut v);
        let Some((m, delta, aspired)) = v.best else {
            return Step::Stuck; // every move is tabu
        };
        if aspired {
            obs.notify(Event::Aspiration, st);
        }
        self.tabu.make_tabu(st.current(), &m, st.iteration());
        st.commit(&m, delta, obs)
    }

    fn start(&mut self, st: &mut State<S, R>) {
        let (cur, rng) = st.current_and_rng();
        self.tabu.start(cur, rng);
        self.neighborhood.start(cur, rng);
    }
}
