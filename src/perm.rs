//! Ready-made pieces for permutation problems (QAP, TSP, scheduling…).
//!
//! * The counterparts of METSlib's `permutation_problem`, `swap_elements`,
//!   `swap_full_neighborhood` and `swap_neighborhood`: [`Permutation`],
//!   [`Swap`], [`SwapNeighborhood`], [`perturb`], [`shuffle`].
//! * [`RobustTabuSearch`]: Taillard's robust tabu search, with tabu
//!   attributes on assignments (element at position).
//! * [`BestRearrangement`]: a very large neighborhood for
//!   [`LargeNeighborhoodSearch`](crate::LargeNeighborhoodSearch): all k!
//!   arrangements of k random positions.
//! * [`UniformCrossover`] and [`CohesiveCrossover`] (Drezner) for
//!   [`Memetic`](crate::Memetic), and
//!   [`adaptive_perturbation`] and [`directed_perturbation`] (Benlic & Hao's
//!   breakout local search) for [`IteratedSearch`](crate::IteratedSearch).

use crate::ils::{Adaptive, Perturbation};
use crate::{
    Crossover, DefaultRng, DenseAttr, Evaluated, Model, Move, Neighborhood, Observer, Rebuild,
    Sampler, Search, State, Step, TabuAttr, Visitor,
};
use num_traits::Bounded;
use rand::{Rng, RngCore, SeedableRng};
use std::ops::ControlFlow;

/// A solution encoded as a permutation.
pub trait Permutation: Model {
    /// The permutation: element at each position.
    fn as_slice(&self) -> &[usize];
    #[inline]
    fn size(&self) -> usize {
        self.as_slice().len()
    }
    /// Exchange the elements at positions `i` and `j`.
    fn swap(&mut self, i: usize, j: usize);
    /// `cost(after swap) - cost(before)`, if you have a formula. Default:
    /// none, the library evaluates the neighbor in full.
    #[inline]
    fn swap_delta(&self, _i: usize, _j: usize) -> Option<Self::Cost> {
        None
    }
}

/// Exchange the elements at positions `i < j`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Swap {
    pub i: usize,
    pub j: usize,
}

impl Swap {
    /// Normalises the order so that `i < j`.
    pub fn new(a: usize, b: usize) -> Self {
        Swap {
            i: a.min(b),
            j: a.max(b),
        }
    }
}

impl<S: Permutation> Move<S> for Swap {
    #[inline]
    fn apply(&self, s: &mut S) {
        s.swap(self.i, self.j)
    }
    #[inline]
    fn delta(&self, s: &Evaluated<S>) -> Option<S::Cost> {
        s.swap_delta(self.i, self.j)
    }
    #[inline]
    fn inverse(&self, _: &S) -> Option<Self> {
        Some(*self)
    }
}

/// Swaps are their own inverse: forbid the same pair. Dense key `i * n + j`.
impl<S: Permutation> TabuAttr<S> for Swap {
    type Key = usize;
    #[inline]
    fn key(&self, s: &S) -> usize {
        self.i * s.size() + self.j
    }
}
impl<S: Permutation> DenseAttr<S> for Swap {
    fn key_count(s: &S) -> usize {
        s.size() * s.size()
    }
}

/// All n(n-1)/2 swaps, generated on the fly; also a uniform O(1) [`Sampler`].
#[derive(Clone, Copy, Debug, Default)]
pub struct SwapNeighborhood;

impl<S: Permutation> Neighborhood<S> for SwapNeighborhood {
    type Move = Swap;
    #[inline]
    fn explore<V: Visitor<Swap, S::Cost>>(
        &mut self,
        s: &Evaluated<S>,
        v: &mut V,
    ) -> ControlFlow<()> {
        let n = s.size();
        for i in 0..n {
            for j in i + 1..n {
                let m = Swap { i, j };
                v.visit(m, s.delta_of(&m))?;
            }
        }
        ControlFlow::Continue(())
    }
}

impl<S: Permutation> Sampler<S> for SwapNeighborhood {
    type Move = Swap;
    #[inline]
    fn sample<R: Rng + ?Sized>(&mut self, s: &Evaluated<S>, rng: &mut R) -> Option<Swap> {
        random_swap(s.size(), rng)
    }
}

/// A uniformly random swap of two distinct positions.
#[inline]
pub fn random_swap<R: Rng + ?Sized>(n: usize, rng: &mut R) -> Option<Swap> {
    if n < 2 {
        return None;
    }
    let i = rng.random_range(0..n);
    let mut j = rng.random_range(0..n - 1);
    if j >= i {
        j += 1;
    }
    Some(Swap::new(i, j))
}

/// Apply `k` random swaps (METSlib's `mets::perturbate`), re-evaluating once.
pub fn perturb<S: Permutation, R: Rng + ?Sized>(s: &mut Evaluated<S>, k: usize, rng: &mut R) {
    s.modify(|p| {
        for _ in 0..k {
            if let Some(m) = random_swap(p.size(), rng) {
                p.swap(m.i, m.j);
            }
        }
    })
}

/// Shuffle uniformly (Fisher–Yates), re-evaluating once.
pub fn shuffle<S: Permutation, R: Rng + ?Sized>(s: &mut Evaluated<S>, rng: &mut R) {
    s.modify(|p| {
        for i in (1..p.size()).rev() {
            let j = rng.random_range(0..=i);
            if i != j {
                p.swap(j, i);
            }
        }
    })
}

/// `perturb(s, strength, rng)` as a function pointer, for [`Adaptive`].
pub type PerturbFn<S> = fn(&mut Evaluated<S>, usize, &mut dyn RngCore);

/// [`perturb`] with an adaptive number of swaps: from `max(2, n/10)`, growing
/// by `max(1, n/20)` up to `n/2` while the incumbent does not change.
pub fn adaptive_perturbation<S: Permutation>(n: usize) -> Adaptive<PerturbFn<S>, S::Cost> {
    fn apply<S: Permutation>(s: &mut Evaluated<S>, k: usize, rng: &mut dyn RngCore) {
        perturb(s, k, rng)
    }
    Adaptive::new((n / 10).max(2), (n / 20).max(1), (n / 2).max(3), apply::<S>)
}

/// Number of positions where `a` and `b` differ: a distance for
/// `Memetic::with_distance`.
/// Breakout local search's perturbation (Benlic & Hao, BLS for the QAP).
/// Each of the `strength` swaps is directed with probability
/// max(exp(−ω/threshold), p_min), where ω counts the perturbations since the
/// best cost seen last improved, else random. A directed swap is the best
/// swap (by delta) not used in the last `tenure` perturbation swaps, unless
/// it leads below the best cost seen during this perturbation (aspiration).
/// The strength grows by `step` (up to `max`) when the search came back to
/// the incumbent cost of the previous call, and is back to `min` otherwise.
/// After `threshold` perturbations without a new best, one strong random
/// perturbation of `max` swaps, and ω starts again. Parameter values follow
/// the BLS paper as recalled, not checked against it: T = 2,500, P₀ = 0.75,
/// L₀ = 0.15n, L_max = n/2.
///
/// A directed swap scans all n(n−1)/2 swap deltas: cheap with O(1) deltas
/// (Taillard's delta matrix); a model without `swap_delta` gets random swaps
/// only.
#[derive(Clone, Debug)]
pub struct DirectedPerturbation<C> {
    pub min: usize,
    pub step: usize,
    pub max: usize,
    /// Perturbation swaps during which a used swap stays forbidden.
    pub tenure: u64,
    /// T: perturbations without a new best before a strong one; also the
    /// scale of the decay of the directed probability.
    pub threshold: u64,
    /// P₀: the directed probability never goes below it.
    pub p_min: f64,
    strength: usize,
    last: Option<C>,
    best: Option<C>,
    /// ω: perturbations since `best` last improved.
    since_best: u64,
    last_used: Vec<u64>,
    moves: u64,
}

impl<C> DirectedPerturbation<C> {
    pub fn new(min: usize, step: usize, max: usize, tenure: u64) -> Self {
        DirectedPerturbation {
            min,
            step,
            max: max.max(min),
            tenure,
            threshold: 2500,
            p_min: 0.75,
            strength: min,
            last: None,
            best: None,
            since_best: 0,
            last_used: Vec::new(),
            moves: 0,
        }
    }

    /// The strength used by the last perturbation.
    pub fn strength(&self) -> usize {
        self.strength
    }

    /// Probability of a directed (rather than random) swap.
    fn p_directed(&self) -> f64 {
        (-(self.since_best as f64) / self.threshold.max(1) as f64)
            .exp()
            .max(self.p_min)
    }
}

/// [`DirectedPerturbation`] with BLS's strengths (from 0.15n, +1 per return
/// to the same optimum, up to n/2) and a tenure of n perturbation swaps.
pub fn directed_perturbation<S: Permutation>(n: usize) -> DirectedPerturbation<S::Cost> {
    DirectedPerturbation::new(
        ((0.15 * n as f64) as usize).max(2),
        1,
        (n / 2).max(3),
        n as u64,
    )
}

impl<S: Permutation> Perturbation<S> for DirectedPerturbation<S::Cost> {
    fn perturb(&mut self, s: &mut Evaluated<S>, rng: &mut dyn RngCore) {
        let cost = s.cost();
        self.strength = if self.last == Some(cost) {
            (self.strength + self.step).min(self.max)
        } else {
            self.min
        };
        self.last = Some(cost);
        if self.best.is_none_or(|b| cost < b) {
            (self.best, self.since_best) = (Some(cost), 0);
        } else {
            self.since_best += 1;
        }
        let n = s.size();
        if n < 2 {
            return;
        }
        if self.since_best > self.threshold {
            // strong perturbation: max random swaps, then ω starts again
            self.since_best = 0;
            perturb(s, self.max, rng);
            return;
        }
        if self.last_used.len() != n * n {
            // "never used": far enough in the past for any tenure
            self.last_used = vec![0; n * n];
            self.moves = self.tenure + 1;
        }
        let p = self.p_directed();
        let mut best_seen = s.cost();
        for _ in 0..self.strength {
            let directed = rng.random::<f64>() < p && s.swap_delta(0, 1).is_some();
            let m = if directed {
                let mut chosen: Option<(Swap, S::Cost)> = None;
                let mut tied = 0u32;
                for i in 0..n {
                    for j in i + 1..n {
                        let d = s.swap_delta(i, j).expect("swap deltas");
                        let tabu = self.last_used[i * n + j] + self.tenure > self.moves;
                        if tabu && s.cost() + d >= best_seen {
                            continue;
                        }
                        match chosen {
                            Some((_, b)) if d > b => {}
                            Some((_, b)) if d == b => {
                                // reservoir sampling among equal deltas
                                tied += 1;
                                if rng.random_range(0..tied) == 0 {
                                    chosen = Some((Swap::new(i, j), d));
                                }
                            }
                            _ => {
                                chosen = Some((Swap::new(i, j), d));
                                tied = 1;
                            }
                        }
                    }
                }
                match chosen {
                    Some((m, _)) => m,
                    None => random_swap(n, rng).expect("n >= 2"),
                }
            } else {
                random_swap(n, rng).expect("n >= 2")
            };
            s.apply(&m);
            self.last_used[m.i * n + m.j] = self.moves;
            self.moves += 1;
            if s.cost() < best_seen {
                best_seen = s.cost();
            }
        }
    }
}

pub fn hamming<S: Permutation>(a: &S, b: &S) -> usize {
    a.as_slice()
        .iter()
        .zip(b.as_slice())
        .filter(|(x, y)| x != y)
        .count()
}

/// Rearrange `s` so that its element at each position is `target[position]`,
/// with at most n − 1 swaps (through `Permutation::swap`, so incremental
/// structures of the model stay valid). Call it inside `Evaluated::modify`.
pub fn rearrange<S: Permutation>(s: &mut S, target: &[usize]) {
    let n = s.size();
    debug_assert_eq!(target.len(), n);
    let mut position = vec![0; n];
    for (i, &e) in s.as_slice().iter().enumerate() {
        position[e] = i;
    }
    for (i, &wanted) in target.iter().enumerate() {
        let current = s.as_slice()[i];
        if current != wanted {
            let j = position[wanted];
            s.swap(i, j);
            position[current] = j;
            position[wanted] = i;
        }
    }
}

// ----------------------------------------------------------- robust tabu search

/// Taillard's robust tabu search (1991) on the swap neighborhood, for any
/// [`Permutation`].
///
/// The tabu memory is about *assignments*, not position pairs: a swap is tabu
/// when it would put both elements back into positions they left within the
/// tenure. The tenure is re-drawn from `[0.9n, 1.1n]` every `2·1.1n`
/// iterations. A long-term aspiration *forces* a move that re-creates an
/// assignment not seen for `5n²` iterations, even if it is worsening: that is
/// why this is its own `Search` and not a `TabuList` for `TabuSearch`, which
/// can only make moves admissible.
///
/// The memory persists across runs (`start` only re-seeds the tenure draw),
/// so as the inner search of an iterated or memetic search it remembers the
/// whole history. Every iteration scans all n(n−1)/2 swaps: use a model with
/// O(1) swap deltas (a delta matrix) for large n.
pub struct RobustTabuSearch {
    /// Tenure range as fractions of n.
    pub tenure: (f64, f64),
    /// Long-term aspiration age as a multiple of n².
    pub aspiration: f64,
    n: usize,
    t: Vec<i64>,
    current_tenure: i64,
    next_redraw: u64,
    aspiration_age: i64,
    rng: DefaultRng,
}

impl Default for RobustTabuSearch {
    fn default() -> Self {
        Self::new()
    }
}

impl RobustTabuSearch {
    pub fn new() -> Self {
        RobustTabuSearch {
            tenure: (0.9, 1.1),
            aspiration: 5.0,
            n: 0,
            t: Vec::new(),
            current_tenure: 1,
            next_redraw: 0,
            aspiration_age: 0,
            rng: DefaultRng::seed_from_u64(0),
        }
    }

    /// Size the memory for permutations of size n (done by `start`).
    fn prepare(&mut self, n: usize) {
        if self.n != n {
            self.n = n;
            // staggered "never" so that unseen assignments become aspired one at a time
            self.t = (0..n * n).map(|k| -(k as i64)).collect();
            self.aspiration_age = (self.aspiration * (n * n) as f64) as i64;
        }
    }

    fn redraw(&mut self, it: u64) {
        let n = self.n as f64;
        let lo = ((self.tenure.0 * n) as i64).max(1);
        let hi = ((self.tenure.1 * n) as i64).max(lo);
        self.current_tenure = self.rng.random_range(lo..=hi);
        self.next_redraw = it + 2 * hi as u64;
    }

    /// (authorized, aspired) for swapping positions i and j of `p` at iteration `it`.
    #[inline]
    fn classify(&self, p: &[usize], i: usize, j: usize, it: i64) -> (bool, bool) {
        let n = self.n;
        let (ti, tj) = (self.t[i * n + p[j]], self.t[j * n + p[i]]); // p[j] back to i, p[i] back to j
        let authorized = ti < it || tj < it;
        let aspired = ti < it - self.aspiration_age || tj < it - self.aspiration_age;
        (authorized, aspired)
    }
}

impl<S: Permutation, R: Rng> Search<S, R> for RobustTabuSearch {
    fn step(&mut self, st: &mut State<S, R>, obs: &mut dyn Observer<S, R>) -> Step {
        let it = st.iteration();
        if it >= self.next_redraw {
            self.redraw(it);
        }
        let cur = st.current();
        let p = cur.as_slice();
        let (n, best) = (self.n, st.best().cost());
        let mut retained = None;
        let (mut min_delta, mut already_aspired) = (S::Cost::max_value(), false);
        for i in 0..n {
            for j in i + 1..n {
                let delta = cur.delta_of(&Swap { i, j });
                let (authorized, long_term) = self.classify(p, i, j, it as i64);
                let aspired = long_term || st.is_better(cur.cost() + delta, best);
                // Taillard's rule: the first aspired move is taken whatever its cost; among
                // several aspired moves, the best; otherwise the best authorized move.
                let take = if aspired {
                    !already_aspired || delta < min_delta
                } else {
                    !already_aspired && authorized && delta < min_delta
                };
                if take {
                    retained = Some((i, j, delta));
                    min_delta = delta;
                    already_aspired |= aspired;
                }
            }
        }
        let Some((i, j, delta)) = retained else {
            return Step::Stuck; // every move tabu, none aspired
        };
        self.t[i * n + p[i]] = it as i64 + self.current_tenure; // p[i] leaves i
        self.t[j * n + p[j]] = it as i64 + self.current_tenure; // p[j] leaves j
        st.commit(&Swap { i, j }, delta, obs)
    }

    fn start(&mut self, st: &mut State<S, R>) {
        let n = st.current().size();
        self.prepare(n);
        self.rng = DefaultRng::seed_from_u64(st.rng().next_u64());
        self.next_redraw = 0; // re-draw the tenure at the first move
    }
}

// ------------------------------------------------------ very large neighborhood

/// Destroy: pick `k` random positions. Repair: try every other arrangement of
/// their elements (k! − 1 of them, by Heap's algorithm: one swap per
/// neighbor) and move to the best one, even if it is worse than the current
/// arrangement: like a tabu search, the acceptance rule of the
/// [`LargeNeighborhoodSearch`](crate::LargeNeighborhoodSearch) decides
/// (`Improving` gives a descent, `LateAcceptance` or `SimulatedAnnealing` can
/// escape local optima). The neighborhood has C(n, k)·k! members; one rebuild
/// explores k! of them exactly.
///
/// Each neighbor costs one swap delta and one swap: use a model whose swap is
/// cheap (an O(n) delta and O(1) swap; not a delta matrix updated in O(n²)).
/// k = 7 is 5,040 neighbors per rebuild, k = 8 is 40,320.
pub struct BestRearrangement {
    pub k: usize,
    positions: Vec<usize>,
}

impl BestRearrangement {
    pub fn new(k: usize) -> Self {
        assert!(k >= 2, "rearranging fewer than 2 positions changes nothing");
        BestRearrangement {
            k,
            positions: Vec::new(),
        }
    }
}

impl<S: Permutation> Rebuild<S> for BestRearrangement {
    fn rebuild(&mut self, s: &mut Evaluated<S>, rng: &mut dyn RngCore) {
        let n = s.size();
        let k = self.k.min(n);
        // destroy: k distinct random positions (partial Fisher-Yates)
        self.positions.clear();
        self.positions.extend(0..n);
        for i in 0..k {
            let j = rng.random_range(i..n);
            self.positions.swap(i, j);
        }
        self.positions.truncate(k);
        let pos = &self.positions;

        // repair: Heap's algorithm visits the k! − 1 other arrangements, one swap apart
        let mut best_cost = S::Cost::max_value();
        let mut best: Vec<usize> = pos.iter().map(|&q| s.as_slice()[q]).collect();
        let mut c = vec![0usize; k];
        let mut i = 1;
        while i < k {
            if c[i] < i {
                let a = if i % 2 == 0 { 0 } else { c[i] };
                s.apply(&Swap::new(pos[a], pos[i]));
                if s.cost() < best_cost {
                    best_cost = s.cost();
                    for (t, &q) in pos.iter().enumerate() {
                        best[t] = s.as_slice()[q];
                    }
                }
                c[i] += 1;
                i = 1;
            } else {
                c[i] = 0;
                i += 1;
            }
        }

        // move to the best arrangement (at most k − 1 swaps)
        for t in 0..k {
            let want = best[t];
            if s.as_slice()[pos[t]] != want {
                let u = (t + 1..k)
                    .find(|&u| s.as_slice()[pos[u]] == want)
                    .expect("element in the subset");
                s.apply(&Swap::new(pos[t], pos[u]));
            }
        }
    }
}

// ------------------------------------------------------------------- crossover

/// Uniform crossover for permutations (as in Merz & Freisleben's memetic
/// algorithm for the QAP): positions where both parents agree keep their
/// element; every other position, visited in random order, takes the element
/// of one parent chosen at random if it is still free; the remaining positions
/// get the remaining elements at random.
#[derive(Clone, Copy, Debug, Default)]
pub struct UniformCrossover;

impl<S: Permutation> Crossover<S> for UniformCrossover {
    fn cross(&mut self, a: &S, b: &S, child: &mut S, rng: &mut dyn RngCore) {
        let (pa, pb) = (a.as_slice(), b.as_slice());
        let n = pa.len();
        let mut target = vec![usize::MAX; n];
        let mut used = vec![false; n];
        let mut open = Vec::new();
        for i in 0..n {
            if pa[i] == pb[i] {
                target[i] = pa[i];
                used[pa[i]] = true;
            } else {
                open.push(i);
            }
        }
        for t in (1..open.len()).rev() {
            open.swap(t, rng.random_range(0..=t));
        }
        let mut empty = Vec::new();
        for &i in &open {
            let (first, second) = if rng.random::<bool>() {
                (pa[i], pb[i])
            } else {
                (pb[i], pa[i])
            };
            if !used[first] {
                target[i] = first;
                used[first] = true;
            } else if !used[second] {
                target[i] = second;
                used[second] = true;
            } else {
                empty.push(i);
            }
        }
        let mut free: Vec<usize> = (0..n).filter(|&e| !used[e]).collect();
        for t in (1..free.len()).rev() {
            free.swap(t, rng.random_range(0..=t));
        }
        for (&i, e) in empty.iter().zip(free) {
            target[i] = e;
        }
        rearrange(child, &target);
    }
}

/// Drezner's cohesive crossover (the merging step of his hybrid genetic
/// algorithm for the QAP): the child keeps parent `a`'s elements on a set of
/// mutually close positions, a ball around a random pivot under a distance
/// between positions, and takes parent `b`'s elsewhere. Uniform crossover
/// ignores the geometry of instances whose positions are locations on a grid
/// (sko, tho, wil); this one inherits whole neighbourhoods.
///
/// With [`over_elements`](Self::over_elements) the distance is between
/// elements instead: the ball is a set of close elements, and the child
/// keeps `a`'s positions of those elements.
#[derive(Clone, Debug)]
pub struct CohesiveCrossover {
    /// `order[x]`: all indices by increasing distance to `x` (`x` first).
    order: Vec<Vec<u32>>,
    over_elements: bool,
    /// The ball holds a random fraction of the positions in this range.
    pub fraction: (f64, f64),
}

impl CohesiveCrossover {
    /// `distance(x, y)` between positions `0..n`.
    pub fn new(n: usize, distance: impl Fn(usize, usize) -> f64) -> Self {
        let order = (0..n)
            .map(|x| {
                let mut row: Vec<u32> = (0..n as u32).collect();
                // x first, then by distance (ties by index, for determinism)
                row.sort_by(|&p, &q| {
                    let key = |y: u32| (y as usize != x, distance(x, y as usize));
                    key(p).partial_cmp(&key(q)).unwrap().then(p.cmp(&q))
                });
                row
            })
            .collect();
        CohesiveCrossover {
            order,
            over_elements: false,
            fraction: (0.25, 0.75),
        }
    }

    /// The distance passed to `new` is between elements, not positions.
    pub fn over_elements(mut self) -> Self {
        self.over_elements = true;
        self
    }

    /// The child (as a target permutation) for a ball of `m` indices around
    /// `pivot`: `a` on the ball, `b` where its element is still free, then the
    /// remaining elements in the order they appear in `b`.
    pub fn child_with(&self, a: &[usize], b: &[usize], pivot: usize, m: usize) -> Vec<usize> {
        let n = a.len();
        let ball = &self.order[pivot][..m.min(n)];
        let mut keep = vec![false; n]; // positions taken from a
        if self.over_elements {
            let mut close = vec![false; n];
            ball.iter().for_each(|&e| close[e as usize] = true);
            (0..n).for_each(|i| keep[i] = close[a[i]]);
        } else {
            ball.iter().for_each(|&i| keep[i as usize] = true);
        }
        let mut target = vec![usize::MAX; n];
        let mut used = vec![false; n];
        for i in 0..n {
            if keep[i] {
                target[i] = a[i];
                used[a[i]] = true;
            }
        }
        for i in 0..n {
            if !keep[i] && !used[b[i]] {
                target[i] = b[i];
                used[b[i]] = true;
            }
        }
        let mut free = b.iter().copied().filter(|&e| !used[e]);
        for t in target.iter_mut().filter(|t| **t == usize::MAX) {
            *t = free
                .next()
                .expect("as many free elements as empty positions");
        }
        target
    }
}

impl<S: Permutation> Crossover<S> for CohesiveCrossover {
    fn cross(&mut self, a: &S, b: &S, child: &mut S, rng: &mut dyn RngCore) {
        let n = a.size();
        if n < 2 {
            return;
        }
        let lo = ((self.fraction.0 * n as f64).ceil() as usize).clamp(1, n - 1);
        let hi = ((self.fraction.1 * n as f64).floor() as usize).clamp(lo, n - 1);
        let (pivot, m) = (rng.random_range(0..n), rng.random_range(lo..=hi));
        let target = self.child_with(a.as_slice(), b.as_slice(), pivot, m);
        rearrange(child, &target);
    }
}

/// Path relinking, the permutation counterpart of differential evolution's
/// `a + F·(b − a)`: the child starts as `a` and takes `b`'s element at a
/// random `fraction` of the positions where they differ (through swaps, so
/// it stays a permutation; each swap may also fix a second position).
#[derive(Clone, Copy, Debug)]
pub struct PathRelinking {
    pub fraction: f64,
}

impl<S: Permutation> Crossover<S> for PathRelinking {
    fn cross(&mut self, _a: &S, b: &S, child: &mut S, rng: &mut dyn RngCore) {
        let pb = b.as_slice();
        let mut differ: Vec<usize> = (0..pb.len())
            .filter(|&i| child.as_slice()[i] != pb[i])
            .collect();
        for t in (1..differ.len()).rev() {
            differ.swap(t, rng.random_range(0..=t));
        }
        let steps = ((self.fraction * differ.len() as f64).round() as usize).min(differ.len());
        let mut position = vec![0; pb.len()];
        for (i, &e) in child.as_slice().iter().enumerate() {
            position[e] = i;
        }
        for &i in &differ[..steps] {
            let (current, wanted) = (child.as_slice()[i], pb[i]);
            if current != wanted {
                let j = position[wanted];
                child.swap(i, j);
                position[current] = j;
                position[wanted] = i;
            }
        }
    }
}

// ------------------------------------------------------------ swap descent

/// First-improvement descent on swaps with don't-look bits (Bentley; as in
/// Merz & Freisleben's memetic algorithm for the QAP): positions are visited
/// in turn, and the first improving swap involving the current position is
/// taken. A position whose swaps were all checked without improvement gets
/// its don't-look bit set and is skipped until a move touches it again. One
/// `step` is one move; the run ends with `Stop::Stuck` at a local optimum.
///
/// Much cheaper per step than a full best-improvement scan, which makes it the
/// usual improvement method inside a memetic search. Use a model with a
/// cheap swap (O(1) apply).
#[derive(Clone, Debug, Default)]
pub struct SwapDescent {
    dont_look: Vec<bool>,
    next: usize,
}

impl SwapDescent {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<S: Permutation, R: Rng> Search<S, R> for SwapDescent {
    fn step(&mut self, st: &mut State<S, R>, obs: &mut dyn Observer<S, R>) -> Step {
        let n = self.dont_look.len();
        let threshold = -st.tolerance();
        for offset in 0..n {
            let i = (self.next + offset) % n;
            if self.dont_look[i] {
                continue;
            }
            let cur = st.current();
            let found = (0..n).filter(|&j| j != i).find_map(|j| {
                let m = Swap::new(i, j);
                let d = cur.delta_of(&m);
                (d < threshold).then_some((m, d))
            });
            match found {
                Some((m, d)) => {
                    self.dont_look[m.i] = false;
                    self.dont_look[m.j] = false;
                    self.next = i;
                    return st.commit(&m, d, obs);
                }
                None => self.dont_look[i] = true,
            }
        }
        Step::Stuck
    }

    fn start(&mut self, st: &mut State<S, R>) {
        self.dont_look.clear();
        self.dont_look.resize(st.current().size(), false);
        self.next = 0;
    }
}
