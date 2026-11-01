//! Quadratic Assignment Problem: ports of metslib-examples/qap (main_ts.cc and
//! main.cc), plus SA / LAHC / RRT and ILS on the same models, compared fairly:
//! same time budget, several seeds, gap to the best-known value from QAPLIB.
//!
//! Two models of the same problem show where the delta trick lives:
//! * `QapPlain`: O(n) swap delta (like qap_model.hpp). Good for sampling
//!   methods that evaluate one move per iteration.
//! * `QapTaillard`: keeps Taillard's delta matrix, so a swap delta is O(1) and
//!   a full neighborhood scan costs O(n²) instead of O(n³). Each swap updates
//!   the matrix in O(n²). Good for tabu search on the full neighborhood.
//!
//! usage:
//!   qap [options]                       compare methods on a random Taillard-like instance (n = 60)
//!   qap FILE.dat [FILE.sln] [options]   compare methods on a QAPLIB instance
//!   qap --qaplib DATA_DIR SOLN_DIR [options]
//!                                       iterated tabu search on every QAPLIB instance with a solution
//!   qap FILE.dat --bench ITERS          the setup of the C++ METSlib benchmark (DESIGN.md §4)
//! options:
//!   --seconds S   time budget per run (default 1)
//!   --seeds K     runs per method (default 3; --qaplib uses 1)
//!
//! QAPLIB: https://qaplib.mgi.polymtl.ca/ (instances `*.dat`, solutions `*.sln`).

use mets::perm::{perturb, Permutation, SwapNeighborhood};
use mets::rand::seq::SliceRandom;
use mets::rand::{Rng, SeedableRng};
use mets::*;
use std::cell::Cell;
use std::collections::BTreeMap;
use std::ops::ControlFlow;
use std::path::Path;
use std::rc::Rc;
use std::time::{Duration, Instant};

type C = i64;

// ------------------------------------------------------------- instance

struct Instance {
    n: usize,
    a: Vec<C>, // first matrix, row-major n×n
    b: Vec<C>, // second matrix
}

impl Instance {
    /// cost(p) = Σ a[i][j] · b[p[i]][p[j]]
    fn cost_of(&self, p: &[usize]) -> C {
        let n = self.n;
        let mut s = 0;
        for i in 0..n {
            for j in 0..n {
                s += self.a[i * n + j] * self.b[p[i] * n + p[j]];
            }
        }
        s
    }

    /// O(n) delta of swapping positions r and s (valid for asymmetric instances).
    fn swap_delta(&self, p: &[usize], r: usize, s: usize) -> C {
        let n = self.n;
        let (pr, ps) = (p[r], p[s]);
        // row slices: bounds are checked once here, not in the loop
        let (ar, as_) = (&self.a[r * n..][..n], &self.a[s * n..][..n]);
        let (br, bs) = (&self.b[pr * n..][..n], &self.b[ps * n..][..n]);
        let mut d = ar[r] * (bs[ps] - br[pr])
            + ar[s] * (bs[pr] - br[ps])
            + as_[r] * (br[ps] - bs[pr])
            + as_[s] * (br[pr] - bs[ps]);
        for (k, &pk) in p.iter().enumerate() {
            if k == r || k == s {
                continue;
            }
            let bk = &self.b[pk * n..][..n];
            d += self.a[k * n + r] * (bk[ps] - bk[pr])
                + self.a[k * n + s] * (bk[pr] - bk[ps])
                + ar[k] * (bs[pk] - br[pk])
                + as_[k] * (br[pk] - bs[pk]);
        }
        d
    }

    /// Like Taillard's taiXXa: symmetric matrices, zero diagonal, entries
    /// uniform in 0..=99. (Not one of the published instances.)
    fn random(n: usize, seed: u64) -> Self {
        let mut rng = DefaultRng::seed_from_u64(seed);
        let mut sym = || {
            let mut m = vec![0; n * n];
            for i in 0..n {
                for j in i + 1..n {
                    let v = rng.random_range(0..=99);
                    m[i * n + j] = v;
                    m[j * n + i] = v;
                }
            }
            m
        };
        let a = sym();
        let b = sym();
        Instance { n, a, b }
    }

    fn parse(text: &str) -> Result<Self, String> {
        let v: Vec<C> = text
            .split_whitespace()
            .map(|t| t.parse().map_err(|_| format!("bad number {t:?}")))
            .collect::<Result<_, _>>()?;
        let n = *v.first().ok_or("empty file")? as usize;
        if v.len() != 1 + 2 * n * n {
            return Err(format!(
                "expected {} numbers for n = {n}, found {}",
                1 + 2 * n * n,
                v.len()
            ));
        }
        Ok(Instance {
            n,
            a: v[1..1 + n * n].to_vec(),
            b: v[1 + n * n..].to_vec(),
        })
    }

    fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// The best-known value of an instance, from a QAPLIB `.sln` file.
///
/// QAPLIB solution files do not all use the same convention: most list the
/// permutation in the direction of `cost_of`, a few list its inverse, and a
/// few are inconsistent (kra32's header is above the cost of its own
/// permutation; tai40a's permutation is invalid). The permutation is checked
/// both ways, and the reference is the lower of the header and the verified
/// cost of the permutation.
struct BestKnown {
    value: C,
    note: Option<String>,
}

fn read_sln(path: &Path, inst: &Instance) -> Result<BestKnown, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Vec<i64> = text
        .replace(',', " ")
        .split_whitespace()
        .map(|t| t.parse().unwrap_or(-1))
        .collect();
    if v.len() < 2 || v[0] as usize != inst.n {
        return Err(format!(
            "{}: header does not match the instance",
            path.display()
        ));
    }
    let header = v[1];
    let n = inst.n;
    let p: Vec<usize> = v[2..].iter().map(|&x| (x - 1) as usize).collect();
    let mut seen = vec![false; n];
    let valid = p.len() == n
        && p.iter()
            .all(|&x| x < n && !std::mem::replace(&mut seen[x], true));
    if !valid {
        return Ok(BestKnown {
            value: header,
            note: Some("permutation in .sln is invalid; header value used".into()),
        });
    }
    let mut inv = vec![0; n];
    for (i, &x) in p.iter().enumerate() {
        inv[x] = i;
    }
    let (direct, inverse) = (inst.cost_of(&p), inst.cost_of(&inv));
    Ok(if direct == header || inverse == header {
        BestKnown {
            value: header,
            note: None,
        }
    } else {
        let best = direct.min(inverse);
        BestKnown {
            value: best.min(header),
            note: Some(format!(
                "header {header} does not match its permutation ({best})"
            )),
        }
    })
}

// ------------------------------------------------------------ plain model

#[derive(Clone)]
struct QapPlain<'a> {
    inst: &'a Instance,
    p: Vec<usize>,
}

impl Model for QapPlain<'_> {
    type Cost = C;
    fn evaluate(&self) -> C {
        self.inst.cost_of(&self.p)
    }
}

impl Permutation for QapPlain<'_> {
    fn size(&self) -> usize {
        self.p.len()
    }
    fn swap(&mut self, i: usize, j: usize) {
        self.p.swap(i, j);
    }
    fn swap_delta(&self, i: usize, j: usize) -> Option<C> {
        Some(self.inst.swap_delta(&self.p, i, j))
    }
}

// --------------------------------------------------------- Taillard model

#[derive(Clone)]
struct QapTaillard<'a> {
    inst: &'a Instance,
    p: Vec<usize>,
    delta: Vec<C>, // delta[r*n+s] for r<s
}

impl<'a> QapTaillard<'a> {
    fn new(inst: &'a Instance, p: Vec<usize>) -> Self {
        let n = inst.n;
        let mut delta = vec![0; n * n];
        for r in 0..n {
            for s in r + 1..n {
                delta[r * n + s] = inst.swap_delta(&p, r, s);
            }
        }
        QapTaillard { inst, p, delta }
    }
}

impl Model for QapTaillard<'_> {
    type Cost = C;
    fn evaluate(&self) -> C {
        self.inst.cost_of(&self.p)
    }
}

impl Permutation for QapTaillard<'_> {
    fn size(&self) -> usize {
        self.p.len()
    }
    #[inline]
    fn swap_delta(&self, i: usize, j: usize) -> Option<C> {
        Some(self.delta[i * self.inst.n + j]) // O(1)
    }
    fn swap(&mut self, u: usize, v: usize) {
        let (inst, n) = (self.inst, self.inst.n);
        let (a, b) = (&inst.a, &inst.b);
        self.p.swap(u, v);
        let p = &self.p;
        // Taillard's O(1) update for pairs disjoint from {u, v}, O(n) otherwise
        for r in 0..n {
            for s in r + 1..n {
                let d = &mut self.delta[r * n + s];
                if r == u || r == v || s == u || s == v {
                    *d = inst.swap_delta(p, r, s);
                } else {
                    let (pr, ps, pu, pv) = (p[r], p[s], p[u], p[v]);
                    *d += (a[r * n + u] - a[r * n + v] + a[s * n + v] - a[s * n + u])
                        * (b[ps * n + pu] - b[ps * n + pv] + b[pr * n + pv] - b[pr * n + pu])
                        + (a[u * n + r] - a[v * n + r] + a[v * n + s] - a[u * n + s])
                            * (b[pu * n + ps] - b[pv * n + ps] + b[pv * n + pr] - b[pu * n + pr]);
                }
            }
        }
    }
}

// ------------------------------------------------- counting evaluations

/// Counts the moves a neighborhood offers, or a sampler draws: the number of
/// move evaluations, which is what makes methods comparable.
struct Counted<N> {
    inner: N,
    count: Rc<Cell<u64>>,
}

impl<N> Counted<N> {
    fn new(inner: N, count: &Rc<Cell<u64>>) -> Self {
        Counted {
            inner,
            count: count.clone(),
        }
    }
}

struct CountingVisitor<'a, V> {
    inner: &'a mut V,
    count: u64,
}

impl<M, Co, V: Visitor<M, Co>> Visitor<M, Co> for CountingVisitor<'_, V> {
    #[inline]
    fn visit(&mut self, m: M, delta: Co) -> ControlFlow<()> {
        self.count += 1;
        self.inner.visit(m, delta)
    }
    #[inline]
    fn bound(&self) -> Option<Co> {
        self.inner.bound()
    }
}

impl<S: Model, N: Neighborhood<S>> Neighborhood<S> for Counted<N> {
    type Move = N::Move;
    fn explore<V: Visitor<N::Move, S::Cost>>(
        &mut self,
        s: &Evaluated<S>,
        v: &mut V,
    ) -> ControlFlow<()> {
        let mut cv = CountingVisitor { inner: v, count: 0 };
        let r = self.inner.explore(s, &mut cv);
        self.count.set(self.count.get() + cv.count);
        r
    }
    fn start<R: Rng + ?Sized>(&mut self, s: &Evaluated<S>, rng: &mut R) {
        self.inner.start(s, rng)
    }
}

impl<S: Model, N: Sampler<S>> Sampler<S> for Counted<N> {
    type Move = N::Move;
    #[inline]
    fn sample<R: Rng + ?Sized>(&mut self, s: &Evaluated<S>, rng: &mut R) -> Option<N::Move> {
        self.count.set(self.count.get() + 1);
        self.inner.sample(s, rng)
    }
}

// --------------------------------------------------------------- methods

/// One run of a method: best cost found and move evaluations performed.
struct RunResult {
    best: C,
    evals: u64,
}

/// Everything a run needs: the instance, a starting permutation, a seed, and
/// a time budget that every criterion (inner ones included) respects.
struct Run<'a> {
    inst: &'a Instance,
    start: Vec<usize>,
    seed: u64,
    budget: Duration,
}

/// Stops when the state's clock exceeds the budget. Used for inner runs too,
/// so nested searches don't overshoot.
fn deadline<'a, S: Model + 'a>(budget: Duration) -> impl Termination<S> + 'a {
    stop_when(move |st: &State<S>| st.elapsed() >= budget)
}

/// Mean positive delta of random swaps: the scale of a "bad move", used to
/// set the temperature and thresholds of the stochastic methods.
fn typical_delta(r: &Run) -> f64 {
    let s = Evaluated::new(QapPlain {
        inst: r.inst,
        p: r.start.clone(),
    });
    let mut rng = DefaultRng::seed_from_u64(r.seed);
    let (mut sum, mut k) = (0.0, 0);
    for _ in 0..1000 {
        if let Some(m) = perm::random_swap(r.inst.n, &mut rng) {
            let d = s.delta_of(&m);
            if d > 0 {
                sum += d as f64;
                k += 1;
            }
        }
    }
    if k == 0 {
        1.0
    } else {
        sum / k as f64
    }
}

/// Sampled moves per second of a stochastic search, from a short pilot run.
fn sample_rate(r: &Run) -> f64 {
    let mut st = State::new(
        QapPlain {
            inst: r.inst,
            p: r.start.clone(),
        },
        r.seed,
    );
    let pilot = r.budget / 20;
    let out = StochasticSearch::new(SwapNeighborhood, NonWorsening)
        .runner(&mut st)
        .until(TimeLimit::new(pilot))
        .run();
    out.iterations as f64 / out.elapsed.as_secs_f64().max(1e-6)
}

fn tabu_full(r: &Run) -> RunResult {
    let n = r.inst.n;
    let count = Rc::new(Cell::new(0));
    let mut st = State::new(QapTaillard::new(r.inst, r.start.clone()), r.seed);
    TabuSearch::new(
        Counted::new(SwapNeighborhood, &count),
        DenseTabu::new(Tenure::Fixed(n as u64)),
    )
    .runner(&mut st)
    .until(deadline(r.budget))
    .run();
    RunResult {
        best: st.best().cost(),
        evals: count.get(),
    }
}

fn tabu_sampled(r: &Run) -> RunResult {
    let n = r.inst.n;
    let k = ((n as f64) * (n as f64).sqrt()) as usize;
    let count = Rc::new(Cell::new(0));
    let mut st = State::new(
        QapPlain {
            inst: r.inst,
            p: r.start.clone(),
        },
        r.seed,
    );
    TabuSearch::new(
        Counted::new(Sampled::new(SwapNeighborhood, k), &count),
        RecencyTabu::new(Tenure::Fixed(k as u64)),
    )
    .runner(&mut st)
    .until(deadline(r.budget))
    .run();
    RunResult {
        best: st.best().cost(),
        evals: count.get(),
    }
}

fn restarts_its(r: &Run) -> RunResult {
    let n = r.inst.n;
    let count = Rc::new(Cell::new(0));
    let ts = TabuSearch::new(
        Counted::new(SwapNeighborhood, &count),
        DenseTabu::new(Tenure::Random(7.min(n as u64)..=(7 * n as u64 / 4).max(8))),
    );
    let its = IteratedSearch::new(
        ts,
        NoImprovement::new(200).or(deadline(r.budget)),
        perturb_with(|s: &mut Evaluated<QapTaillard>, rng| {
            let k = rng.random_range(2..=(n / 4).max(2));
            perturb(s, k, rng)
        }),
        Improving,
    );
    let mut restarts = IteratedSearch::new(
        its,
        NoImprovement::new(20).or(deadline(r.budget)),
        perturb_with(|s: &mut Evaluated<QapTaillard>, rng| {
            let mut p: Vec<usize> = (0..n).collect();
            p.shuffle(rng);
            s.modify(|q| *q = QapTaillard::new(q.inst, p));
        }),
        Always,
    );
    let mut st = State::new(QapTaillard::new(r.inst, r.start.clone()), r.seed);
    restarts.runner(&mut st).until(deadline(r.budget)).run();
    RunResult {
        best: st.best().cost(),
        evals: count.get(),
    }
}

fn ils_descent(r: &Run) -> RunResult {
    let n = r.inst.n;
    let count = Rc::new(Cell::new(0));
    let mut st = State::new(QapTaillard::new(r.inst, r.start.clone()), r.seed);
    IteratedSearch::new(
        LocalSearch::best_improvement(Counted::new(SwapNeighborhood, &count)),
        deadline(r.budget), // each descent runs to its local optimum, or to the deadline
        perturb_with(|s: &mut Evaluated<QapTaillard>, rng| perturb(s, (n / 8).max(2), rng)),
        Improving,
    )
    .runner(&mut st)
    .until(deadline(r.budget))
    .run();
    RunResult {
        best: st.best().cost(),
        evals: count.get(),
    }
}

fn stochastic<A: Acceptance<C>>(r: &Run, acceptance: A) -> RunResult {
    let count = Rc::new(Cell::new(0));
    let mut st = State::new(
        QapPlain {
            inst: r.inst,
            p: r.start.clone(),
        },
        r.seed,
    );
    StochasticSearch::new(Counted::new(SwapNeighborhood, &count), acceptance)
        .runner(&mut st)
        .until(deadline(r.budget))
        .run();
    RunResult {
        best: st.best().cost(),
        evals: count.get(),
    }
}

/// SA calibrated to the budget: starts where a typical bad move is accepted
/// with probability 1/e, and cools to 1/10000 of that over the whole run.
fn annealing(r: &Run) -> RunResult {
    let t0 = typical_delta(r);
    let total = sample_rate(r) * r.budget.as_secs_f64();
    let alpha = 0.95_f64;
    let steps = ((1e-4_f64).ln() / alpha.ln()).ceil();
    let plateau = ((total / steps) as u64).max(1);
    stochastic(r, SimulatedAnnealing::new(t0, t0 * 1e-4, alpha, plateau))
}

fn late_acceptance(r: &Run) -> RunResult {
    let total = sample_rate(r) * r.budget.as_secs_f64();
    stochastic(
        r,
        LateAcceptance::new(((total / 1000.0) as usize).max(r.inst.n)),
    )
}

fn record_to_record(r: &Run) -> RunResult {
    stochastic(r, RecordToRecord::new(typical_delta(r) / 2.0))
}

type Method = (&'static str, fn(&Run) -> RunResult);

const METHODS: [Method; 7] = [
    ("tabu, full neighborhood (Taillard)", tabu_full),
    ("tabu, sampled neighborhood (main_ts.cc)", tabu_sampled),
    ("restarts(ITS(tabu)) (main.cc)", restarts_its),
    ("ILS(steepest descent)", ils_descent),
    ("simulated annealing", annealing),
    ("late acceptance hill climbing", late_acceptance),
    ("record-to-record travel", record_to_record),
];

fn random_start(n: usize, seed: u64) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    p.shuffle(&mut DefaultRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15));
    p
}

fn gap(cost: C, best_known: C) -> f64 {
    (cost - best_known) as f64 * 100.0 / (best_known as f64).abs().max(1.0)
}

// ------------------------------------------------------------------ modes

struct Options {
    budget: Duration,
    seeds: u64,
}

fn compare(name: &str, inst: &Instance, best_known: Option<BestKnown>, opt: &Options) {
    print!(
        "{name}: n = {}, {:.1} s per run, {} seed(s)",
        inst.n,
        opt.budget.as_secs_f64(),
        opt.seeds
    );
    match &best_known {
        Some(b) => println!(
            ", best known {}{}",
            b.value,
            b.note
                .as_ref()
                .map(|n| format!(" ({n})"))
                .unwrap_or_default()
        ),
        None => println!(", no best-known value"),
    }
    let head = if best_known.is_some() {
        ("mean gap", "best gap")
    } else {
        ("mean cost", "best cost")
    };
    println!(
        "{:<42} {:>10} {:>10} {:>14}",
        "method", head.0, head.1, "evaluations/s"
    );
    for (label, method) in METHODS {
        let mut results = Vec::new();
        for seed in 1..=opt.seeds {
            let run = Run {
                inst,
                start: random_start(inst.n, seed),
                seed,
                budget: opt.budget,
            };
            let t = Instant::now();
            let res = method(&run);
            results.push((res, t.elapsed()));
        }
        let evals_per_s = results
            .iter()
            .map(|(r, t)| r.evals as f64 / t.as_secs_f64())
            .sum::<f64>()
            / results.len() as f64;
        let bests: Vec<C> = results.iter().map(|(r, _)| r.best).collect();
        let best = *bests.iter().min().unwrap();
        let mean = bests.iter().sum::<C>() as f64 / bests.len() as f64;
        match &best_known {
            Some(b) => {
                let mean_gap =
                    bests.iter().map(|&c| gap(c, b.value)).sum::<f64>() / bests.len() as f64;
                println!(
                    "{label:<42} {:>9.2}% {:>9.2}% {evals_per_s:>14.3e}",
                    mean_gap,
                    gap(best, b.value)
                );
            }
            None => println!("{label:<42} {mean:>10.0} {best:>10} {evals_per_s:>14.3e}"),
        }
    }
}

fn sweep(data: &Path, soln: &Path, opt: &Options) {
    let mut names: Vec<String> = std::fs::read_dir(data)
        .expect("cannot read the data directory")
        .filter_map(|e| {
            e.ok()?
                .file_name()
                .to_str()?
                .strip_suffix(".dat")
                .map(String::from)
        })
        .collect();
    names.sort();
    println!(
        "restarts(ITS(tabu)), {:.1} s per instance, {} seed(s); instances without a .sln are skipped",
        opt.budget.as_secs_f64(),
        opt.seeds
    );
    println!(
        "{:<10} {:>4} {:>12} {:>12} {:>8}  note",
        "instance", "n", "best known", "found", "gap"
    );
    let mut families: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for name in names {
        let sln = soln.join(format!("{name}.sln"));
        if !sln.exists() {
            continue;
        }
        let inst = match Instance::read(&data.join(format!("{name}.dat"))) {
            Ok(i) => i,
            Err(e) => {
                println!("{name:<10} skipped: {e}");
                continue;
            }
        };
        let bk = match read_sln(&sln, &inst) {
            Ok(b) => b,
            Err(e) => {
                println!("{name:<10} skipped: {e}");
                continue;
            }
        };
        let found = (1..=opt.seeds)
            .map(|seed| {
                restarts_its(&Run {
                    inst: &inst,
                    start: random_start(inst.n, seed),
                    seed,
                    budget: opt.budget,
                })
                .best
            })
            .min()
            .unwrap();
        let g = gap(found, bk.value);
        println!(
            "{name:<10} {:>4} {:>12} {found:>12} {g:>7.2}%  {}",
            inst.n,
            bk.value,
            bk.note.unwrap_or_default()
        );
        let family: String = name
            .chars()
            .take_while(|c| c.is_ascii_alphabetic())
            .collect();
        families.entry(family).or_default().push(g);
    }
    println!("\nmean gap by family:");
    for (family, gaps) in families {
        let at_best = gaps.iter().filter(|&&g| g <= 0.0).count();
        println!(
            "  {family:<5} {:>3} instances  mean {:>6.2}%  max {:>6.2}%  best known reached on {at_best}",
            gaps.len(),
            gaps.iter().sum::<f64>() / gaps.len() as f64,
            gaps.iter().cloned().fold(f64::MIN, f64::max)
        );
    }
}

/// Same setup as METSlib's tabu_search<swap_full_neighborhood> +
/// simple_tabu_list(n) + best_ever_criteria, starting from the identity.
fn bench(inst: &Instance, iters: u64) {
    fn go<S: Permutation<Cost = C>>(
        name: &str,
        s: S,
        tabu: impl TabuList<S, perm::Swap>,
        iters: u64,
    ) {
        let mut st = State::new(s, 1);
        let mut ts = TabuSearch::new(SwapNeighborhood, tabu);
        let t = Instant::now();
        ts.runner(&mut st).until(MaxIterations::new(iters)).run();
        let secs = t.elapsed().as_secs_f64();
        println!(
            "{name:<34} best {}  {iters} iters  {secs:.4} s  {:.2} us/iter",
            st.best().cost(),
            secs / iters as f64 * 1e6
        );
    }
    let n = inst.n;
    let id: Vec<usize> = (0..n).collect();
    let tenure = Tenure::Fixed(n as u64);
    go(
        "Rust plain    + RecencyTabu(hash)",
        QapPlain {
            inst,
            p: id.clone(),
        },
        RecencyTabu::new(tenure.clone()),
        iters,
    );
    go(
        "Rust plain    + DenseTabu",
        QapPlain {
            inst,
            p: id.clone(),
        },
        DenseTabu::new(tenure.clone()),
        iters,
    );
    go(
        "Rust Taillard + DenseTabu",
        QapTaillard::new(inst, id),
        DenseTabu::new(tenure),
        iters,
    );
}

fn main() {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    let mut take = |flag: &str| -> Option<String> {
        let i = args.iter().position(|a| a == flag)?;
        let v = args.get(i + 1).cloned();
        args.drain(i..(i + 2).min(args.len()));
        v
    };
    let seconds: f64 = take("--seconds")
        .map(|s| s.parse().expect("--seconds S"))
        .unwrap_or(1.0);
    let seeds: Option<u64> = take("--seeds").map(|s| s.parse().expect("--seeds K"));
    let bench_iters: Option<u64> = take("--bench").map(|s| s.parse().expect("--bench ITERS"));
    let qaplib = args.iter().position(|a| a == "--qaplib");
    let budget = Duration::from_secs_f64(seconds);

    if let Some(i) = qaplib {
        let (data, soln) = (
            args.get(i + 1).expect("--qaplib DATA_DIR SOLN_DIR"),
            args.get(i + 2).expect("SOLN_DIR"),
        );
        return sweep(
            Path::new(data),
            Path::new(soln),
            &Options {
                budget,
                seeds: seeds.unwrap_or(1),
            },
        );
    }
    let opt = Options {
        budget,
        seeds: seeds.unwrap_or(3),
    };
    match args.first() {
        None => {
            if let Some(iters) = bench_iters {
                return bench(&Instance::random(60, 1), iters);
            }
            compare(
                "random Taillard-like instance",
                &Instance::random(60, 1),
                None,
                &opt,
            )
        }
        Some(dat) => {
            let path = Path::new(dat);
            let inst = Instance::read(path).unwrap_or_else(|e| panic!("{e}"));
            if let Some(iters) = bench_iters {
                return bench(&inst, iters);
            }
            let best_known = args
                .get(1)
                .map(|s| read_sln(Path::new(s), &inst).unwrap_or_else(|e| panic!("{e}")));
            let name = path.file_stem().and_then(|s| s.to_str()).unwrap_or(dat);
            compare(name, &inst, best_known, &opt)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deltas_match_full_evaluation() {
        let inst = Instance::random(13, 5);
        let mut st = State::new(QapTaillard::new(&inst, (0..13).collect()), 1);
        for _ in 0..200 {
            st.modify_current(|s, rng| perturb(s, 1, rng));
            assert!(check::deltas(&mut SwapNeighborhood, st.current()).is_empty());
            let plain = Evaluated::new(QapPlain {
                inst: &inst,
                p: st.current().p.clone(),
            });
            assert!(check::deltas(&mut SwapNeighborhood, &plain).is_empty());
        }
    }

    #[test]
    fn asymmetric_deltas_match_too() {
        let mut inst = Instance::random(11, 9);
        inst.a[3] += 17; // break the symmetry and the zero diagonal
        inst.b[11 * 4 + 4] = 5;
        let s = Evaluated::new(QapTaillard::new(&inst, random_start(11, 2)));
        assert!(check::deltas(&mut SwapNeighborhood, &s).is_empty());
    }

    #[test]
    fn random_instances_look_like_taillards() {
        let inst = Instance::random(20, 3);
        for i in 0..20 {
            assert_eq!(inst.a[i * 20 + i], 0);
            for j in 0..20 {
                assert_eq!(inst.a[i * 20 + j], inst.a[j * 20 + i]);
                assert!((0..=99).contains(&inst.b[i * 20 + j]));
            }
        }
    }

    #[test]
    fn sln_conventions_are_both_accepted() {
        let inst = Instance::random(8, 4);
        let p = random_start(8, 1);
        let mut inv = vec![0; 8];
        for (i, &x) in p.iter().enumerate() {
            inv[x] = i;
        }
        let dir = std::env::temp_dir().join(format!("mets-sln-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let write = |name: &str, header: C, perm: &[usize]| {
            let body: Vec<String> = perm.iter().map(|x| (x + 1).to_string()).collect();
            let path = dir.join(name);
            std::fs::write(&path, format!("8 {header}\n{}\n", body.join(" "))).unwrap();
            path
        };
        let direct = read_sln(&write("a.sln", inst.cost_of(&p), &p), &inst).unwrap();
        assert!(direct.note.is_none());
        let inverse = read_sln(&write("b.sln", inst.cost_of(&p), &inv), &inst).unwrap();
        assert!(inverse.note.is_none());
        let wrong = read_sln(&write("c.sln", inst.cost_of(&p) + 100, &p), &inst).unwrap();
        // either reading of the permutation is a feasible assignment: the cheaper is used
        assert_eq!(wrong.value, inst.cost_of(&p).min(inst.cost_of(&inv)));
        assert!(wrong.note.is_some());
        let broken = read_sln(&write("d.sln", 42, &p[..7]), &inst).unwrap();
        assert_eq!(broken.value, 42);
        std::fs::remove_dir_all(dir).ok();
    }
}
