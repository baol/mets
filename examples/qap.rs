//! Quadratic Assignment Problem: ports of metslib-examples/qap (main_ts.cc and
//! main.cc), plus SA / LAHC / RRT and ILS on the same models, compared fairly:
//! same time budget, several seeds, gap to the best-known value from QAPLIB.
//! See `qap_advanced` for a stronger solver on the same models.
//!
//! Two models of the same problem (`examples/common/qap.rs`) show where the
//! delta trick lives:
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
//!                                       iterated tabu search on every QAPLIB instance
//!   qap FILE.dat --bench ITERS          the setup of the C++ METSlib benchmark (DESIGN.md §4)
//! options:
//!   --seconds S   time budget per run (default 1)
//!   --seeds K     runs per method (default 3; --qaplib uses 1)
//!   --restarts R  --qaplib only: R random restarts per instance, every inner
//!                 search stopping on no-improvement (no time limit unless --seconds)
//!   --jobs J      --qaplib only: instances solved in parallel (default: all cores)
//!   --markdown    --qaplib only: print the results as a Markdown table
//!
//! QAPLIB: https://qaplib.mgi.polymtl.ca/ (instances `*.dat`, solutions `*.sln`).

#[path = "common/qap.rs"]
mod common;

use common::*;
use mets::perm::{perturb, Permutation, SwapNeighborhood};
use mets::rand::seq::SliceRandom;
use mets::rand::{Rng, SeedableRng};
use mets::*;
use std::cell::Cell;
use std::ops::ControlFlow;
use std::path::Path;
use std::rc::Rc;
use std::time::{Duration, Instant};

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
    #[inline]
    fn can_skip(&self, lower: Co) -> bool
    where
        Co: PartialOrd,
    {
        self.inner.can_skip(lower)
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
    /// Wall-clock limit (`Duration::MAX`: none).
    budget: Duration,
    /// restarts(ITS) only: number of random restarts (`None`: until the budget).
    restarts: Option<u64>,
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

/// The three nested loops of metslib-examples qap/main.cc: random restarts of
/// an iterated tabu search with random tenure.
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
    restarts
        .runner(&mut st)
        .until(MaxIterations::new(r.restarts.unwrap_or(u64::MAX)).or(deadline(r.budget)))
        .run();
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

// ------------------------------------------------------------------ modes

fn compare(
    name: &str,
    inst: &Instance,
    best_known: Option<BestKnown>,
    budget: Duration,
    seeds: u64,
) {
    print!(
        "{name}: n = {}, {:.1} s per run, {seeds} seed(s)",
        inst.n,
        budget.as_secs_f64()
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
        for seed in 1..=seeds {
            let run = Run {
                inst,
                start: random_start(inst.n, seed),
                seed,
                budget,
                restarts: None,
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
    let mut args = Args::from_env();
    let markdown = args.flag("--markdown");
    let seconds: Option<f64> = args.value("--seconds");
    let restarts: Option<u64> = args.value("--restarts");
    let jobs: usize = args
        .value("--jobs")
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    let seeds: Option<u64> = args.value("--seeds");
    let bench_iters: Option<u64> = args.value("--bench");
    let args = args.0;
    // with --restarts and no --seconds, runs are not time-limited
    let budget = match (seconds, restarts) {
        (Some(s), _) => Duration::from_secs_f64(s),
        (None, Some(_)) => Duration::MAX,
        (None, None) => Duration::from_secs(1),
    };

    if let Some(i) = args.iter().position(|a| a == "--qaplib") {
        let (data, soln) = (
            args.get(i + 1).expect("--qaplib DATA_DIR SOLN_DIR"),
            args.get(i + 2).expect("SOLN_DIR"),
        );
        let seeds = seeds.unwrap_or(1);
        let stop = match restarts {
            Some(r) => format!(
                "{r} random restarts, each an iterated tabu search (tabu runs stop after 200 iterations \
                 without improvement, the iterated search after 20 perturbations without improvement)"
            ),
            None => format!("{} s per instance", budget.as_secs_f64()),
        };
        let opt = SweepOptions {
            setup: format!("restarts(ITS(tabu)), {stop}, best of {seeds} seed(s)"),
            instance_jobs: jobs,
            markdown,
        };
        return sweep(
            Path::new(data),
            Path::new(soln),
            &opt,
            |inst, _best_known| {
                let found = (1..=seeds)
                    .map(|seed| {
                        restarts_its(&Run {
                            inst,
                            start: random_start(inst.n, seed),
                            seed,
                            budget,
                            restarts,
                        })
                        .best
                    })
                    .min()
                    .unwrap();
                Solved {
                    found,
                    to_found: None,
                }
            },
        );
    }
    let seeds = seeds.unwrap_or(3);
    match args.first() {
        None => {
            if let Some(iters) = bench_iters {
                return bench(&Instance::random(60, 1), iters);
            }
            compare(
                "random Taillard-like instance",
                &Instance::random(60, 1),
                None,
                budget,
                seeds,
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
            compare(name, &inst, best_known, budget, seeds)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Taillard's grey-density layout: an s×s torus, similarity
    /// 100,000 / (dx² + dy²) between cells, and a 0/1 flow among m cells.
    fn grey_density(s: usize, m: usize) -> Instance {
        let n = s * s;
        let wrap = |d: usize| d.min(s - d) as f64;
        let b: Vec<C> = (0..n * n)
            .map(|k| {
                let (p, q) = (k / n, k % n);
                if p == q {
                    return 0;
                }
                let (dx, dy) = (wrap((p / s).abs_diff(q / s)), wrap((p % s).abs_diff(q % s)));
                (100_000.0 / (dx * dx + dy * dy)).round() as C
            })
            .collect();
        let a: Vec<C> = (0..n * n)
            .map(|k| ((k / n) < m && (k % n) < m) as C)
            .collect();
        Instance::new(n, a, b)
    }

    #[test]
    fn proximity_detects_distances_and_similarities() {
        // sko-like: grid distances in the first matrix, random flows in the second.
        let s: usize = 6;
        let n = s * s;
        let a: Vec<C> = (0..n * n)
            .map(|k| {
                let (p, q) = (k / n, k % n);
                ((p / s).abs_diff(q / s) + (p % s).abs_diff(q % s)) as C
            })
            .collect();
        let mut rng = mets::DefaultRng::seed_from_u64(3);
        let b: Vec<C> = (0..n * n)
            .map(|k| {
                if k / n == k % n {
                    0
                } else {
                    rng.random_range(0..10)
                }
            })
            .collect();
        let sko = Instance::new(n, a, b);
        let px = sko.proximity().expect("grid distances");
        assert!(!px.over_elements && !px.similarity);

        let grey = grey_density(8, 20);
        assert!(grey.distance().is_none(), "1/d² is not a metric");
        let px = grey.proximity().expect("similarity on the torus");
        assert!(px.over_elements && px.similarity, "{}", px.describe());
        // The crossover ranks the pivot's torus neighbours first: a ball of 5
        // around cell 0 is the cell and its 4 neighbours (1, 7, 8, 56). With
        // a = identity the child keeps a at those positions and takes b
        // elsewhere where the element is still free.
        let x = grey.cohesive_crossover().unwrap();
        let a: Vec<usize> = (0..64).collect();
        let b: Vec<usize> = (0..64).rev().collect();
        let child = x.child_with(&a, &b, 0, 5);
        for i in [0, 1, 7, 8, 56] {
            assert_eq!(child[i], i, "position {i} comes from a");
        }
        assert_eq!(child[2], b[2]);
        assert_eq!(child[61], b[61], "element 2 is free");
        let mut sorted = child.clone();
        sorted.sort();
        assert_eq!(sorted, a);

        assert!(Instance::random(20, 1).proximity().is_none());
    }

    #[test]
    fn deltas_match_full_evaluation() {
        let inst = Instance::random(13, 5);
        assert!(inst.symmetric);
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
        let sym = Instance::random(11, 9);
        let (mut a, mut b) = (sym.a.clone(), sym.b.clone());
        a[3] += 17; // break the symmetry and the zero diagonal
        b[11 * 4 + 4] = 5;
        let inst = Instance::new(11, a, b);
        assert!(!inst.symmetric);
        let mut st = State::new(QapTaillard::new(&inst, random_start(11, 2)), 1);
        for _ in 0..50 {
            st.modify_current(|s, rng| perturb(s, 1, rng));
            assert!(check::deltas(&mut SwapNeighborhood, st.current()).is_empty());
        }
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
