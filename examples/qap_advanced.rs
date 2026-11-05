//! Stronger QAP solvers, assembled from library pieces on the models of
//! `examples/common/qap.rs` (the symmetric fast path halves the delta cost on
//! most QAPLIB instances):
//!
//! * `memetic` (default): a population of 10 solutions improved by an
//!   iterated robust tabu search (`perm::RobustTabuSearch` +
//!   `perm::adaptive_perturbation`); new starting points come from a
//!   `perm::UniformCrossover` of two members instead of a random restart;
//!   members stay at least n/10 positions apart (`perm::hamming`).
//! * `restarts`: the same improvement, restarted from random permutations
//!   (the baseline the crossover replaces).
//! * `ma`: Merz & Freisleben's memetic algorithm (1999): population 40,
//!   uniform crossover, `perm::SwapDescent` (first improvement, don't-look
//!   bits) on the plain model, renewal after 30 children without insertion.
//! * `lns`: large neighborhood search with `perm::BestRearrangement`: every
//!   step moves to the best other arrangement of k random positions (all
//!   k! − 1 of them are tried), accepted by late acceptance over 10n
//!   iterations.
//!
//! `--runs` independent runs (seeds 1..=runs) execute on `--jobs` threads;
//! the best is reported. Every stopping rule is a count, so results depend
//! only on the seeds. With a `.sln`, all runs stop when the best-known value
//! is reached.
//!
//! usage:
//!   qap_advanced FILE.dat [FILE.sln] [options]         solve one instance
//!   qap_advanced --qaplib DATA_DIR SOLN_DIR [options]  solve every instance
//! options:
//!   --method M     memetic | restarts | lns | ma (default memetic)
//!   --steps N      per run: children / restarts (default 50; ma 100000), or rebuilds for lns (default 1500000)
//!   --population P memetic / ma: population size (default 10; ma 40)
//!   --crossover X  memetic / ma: uniform | path (path relinking half-way, DE-style) |
//!                  cohesive (Drezner: one parent's elements on a ball of close positions;
//!                  uniform on instances without a distance matrix) (default uniform)
//!   --replacement Y  memetic: niche (closest member or worst) | parent (DE one-to-one) (default niche)
//!   --k K          lns: positions rearranged per step (default 5)
//!   --runs R       independent runs (default 4)
//!   --first-seed S seed of the first run; runs use S, S+1, … (default 1)
//!   --migrate-every G  ma: the runs become islands in a ring, the best member
//!                  migrating every G children (one thread per island; default 0: independent)
//!   --migration M  ma islands: replace (migrant replaces the worst member) | cross (migrant
//!                  is the second parent of the next child) | part (the child takes only 20%
//!                  of the migrant, by path relinking) (default replace)
//!   --async-migration  ma islands: mailboxes instead of synchronised exchanges (no waiting;
//!                  results then depend on thread timing)
//!   --jobs J       threads (default: all cores)
//!   --run-to-end   do not stop at the best-known value
//!   --max-seconds T  also stop each run after T seconds and report its best
//!                  (results then depend on timing)
//!   --markdown     --qaplib only: print a Markdown table

#[path = "common/qap.rs"]
mod common;

use common::*;
use mets::parallel::{self, run_many, Cancel};
use mets::perm::{
    adaptive_perturbation, shuffle, BestRearrangement, RobustTabuSearch, UniformCrossover,
};
use mets::*;
use std::path::Path;
use std::time::Instant;

#[derive(Clone, Copy, PartialEq)]
enum Method {
    Memetic,
    Restarts,
    Lns,
    /// Merz & Freisleben (1999): population 40, swap descent with don't-look bits.
    Ma,
}

struct Settings {
    method: Method,
    steps: u64,
    runs: u64,
    /// Seed of the first run (runs use first_seed, first_seed + 1, …).
    first_seed: u64,
    /// ma: run the runs as islands, migrating every this many children (0: independent).
    migrate_every: u64,
    /// ma islands: what a migrant does on arrival.
    migration: Migration,
    /// ma islands: post/take mailboxes instead of synchronised exchanges.
    async_migration: bool,
    /// Wall-clock cap per run, in seconds (results then depend on timing).
    max_seconds: Option<f64>,
    jobs: usize,
    stop_at_best: bool,
    /// lns: positions rearranged per step.
    k: usize,
    /// memetic: population size.
    population: usize,
    /// memetic: recombination.
    crossover: Recombine,
    /// memetic: differential-evolution one-to-one selection.
    replace_parent: bool,
}

/// What a migrant does when it reaches an island.
#[derive(Clone, Copy, PartialEq)]
enum Migration {
    /// It replaces the island's worst member (if better).
    Replace,
    /// It is the second parent of the next child (the island's crossover).
    Cross,
    /// Same, but the child takes only 20% of the migrant's differing
    /// assignments (path relinking): a guided perturbation.
    Part,
}

/// The permutation recombinations of the library, chosen at run time.
#[derive(Clone, Copy, PartialEq)]
enum Recombine {
    Uniform,
    /// Path relinking half-way: DE's `a + F·(b − a)` with F = 0.5.
    Path,
    /// Drezner's cohesive crossover on the instance's distance matrix
    /// (uniform when the instance has none).
    Cohesive,
}

/// A [`Recombine`] made for one instance (the cohesive crossover needs its
/// distances).
enum Recombiner {
    Uniform,
    Path,
    Cohesive(perm::CohesiveCrossover),
}

impl Recombiner {
    fn new(r: Recombine, inst: &Instance) -> Self {
        match (r, inst.cohesive_crossover()) {
            (Recombine::Uniform, _) | (Recombine::Cohesive, None) => Recombiner::Uniform,
            (Recombine::Path, _) => Recombiner::Path,
            (Recombine::Cohesive, Some(x)) => Recombiner::Cohesive(x),
        }
    }
}

impl<S: perm::Permutation> Crossover<S> for Recombiner {
    fn cross(&mut self, a: &S, b: &S, child: &mut S, rng: &mut dyn mets::rand::RngCore) {
        match self {
            Recombiner::Uniform => UniformCrossover.cross(a, b, child, rng),
            Recombiner::Path => perm::PathRelinking { fraction: 0.5 }.cross(a, b, child, rng),
            Recombiner::Cohesive(x) => x.cross(a, b, child, rng),
        }
    }
}

/// The improvement step of `memetic` and `restarts`: iterated robust tabu
/// search, tabu runs of up to max(200, 2n) non-improving iterations, stopped
/// after 20 perturbations without improvement.
fn improvement<'a>(n: usize, cancel: &'a Cancel) -> impl Search<QapTaillard<'a>> + 'a {
    IteratedSearch::new(
        RobustTabuSearch::new(),
        NoImprovement::new((2 * n as u64).max(200)).or(cancel),
        adaptive_perturbation(n),
        Improving,
    )
}

/// One run with seed `seed`; returns the best cost found. Cancels every run
/// when `target` is reached.
fn run(inst: &Instance, seed: u64, target: Option<C>, cancel: &Cancel, set: &Settings) -> C {
    let n = inst.n;
    let fresh = || perturb_with(|s: &mut Evaluated<QapTaillard>, rng| shuffle(s, rng));
    let start = random_start(n, seed);
    // count-based, plus an optional wall-clock cap that still reports the result
    let cap = set
        .max_seconds
        .map_or(std::time::Duration::MAX, std::time::Duration::from_secs_f64);
    let steps = MaxIterations::new(set.steps)
        .or(cancel)
        .or(TimeLimit::new(cap));
    match set.method {
        Method::Memetic => {
            let mut st = State::new(QapTaillard::new(inst, start), seed);
            let memetic = Memetic::new(
                improvement(n, cancel),
                NoImprovement::new(20).or(cancel),
                fresh(),
                Recombiner::new(set.crossover, inst),
                set.population,
            )
            .with_distance(
                |a: &QapTaillard, b: &QapTaillard| perm::hamming(a, b),
                n / 10,
            )
            .renew_after(20);
            let mut memetic = if set.replace_parent {
                memetic.replace_parent()
            } else {
                memetic
            };
            memetic
                .runner(&mut st)
                .until(steps)
                .on_global_improvement(|st| stop_at(st.global_best().cost(), target, cancel))
                .run();
            st.global_best().cost()
        }
        Method::Restarts => {
            let mut st = State::new(QapTaillard::new(inst, start), seed);
            IteratedSearch::new(
                improvement(n, cancel),
                NoImprovement::new(20).or(cancel),
                fresh(),
                Always,
            )
            .runner(&mut st)
            .until(steps)
            .on_global_improvement(|st| stop_at(st.global_best().cost(), target, cancel))
            .run();
            st.global_best().cost()
        }
        Method::Lns => {
            // the rearrangement swaps a lot: the plain model's O(1) swap fits it better
            let mut st = State::new(QapPlain { inst, p: start }, seed);
            LargeNeighborhoodSearch::new(
                BestRearrangement::new(set.k),
                LateAcceptance::new(10 * n),
            )
            .runner(&mut st)
            .until(steps)
            .on_global_improvement(|st| stop_at(st.global_best().cost(), target, cancel))
            .run();
            st.global_best().cost()
        }
        Method::Ma => {
            // many cheap moves: the plain model's O(1) swap fits it better
            let mut st = State::new(QapPlain { inst, p: start }, seed);
            let fresh = perturb_with(|s: &mut Evaluated<QapPlain>, rng| shuffle(s, rng));
            Memetic::new(
                perm::SwapDescent::new(),
                cancel,
                fresh,
                Recombiner::new(set.crossover, inst),
                set.population,
            )
            .with_distance(|a: &QapPlain, b: &QapPlain| perm::hamming(a, b), n / 10)
            .renew_after(30)
            .runner(&mut st)
            .until(steps)
            .on_global_improvement(|st| stop_at(st.global_best().cost(), target, cancel))
            .run();
            st.global_best().cost()
        }
    }
}

fn stop_at(cost: C, target: Option<C>, cancel: &Cancel) {
    if target.is_some_and(|t| cost <= t) {
        cancel.cancel();
    }
}

/// `set.runs` runs on `set.jobs` threads; the best of them.
/// `ma` as an island model: `set.runs` islands in a ring, the best member
/// migrating every `set.migrate_every` children.
fn ma_islands(inst: &Instance, target: Option<C>, set: &Settings) -> Vec<C> {
    let n = inst.n;
    parallel::islands(set.runs as usize, |island| {
        let seed = set.first_seed + island.index() as u64;
        let mut st = State::new(
            QapPlain {
                inst,
                p: random_start(n, seed),
            },
            seed,
        );
        let fresh = perturb_with(|s: &mut Evaluated<QapPlain>, rng| shuffle(s, rng));
        let mut ma = Memetic::new(
            perm::SwapDescent::new(),
            island.cancel,
            fresh,
            Recombiner::new(set.crossover, inst),
            set.population,
        )
        .with_distance(|a: &QapPlain, b: &QapPlain| perm::hamming(a, b), n / 10)
        .renew_after(30);
        if set.migration == Migration::Part {
            // take only a fifth of the migrant's differing assignments
            ma = ma.with_migration_crossover(perm::PathRelinking { fraction: 0.2 });
        }
        let epochs = set.steps.div_ceil(set.migrate_every);
        for epoch in 0..epochs {
            let r = ma
                .runner(&mut st)
                .until(MaxIterations::new(set.migrate_every).or(island.cancel))
                .on_global_improvement(|st| {
                    stop_at(st.global_best().cost(), target, island.cancel)
                });
            if epoch == 0 {
                r.run()
            } else {
                r.resume()
            };
            let migrant = if set.async_migration {
                if let Some(b) = ma.best() {
                    island.post(b.clone());
                }
                island.take()
            } else {
                island.exchange(ma.best().cloned())
            };
            if let Some(m) = migrant {
                match set.migration {
                    Migration::Replace => {
                        ma.immigrate(m);
                    }
                    Migration::Cross | Migration::Part => ma.receive(m),
                }
            }
        }
        st.global_best().cost()
    })
}

fn solve(inst: &Instance, best_known: Option<C>, set: &Settings) -> Solved {
    let target = best_known.filter(|_| set.stop_at_best);
    if set.crossover == Recombine::Cohesive && inst.distance().is_none() {
        eprintln!(
            "note: no distance matrix in this instance; cohesive crossover falls back to uniform"
        );
    }
    let t0 = Instant::now();
    if set.method == Method::Ma && set.migrate_every > 0 {
        let found = ma_islands(inst, target, set)
            .into_iter()
            .min()
            .expect("at least one island");
        return Solved {
            found,
            to_found: Some(t0.elapsed().as_secs_f64()),
        };
    }
    let results = run_many(set.runs, set.jobs, |r, cancel| {
        (
            run(inst, set.first_seed + r, target, cancel, set),
            t0.elapsed(),
        )
    });
    let (found, at) = results
        .into_iter()
        .map(|(_, x)| x)
        .min_by_key(|(c, _)| *c)
        .expect("at least one run");
    Solved {
        found,
        to_found: Some(at.as_secs_f64()),
    }
}

fn main() {
    let mut args = Args::from_env();
    let markdown = args.flag("--markdown");
    let run_to_end = args.flag("--run-to-end");
    let method = match args.value::<String>("--method").as_deref() {
        None | Some("memetic") => Method::Memetic,
        Some("restarts") => Method::Restarts,
        Some("lns") => Method::Lns,
        Some("ma") => Method::Ma,
        Some(m) => panic!("unknown method {m:?} (memetic, restarts, lns)"),
    };
    let set = Settings {
        method,
        steps: args.value("--steps").unwrap_or(match method {
            Method::Lns => 1_500_000,
            Method::Ma => 100_000,
            _ => 50,
        }),
        runs: args.value("--runs").unwrap_or(4),
        first_seed: args.value("--first-seed").unwrap_or(1),
        migrate_every: args.value("--migrate-every").unwrap_or(0),
        migration: match args.value::<String>("--migration").as_deref() {
            None | Some("replace") => Migration::Replace,
            Some("cross") => Migration::Cross,
            Some("part") => Migration::Part,
            Some(x) => panic!("unknown migration {x:?} (replace, cross, part)"),
        },
        async_migration: args.flag("--async-migration"),
        max_seconds: args.value("--max-seconds"),
        jobs: args
            .value("--jobs")
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get())),
        stop_at_best: !run_to_end,
        k: args.value("--k").unwrap_or(5),
        population: args
            .value("--population")
            .unwrap_or(if method == Method::Ma { 40 } else { 10 }),
        crossover: match args.value::<String>("--crossover").as_deref() {
            None | Some("uniform") => Recombine::Uniform,
            Some("path") => Recombine::Path,
            Some("cohesive") => Recombine::Cohesive,
            Some(x) => panic!("unknown crossover {x:?} (uniform, path, cohesive)"),
        },
        replace_parent: match args.value::<String>("--replacement").as_deref() {
            None | Some("niche") => false,
            Some("parent") => true,
            Some(x) => panic!("unknown replacement {x:?} (niche, parent)"),
        },
    };
    let args = args.0;
    let what = match set.method {
        Method::Memetic => {
            "memetic of iterated robust tabu search"
        }
        Method::Restarts => "random restarts of iterated robust tabu search",
        Method::Ma => "Merz & Freisleben memetic (swap descent with don't-look bits)",
        Method::Lns => {
            "large neighborhood search (best other arrangement of k random positions, late acceptance)"
        }
    };
    let setup = format!(
        "{what}, {} runs × {} steps, {} threads",
        set.runs, set.steps, set.jobs
    );

    if let Some(i) = args.iter().position(|a| a == "--qaplib") {
        let (data, soln) = (
            args.get(i + 1).expect("--qaplib DATA_DIR SOLN_DIR"),
            args.get(i + 2).expect("SOLN_DIR"),
        );
        let early = if set.stop_at_best {
            "; stops at the best-known value"
        } else {
            ""
        };
        let opt = SweepOptions {
            setup: format!("{setup}{early}"),
            instance_jobs: 1,
            markdown,
        };
        return sweep(Path::new(data), Path::new(soln), &opt, |inst, bk| {
            solve(inst, bk, &set)
        });
    }

    let dat = args
        .first()
        .expect("usage: qap_advanced FILE.dat [FILE.sln] | --qaplib DATA SOLN");
    let inst = Instance::read(Path::new(dat)).unwrap_or_else(|e| panic!("{e}"));
    let best_known = args
        .get(1)
        .map(|s| read_sln(Path::new(s), &inst).unwrap_or_else(|e| panic!("{e}")));
    let t = Instant::now();
    let s = solve(&inst, best_known.as_ref().map(|b| b.value), &set);
    print!(
        "{setup}\n{dat}: n = {}, found {} in {:.1} s",
        inst.n,
        s.found,
        t.elapsed().as_secs_f64()
    );
    match best_known {
        Some(b) => println!(
            ", best known {} (gap {:.2}%)",
            b.value,
            gap(s.found, b.value)
        ),
        None => println!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_method_is_deterministic_whatever_the_thread_count() {
        let inst = Instance::random(12, 7);
        for method in [Method::Memetic, Method::Restarts, Method::Lns] {
            let set = |jobs| Settings {
                method,
                steps: 6,
                runs: 3,
                first_seed: 1,
                migrate_every: 0,
                migration: Migration::Replace,
                async_migration: false,
                max_seconds: None,
                jobs,
                k: 5,
                population: 4,
                crossover: Recombine::Uniform,
                replace_parent: false,
                stop_at_best: false,
            };
            assert_eq!(
                solve(&inst, None, &set(3)).found,
                solve(&inst, None, &set(1)).found
            );
        }
    }

    #[test]
    fn ma_islands_are_deterministic() {
        let inst = Instance::random(14, 3);
        let set = Settings {
            method: Method::Ma,
            steps: 40,
            runs: 3,
            first_seed: 1,
            migrate_every: 10,
            migration: Migration::Replace,
            async_migration: false,
            max_seconds: None,
            jobs: 3,
            stop_at_best: false,
            k: 5,
            population: 6,
            crossover: Recombine::Uniform,
            replace_parent: false,
        };
        assert_eq!(ma_islands(&inst, None, &set), ma_islands(&inst, None, &set));
    }

    #[test]
    fn stops_at_the_target() {
        let inst = Instance::random(12, 7);
        let set = Settings {
            method: Method::Memetic,
            steps: 6,
            runs: 3,
            first_seed: 1,
            migrate_every: 0,
            migration: Migration::Replace,
            async_migration: false,
            max_seconds: None,
            jobs: 2,
            k: 5,
            population: 4,
            crossover: Recombine::Uniform,
            replace_parent: false,
            stop_at_best: true,
        };
        let found = solve(
            &inst,
            None,
            &Settings {
                stop_at_best: false,
                ..set
            },
        )
        .found;
        let s = solve(&inst, Some(found), &set);
        assert!(s.found <= found);
    }
}
