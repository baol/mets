//! A stronger QAP solver on the models of `examples/qap.rs`, built from the
//! library's iterated search plus two problem-specific components:
//!
//! * **Robust Tabu Search** (Taillard 1991), written as a custom `Search` in
//!   about 80 lines. The tabu memory is about *assignments*, not position
//!   pairs: a swap is tabu when it would put both facilities back into
//!   locations they left within the tenure. The tenure is re-drawn from
//!   [0.9n, 1.1n] every 2·1.1n iterations, and a long-term aspiration *forces*
//!   a move that re-creates an assignment not seen for 5n² iterations, even
//!   if it is worsening. The library's `TabuSearch` can't express forcing (a
//!   `TabuList` only says tabu or not), which is why this is its own `Search`.
//! * **Adaptive perturbation.** The iterated search perturbs the incumbent
//!   with k random swaps; k starts at n/10, grows while the incumbent does
//!   not change, and resets when it does (the idea of breakout local search).
//!
//! Plus: the symmetric fast path of the shared models (2× on most QAPLIB
//! instances), restarts run in parallel threads, and the sweep stops an
//! instance as soon as the best-known value is reached (reporting the time to
//! reach it). All stopping rules are counts, not time, so the costs found
//! depend only on the seeds, whatever the thread count.
//!
//! usage:
//!   qap_advanced FILE.dat [FILE.sln] [options]         solve one QAPLIB instance
//!   qap_advanced --qaplib DATA_DIR SOLN_DIR [options]  solve every QAPLIB instance
//! options:
//!   --restarts R   random restarts per instance (default 200)
//!   --jobs J       threads (default: all cores)
//!   --run-to-end   do not stop when the best-known value is reached
//!   --stall K      a tabu run stops after K·n iterations without improvement (default 2, min 200)
//!   --perturbations P  the iterated search stops after P perturbations without improvement (default 20)
//!   --markdown     --qaplib only: print the results as a Markdown table

#[path = "common/qap.rs"]
mod common;

use common::*;
use mets::perm::{Permutation, Swap};
use mets::rand::{Rng, SeedableRng};
use mets::*;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

// ------------------------------------------------------- robust tabu search

/// Taillard's robust tabu search for the QAP, one iteration per `step`.
///
/// `t[l][f]` is the iteration until which facility `f` may not return to
/// location `l`; it is set when `f` leaves `l`. The same matrix drives the
/// long-term aspiration: an assignment whose entry is older than
/// `aspiration` iterations (or was never made: staggered negative initial
/// values) is forced. The memory persists across the inner runs of an
/// iterated search (`start` only re-draws the tenure); build a new value per
/// restart.
struct RobustTabuSearch {
    n: usize,
    t: Vec<i64>,
    tenure: i64,
    next_redraw: u64,
    aspiration: i64,
    rng: DefaultRng,
}

impl RobustTabuSearch {
    fn new(n: usize) -> Self {
        RobustTabuSearch {
            n,
            t: (0..n * n).map(|k| -(k as i64)).collect(),
            tenure: n as i64,
            next_redraw: 0,
            aspiration: 5 * (n * n) as i64,
            rng: DefaultRng::seed_from_u64(0),
        }
    }

    fn redraw(&mut self, it: u64) {
        let n = self.n as f64;
        let (lo, hi) = ((0.9 * n) as i64, ((1.1 * n) as i64).max(1));
        self.tenure = self.rng.random_range(lo.min(hi)..=hi).max(1);
        self.next_redraw = it + 2 * hi as u64;
    }

    /// (authorized, aspired) for swapping positions i and j of `p` at iteration `it`.
    #[inline]
    fn classify(&self, p: &[usize], i: usize, j: usize, it: i64) -> (bool, bool) {
        let n = self.n;
        let (ti, tj) = (self.t[i * n + p[j]], self.t[j * n + p[i]]); // p[j] back to i, p[i] back to j
        let authorized = ti < it || tj < it;
        let aspired = ti < it - self.aspiration || tj < it - self.aspiration;
        (authorized, aspired)
    }
}

impl<S: Permutation<Cost = C> + AsRef<[usize]>, R: Rng> Search<S, R> for RobustTabuSearch {
    fn step(&mut self, st: &mut State<S, R>, obs: &mut dyn Observer<S, R>) -> Step {
        let it = st.iteration();
        if it >= self.next_redraw {
            self.redraw(it);
        }
        let cur = st.current();
        let p = cur.as_ref();
        let n = self.n;
        let best = st.best().cost();
        let (mut retained, mut min_delta, mut already_aspired) = (None, C::MAX, false);
        for i in 0..n {
            for j in i + 1..n {
                let delta = cur.swap_delta(i, j).expect("QAP models have a swap delta");
                let (authorized, long_term) = self.classify(p, i, j, it as i64);
                let aspired = long_term || cur.cost() + delta < best;
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
        self.t[i * n + p[i]] = it as i64 + self.tenure; // p[i] leaves i
        self.t[j * n + p[j]] = it as i64 + self.tenure; // p[j] leaves j
        st.commit(&Swap { i, j }, delta, obs)
    }

    fn start(&mut self, st: &mut State<S, R>) {
        self.rng = DefaultRng::seed_from_u64(st.rng().next_u64());
        self.next_redraw = 0; // re-draw the tenure at the first move
    }
}

impl AsRef<[usize]> for QapTaillard<'_> {
    fn as_ref(&self) -> &[usize] {
        &self.p
    }
}

// ------------------------------------------------- adaptive perturbation

/// k random swaps, with k growing while the incumbent is unchanged.
struct AdaptivePerturbation {
    k0: usize,
    step: usize,
    kmax: usize,
    k: usize,
    last_cost: Option<C>,
}

impl AdaptivePerturbation {
    fn new(n: usize) -> Self {
        let k0 = (n / 10).max(2);
        AdaptivePerturbation {
            k0,
            step: (n / 20).max(1),
            kmax: (n / 2).max(3),
            k: k0,
            last_cost: None,
        }
    }
}

impl<S: Permutation<Cost = C>> Perturbation<S> for AdaptivePerturbation {
    fn perturb(&mut self, s: &mut Evaluated<S>, rng: &mut dyn mets::rand::RngCore) {
        let cost = s.cost();
        self.k = match self.last_cost {
            Some(c) if c == cost => (self.k + self.step).min(self.kmax), // stuck: shake harder
            _ => self.k0, // new incumbent: gentle again
        };
        self.last_cost = Some(cost);
        perm::perturb(s, self.k, rng);
    }
}

// ------------------------------------------------------------ the solver

struct Settings {
    restarts: u64,
    jobs: usize,
    /// Stop as soon as the best-known value is reached.
    stop_at_best: bool,
    /// A tabu run stops after `stall · n` iterations without improvement (at least 200).
    stall: u64,
    /// The iterated search stops after this many perturbations without improvement.
    perturbations: u64,
}

/// One restart: an iterated robust tabu search from a random permutation.
/// Deterministic given `seed`.
fn restart(inst: &Instance, seed: u64, target: Option<C>, done: &AtomicBool, set: &Settings) -> C {
    let n = inst.n;
    let no_improve = (set.stall * n as u64).max(200);
    let stop = || stop_when(move |_: &State<QapTaillard>| done.load(Ordering::Relaxed));
    let mut its = IteratedSearch::new(
        RobustTabuSearch::new(n),
        NoImprovement::new(no_improve).or(stop()),
        AdaptivePerturbation::new(n),
        Improving,
    );
    let mut st = State::new(QapTaillard::new(inst, random_start(n, seed)), seed);
    its.runner(&mut st)
        .until(NoImprovement::new(set.perturbations).or(stop()))
        .on_global_improvement(|st| {
            if target.is_some_and(|t| st.global_best().cost() <= t) {
                done.store(true, Ordering::Relaxed);
            }
        })
        .run();
    st.global_best().cost()
}

/// All restarts, `jobs` at a time. Restart `r` always uses seed `r`, so the
/// set of costs explored does not depend on the thread count.
fn solve(inst: &Instance, best_known: Option<C>, set: &Settings) -> Solved {
    let target = if set.stop_at_best { best_known } else { None };
    let done = AtomicBool::new(false);
    let next = AtomicUsize::new(0);
    let best = Mutex::new((C::MAX, None::<f64>));
    let t0 = Instant::now();
    std::thread::scope(|scope| {
        for _ in 0..set.jobs.max(1) {
            scope.spawn(|| loop {
                let r = next.fetch_add(1, Ordering::Relaxed) as u64;
                if r >= set.restarts || done.load(Ordering::Relaxed) {
                    break;
                }
                let c = restart(inst, r + 1, target, &done, set);
                let mut b = best.lock().unwrap();
                if c < b.0 {
                    *b = (c, Some(t0.elapsed().as_secs_f64()));
                }
            });
        }
    });
    let (found, to_found) = best.into_inner().unwrap();
    Solved { found, to_found }
}

fn main() {
    let mut args = Args::from_env();
    let markdown = args.flag("--markdown");
    let run_to_end = args.flag("--run-to-end");
    let set = Settings {
        restarts: args.value("--restarts").unwrap_or(200),
        jobs: args
            .value("--jobs")
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get())),
        stop_at_best: !run_to_end,
        stall: args.value("--stall").unwrap_or(2),
        perturbations: args.value("--perturbations").unwrap_or(20),
    };
    let args = args.0;

    if let Some(i) = args.iter().position(|a| a == "--qaplib") {
        let (data, soln) = (
            args.get(i + 1).expect("--qaplib DATA_DIR SOLN_DIR"),
            args.get(i + 2).expect("SOLN_DIR"),
        );
        let early = if set.stop_at_best {
            "; an instance stops when the best-known value is reached"
        } else {
            ""
        };
        let opt = SweepOptions {
            setup: format!(
                "{} random restarts of an iterated robust tabu search (RoTS: assignment attributes, tenure in \
                 [0.9n, 1.1n], long-term aspiration; tabu runs stop after max(200, {}n) iterations without \
                 improvement, the iterated search after {} perturbations without improvement, adaptive \
                 perturbation), {} threads{early}",
                set.restarts, set.stall, set.perturbations, set.jobs
            ),
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
        "{dat}: n = {}, found {} in {:.1} s",
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
    fn tabu_forbids_both_returns_only_and_expires() {
        let n = 6;
        let mut rts = RobustTabuSearch::new(n);
        rts.aspiration = i64::MAX / 2; // no long-term aspiration here
        rts.tenure = 10;
        let mut p: Vec<usize> = (0..n).collect();
        // swap positions 0 and 1 at iteration 5
        rts.t[p[0]] = 5 + rts.tenure; // location 0, facility p[0]
        rts.t[n + p[1]] = 5 + rts.tenure;
        p.swap(0, 1);
        let (authorized, aspired) = rts.classify(&p, 0, 1, 6);
        assert!(
            !authorized && !aspired,
            "swapping back puts both facilities where they were"
        );
        let (authorized, _) = rts.classify(&p, 0, 2, 6);
        assert!(authorized, "moving only one of them back is allowed");
        let (authorized, _) = rts.classify(&p, 0, 1, 16);
        assert!(authorized, "allowed again after the tenure");
    }

    #[test]
    fn never_made_assignments_become_aspired_over_time() {
        let n = 6;
        let mut rts = RobustTabuSearch::new(n);
        rts.aspiration = 10;
        let p: Vec<usize> = (0..n).collect();
        // entry for (location 5, facility 0) is -(5n) = -30: aspired once it - 10 > -30
        let (_, aspired) = rts.classify(&p, 0, 5, 0);
        assert!(aspired);
        let p2: Vec<usize> = (0..n).rev().collect(); // entry (0, p2[1]=4) = -4, (1, p2[0]=5) = -(n+5) = -11
        let (_, aspired) = rts.classify(&p2, 0, 1, 0);
        assert!(aspired, "-11 < 0 - 10");
        let (_, aspired) = rts.classify(&p2, 0, 1, -2);
        assert!(!aspired, "-11 < -12 is false and -4 < -12 is false");
    }

    #[test]
    fn robust_tabu_search_descends_and_leaves_local_optima() {
        let inst = Instance::random(15, 3);
        let mut st = State::new(QapTaillard::new(&inst, random_start(15, 1)), 1);
        let start = st.current().cost();
        let out = RobustTabuSearch::new(15)
            .runner(&mut st)
            .until(MaxIterations::new(300))
            .run();
        assert_eq!(
            out.stop,
            Stop::Terminated,
            "tabu search keeps moving at local optima"
        );
        assert!(st.best().cost() < start);
        assert_eq!(st.best().cost(), st.best().evaluate());
    }

    #[test]
    fn adaptive_perturbation_grows_then_resets() {
        let inst = Instance::random(20, 2);
        let mut s = Evaluated::new(QapTaillard::new(&inst, (0..20).collect()));
        let mut p = AdaptivePerturbation::new(20);
        let mut rng = DefaultRng::seed_from_u64(1);
        let frozen = s.clone();
        for expected in [2, 3, 4, 5] {
            s.clone_from(&frozen); // incumbent unchanged since last time
            p.perturb(&mut s, &mut rng);
            assert_eq!(p.k, expected);
        }
        p.perturb(&mut s, &mut rng); // s now differs from the last incumbent: reset
        assert_eq!(p.k, p.k0);
    }

    #[test]
    fn solve_is_deterministic_whatever_the_thread_count() {
        let inst = Instance::random(12, 7);
        let a = solve(
            &inst,
            None,
            &Settings {
                restarts: 6,
                jobs: 3,
                stop_at_best: false,
            },
        )
        .found;
        let b = solve(
            &inst,
            None,
            &Settings {
                restarts: 6,
                jobs: 1,
                stop_at_best: false,
            },
        )
        .found;
        assert_eq!(a, b);
        // with a target, stops at or below it and reports when
        let s = solve(
            &inst,
            Some(a),
            &Settings {
                restarts: 6,
                jobs: 2,
                stop_at_best: true,
            },
        );
        assert!(s.found <= a && s.to_found.is_some());
    }
}
