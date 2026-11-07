//! Capacitated vehicle routing (PLAN-problems.md P7): the iterated tabu
//! search of Cordeau & Maischberger (2012), sequential version, on the
//! `mets` pieces: `TabuSearch` over `vrp::InterRoute` relocations with the
//! (customer, route) tabu attribute, an attribute aspiration, frequency
//! diversification through `TabuSearch::with_score`, and the per-iteration
//! bookkeeping (penalty weights, aspiration table, frequencies, best
//! feasible solution, route refinement every 200 iterations) in an
//! `Adjusting` hook. Insertion is best insertion between consecutive
//! customers (the paper's time-window variant) or, with `--insertion geni`,
//! GENI types I and II (`common/geni.rs`); removal is by reconnection or,
//! with `--removal unstring`, GENI's unstringing (types I and II).
//!
//! Per phase: δ, ζ uniform in [0, 1] and the tenure uniform in
//! [0, ⌈√(n·m)⌉] are drawn; the tabu search runs until the best feasible
//! solution s* has not improved for μ = √((η − λ)·π) iterations; the final
//! solution is kept with probability 1 − (λ/η)², else the search returns to
//! s*; then a cluster of π + 1 customers (π uniform in [0, ⌈√n⌉]) is removed
//! and reinserted. The run stops after η tabu iterations in total.
//!
//! usage:
//!   cvrp FILE.vrp [FILE.sol] [--eta N] [--seed S] [--vehicles M | --no-spare] [--max-seconds T]
//!        [--insertion best|geni] [--removal reconnect|unstring] [--geni-p P]
//!        (defaults best, reconnect; P = 5 for both GENI insertion and unstringing)
//!        [--phase-log]  one line per phase: feasibility, excesses, α, β
//!       solve; the best feasible solution is saved as NAME.sol (CVRPLIB format,
//!       or the first free NAME.2.sol, …); a .sol gives the gap to its cost and,
//!       by default, the fleet size (its routes, plus one spare vehicle)
//!   cvrp --check VRP_DIR SOL_DIR [--real | --round]
//!       read every FILE.vrp and its FILE.sol, build the routes, and compare
//!       the distance and feasibility with the stated cost
//!
//! `scripts/verify_vrp.py` checks solutions with none of this code.

#[path = "common/qap.rs"]
mod common; // for its argument parser and fresh_path
#[path = "common/vrp.rs"]
mod vrp;

use common::{fresh_path, Args};
use mets::rand::{Rng, SeedableRng};
use mets::*;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::time::Instant;
use vrp::*;

/// The long-term memory shared by the aspiration, the score and the hook.
struct Memory {
    /// Best f(s) over visited solutions with customer c in route k (c·m + k).
    best_with: Vec<f64>,
    /// How often each attribute (c, k) was added.
    freq: Vec<u32>,
    /// Tabu iterations so far, over all phases (λ).
    lambda: u64,
    /// This phase's ζ and δ.
    zeta: f64,
    delta: f64,
    /// Best feasible solution (distance, routes), and when it last improved.
    best: (f64, Vec<Vec<usize>>),
    last_improvement: u64,
    /// This phase's diagnostics (`--phase-log`): feasible iterations, the
    /// smallest load and duration excess seen, and the largest β.
    feasible_its: u64,
    min_load: f64,
    min_duration: f64,
    max_beta: f64,
}

/// Aspiration by attribute: a tabu move is allowed if f(s′) is below the
/// best cost seen with the attribute it adds.
struct AttributeAspiration(Rc<RefCell<Memory>>);

impl<'a> Aspiration<Routes<'a>, Relocate> for AttributeAspiration {
    fn allows<R>(&self, st: &State<Routes<'a>, R>, m: &Relocate, new_cost: f64) -> bool {
        new_cost < self.0.borrow().best_with[m.c * st.current().m + m.to] - 1e-9
    }
}

/// One iterated tabu search run; returns the best feasible (distance, routes).
#[allow(clippy::too_many_arguments)]
fn its(
    inst: &Instance,
    m: usize,
    eta: u64,
    seed: u64,
    max_seconds: f64,
    reference: Option<f64>,
    geni_p: usize,
    unstring_p: usize,
    phase_log: bool,
) -> (f64, Vec<Vec<usize>>) {
    let n = inst.n;
    let t0 = Instant::now();
    let mut rng = DefaultRng::seed_from_u64(seed);
    let mut s = sweep(inst, m, &mut rng);
    let mem = Rc::new(RefCell::new(Memory {
        best_with: vec![f64::INFINITY; n * m],
        freq: vec![0; n * m],
        lambda: 0,
        zeta: 0.0,
        delta: 0.0,
        best: if s.is_feasible() {
            (s.distance(), s.routes())
        } else {
            (f64::INFINITY, Vec::new())
        },
        last_improvement: 0,
        feasible_its: 0,
        min_load: f64::INFINITY,
        min_duration: f64::INFINITY,
        max_beta: 0.0,
    }));
    let tau_max = ((n * m) as f64).sqrt().ceil() as u64;
    let pi_max = (n as f64).sqrt().ceil() as usize;
    let mut pi = pi_max; // the first phase's μ
    let mut phase = 0u64;
    loop {
        let lambda = mem.borrow().lambda;
        if lambda >= eta || t0.elapsed().as_secs_f64() > max_seconds {
            break;
        }
        phase += 1;
        let mu = (((eta - lambda) as f64) * pi.max(1) as f64).sqrt() as u64;
        {
            let mut mm = mem.borrow_mut();
            mm.zeta = rng.random();
            mm.delta = rng.random();
            mm.last_improvement = lambda;
            (mm.feasible_its, mm.min_load, mm.min_duration, mm.max_beta) =
                (0, f64::INFINITY, f64::INFINITY, 0.0);
        }
        // the tabu phase
        let mut st = State::new(s.clone(), rng.random());
        let (score_mem, hook_mem, stop_mem) = (mem.clone(), mem.clone(), mem.clone());
        let tabu = TabuSearch::new(
            InterRoute::with_unstring(geni_p, unstring_p),
            DenseTabu::new(Tenure::Random(0..=tau_max)),
        )
        .with_aspiration(AttributeAspiration(mem.clone()))
        .with_score(move |s: &Evaluated<Routes>, mv: &Relocate, delta: f64| {
            if delta < 0.0 {
                return delta; // improving moves are not penalised
            }
            let mm = score_mem.borrow();
            let r = mm.freq[mv.c * s.m + mv.to] as f64;
            let g = (s.cost() + delta) * (1.0 + mm.zeta * r / mm.lambda.max(1) as f64);
            g - s.cost()
        });
        let has_duration = inst.max_duration.is_some();
        let mut search = adjusting(tabu, move |st: &mut State<Routes>, _: Step| {
            let mut mm = hook_mem.borrow_mut();
            mm.lambda += 1;
            let cur = st.current();
            let (km, f) = (cur.m, cur.cost());
            if let Some((c, _, to)) = cur.last {
                mm.freq[c * km + to] += 1;
            }
            for c in 1..cur.inst.n {
                let a = c * km + cur.route[c] as usize;
                if f < mm.best_with[a] {
                    mm.best_with[a] = f;
                }
            }
            let record = |mm: &mut Memory, r: &Routes| {
                if r.is_feasible() && r.distance() < mm.best.0 - 1e-9 {
                    mm.best = (r.distance(), r.routes());
                    mm.last_improvement = mm.lambda;
                }
            };
            record(&mut mm, cur);
            let (xl, xd) = (cur.excess_load(), cur.excess_duration());
            mm.feasible_its += (xl <= 1e-9 && xd <= 1e-9) as u64;
            mm.min_load = mm.min_load.min(xl);
            mm.min_duration = mm.min_duration.min(xd);
            mm.max_beta = mm.max_beta.max(cur.penalties.duration);
            // self-adjusting penalty weights, then the cached cost follows
            let (q, t) = (cur.excess_load() > 1e-9, cur.excess_duration() > 1e-9);
            let up = 1.0 + mm.delta;
            let refine = mm.lambda.is_multiple_of(200);
            st.modify_current(|e, _| {
                e.modify(|r| {
                    let w = &mut r.penalties;
                    w.capacity =
                        (if q { w.capacity * up } else { w.capacity / up }).clamp(1e-6, 1e9);
                    if has_duration {
                        w.duration =
                            (if t { w.duration * up } else { w.duration / up }).clamp(1e-6, 1e9);
                    }
                    if refine {
                        r.refine();
                    }
                })
            });
            if refine {
                record(&mut mm, st.current());
            }
        });
        search
            .runner(&mut st)
            .until(stop_when(move |_: &State<Routes>| {
                let mm = stop_mem.borrow();
                mm.lambda >= eta || mm.lambda - mm.last_improvement >= mu
            }))
            .run();
        // acceptance: keep the final solution with probability 1 − (λ/η)², else s*
        let (lambda, best) = {
            let mm = mem.borrow();
            (mm.lambda, mm.best.clone())
        };
        let keep = rng.random::<f64>() < 1.0 - (lambda as f64 / eta as f64).powi(2);
        s = if keep || best.1.is_empty() {
            st.current().clone().into_inner()
        } else {
            Routes::with_vehicles(inst, best.1, m, st.current().penalties)
        };
        if phase_log {
            let mm = mem.borrow();
            let e = st.current();
            println!(
                "phase={phase} lambda={} mu={mu} pi={pi} feasible_its={} min_excess_load={:.3} min_excess_duration={:.3} end_excess_load={:.3} end_excess_duration={:.3} alpha={:.4e} beta={:.4e} max_beta={:.4e} distance={:.2} best={:.2} kept={keep}",
                mm.lambda,
                mm.feasible_its,
                mm.min_load,
                mm.min_duration,
                e.excess_load(),
                e.excess_duration(),
                e.penalties.capacity,
                e.penalties.duration,
                mm.max_beta,
                e.distance(),
                best.0,
            );
        }
        pi = rng.random_range(0..=pi_max);
        cluster_perturbation(&mut s, pi, &mut rng);
        if phase.is_multiple_of(50) {
            let g = reference.map_or(String::new(), |r| {
                format!(" (gap {:.2}%)", (best.0 - r) / r * 100.0)
            });
            println!(
                "  phase {phase:5}, λ = {lambda:8}: best {:.2}{g}, {:.1} s",
                best.0,
                t0.elapsed().as_secs_f64()
            );
        }
    }
    let best = mem.borrow().best.clone();
    best
}

fn main() {
    let mut args = Args::from_env();
    let rounding = if args.flag("--real") {
        Some(Rounding::Real)
    } else if args.flag("--round") {
        Some(Rounding::Nearest)
    } else {
        None
    };
    if args.flag("--check") {
        return check(&args, rounding);
    }
    let eta: u64 = args.value("--eta").unwrap_or(100_000);
    let seed: u64 = args.value("--seed").unwrap_or(1);
    let max_seconds: f64 = args.value("--max-seconds").unwrap_or(f64::INFINITY);
    let vehicles: Option<usize> = args.value("--vehicles");
    let p: usize = args.value("--geni-p").unwrap_or(5);
    let phase_log = args.flag("--phase-log");
    let geni_p = match args.value::<String>("--insertion").as_deref() {
        None | Some("best") => 0,
        Some("geni") => p,
        Some(x) => panic!("--insertion best|geni, not {x:?}"),
    };
    let unstring_p = match args.value::<String>("--removal").as_deref() {
        None | Some("reconnect") => 0,
        Some("unstring") => p,
        Some(x) => panic!("--removal reconnect|unstring, not {x:?}"),
    };
    // default fleet: the reference routes (or the load bound) plus one spare
    let spare = if args.flag("--no-spare") { 0 } else { 1 };
    let file = args
        .0
        .first()
        .expect("usage: cvrp FILE.vrp [FILE.sol] [options]")
        .clone();
    let inst = Instance::read(Path::new(&file), rounding).unwrap_or_else(|e| panic!("{e}"));
    let reference = args
        .0
        .get(1)
        .map(|f| read_sol(Path::new(f)).unwrap_or_else(|e| panic!("{e}")));
    let total: f64 = inst.demand.iter().sum();
    let m = vehicles.unwrap_or_else(|| {
        let lower = (total / inst.capacity).ceil() as usize;
        reference.as_ref().map_or(lower, |r| r.0.len()).max(lower) + spare
    });
    let ref_cost = reference.as_ref().and_then(|r| r.1);
    let insertion = if geni_p > 0 {
        format!("GENI insertion, p = {geni_p}")
    } else {
        "best insertion".to_string()
    };
    let removal = if unstring_p > 0 {
        format!("unstringing removal, p = {unstring_p}")
    } else {
        "removal by reconnection".to_string()
    };
    println!(
        "{}: {} customers, capacity {}, {}route limit, {m} vehicles; {insertion}, {removal}; η = {eta}, seed {seed}; build {}",
        inst.name,
        inst.n - 1,
        inst.capacity,
        inst.max_duration.map_or("no ".to_string(), |d| format!("{d} ")),
        env!("METS_GIT_HASH")
    );
    let t0 = Instant::now();
    let (dist, routes) = its(
        &inst,
        m,
        eta,
        seed,
        max_seconds,
        ref_cost,
        geni_p,
        unstring_p,
        phase_log,
    );
    let secs = t0.elapsed().as_secs_f64();
    if routes.is_empty() {
        println!("{}: no feasible solution in {secs:.1} s", inst.name);
        return;
    }
    // recount from the lists before saving
    let s = Routes::with_vehicles(&inst, routes.clone(), m, Penalties::default());
    assert!(
        s.is_feasible() && (s.distance() - dist).abs() < 1e-6,
        "verified"
    );
    let out = fresh_path(&inst.name, "sol");
    let mut text = String::new();
    for (k, r) in routes.iter().enumerate() {
        let r: Vec<String> = r.iter().map(|c| c.to_string()).collect();
        text += &format!(
            "Route #{}: {}
",
            k + 1,
            r.join(" ")
        );
    }
    let cost = if inst.rounding == Some(Rounding::Nearest) || inst.rounding.is_none() {
        format!("{}", dist.round() as i64)
    } else {
        format!("{dist:.2}")
    };
    text += &format!(
        "Cost {cost}
"
    );
    std::fs::write(&out, text).unwrap_or_else(|e| panic!("{out}: {e}"));
    let g = ref_cost.map_or(String::new(), |r| {
        format!(", reference {r} (gap {:.2}%)", (dist - r) / r * 100.0)
    });
    println!(
        "{}: best {dist:.2} ({} routes){g} in {secs:.1} s, saved to {out}",
        inst.name,
        routes.len()
    );
}

/// `--check VRP_DIR SOL_DIR`: rebuild every `.sol` and compare.
fn check(args: &Args, rounding: Option<Rounding>) {
    let (vrp_dir, sol_dir) = (
        Path::new(args.0.first().expect("VRP_DIR")).to_path_buf(),
        Path::new(args.0.get(1).expect("SOL_DIR")).to_path_buf(),
    );
    let mut files: Vec<_> = std::fs::read_dir(&vrp_dir)
        .expect("VRP_DIR")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "vrp"))
        .collect();
    files.sort();
    let (mut ok, mut all) = (0, 0);
    for f in files {
        all += 1;
        let sol = sol_dir.join(f.with_extension("sol").file_name().expect("name"));
        let inst = match Instance::read(&f, rounding) {
            Ok(i) => i,
            Err(e) => {
                println!("FAIL {e}");
                continue;
            }
        };
        let (routes, stated, precision) = read_sol(&sol).unwrap_or_else(|e| panic!("{e}"));
        let s = Routes::new(&inst, routes, Penalties::default());
        let (dist, eval) = (s.distance(), s.evaluate());
        let tol = if inst.rounding == Some(Rounding::Real) {
            // equal at the precision the cost is written with, but never stricter
            // than 1e-3 (six-decimal files carry the writer's float error)
            precision.max(1e-3)
        } else {
            1e-6
        };
        let good = s.is_feasible()
            && stated.is_some_and(|c| (dist - c).abs() <= tol)
            && (eval - dist).abs() <= 1e-9;
        ok += good as usize;
        println!(
            "{} {}: {} routes, distance {dist:.4}, stated {stated:?}, feasible {}",
            if good { "OK" } else { "FAIL" },
            inst.name,
            s.routes().len(),
            s.is_feasible()
        );
    }
    println!("{ok} / {all} OK");
}

#[cfg(test)]
mod tests {
    use super::*;
    use mets::rand::{Rng, SeedableRng};

    /// A random Euclidean instance with tight capacity and a route length limit.
    fn instance(n: usize, seed: u64) -> Instance {
        let mut rng = DefaultRng::seed_from_u64(seed);
        let xy: Vec<(f64, f64)> = (0..n)
            .map(|_| (rng.random::<f64>() * 100.0, rng.random::<f64>() * 100.0))
            .collect();
        let mut dist = vec![0.0; n * n];
        for a in 0..n {
            for b in 0..n {
                dist[a * n + b] =
                    ((xy[a].0 - xy[b].0).powi(2) + (xy[a].1 - xy[b].1).powi(2)).sqrt();
            }
        }
        let demand = (0..n)
            .map(|i| {
                if i == 0 {
                    0.0
                } else {
                    rng.random_range(1..10) as f64
                }
            })
            .collect();
        let service = (0..n).map(|i| if i == 0 { 0.0 } else { 2.0 }).collect();
        Instance {
            name: "test".into(),
            n,
            capacity: 25.0,
            max_duration: Some(180.0),
            demand,
            service,
            windows: None,
            dist,
            rounding: Some(Rounding::Real),
            xy: Some(xy),
        }
    }

    #[test]
    fn relocation_deltas_match_full_evaluations() {
        let inst = instance(30, 1);
        let mut rng = DefaultRng::seed_from_u64(2);
        let pen = Penalties {
            capacity: 3.0,
            duration: 2.0,
        };
        let mut s = Evaluated::new(Routes::with_vehicles(
            &inst,
            sweep(&inst, 6, &mut rng).routes(),
            7,
            pen,
        ));
        for _ in 0..2000 {
            let c = rng.random_range(1..inst.n);
            let to = rng.random_range(0..s.m);
            let nodes: Vec<usize> = std::iter::once(inst.n + to)
                .chain(s.customers(to))
                .filter(|&a| a != c)
                .collect();
            let after = nodes[rng.random_range(0..nodes.len())];
            let m = Relocate {
                c,
                to,
                after,
                geni: None,
                us: None,
            };
            let expected = m.delta_of(&s);
            let before = s.cost();
            s.apply(&m);
            let fresh = s.evaluate();
            assert!(
                (fresh - (before + expected)).abs() < 1e-6,
                "delta {expected} vs {}",
                fresh - before
            );
            assert!((s.cost() - fresh).abs() < 1e-6);
            let mut tables = (*s).clone();
            tables.rebuild();
            for k in 0..s.m {
                assert!((tables.length[k] - s.length[k]).abs() < 1e-6);
                assert!((tables.load[k] - s.load[k]).abs() < 1e-9);
                assert_eq!(tables.size[k], s.size[k]);
            }
        }
        assert!(check::deltas(&mut InterRoute::new(0), &s).is_empty());
    }

    /// GENI relocations into routes of 0 (empty), 1, 2, 3, 4, 5 and 14
    /// customers: every valid GENI move of every type, both orientations,
    /// at every position, for a customer from another route; the move's
    /// delta must match a full evaluation and the route tables a rebuild.
    #[test]
    fn geni_relocations_into_routes_of_every_small_size() {
        let inst = instance(30, 5);
        let pen = Penalties {
            capacity: 3.0,
            duration: 2.0,
        };
        let mut routes: Vec<Vec<usize>> = Vec::new();
        let mut next = 1;
        for size in [1, 2, 3, 4, 5, 14] {
            routes.push((next..next + size).collect());
            next += size;
        }
        assert_eq!(next, inst.n);
        let s = Evaluated::new(Routes::with_vehicles(&inst, routes, 7, pen));
        let mut tried = [0usize; 3];
        for to in 0..s.m {
            let tour = s.tour(to);
            // a customer from the first other non-empty route, and one from the long route
            for c in [if to == 0 { 2 } else { 1 }, 29] {
                if s.route[c] as usize == to {
                    continue;
                }
                for mut g in vrp::geni::moves(tour.len()) {
                    g.added =
                        vrp::geni::added(&vrp::geni::oriented(&tour, g.rev), c, &g, &|a, b| {
                            s.d(a, b)
                        });
                    tried[g.kind as usize] += 1;
                    let m = Relocate {
                        c,
                        to,
                        after: c,
                        geni: Some(g),
                        us: None,
                    };
                    let expected = m.delta_of(&s);
                    let mut t = s.clone();
                    t.apply(&m);
                    let fresh = t.evaluate();
                    assert!(
                        (fresh - (s.cost() + expected)).abs() < 1e-6,
                        "route of {} customers, {g:?}: delta {expected} vs {}",
                        tour.len() - 1,
                        fresh - s.cost()
                    );
                    assert_eq!(t.route[c] as usize, to);
                    assert_eq!(t.customers(to).len(), tour.len());
                    let mut tables = (*t).clone();
                    tables.rebuild();
                    for k in 0..t.m {
                        assert!((tables.length[k] - t.length[k]).abs() < 1e-6);
                        assert!((tables.load[k] - t.load[k]).abs() < 1e-9);
                        assert_eq!(tables.size[k], t.size[k]);
                    }
                }
            }
        }
        assert!(tried.iter().all(|&x| x > 0), "every kind tried: {tried:?}");
    }

    /// Unstring relocations out of routes of 1, 2, 3, 4, 5 and 14 customers:
    /// every valid unstringing of every type, both orientations, for every
    /// customer of the route, with plain and GENI insertion into another
    /// route; deltas against full evaluations, tables against a rebuild.
    #[test]
    fn unstring_relocations_from_routes_of_every_small_size() {
        let inst = instance(30, 8);
        let pen = Penalties {
            capacity: 3.0,
            duration: 2.0,
        };
        let mut routes: Vec<Vec<usize>> = Vec::new();
        let mut next = 1;
        for size in [1, 2, 3, 4, 5, 14] {
            routes.push((next..next + size).collect());
            next += size;
        }
        let s = Evaluated::new(Routes::with_vehicles(&inst, routes, 7, pen));
        let d = |a: usize, b: usize| s.d(a, b);
        let mut tried = [0usize; 3];
        for from in 0..6 {
            let tour = s.tour(from);
            for &c in &tour[1..] {
                for to in [(from + 1) % 6, 6] {
                    let target = s.tour(to);
                    let g = vrp::geni::best(&target, c, 5, &d);
                    for mut u in vrp::geni::unstring_moves(tour.len()) {
                        let o = vrp::geni::oriented(&tour, u.rev);
                        let p = o.iter().position(|&x| x == c).unwrap();
                        u.delta = vrp::geni::unstring_delta(&o, p, &u, &d);
                        tried[u.kind as usize] += 1;
                        let (after, geni) = if g.kind == 0 {
                            (target[g.i as usize], None)
                        } else {
                            (c, Some(g))
                        };
                        let m = Relocate {
                            c,
                            to,
                            after,
                            geni,
                            us: Some(u),
                        };
                        let expected = m.delta_of(&s);
                        let mut t = s.clone();
                        t.apply(&m);
                        let fresh = t.evaluate();
                        assert!(
                            (fresh - (s.cost() + expected)).abs() < 1e-6,
                            "from a route of {} customers, {u:?}: delta {expected} vs {}",
                            tour.len() - 1,
                            fresh - s.cost()
                        );
                        assert_eq!(t.route[c] as usize, to);
                        assert_eq!(t.customers(from).len(), tour.len() - 2);
                        let mut tables = (*t).clone();
                        tables.rebuild();
                        for k in 0..t.m {
                            assert!((tables.length[k] - t.length[k]).abs() < 1e-6);
                            assert!((tables.load[k] - t.load[k]).abs() < 1e-9);
                            assert_eq!(tables.size[k], t.size[k]);
                        }
                    }
                }
            }
        }
        assert!(tried.iter().all(|&x| x > 0), "every kind tried: {tried:?}");
    }

    /// The GENI neighbourhood's deltas agree with full evaluations along a
    /// search, as the plain one's do.
    #[test]
    fn geni_neighbourhood_deltas_along_a_search() {
        // GENI insertion alone, unstringing alone, and both
        for (gp, up) in [(5, 0), (0, 5), (5, 5), (2, 3)] {
            let inst = instance(30, 6);
            let mut rng = DefaultRng::seed_from_u64(7);
            let pen = Penalties {
                capacity: 3.0,
                duration: 2.0,
            };
            let mut s = Evaluated::new(Routes::with_vehicles(
                &inst,
                sweep(&inst, 6, &mut rng).routes(),
                7,
                pen,
            ));
            let mut hood = InterRoute::with_unstring(gp, up);
            let saved = s.clone();
            for step in 0..60 {
                if step == 40 {
                    // return to an earlier solution, as the search does to s*
                    s = saved.clone();
                }
                // the cached neighbourhood visits exactly what a fresh one does
                let visits = |h: &mut InterRoute, s: &Evaluated<Routes>| {
                    let mut out = Vec::new();
                    let _ = h.explore(s, &mut |m: Relocate, d: f64| {
                        out.push((m.c, m.to, m.after, m.geni, m.us, d));
                        std::ops::ControlFlow::<()>::Continue(())
                    });
                    out
                };
                let all = visits(&mut hood, &s);
                assert_eq!(
                    all,
                    visits(&mut InterRoute::with_unstring(gp, up), &s),
                    "({gp}, {up}) step {step}"
                );
                assert!(
                    check::deltas(&mut hood, &s).is_empty(),
                    "({gp}, {up}) step {step}"
                );
                // apply a random visited move
                let (c, to, after, geni, us, _) = all[rng.random_range(0..all.len())];
                s.apply(&Relocate {
                    c,
                    to,
                    after,
                    geni,
                    us,
                });
                assert!((s.cost() - s.evaluate()).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn refinement_never_lengthens_and_the_sweep_is_feasible_until_the_last_route() {
        let inst = instance(40, 3);
        let mut rng = DefaultRng::seed_from_u64(4);
        let mut s = sweep(&inst, 12, &mut rng);
        let used = (0..s.m).filter(|&k| s.size[k] > 0).count();
        for k in 0..used.saturating_sub(1) {
            assert!(s.load_excess(s.load[k]) == 0.0, "route {k} over capacity");
        }
        assert_eq!(s.routes().iter().map(Vec::len).sum::<usize>(), inst.n - 1);
        let before = s.length.clone();
        s.refine();
        for k in 0..s.m {
            assert!(s.length[k] <= before[k] + 1e-9);
        }
        let mut fresh = s.clone();
        fresh.rebuild();
        assert!((fresh.distance() - s.distance()).abs() < 1e-6);
    }
}
