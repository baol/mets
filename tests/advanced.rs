//! Large neighborhood search, memetic search, robust tabu search, adaptive
//! perturbation and parallel runs.
mod common;

use common::*;
use mets::parallel::{run_many, Cancel};
use mets::perm::{
    adaptive_perturbation, rearrange, shuffle, BestRearrangement, Permutation, RobustTabuSearch,
    SwapNeighborhood, UniformCrossover,
};
use mets::rand::SeedableRng;
use mets::*;

fn is_permutation(p: &[usize]) -> bool {
    let mut seen = vec![false; p.len()];
    p.iter()
        .all(|&e| e < p.len() && !std::mem::replace(&mut seen[e], true))
}

fn rng(seed: u64) -> DefaultRng {
    DefaultRng::seed_from_u64(seed)
}

#[test]
fn replace_current_records_the_best_and_notifies() {
    let mut st = State::new(Sort(reversed(8)), 1);
    let better = Evaluated::new(Sort((0..8).collect()));
    let mut events = Vec::new();
    let mut obs = on_event(|e: Event<i64>, _: &State<Sort>| events.push(e));
    assert_eq!(st.replace_current(&better, &mut obs), Step::Improved);
    assert_eq!(st.best().cost(), 0);
    let delta = -Sort(reversed(8)).evaluate();
    assert_eq!(
        events,
        [
            Event::Moved { delta },
            Event::Improved,
            Event::GlobalImproved
        ]
    );
}

#[test]
fn rearrange_reaches_any_target() {
    let mut s = Sort(reversed(9));
    let mut r = rng(3);
    for _ in 0..20 {
        let mut target: Vec<usize> = (0..9).collect();
        for t in (1..9).rev() {
            target.swap(t, mets::rand::Rng::random_range(&mut r, 0..=t));
        }
        rearrange(&mut s, &target);
        assert_eq!(s.as_slice(), &target[..]);
    }
}

#[test]
fn best_rearrangement_of_all_positions_is_the_optimum() {
    // k = n: the rebuild enumerates every permutation, so it must find the optimum (0)
    for start in [reversed(6), vec![3, 5, 0, 1, 4, 2], vec![1, 0, 3, 2, 5, 4]] {
        let mut s = Evaluated::new(Sort(start));
        BestRearrangement::new(6).rebuild(&mut s, &mut rng(1));
        assert_eq!(s.cost(), 0);
        assert_eq!(s.cost(), s.evaluate());
    }
}

#[test]
fn best_rearrangement_moves_to_the_best_other_arrangement() {
    let mut s = Evaluated::new(Sort(reversed(30)));
    let mut op = BestRearrangement::new(5);
    let mut r = rng(7);
    let mut worsened = false;
    for _ in 0..300 {
        let before = s.as_slice().to_vec();
        op.rebuild(&mut s, &mut r);
        assert_ne!(
            s.as_slice(),
            &before[..],
            "the current arrangement is never kept"
        );
        let changed = before
            .iter()
            .zip(s.as_slice())
            .filter(|(a, b)| a != b)
            .count();
        assert!(changed <= 5, "only the k chosen positions change");
        assert!(is_permutation(s.as_slice()));
        assert_eq!(s.cost(), s.evaluate());
        worsened |= s.cost() > Sort(before).evaluate();
    }
    assert!(
        worsened,
        "at a local optimum of the neighborhood, the best other arrangement is worse"
    );
}

#[test]
fn large_neighborhood_descent_never_worsens() {
    let mut st = State::new(Sort(reversed(30)), 4);
    let mut last = st.current().cost();
    LargeNeighborhoodSearch::new(BestRearrangement::new(4), Improving)
        .runner(&mut st)
        .until(MaxIterations::new(500))
        .on_move(|st| {
            assert!(st.current().cost() < last);
            last = st.current().cost();
        })
        .run();
}

#[test]
fn large_neighborhood_search_solves_the_toy_problem() {
    for acceptance_is_sa in [false, true] {
        let mut st = State::new(Sort(reversed(25)), 2);
        let op = BestRearrangement::new(6);
        let out = if acceptance_is_sa {
            LargeNeighborhoodSearch::new(op, SimulatedAnnealing::new(5.0, 0.01, 0.9, 20))
                .runner(&mut st)
                .until(Target::new(0).or(MaxIterations::new(5_000)))
                .run()
        } else {
            LargeNeighborhoodSearch::new(op, LateAcceptance::new(20))
                .runner(&mut st)
                .until(Target::new(0).or(MaxIterations::new(5_000)))
                .run()
        };
        assert_eq!(st.best().cost(), 0, "{out:?}");
    }
}

#[test]
fn large_neighborhood_search_works_without_delta_formulas() {
    let mut st = State::new(SortFull(reversed(12)), 2);
    LargeNeighborhoodSearch::new(BestRearrangement::new(4), LateAcceptance::new(20))
        .runner(&mut st)
        .until(Target::new(0).or(MaxIterations::new(2_000)))
        .run();
    assert_eq!(st.best().cost(), 0);
}

#[test]
fn rebuild_with_closures() {
    let mut st = State::new(Sort(reversed(10)), 1);
    let op = rebuild_with(|s: &mut Evaluated<Sort>, rng| perm::perturb(s, 2, rng));
    let out = LargeNeighborhoodSearch::new(op, Improving)
        .runner(&mut st)
        .until(MaxIterations::new(500))
        .run();
    assert_eq!(out.iterations, 500);
    assert!(st.best().cost() < Sort(reversed(10)).evaluate());
}

#[test]
fn uniform_crossover_keeps_common_assignments() {
    let mut r = rng(5);
    for _ in 0..50 {
        let mut a = Evaluated::new(Sort((0..20).collect()));
        let mut b = Evaluated::new(Sort((0..20).collect()));
        perm::perturb(&mut a, 6, &mut r);
        perm::perturb(&mut b, 6, &mut r);
        let mut child = (*a).clone();
        UniformCrossover.cross(&*a, &*b, &mut child, &mut r);
        let (pa, pb, pc) = (a.as_slice(), b.as_slice(), child.as_slice());
        assert!(is_permutation(pc));
        for i in 0..20 {
            if pa[i] == pb[i] {
                assert_eq!(pc[i], pa[i], "common assignment lost at {i}");
            }
        }
        // most positions take an element from one of the parents
        let inherited = (0..20)
            .filter(|&i| pc[i] == pa[i] || pc[i] == pb[i])
            .count();
        assert!(inherited >= 14, "{inherited}");
    }
}

#[test]
fn memetic_search_with_crossover_solves_the_toy_problem() {
    let mut st = State::new(Sort(reversed(30)), 3);
    let mut memetic = Memetic::new(
        LocalSearch::first_improvement(SwapNeighborhood),
        Unlimited,
        perturb_with(|s: &mut Evaluated<Sort>, rng| shuffle(s, rng)),
        UniformCrossover,
        6,
    )
    .renew_after(10);
    let mut restarts = 0;
    memetic
        .runner(&mut st)
        .until(Target::new(0).or(MaxIterations::new(300)))
        .on_event(|e, _| restarts += (e == Event::Restart) as u32)
        .run();
    assert_eq!(st.best().cost(), 0);
    assert!(restarts > 0);
    let pop = memetic.population();
    assert!(pop.len() <= 6);
    for (i, m) in pop.iter().enumerate() {
        assert_eq!(m.cost(), m.evaluate());
        assert!(
            pop[..i].iter().all(|o| o.cost() != m.cost()),
            "duplicate cost in the population"
        );
    }
}

#[test]
fn robust_tabu_search_keeps_moving_and_solves_the_toy_problem() {
    let run = |seed| {
        let mut st = State::new(Sort(reversed(20)), seed);
        let out = RobustTabuSearch::new()
            .runner(&mut st)
            .until(Target::new(0).or(MaxIterations::new(3_000)))
            .run();
        (st.best().cost(), out.iterations)
    };
    let (cost, iters) = run(1);
    assert_eq!(cost, 0);
    assert!(iters < 3_000);
    assert_eq!(run(1), (cost, iters), "deterministic");
}

#[test]
fn robust_tabu_memory_persists_across_runs() {
    let mut st = State::new(Sort(reversed(15)), 1);
    let mut rts = RobustTabuSearch::new();
    rts.runner(&mut st).until(MaxIterations::new(50)).run();
    // a second run in the same state continues with the same memory (no reset)
    let out = rts.runner(&mut st).until(MaxIterations::new(50)).run();
    assert_eq!(out.stop, Stop::Terminated);
}

#[test]
fn adaptive_perturbation_grows_while_stuck_and_resets() {
    let mut p = adaptive_perturbation::<Sort>(40);
    let mut r = rng(1);
    let frozen = Evaluated::new(Sort(reversed(40)));
    let mut s = frozen.clone();
    let mut seen = Vec::new();
    for _ in 0..4 {
        s.clone_from(&frozen); // incumbent unchanged
        p.perturb(&mut s, &mut r);
        seen.push(p.strength());
    }
    assert_eq!(seen, [4, 6, 8, 10]);
    p.perturb(&mut s, &mut r); // s differs from the last incumbent
    assert_eq!(p.strength(), 4);
    // the generic form takes any strength-dependent operation
    let mut calls = Vec::new();
    let mut g = Adaptive::new(
        1,
        1,
        3,
        |_: &mut Evaluated<Sort>, k, _: &mut dyn mets::rand::RngCore| calls.push(k),
    );
    for _ in 0..5 {
        g.perturb(&mut frozen.clone(), &mut r);
    }
    assert_eq!(calls, [1, 2, 3, 3, 3]);
}

#[test]
fn run_many_is_ordered_deterministic_and_cancellable() {
    let solve = |run: u64, cancel: &Cancel| {
        let mut st = State::new(Sort(reversed(12)), run);
        StochasticSearch::new(SwapNeighborhood, LateAcceptance::new(10))
            .runner(&mut st)
            .until(MaxIterations::new(2_000).or(cancel))
            .run();
        st.best().cost()
    };
    let one = run_many(8, 1, solve);
    let four = run_many(8, 4, solve);
    assert_eq!(one, four);
    assert_eq!(
        one.iter().map(|(r, _)| *r).collect::<Vec<_>>(),
        (0..8).collect::<Vec<_>>()
    );

    let cancelled = run_many(100, 2, |run, cancel| {
        if run == 3 {
            cancel.cancel();
        }
        run
    });
    assert!(cancelled.len() < 100);
    assert!(cancelled.iter().any(|&(r, _)| r == 3));
}

#[test]
fn path_relinking_moves_a_fraction_of_the_way() {
    let mut r = rng(11);
    for fraction in [0.0, 0.3, 0.5, 1.0] {
        let a = Sort(reversed(30));
        let b = Sort((0..30).collect());
        let mut child = a.clone();
        perm::PathRelinking { fraction }.cross(&a, &b, &mut child, &mut r);
        assert!(is_permutation(child.as_slice()));
        let from_a = perm::hamming(&a, &child);
        let to_b = perm::hamming(&child, &b);
        assert!(
            from_a + to_b >= perm::hamming(&a, &b),
            "triangle inequality"
        );
        if fraction == 0.0 {
            assert_eq!(from_a, 0);
        }
        if fraction == 1.0 {
            assert_eq!(to_b, 0);
        }
        if fraction == 0.5 {
            assert!(to_b > 0 && from_a > 0 && to_b < 30);
        }
    }
}

#[test]
fn memetic_with_one_to_one_replacement_and_path_relinking() {
    let mut st = State::new(Sort(reversed(30)), 5);
    let mut memetic = Memetic::new(
        LocalSearch::first_improvement(SwapNeighborhood),
        Unlimited,
        perturb_with(|s: &mut Evaluated<Sort>, rng| shuffle(s, rng)),
        perm::PathRelinking { fraction: 0.5 },
        6,
    )
    .replace_parent();
    // a fixed number of steps, so the population fills and children replace parents
    memetic.runner(&mut st).until(MaxIterations::new(60)).run();
    assert_eq!(st.best().cost(), 0);
    assert_eq!(
        memetic.population().len(),
        6,
        "fills even when members share a cost"
    );
}

#[test]
fn memetic_with_distance_keeps_members_apart() {
    let mut st = State::new(Sort(reversed(30)), 6);
    let mut memetic = Memetic::new(
        LocalSearch::first_improvement(SwapNeighborhood),
        Unlimited,
        perturb_with(|s: &mut Evaluated<Sort>, rng| shuffle(s, rng)),
        UniformCrossover,
        6,
    )
    .with_distance(|a: &Sort, b: &Sort| perm::hamming(a, b), 3);
    memetic.runner(&mut st).until(MaxIterations::new(200)).run();
    assert!(st.best().cost() < Sort(reversed(30)).evaluate());
    let pop = memetic.population();
    let close = (0..pop.len())
        .flat_map(|i| (i + 1..pop.len()).map(move |j| (i, j)))
        .filter(|&(i, j)| perm::hamming(&*pop[i], &*pop[j]) < 3)
        .count();
    assert!(close <= 1, "{close} pairs closer than the minimum distance");
}

#[test]
fn dynamic_tabu_tenure_is_computed_per_move() {
    let mut st = State::new(Sort(reversed(12)), 1);
    let mut calls = 0u64;
    {
        let tabu = DynamicTabu::new(|_: &Sort, _rng| {
            calls += 1;
            3
        });
        let mut ts = TabuSearch::new(SwapNeighborhood, tabu);
        ts.runner(&mut st).until(MaxIterations::new(50)).run();
    }
    assert_eq!(calls, 50);
}

#[test]
fn head_style_memetic_keeps_moving_and_reinjects_elites() {
    // population 2, children always replace their parent, elites every 3 steps
    let mut st = State::new(Sort(reversed(40)), 8);
    let mut memetic = Memetic::new(
        LocalSearch::first_improvement(SwapNeighborhood),
        MaxIterations::new(5), // short improvement: children are often worse than parents
        perturb_with(|s: &mut Evaluated<Sort>, rng| shuffle(s, rng)),
        UniformCrossover,
        2,
    )
    .replace_parent_always()
    .elites(3);
    let mut costs = Vec::new();
    memetic
        .runner(&mut st)
        .until(MaxIterations::new(60))
        .on_event(|e, st| {
            if e == Event::Restart {
                costs.push(st.current().cost())
            }
        })
        .run();
    assert_eq!(memetic.population().len(), 2);
    for m in memetic.population() {
        assert_eq!(m.cost(), m.evaluate());
    }
    // the global best is kept even though members are replaced unconditionally
    assert!(st.best().cost() <= memetic.population().iter().map(|m| m.cost()).min().unwrap());
    assert!(costs.len() == 60);
}

#[test]
fn random_ties_are_reproducible_and_explore_plateaus() {
    // Sort has many equal swap deltas: random tie-breaking changes the walk,
    // but a given seed always gives the same one.
    let walk = |seed| {
        let mut st = State::new(Sort(reversed(16)), seed);
        let mut trace = Vec::new();
        TabuSearch::new(SwapNeighborhood, DenseTabu::new(Tenure::Fixed(5)))
            .random_ties()
            .runner(&mut st)
            .until(MaxIterations::new(40))
            .on_move(|st| trace.push(st.current().cost()))
            .run();
        (trace, st.current().as_slice().to_vec())
    };
    assert_eq!(walk(1), walk(1));
    assert!(
        (2..6).any(|s| walk(s).1 != walk(1).1),
        "different seeds take different tied moves"
    );
    // the best move is still chosen: the first move is the steepest one either way
    let mut plain = State::new(Sort(reversed(16)), 1);
    TabuSearch::new(SwapNeighborhood, DenseTabu::new(Tenure::Fixed(5)))
        .runner(&mut plain)
        .until(MaxIterations::new(1))
        .run();
    assert_eq!(walk(3).0[0], plain.current().cost());
}
