mod common;

use common::*;
use mets::perm::{perturb, shuffle, Permutation, SwapNeighborhood};
use mets::*;

fn assert_local_optimum<S: Permutation>(s: &Evaluated<S>) {
    for (m, d) in collect_moves(&mut SwapNeighborhood, s) {
        assert!(d >= num_traits::Zero::zero(), "{m:?} improves by {d:?}");
    }
}

#[test]
fn local_search_stops_at_a_local_optimum() {
    for first in [false, true] {
        let mut st = State::new(Sort(reversed(30)), 1);
        let mut ls = LocalSearch {
            neighborhood: SwapNeighborhood,
            first_improvement: first,
        };
        let out = ls.runner(&mut st).run();
        assert_eq!(out.stop, Stop::Stuck);
        assert_eq!(out.iterations, st.iteration());
        assert_local_optimum(st.current());
        assert_eq!(st.current().cost(), st.current().evaluate());
    }
}

#[test]
fn tabu_search_reaches_the_optimum_and_reports_improvements() {
    let mut st = State::new(Sort(reversed(30)), 1);
    let mut ts = TabuSearch::new(SwapNeighborhood, DenseTabu::new(Tenure::Fixed(10)));
    let (mut moves, mut improved, mut global) = (0, 0, 0);
    let out = ts
        .runner(&mut st)
        .until(Target::new(0).or(MaxIterations::new(10_000)))
        .on_move(|_| moves += 1)
        .on_improvement(|_| improved += 1)
        .on_global_improvement(|_| global += 1)
        .run();
    assert_eq!(out.stop, Stop::Terminated);
    assert_eq!(st.best().cost(), 0);
    assert_eq!(moves, out.iterations);
    assert!(improved > 0);
    assert_eq!(
        improved, global,
        "no scope open: every improvement is global"
    );
}

#[test]
fn tabu_on_a_sampled_neighborhood_with_a_hash_list() {
    let mut st = State::new(Sort(reversed(30)), 1);
    let mut ts = TabuSearch::new(
        Sampled::new(SwapNeighborhood, 50),
        RecencyTabu::new(Tenure::Random(5..=15)),
    );
    ts.runner(&mut st).until(MaxIterations::new(5_000)).run();
    assert!(st.best().cost() < Sort(reversed(30)).evaluate() / 4);
    // the sampled neighborhood is seeded from the state: same seed, same result
    let mut st2 = State::new(Sort(reversed(30)), 1);
    let mut ts2 = TabuSearch::new(
        Sampled::new(SwapNeighborhood, 50),
        RecencyTabu::new(Tenure::Random(5..=15)),
    );
    ts2.runner(&mut st2).until(MaxIterations::new(5_000)).run();
    assert_eq!(st.best().0, st2.best().0);
}

#[test]
fn every_acceptance_rule_improves() {
    fn go<A: Acceptance<i64>>(a: A) -> i64 {
        let mut st = State::new(Sort(reversed(30)), 3);
        StochasticSearch::new(SwapNeighborhood, a)
            .runner(&mut st)
            .until(MaxIterations::new(200_000))
            .run();
        assert_eq!(st.current().cost(), st.current().evaluate());
        st.best().cost()
    }
    let initial = Sort(reversed(30)).evaluate();
    for best in [
        go(Improving),
        go(NonWorsening),
        go(SimulatedAnnealing::new(50.0, 0.01, 0.95, 1000)),
        go(LateAcceptance::new(50)),
        go(RecordToRecord::new(5.0)),
        go(ThresholdAccepting::new(20.0, 0.9999)),
        go(GreatDeluge::new(0.01)),
    ] {
        assert!(best < initial / 4, "{best} vs {initial}");
    }
}

#[test]
fn simulated_annealing_ends_exhausted_when_frozen() {
    let mut st = State::new(Sort(reversed(20)), 1);
    let mut sa = StochasticSearch::new(
        SwapNeighborhood,
        SimulatedAnnealing::new(10.0, 1.0, 0.5, 10),
    );
    let out = sa.runner(&mut st).run(); // no termination criterion
    assert_eq!(out.stop, Stop::Exhausted);
    assert!(sa.acceptance.temperature() < 1.0);
}

#[test]
fn resume_continues_without_restarting() {
    let mut st = State::new(Sort(reversed(20)), 1);
    let mut sa = StochasticSearch::new(
        SwapNeighborhood,
        SimulatedAnnealing::new(100.0, 1e-9, 0.5, 100),
    );
    sa.runner(&mut st).until(MaxIterations::new(500)).run();
    let t = sa.acceptance.temperature();
    sa.runner(&mut st).until(MaxIterations::new(500)).resume();
    assert!(sa.acceptance.temperature() < t, "resume must not reheat");
    sa.runner(&mut st).until(MaxIterations::new(500)).run();
    assert_eq!(sa.acceptance.temperature(), t, "run starts afresh");
    assert_eq!(st.iteration(), 1500);
}

#[test]
fn algorithms_can_be_chosen_at_runtime() {
    let algorithms: Vec<Box<dyn Search<Sort>>> = vec![
        Box::new(LocalSearch::best_improvement(SwapNeighborhood)),
        Box::new(TabuSearch::new(
            SwapNeighborhood,
            DenseTabu::new(Tenure::Fixed(7)),
        )),
        Box::new(StochasticSearch::new(
            SwapNeighborhood,
            LateAcceptance::new(20),
        )),
    ];
    for mut alg in algorithms {
        let mut st = State::new(Sort(reversed(15)), 2);
        alg.runner(&mut st)
            .until(MaxIterations::new(20_000))
            .on_move(|_| {})
            .run();
        assert!(st.best().cost() < Sort(reversed(15)).evaluate());
    }
}

#[test]
fn nested_searches_keep_scoped_and_global_best_apart() {
    let mut st = State::new(Sort(reversed(30)), 4);
    let ils = IteratedSearch::new(
        LocalSearch::first_improvement(SwapNeighborhood),
        Unlimited,
        perturb_with(|s: &mut Evaluated<Sort>, rng| perturb(s, 3, rng)),
        Improving,
    );
    let mut restarts = IteratedSearch::new(
        ils,
        NoImprovement::new(5),
        perturb_with(|s: &mut Evaluated<Sort>, rng| shuffle(s, rng)),
        Always,
    );
    let (mut improved, mut global, mut restarts_seen) = (0, 0, 0);
    let mut lowest = i64::MAX;
    restarts
        .runner(&mut st)
        .until(MaxIterations::new(4))
        .on_event(|e, st| {
            // the global best is never worse than the scoped one
            assert!(st.global_best().cost() <= st.best().cost());
            lowest = lowest.min(st.current().cost());
            match e {
                Event::Improved => improved += 1,
                Event::GlobalImproved => global += 1,
                Event::Restart => restarts_seen += 1,
                _ => {}
            }
        })
        .run();
    assert!(global >= 1 && global <= improved);
    assert!(restarts_seen > 4, "inner restarts are reported too");
    // back at top level, best() is the global best, and it is the lowest cost seen
    assert_eq!(st.best().cost(), st.global_best().cost());
    assert_eq!(st.best().cost(), lowest);
    assert_eq!(st.best().cost(), st.best().evaluate());
}

#[test]
fn scopes_return_the_run_best_and_restore_the_outer_best() {
    let mut st = State::new(Sort(reversed(10)), 1);
    let outer = st.best().cost();
    let mut scope = st.scope();
    let inner = scope.state();
    LocalSearch::best_improvement(SwapNeighborhood)
        .runner(inner)
        .run();
    let run_cost = inner.best().cost();
    assert!(run_cost < outer);
    let (run_best, improved) = scope.finish();
    assert!(improved);
    assert_eq!(run_best.cost(), run_cost);
    assert_eq!(st.best().cost(), run_cost);

    // a scope dropped without finish() also merges back
    {
        let mut scope = st.scope();
        scope.state().modify_current(shuffle);
    }
    assert_eq!(st.best().cost(), run_cost);
}

#[test]
fn termination_criteria_and_counters() {
    let mut st = State::new(Sort(reversed(20)), 1);
    let mut ls = LocalSearch::best_improvement(SwapNeighborhood);
    assert_eq!(
        ls.runner(&mut st)
            .until(MaxIterations::new(3))
            .run()
            .iterations,
        3
    );
    assert_eq!(
        ls.runner(&mut st)
            .until(MaxIterations::new(3))
            .run()
            .iterations,
        3
    );
    assert_eq!(st.iteration(), 6);
    let out = ls
        .runner(&mut st)
        .stop_when(|st| st.best().cost() < 40)
        .run();
    assert!(out.stop == Stop::Terminated && st.best().cost() < 40);
    let mut both = MaxIterations::new(2).and(MaxIterations::new(5));
    assert_eq!(ls.runner(&mut st).until(&mut both).run().iterations, 5);
}

#[test]
fn log_improvements_writes_one_line_per_global_improvement() {
    let mut out = Vec::new();
    let mut st = State::new(Sort(reversed(10)), 1);
    LocalSearch::best_improvement(SwapNeighborhood)
        .runner(&mut st)
        .observe(log_improvements(&mut out))
        .run();
    let text = String::from_utf8(out).unwrap();
    assert_eq!(text.lines().count() as u64, st.iteration());
    assert!(text
        .lines()
        .last()
        .unwrap()
        .ends_with(&format!("\t{}", st.best().cost())));
}
