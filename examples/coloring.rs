//! Graph colouring on DIMACS instances: the hybrid evolutionary algorithm of
//! Galinier & Hao (1999), assembled from library pieces.
//!
//! * **TabuCol** (Hertz & de Werra; Galinier & Hao's tenure): `TabuSearch`
//!   on recolourings of conflicting vertices, with a `DynamicTabu` tenure of
//!   `random(0..10) + 0.6 · conflicting vertices` and random tie-breaking.
//! * **HEA**: `Memetic` with the GPX crossover (alternately inherit the
//!   largest remaining colour class of each parent), TabuCol as the
//!   improvement step, and a partition distance for population diversity.
//!
//! For a given k the cost is the number of conflicting edges; a k-colouring is
//! legal at cost 0. The program starts from a greedy (DSATUR-like) colouring
//! and tries k − 1 colours after each success. `--runs` independent runs share
//! the work for each k; the first legal colouring cancels the others.
//!
//! usage: coloring FILE.col [options]
//!   --k K          first k to try (default: greedy colouring − 1)
//!   --stop-at K    stop after a legal K-colouring (default: never)
//!   --generations G   children per run and per k (default 2000)
//!   --iters L      TabuCol iterations per child, in units of n (default 10; 60 with --head)
//!   --runs R       independent runs per k (default 4)
//!   --jobs J       threads (default: all cores)
//!   --population P (default 10)
//!   --replace-parent  one-to-one replacement (a child replaces its parent if better)
//!   --head         HEAD's scheme: population 2, children always replace their parent, elites every 10
//!   --bench N      time N plain TabuCol iterations at --k colours, then exit

#[path = "common/qap.rs"]
mod common; // only for its argument parser
#[path = "common/coloring.rs"]
mod graph_coloring;

use common::Args;
use graph_coloring::*;
use mets::parallel::{run_many, Cancel};
use mets::rand::SeedableRng;
use mets::*;
use std::path::Path;
use std::time::Instant;

// ---------------------------------------------------------------- solver

struct Settings {
    generations: u64,
    iters: u64,
    runs: u64,
    jobs: usize,
    population: usize,
    /// One-to-one replacement (a child replaces its first parent): with a
    /// population of 2 this is the scheme of HEAD (Moalic & Gondran 2018).
    replace_parent: bool,
    /// HEAD: population 2, each child always replaces its parent, elites
    /// reintroduced every 10 children.
    head: bool,
}

/// One HEA run for k colours from seed `seed`. Returns a legal colouring if found.
fn hea(g: &Graph, k: usize, seed: u64, cancel: &Cancel, set: &Settings) -> Option<Vec<usize>> {
    let n = g.n();
    let mut start = Coloring::new(g, k, vec![0; n]);
    start.randomize(&mut DefaultRng::seed_from_u64(seed));
    let tabucol = tabucol();
    let mut st = State::new(start, seed);
    let memetic = Memetic::new(
        tabucol,
        MaxIterations::new(set.iters * n as u64)
            .or(Target::new(0))
            .or(cancel),
        perturb_with(|s: &mut Evaluated<Coloring>, rng| s.modify(|c| c.randomize(rng))),
        Gpx,
        set.population,
    )
    .with_distance(
        |a: &Coloring, b: &Coloring| partition_distance(a, b),
        n / 10,
    );
    let mut memetic = if set.head {
        memetic.replace_parent_always().elites(10)
    } else if set.replace_parent {
        memetic.replace_parent()
    } else {
        memetic
    };
    memetic
        .runner(&mut st)
        .until(MaxIterations::new(set.generations).or(cancel))
        .stop_when(|st| st.global_best().cost() == 0)
        .run();
    let best = st.global_best();
    (best.cost() == 0).then(|| {
        cancel.cancel();
        best.color.clone()
    })
}

fn main() {
    let mut args = Args::from_env();
    let first_k: Option<usize> = args.value("--k");
    let stop_at: Option<usize> = args.value("--stop-at");
    let bench: Option<u64> = args.value("--bench");
    let mut set = Settings {
        generations: args.value("--generations").unwrap_or(2000),
        iters: 0, // set below: 10, or 60 for --head
        runs: args.value("--runs").unwrap_or(4),
        jobs: args
            .value("--jobs")
            .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get())),
        population: args.value("--population").unwrap_or(10),
        replace_parent: args.flag("--replace-parent"),
        head: args.flag("--head"),
    };
    let iters: Option<u64> = args.value("--iters");
    set.iters = iters.unwrap_or(if set.head { 60 } else { 10 });
    if set.head {
        set.population = 2;
    }
    let file = args.0.first().expect("usage: coloring FILE.col [options]");
    let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{file}: {e}"));
    let g = Graph::parse(&text).unwrap_or_else(|e| panic!("{file}: {e}"));
    let name = Path::new(file)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or(file);

    let greedy = g.dsatur();
    let greedy_k = greedy.iter().max().unwrap() + 1;
    assert!(is_legal(&g, &greedy));
    println!(
        "{name}: {} vertices, {} edges; DSATUR {greedy_k} colours; HEA: population {}, {} children × {}n TabuCol iterations, {} runs",
        g.n(),
        g.edges,
        set.population,
        set.generations,
        set.iters,
        set.runs
    );
    if let Some(iters) = bench {
        // plain TabuCol from a random colouring: iterations per second
        let k = first_k.unwrap_or(greedy_k - 1);
        let mut start = Coloring::new(&g, k, vec![0; g.n()]);
        start.randomize(&mut DefaultRng::seed_from_u64(1));
        let mut st = State::new(start, 1);
        let t = Instant::now();
        tabucol()
            .runner(&mut st)
            .until(MaxIterations::new(iters).or(Target::new(0)))
            .run();
        let secs = t.elapsed().as_secs_f64();
        println!(
            "  TabuCol, {k} colours: {} iterations in {secs:.2} s ({:.0} it/s), best {} conflicts",
            st.iteration(),
            st.iteration() as f64 / secs,
            st.global_best().cost()
        );
        return;
    }
    let t0 = Instant::now();
    let mut k = first_k.unwrap_or(greedy_k - 1);
    let mut best = greedy_k;
    while k >= 1 && stop_at.is_none_or(|s| best > s) {
        let t = Instant::now();
        let found = run_many(set.runs, set.jobs, |r, cancel| {
            hea(&g, k, r + 1, cancel, &set)
        })
        .into_iter()
        .find_map(|(_, c)| c);
        match found {
            Some(color) => {
                assert!(
                    is_legal(&g, &color) && color.iter().all(|&c| c < k),
                    "verified legal"
                );
                println!(
                    "  {k} colours: legal colouring in {:.1} s",
                    t.elapsed().as_secs_f64()
                );
                best = k;
                k -= 1;
            }
            None => {
                println!(
                    "  {k} colours: not found in {:.1} s",
                    t.elapsed().as_secs_f64()
                );
                break;
            }
        }
    }
    println!(
        "{name}: best {best} colours in {:.1} s",
        t0.elapsed().as_secs_f64()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use mets::rand::Rng;

    fn random_graph(n: usize, p: f64, seed: u64) -> Graph {
        let mut rng = DefaultRng::seed_from_u64(seed);
        let mut text = format!("p edge {n} 0\n");
        for v in 1..=n {
            for u in v + 1..=n {
                if rng.random::<f64>() < p {
                    text += &format!("e {v} {u}\n");
                }
            }
        }
        Graph::parse(&text).unwrap()
    }

    #[test]
    fn incremental_conflicts_match_a_recount() {
        let g = random_graph(40, 0.4, 1);
        let mut s = Evaluated::new(Coloring::new(&g, 5, vec![0; 40]));
        let mut rng = DefaultRng::seed_from_u64(2);
        s.modify(|c| c.randomize(&mut rng));
        for _ in 0..300 {
            let m = Recolor {
                v: rng.random_range(0..40),
                c: rng.random_range(0..5),
            };
            if m.c != s.color[m.v] {
                s.apply(&m);
            }
            let fresh = Coloring::new(&g, 5, s.color.clone());
            assert_eq!(s.gamma, fresh.gamma);
            assert_eq!((&s.low, &s.at_low), (&fresh.low, &fresh.at_low));
            let mut listed: Vec<u32> = s.list.clone();
            listed.sort_unstable();
            assert_eq!(listed, fresh.list); // fresh lists in vertex order
            assert!(listed
                .iter()
                .all(|&v| s.list[s.pos[v as usize] as usize] == v));
            assert_eq!(s.cost(), s.evaluate());
        }
        assert!(check::deltas(&mut ConflictingVertices, &s).is_empty());
    }

    #[test]
    fn gpx_keeps_k_colours_and_inherits_classes() {
        let g = random_graph(60, 0.3, 3);
        let mut rng = DefaultRng::seed_from_u64(4);
        let mut a = Coloring::new(&g, 6, vec![0; 60]);
        let mut b = a.clone();
        a.randomize(&mut rng);
        b.randomize(&mut rng);
        let mut child = a.clone();
        Gpx.cross(&a, &b, &mut child, &mut rng);
        assert!(child.color.iter().all(|&c| c < 6));
        assert_eq!(child.gamma, Coloring::new(&g, 6, child.color.clone()).gamma);
        // colour 0 is exactly a's largest class
        let largest = (0..6)
            .max_by_key(|&q| a.color.iter().filter(|&&c| c == q).count())
            .unwrap();
        for v in 0..60 {
            assert_eq!(child.color[v] == 0, a.color[v] == largest);
        }
    }

    #[test]
    fn partition_distance_ignores_colour_names() {
        let g = random_graph(30, 0.3, 5);
        let mut rng = DefaultRng::seed_from_u64(6);
        let mut a = Coloring::new(&g, 4, vec![0; 30]);
        a.randomize(&mut rng);
        let renamed = Coloring::new(&g, 4, a.color.iter().map(|&c| (c + 1) % 4).collect());
        assert_eq!(partition_distance(&a, &renamed), 0);
        let mut moved = renamed.clone();
        moved.color[0] = (moved.color[0] + 1) % 4;
        assert_eq!(partition_distance(&a, &moved), 1);
    }

    #[test]
    fn dsatur_is_legal_and_hea_colours_a_small_graph() {
        let g = random_graph(50, 0.5, 7);
        let greedy = g.dsatur();
        assert!(is_legal(&g, &greedy));
        let k = greedy.iter().max().unwrap() + 1;
        let set = Settings {
            generations: 200,
            iters: 10,
            runs: 2,
            jobs: 2,
            population: 6,
            replace_parent: false,
            head: false,
        };
        let found = run_many(set.runs, set.jobs, |r, c| hea(&g, k - 1, r + 1, c, &set))
            .into_iter()
            .find_map(|(_, c)| c)
            .expect("one colour fewer than greedy on a small random graph");
        assert!(is_legal(&g, &found) && found.iter().all(|&c| c < k - 1));
    }
}
