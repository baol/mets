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

#[path = "common/qap.rs"]
mod common; // only for its argument parser

use common::Args;
use mets::parallel::{run_many, Cancel};
use mets::rand::{Rng, RngCore, SeedableRng};
use mets::*;
use std::ops::ControlFlow;
use std::path::Path;
use std::time::Instant;

// ---------------------------------------------------------------- graph

struct Graph {
    adj: Vec<Vec<usize>>,
    edges: usize,
}

impl Graph {
    /// DIMACS `.col` text: `p edge n m`, then `e u v` lines (1-based).
    fn parse(text: &str) -> Result<Self, String> {
        let mut adj: Vec<Vec<usize>> = Vec::new();
        for line in text.lines() {
            let mut w = line.split_whitespace();
            match w.next() {
                Some("p") => {
                    let n: usize = w.nth(1).and_then(|t| t.parse().ok()).ok_or("bad p line")?;
                    adj = vec![Vec::new(); n];
                }
                Some("e") => {
                    let mut num = || {
                        w.next()
                            .and_then(|t| t.parse::<usize>().ok())
                            .ok_or("bad e line")
                    };
                    let (u, v) = (num()? - 1, num()? - 1);
                    if u != v && !adj[u].contains(&v) {
                        adj[u].push(v);
                        adj[v].push(u);
                    }
                }
                _ => {}
            }
        }
        let edges = adj.iter().map(Vec::len).sum::<usize>() / 2;
        Ok(Graph { adj, edges })
    }

    fn n(&self) -> usize {
        self.adj.len()
    }

    /// Greedy colouring, highest saturation first (DSATUR). Returns the colours.
    fn dsatur(&self) -> Vec<usize> {
        let n = self.n();
        let mut color = vec![usize::MAX; n];
        let mut used: Vec<Vec<bool>> = vec![Vec::new(); n]; // colours seen by each vertex
        let mut sat = vec![0usize; n];
        for _ in 0..n {
            let v = (0..n)
                .filter(|&v| color[v] == usize::MAX)
                .max_by_key(|&v| (sat[v], self.adj[v].len()))
                .unwrap();
            let c = (0..)
                .find(|&c| !used[v].get(c).copied().unwrap_or(false))
                .unwrap();
            color[v] = c;
            for &u in &self.adj[v] {
                if used[u].len() <= c {
                    used[u].resize(c + 1, false);
                }
                if !used[u][c] {
                    used[u][c] = true;
                    sat[u] += 1;
                }
            }
        }
        color
    }
}

// ---------------------------------------------------------------- model

/// A k-colouring with its conflict table: `gamma[v·k + c]` = neighbours of v
/// coloured c. Tracks the number of conflicting vertices for the tenure.
#[derive(Clone)]
struct Coloring<'g> {
    g: &'g Graph,
    k: usize,
    color: Vec<usize>,
    gamma: Vec<u32>,
    conflicting: usize,
}

impl<'g> Coloring<'g> {
    fn new(g: &'g Graph, k: usize, color: Vec<usize>) -> Self {
        let mut s = Coloring {
            g,
            k,
            color,
            gamma: Vec::new(),
            conflicting: 0,
        };
        s.rebuild();
        s
    }

    /// Recompute the conflict table from the colours.
    fn rebuild(&mut self) {
        let (n, k) = (self.g.n(), self.k);
        self.gamma.clear();
        self.gamma.resize(n * k, 0);
        for v in 0..n {
            for &u in &self.g.adj[v] {
                self.gamma[v * k + self.color[u]] += 1;
            }
        }
        self.conflicting = (0..n).filter(|&v| self.in_conflict(v)).count();
    }

    #[inline]
    fn gamma(&self, v: usize, c: usize) -> i64 {
        self.gamma[v * self.k + c] as i64
    }

    #[inline]
    fn in_conflict(&self, v: usize) -> bool {
        self.gamma[v * self.k + self.color[v]] > 0
    }

    fn randomize(&mut self, rng: &mut dyn RngCore) {
        for c in &mut self.color {
            *c = rng.random_range(0..self.k);
        }
        self.rebuild();
    }
}

impl Model for Coloring<'_> {
    type Cost = i64;
    /// Conflicting edges, from scratch.
    fn evaluate(&self) -> i64 {
        let adj = &self.g.adj;
        (0..adj.len())
            .map(|v| {
                adj[v]
                    .iter()
                    .filter(|&&u| u > v && self.color[u] == self.color[v])
                    .count() as i64
            })
            .sum()
    }
}

/// Give vertex v colour c.
#[derive(Clone, Copy, Debug)]
struct Recolor {
    v: usize,
    c: usize,
}

impl<'g> Move<Coloring<'g>> for Recolor {
    fn apply(&self, s: &mut Coloring<'g>) {
        let (v, new, k) = (self.v, self.c, s.k);
        let old = s.color[v];
        let was = s.in_conflict(v) as isize;
        let mut change = 0isize;
        for &u in &s.g.adj[v] {
            let before = s.in_conflict(u) as isize;
            s.gamma[u * k + old] -= 1;
            s.gamma[u * k + new] += 1;
            change += s.in_conflict(u) as isize - before;
        }
        s.color[v] = new;
        change += s.in_conflict(v) as isize - was;
        s.conflicting = (s.conflicting as isize + change) as usize;
    }
    #[inline]
    fn delta(&self, s: &Evaluated<Coloring<'g>>) -> Option<i64> {
        Some(s.gamma(self.v, self.c) - s.gamma(self.v, s.color[self.v]))
    }
}

/// Forbid giving v back the colour it had (key: v·k + colour).
impl<'g> TabuAttr<Coloring<'g>> for Recolor {
    type Key = usize;
    #[inline]
    fn key(&self, s: &Coloring<'g>) -> usize {
        self.v * s.k + self.c
    }
    #[inline]
    fn reverse_key(&self, before: &Coloring<'g>) -> usize {
        self.v * before.k + before.color[self.v]
    }
}
impl<'g> DenseAttr<Coloring<'g>> for Recolor {
    fn key_count(s: &Coloring<'g>) -> usize {
        s.g.n() * s.k
    }
}

/// Recolour a conflicting vertex; vertices that cannot beat the best candidate
/// so far are skipped (their best possible delta is −gamma[v][colour]).
struct ConflictingVertices;

impl<'g> Neighborhood<Coloring<'g>> for ConflictingVertices {
    type Move = Recolor;
    fn explore<V: Visitor<Recolor, i64>>(
        &mut self,
        s: &Evaluated<Coloring<'g>>,
        vis: &mut V,
    ) -> ControlFlow<()> {
        for v in 0..s.color.len() {
            let own = s.gamma(v, s.color[v]);
            if own == 0 || vis.bound().is_some_and(|b| -own >= b) {
                continue;
            }
            for c in 0..s.k {
                if c != s.color[v] {
                    vis.visit(Recolor { v, c }, s.gamma(v, c) - own)?;
                }
            }
        }
        ControlFlow::Continue(())
    }
}

// ------------------------------------------------------------- crossover

/// Galinier & Hao's greedy partition crossover: for colour 0..k, alternately
/// from each parent, take its largest class among the vertices not yet
/// coloured; the vertices left get random colours.
struct Gpx;

impl<'g> Crossover<Coloring<'g>> for Gpx {
    fn cross(
        &mut self,
        a: &Coloring<'g>,
        b: &Coloring<'g>,
        child: &mut Coloring<'g>,
        rng: &mut dyn RngCore,
    ) {
        let (n, k) = (a.color.len(), a.k);
        let parents = [&a.color, &b.color];
        let mut size = [vec![0usize; k], vec![0usize; k]];
        for v in 0..n {
            size[0][a.color[v]] += 1;
            size[1][b.color[v]] += 1;
        }
        let mut done = vec![false; n];
        for c in 0..k {
            let p = c % 2;
            let class = (0..k).max_by_key(|&q| size[p][q]).unwrap();
            for v in 0..n {
                if !done[v] && parents[p][v] == class {
                    done[v] = true;
                    child.color[v] = c;
                    size[0][a.color[v]] -= 1;
                    size[1][b.color[v]] -= 1;
                }
            }
        }
        for (v, &d) in done.iter().enumerate() {
            if !d {
                child.color[v] = rng.random_range(0..k);
            }
        }
        child.rebuild();
    }
}

/// Vertices to move to turn one colouring into the other, up to renaming the
/// colours (n minus a greedy maximum matching of the classes' overlaps).
fn partition_distance(a: &Coloring, b: &Coloring) -> usize {
    let (n, k) = (a.color.len(), a.k);
    let mut overlap = vec![0usize; k * k];
    for v in 0..n {
        overlap[a.color[v] * k + b.color[v]] += 1;
    }
    let mut pairs: Vec<(usize, usize)> = (0..k * k)
        .map(|i| (overlap[i], i))
        .filter(|&(o, _)| o > 0)
        .collect();
    pairs.sort_unstable_by(|x, y| y.cmp(x));
    let (mut used_a, mut used_b, mut matched) = (vec![false; k], vec![false; k], 0);
    for (o, i) in pairs {
        let (ca, cb) = (i / k, i % k);
        if !used_a[ca] && !used_b[cb] {
            used_a[ca] = true;
            used_b[cb] = true;
            matched += o;
        }
    }
    n - matched
}

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
    let tabucol = TabuSearch::new(
        ConflictingVertices,
        DynamicTabu::new(|s: &Coloring, rng| {
            rng.random_range(0..10) + (0.6 * s.conflicting as f64) as u64
        }),
    )
    .random_ties();
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

fn is_legal(g: &Graph, color: &[usize]) -> bool {
    (0..g.n()).all(|v| g.adj[v].iter().all(|&u| color[u] != color[v]))
}

fn main() {
    let mut args = Args::from_env();
    let first_k: Option<usize> = args.value("--k");
    let stop_at: Option<usize> = args.value("--stop-at");
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
            assert_eq!(s.conflicting, fresh.conflicting);
            assert_eq!(s.gamma, fresh.gamma);
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
