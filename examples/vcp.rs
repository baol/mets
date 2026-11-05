//! Vertex colouring: port of metslib-examples/vcp (a TabuCol-like iterated
//! tabu search).
//!
//! Shows an incremental conflict table (`gamma[v][c]` = neighbours of v with
//! colour c) giving O(1) deltas, a neighborhood restricted to conflicting
//! vertices, pruning with `Visitor::bound`, and a tabu attribute computed on
//! the solution before the move ("v may not go back to its old colour").
//!
//! usage: vcp [n p k seed]   (default: random G(125, 0.5), 17 colours)

use mets::rand::{Rng, SeedableRng};
use mets::*;
use std::ops::ControlFlow;

struct Graph {
    adj: Vec<Vec<usize>>,
}

#[derive(Clone)]
struct Coloring<'g> {
    g: &'g Graph,
    k: usize,
    color: Vec<usize>,
    gamma: Vec<u32>, // n × k
}

impl<'g> Coloring<'g> {
    fn new(g: &'g Graph, k: usize, color: Vec<usize>) -> Self {
        let n = g.adj.len();
        let mut gamma = vec![0; n * k];
        for v in 0..n {
            for &u in &g.adj[v] {
                gamma[v * k + color[u]] += 1;
            }
        }
        Coloring { g, k, color, gamma }
    }
    #[inline]
    fn gamma(&self, v: usize, c: usize) -> i64 {
        self.gamma[v * self.k + c] as i64
    }
}

impl Model for Coloring<'_> {
    type Cost = i64;
    /// Number of monochromatic edges, counted from scratch.
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

#[derive(Clone, Copy, Debug)]
struct Recolor {
    v: usize,
    c: usize,
}

impl<'g> Move<Coloring<'g>> for Recolor {
    fn apply(&self, s: &mut Coloring<'g>) {
        let old = s.color[self.v];
        for &u in &s.g.adj[self.v] {
            s.gamma[u * s.k + old] -= 1;
            s.gamma[u * s.k + self.c] += 1;
        }
        s.color[self.v] = self.c;
    }
    #[inline]
    fn delta(&self, s: &Evaluated<Coloring<'g>>) -> Option<i64> {
        Some(s.gamma(self.v, self.c) - s.gamma(self.v, s.color[self.v]))
    }
}

impl<'g> TabuAttr<Coloring<'g>> for Recolor {
    type Key = usize;
    #[inline]
    fn key(&self, s: &Coloring<'g>) -> usize {
        self.v * s.k + self.c
    }
    /// After moving v away from its colour, forbid (v, old colour).
    #[inline]
    fn reverse_key(&self, before: &Coloring<'g>) -> usize {
        self.v * before.k + before.color[self.v]
    }
}
impl<'g> DenseAttr<Coloring<'g>> for Recolor {
    fn key_count(s: &Coloring<'g>) -> usize {
        s.color.len() * s.k
    }
}

/// Recolour a conflicting vertex. Vertices that cannot beat the best candidate
/// so far (their best possible delta is -gamma[v][colour]) are skipped.
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
            if own == 0 {
                continue; // not conflicting
            }
            if vis.can_skip(-own) {
                continue; // pruned: no colour of v can be selected
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let get = |i: usize, d: &str| args.get(i).cloned().unwrap_or_else(|| d.into());
    let n: usize = get(0, "125").parse().expect("n");
    let p: f64 = get(1, "0.5").parse().expect("p");
    let k: usize = get(2, "17").parse().expect("k");
    let seed: u64 = get(3, "1").parse().expect("seed");

    let mut rng = DefaultRng::seed_from_u64(seed);
    let mut adj = vec![Vec::new(); n];
    for v in 0..n {
        for u in v + 1..n {
            if rng.random::<f64>() < p {
                adj[v].push(u);
                adj[u].push(v);
            }
        }
    }
    let g = Graph { adj };
    let edges: usize = g.adj.iter().map(Vec::len).sum::<usize>() / 2;
    let start = Coloring::new(&g, k, (0..n).map(|_| rng.random_range(0..k)).collect());
    println!(
        "G({n}, {p}) with {edges} edges, {k} colours, initial conflicts {}",
        start.evaluate()
    );

    let t = std::time::Instant::now();
    let ts = TabuSearch::new(
        ConflictingVertices,
        DenseTabu::new(Tenure::Random(1..=n as u64 / 4)),
    );
    let mut its = IteratedSearch::new(
        ts,
        NoImprovement::new(5000).or(Target::new(0)),
        perturb_with(|s: &mut Evaluated<Coloring>, rng| {
            // recolour n/10 random vertices (several moves: one evaluation)
            s.modify(|c| {
                for _ in 0..c.color.len() / 10 {
                    let m = Recolor {
                        v: rng.random_range(0..c.color.len()),
                        c: rng.random_range(0..c.k),
                    };
                    m.apply(c);
                }
            })
        }),
        Improving,
    );
    let mut st = State::new(start, seed);
    its.runner(&mut st)
        .until(MaxIterations::new(200))
        .stop_when(|st| st.global_best().cost() == 0)
        .run();

    // the cached cost must match a full recount
    assert_eq!(st.best().evaluate(), st.best().cost());
    println!(
        "best conflicts {} after {} tabu iterations in {:.2}s",
        st.best().cost(),
        st.iteration(),
        t.elapsed().as_secs_f64()
    );
}
