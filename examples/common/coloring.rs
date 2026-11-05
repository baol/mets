//! Graph colouring pieces shared by `coloring` and `coloring_gpu`: DIMACS
//! graphs, the TabuCol model (conflict table, conflicting-vertex list,
//! per-vertex lower bounds), TabuCol itself, the GPX crossover and the
//! partition distance.

#![allow(dead_code)] // each example uses part of it

use mets::rand::{Rng, RngCore};
use mets::*;
use std::ops::ControlFlow;

// ---------------------------------------------------------------- graph

pub struct Graph {
    pub adj: Vec<Vec<usize>>,
    pub edges: usize,
}

impl Graph {
    /// DIMACS `.col` text: `p edge n m`, then `e u v` lines (1-based).
    pub fn parse(text: &str) -> Result<Self, String> {
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

    pub fn n(&self) -> usize {
        self.adj.len()
    }

    /// Greedy colouring, highest saturation first (DSATUR). Returns the colours.
    pub fn dsatur(&self) -> Vec<usize> {
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
/// coloured c. Also keeps, incrementally:
/// * the list of conflicting vertices, so a scan does not visit all n;
/// * for each vertex, the smallest gamma over all colours and how many
///   colours reach it: `low[v] − gamma[v][colour]` bounds the delta of every
///   recolouring of v from below, which lets the scan skip vertices.
#[derive(Clone)]
pub struct Coloring<'g> {
    pub g: &'g Graph,
    pub k: usize,
    pub color: Vec<usize>,
    pub gamma: Vec<u32>,
    pub low: Vec<u32>,
    pub at_low: Vec<u32>,
    /// Conflicting vertices, in any order, and each vertex's index in it.
    pub list: Vec<u32>,
    pub pos: Vec<u32>,
}

pub const OUT: u32 = u32::MAX;

impl<'g> Coloring<'g> {
    pub fn new(g: &'g Graph, k: usize, color: Vec<usize>) -> Self {
        let mut s = Coloring {
            g,
            k,
            color,
            gamma: Vec::new(),
            low: Vec::new(),
            at_low: Vec::new(),
            list: Vec::new(),
            pos: Vec::new(),
        };
        s.rebuild();
        s
    }

    /// Recompute every table from the colours.
    pub fn rebuild(&mut self) {
        let (n, k) = (self.g.n(), self.k);
        self.gamma.clear();
        self.gamma.resize(n * k, 0);
        for v in 0..n {
            for &u in &self.g.adj[v] {
                self.gamma[v * k + self.color[u]] += 1;
            }
        }
        self.low.resize(n, 0);
        self.at_low.resize(n, 0);
        (0..n).for_each(|v| self.rescan_low(v));
        self.list.clear();
        self.pos.clear();
        self.pos.resize(n, OUT);
        for v in 0..n {
            if self.in_conflict(v) {
                self.pos[v] = self.list.len() as u32;
                self.list.push(v as u32);
            }
        }
    }

    fn rescan_low(&mut self, v: usize) {
        let row = &self.gamma[v * self.k..][..self.k];
        let low = *row.iter().min().expect("k >= 1");
        self.low[v] = low;
        self.at_low[v] = row.iter().filter(|&&g| g == low).count() as u32;
    }

    /// Number of conflicting vertices (for the tenure).
    #[inline]
    pub fn conflicting(&self) -> usize {
        self.list.len()
    }

    #[inline]
    pub fn gamma(&self, v: usize, c: usize) -> i64 {
        self.gamma[v * self.k + c] as i64
    }

    #[inline]
    pub fn in_conflict(&self, v: usize) -> bool {
        self.gamma[v * self.k + self.color[v]] > 0
    }

    /// Put v in or out of the conflict list to match its state.
    #[inline]
    fn sync(&mut self, v: usize) {
        match (self.in_conflict(v), self.pos[v] != OUT) {
            (true, false) => {
                self.pos[v] = self.list.len() as u32;
                self.list.push(v as u32);
            }
            (false, true) => {
                let i = self.pos[v] as usize;
                let last = self.list.pop().expect("v is listed");
                if last as usize != v {
                    self.list[i] = last;
                    self.pos[last as usize] = i as u32;
                }
                self.pos[v] = OUT;
            }
            _ => {}
        }
    }

    pub fn randomize(&mut self, rng: &mut dyn RngCore) {
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
pub struct Recolor {
    pub v: usize,
    pub c: usize,
}

impl<'g> Move<Coloring<'g>> for Recolor {
    fn apply(&self, s: &mut Coloring<'g>) {
        let (v, new, k) = (self.v, self.c, s.k);
        let old = s.color[v];
        s.color[v] = new;
        for &u in &s.g.adj[v] {
            // gamma[u][old] − 1: may become the new minimum
            let g = &mut s.gamma[u * k + old];
            *g -= 1;
            if *g < s.low[u] {
                (s.low[u], s.at_low[u]) = (*g, 1);
            } else if *g == s.low[u] {
                s.at_low[u] += 1;
            }
            // gamma[u][new] + 1: may leave the minimum
            let g = &mut s.gamma[u * k + new];
            *g += 1;
            if *g - 1 == s.low[u] {
                s.at_low[u] -= 1;
                if s.at_low[u] == 0 {
                    s.rescan_low(u);
                }
            }
            if s.color[u] == old || s.color[u] == new {
                s.sync(u);
            }
        }
        s.sync(v);
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

/// Recolour a conflicting vertex. A vertex whose smallest gamma cannot beat
/// the best candidate so far is skipped, and so is each colour that cannot.
pub struct ConflictingVertices;

impl<'g> Neighborhood<Coloring<'g>> for ConflictingVertices {
    type Move = Recolor;
    fn explore<V: Visitor<Recolor, i64>>(
        &mut self,
        s: &Evaluated<Coloring<'g>>,
        vis: &mut V,
    ) -> ControlFlow<()> {
        let k = s.k;
        for &v in &s.list {
            let v = v as usize;
            let own = s.gamma(v, s.color[v]);
            if vis.can_skip(s.low[v] as i64 - own) {
                continue;
            }
            let row = &s.gamma[v * k..][..k];
            for (c, &g) in row.iter().enumerate() {
                let delta = g as i64 - own;
                if c != s.color[v] && !vis.can_skip(delta) {
                    vis.visit(Recolor { v, c }, delta)?;
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
pub struct Gpx;

impl<'g> Crossover<Coloring<'g>> for Gpx {
    fn cross(
        &mut self,
        a: &Coloring<'g>,
        b: &Coloring<'g>,
        child: &mut Coloring<'g>,
        rng: &mut dyn RngCore,
    ) {
        gpx(&a.color, &b.color, a.k, &mut child.color, rng);
        child.rebuild();
    }
}

/// GPX on plain colour vectors (colours `0..k`).
pub fn gpx(a: &[usize], b: &[usize], k: usize, child: &mut [usize], rng: &mut dyn RngCore) {
    let n = a.len();
    let parents = [a, b];
    let mut size = [vec![0usize; k], vec![0usize; k]];
    for v in 0..n {
        size[0][a[v]] += 1;
        size[1][b[v]] += 1;
    }
    let mut done = vec![false; n];
    for c in 0..k {
        let p = c % 2;
        let class = (0..k).max_by_key(|&q| size[p][q]).unwrap();
        for v in 0..n {
            if !done[v] && parents[p][v] == class {
                done[v] = true;
                child[v] = c;
                size[0][a[v]] -= 1;
                size[1][b[v]] -= 1;
            }
        }
    }
    for (v, &d) in done.iter().enumerate() {
        if !d {
            child[v] = rng.random_range(0..k);
        }
    }
}

/// Vertices to move to turn one colouring into the other, up to renaming the
/// colours (n minus a greedy maximum matching of the classes' overlaps).
pub fn partition_distance(a: &Coloring, b: &Coloring) -> usize {
    partition_distance_of(&a.color, &b.color, a.k)
}

/// [`partition_distance`] on plain colour vectors.
pub fn partition_distance_of(a: &[usize], b: &[usize], k: usize) -> usize {
    let n = a.len();
    let mut overlap = vec![0usize; k * k];
    for v in 0..n {
        overlap[a[v] * k + b[v]] += 1;
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

/// TabuCol (Hertz & de Werra; tenure as in Galinier & Hao): recolour a
/// conflicting vertex, tenure 0..10 + 0.6 × conflicting vertices, random ties.
pub fn tabucol<'g>() -> impl Search<Coloring<'g>> {
    TabuSearch::new(
        ConflictingVertices,
        DynamicTabu::new(|s: &Coloring, rng| {
            rng.random_range(0..10) + (0.6 * s.conflicting() as f64) as u64
        }),
    )
    .random_ties()
}

pub fn is_legal(g: &Graph, color: &[usize]) -> bool {
    (0..g.n()).all(|v| g.adj[v].iter().all(|&u| color[u] != color[v]))
}
