//! Graph colouring pieces shared by `coloring` and `coloring_gpu`: DIMACS
//! graphs, the TabuCol model (conflict table, conflicting-vertex list,
//! per-vertex lower bounds), TabuCol itself, the GPX crossover, the
//! partition distance, and the PartialCol space with Variable Space Search
//! (alternating TabuCol and PartialCol).

#![allow(dead_code)] // each example uses part of it

use mets::rand::{Rng, RngCore, SeedableRng};
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

// ------------------------------------------------------ partial colourings

/// No colour, in a [`PartialColoring`].
pub const UNCOLORED: usize = usize::MAX;

/// A legal partial k-colouring (PartialCol, Blöchliger & Zufferey 2008):
/// some vertices are uncoloured, coloured vertices never conflict, and the
/// cost is the number of uncoloured vertices (0: a legal k-colouring).
/// `gamma[v·k + c]` counts the *coloured* neighbours of v with colour c, so
/// giving c to v uncolours gamma[v][c] vertices. Like [`Coloring`], keeps the
/// uncoloured vertices in a list and, per vertex, the smallest gamma (a lower
/// bound on the deltas of its moves).
#[derive(Clone)]
pub struct PartialColoring<'g> {
    pub g: &'g Graph,
    pub k: usize,
    /// `UNCOLORED` or `0..k`.
    pub color: Vec<usize>,
    pub gamma: Vec<u32>,
    pub low: Vec<u32>,
    pub at_low: Vec<u32>,
    /// Uncoloured vertices, in any order, and each vertex's index in it.
    pub list: Vec<u32>,
    pub pos: Vec<u32>,
}

impl<'g> PartialColoring<'g> {
    /// `color` must be legal on its coloured vertices.
    pub fn new(g: &'g Graph, k: usize, color: Vec<usize>) -> Self {
        let mut s = PartialColoring {
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
        debug_assert!(s.is_legal_partial(), "coloured vertices conflict");
        s
    }

    /// Recompute every table from the colours.
    pub fn rebuild(&mut self) {
        let (n, k) = (self.g.n(), self.k);
        self.gamma.clear();
        self.gamma.resize(n * k, 0);
        for v in 0..n {
            for &u in &self.g.adj[v] {
                if self.color[u] != UNCOLORED {
                    self.gamma[v * k + self.color[u]] += 1;
                }
            }
        }
        self.low.resize(n, 0);
        self.at_low.resize(n, 0);
        (0..n).for_each(|v| self.rescan_low(v));
        self.list.clear();
        self.pos.clear();
        self.pos.resize(n, OUT);
        for v in 0..n {
            if self.color[v] == UNCOLORED {
                self.list_add(v);
            }
        }
    }

    /// From a colouring with conflicts: uncolour, one at a time, a vertex with
    /// the most conflicting neighbours (ties at random) until none conflict.
    /// A colouring with one conflicting edge gives one uncoloured vertex.
    pub fn from_coloring(c: &Coloring<'g>, rng: &mut dyn RngCore) -> Self {
        let g = c.g;
        let mut color = c.color.clone();
        let mut conf: Vec<usize> = (0..g.n())
            .map(|v| g.adj[v].iter().filter(|&&u| color[u] == color[v]).count())
            .collect();
        loop {
            let most = *conf.iter().max().expect("n >= 1");
            if most == 0 {
                break;
            }
            let ties: Vec<usize> = (0..g.n()).filter(|&v| conf[v] == most).collect();
            let v = ties[rng.random_range(0..ties.len())];
            for &u in &g.adj[v] {
                if color[u] == color[v] {
                    conf[u] -= 1;
                }
            }
            (color[v], conf[v]) = (UNCOLORED, 0);
        }
        PartialColoring::new(g, c.k, color)
    }

    /// Back to a full colouring: each uncoloured vertex, in random order, takes
    /// a colour used by the fewest of its neighbours (ties at random).
    pub fn complete(&self, rng: &mut dyn RngCore) -> Coloring<'g> {
        let (g, k) = (self.g, self.k);
        let mut color = self.color.clone();
        let mut order: Vec<usize> = self.list.iter().map(|&v| v as usize).collect();
        for i in (1..order.len()).rev() {
            order.swap(i, rng.random_range(0..=i));
        }
        let mut count = vec![0usize; k];
        for v in order {
            count.iter_mut().for_each(|x| *x = 0);
            for &u in &g.adj[v] {
                if color[u] != UNCOLORED {
                    count[color[u]] += 1;
                }
            }
            let fewest = *count.iter().min().expect("k >= 1");
            let ties: Vec<usize> = (0..k).filter(|&c| count[c] == fewest).collect();
            color[v] = ties[rng.random_range(0..ties.len())];
        }
        Coloring::new(g, k, color)
    }

    /// Coloured vertices do not conflict (the invariant).
    pub fn is_legal_partial(&self) -> bool {
        (0..self.g.n())
            .all(|v| self.color[v] == UNCOLORED || self.gamma[v * self.k + self.color[v]] == 0)
    }

    /// Number of uncoloured vertices (for the tenure).
    #[inline]
    pub fn uncolored(&self) -> usize {
        self.list.len()
    }

    #[inline]
    pub fn gamma(&self, v: usize, c: usize) -> i64 {
        self.gamma[v * self.k + c] as i64
    }

    fn rescan_low(&mut self, v: usize) {
        let row = &self.gamma[v * self.k..][..self.k];
        let low = *row.iter().min().expect("k >= 1");
        self.low[v] = low;
        self.at_low[v] = row.iter().filter(|&&g| g == low).count() as u32;
    }

    #[inline]
    fn list_add(&mut self, v: usize) {
        self.pos[v] = self.list.len() as u32;
        self.list.push(v as u32);
    }

    #[inline]
    fn list_remove(&mut self, v: usize) {
        let i = self.pos[v] as usize;
        let last = self.list.pop().expect("v is listed");
        if last as usize != v {
            self.list[i] = last;
            self.pos[last as usize] = i as u32;
        }
        self.pos[v] = OUT;
    }

    /// gamma[w][c] − 1: may become the new minimum.
    #[inline]
    fn dec(&mut self, w: usize, c: usize) {
        let g = &mut self.gamma[w * self.k + c];
        *g -= 1;
        if *g < self.low[w] {
            (self.low[w], self.at_low[w]) = (*g, 1);
        } else if *g == self.low[w] {
            self.at_low[w] += 1;
        }
    }

    /// gamma[w][c] + 1: may leave the minimum.
    #[inline]
    fn inc(&mut self, w: usize, c: usize) {
        let g = &mut self.gamma[w * self.k + c];
        *g += 1;
        if *g - 1 == self.low[w] {
            self.at_low[w] -= 1;
            if self.at_low[w] == 0 {
                self.rescan_low(w);
            }
        }
    }
}

impl Model for PartialColoring<'_> {
    type Cost = i64;
    /// Uncoloured vertices, from scratch.
    fn evaluate(&self) -> i64 {
        self.color.iter().filter(|&&c| c == UNCOLORED).count() as i64
    }
}

/// Colour the uncoloured vertex v with c, uncolouring its neighbours coloured
/// c: the delta is gamma[v][c] − 1.
#[derive(Clone, Copy, Debug)]
pub struct Assign {
    pub v: usize,
    pub c: usize,
}

impl<'g> Move<PartialColoring<'g>> for Assign {
    fn apply(&self, s: &mut PartialColoring<'g>) {
        let (v, c, g) = (self.v, self.c, s.g);
        debug_assert_eq!(s.color[v], UNCOLORED);
        for &u in &g.adj[v] {
            if s.color[u] == c {
                s.color[u] = UNCOLORED;
                s.list_add(u);
                for &w in &g.adj[u] {
                    s.dec(w, c);
                }
            }
        }
        s.color[v] = c;
        s.list_remove(v);
        for &w in &g.adj[v] {
            s.inc(w, c);
        }
    }
    #[inline]
    fn delta(&self, s: &Evaluated<PartialColoring<'g>>) -> Option<i64> {
        Some(s.gamma(self.v, self.c) - 1)
    }
}

/// PartialCol's tabu rule: after `Assign { v, c }`, the neighbours it
/// uncoloured may not take colour c again for random(0..10) + 0.6 ×
/// uncoloured vertices iterations. One move makes several keys tabu, so this
/// is a `TabuList` rather than a `TabuAttr`.
pub struct PartialTabu {
    until: Vec<u64>,
    rng: DefaultRng,
}

impl PartialTabu {
    pub fn new() -> Self {
        PartialTabu {
            until: Vec::new(),
            rng: DefaultRng::seed_from_u64(0),
        }
    }
}

impl Default for PartialTabu {
    fn default() -> Self {
        Self::new()
    }
}

impl<'g> TabuList<PartialColoring<'g>, Assign> for PartialTabu {
    #[inline]
    fn is_tabu(&self, s: &PartialColoring<'g>, m: &Assign, it: u64) -> bool {
        self.until[m.v * s.k + m.c] > it
    }
    fn make_tabu(&mut self, before: &PartialColoring<'g>, m: &Assign, it: u64) {
        let tenure = self.rng.random_range(0..10) + (0.6 * before.uncolored() as f64) as u64;
        for &u in &before.g.adj[m.v] {
            if before.color[u] == m.c {
                self.until[u * before.k + m.c] = it + 1 + tenure;
            }
        }
    }
    fn start<R: Rng + ?Sized>(&mut self, s: &PartialColoring<'g>, rng: &mut R) {
        self.rng = DefaultRng::seed_from_u64(rng.next_u64());
        self.until.clear();
        self.until.resize(s.g.n() * s.k, 0);
    }
}

/// Colour an uncoloured vertex. A vertex whose smallest gamma cannot beat the
/// best candidate so far is skipped, and so is each colour that cannot.
pub struct UncoloredVertices;

impl<'g> Neighborhood<PartialColoring<'g>> for UncoloredVertices {
    type Move = Assign;
    fn explore<V: Visitor<Assign, i64>>(
        &mut self,
        s: &Evaluated<PartialColoring<'g>>,
        vis: &mut V,
    ) -> ControlFlow<()> {
        let k = s.k;
        for &v in &s.list {
            let v = v as usize;
            if vis.can_skip(s.low[v] as i64 - 1) {
                continue;
            }
            for (c, &g) in s.gamma[v * k..][..k].iter().enumerate() {
                let delta = g as i64 - 1;
                if !vis.can_skip(delta) {
                    vis.visit(Assign { v, c }, delta)?;
                }
            }
        }
        ControlFlow::Continue(())
    }
}

/// PartialCol: tabu search on legal partial colourings, random ties.
pub fn partialcol<'g>() -> impl Search<PartialColoring<'g>> {
    TabuSearch::new(UncoloredVertices, PartialTabu::new()).random_ties()
}

/// Variable Space Search (Hertz, Plumettaz & Zufferey 2008), two spaces:
/// from `start`, alternate `iters` TabuCol iterations (minimise conflicting
/// edges) and `iters` PartialCol iterations (minimise uncoloured vertices of
/// a legal partial colouring), `rounds` times, stopping at a legal
/// colouring. Each space's local optima are ordinary points of the other.
/// Returns the best colouring found, as (conflicting edges, colours).
pub fn vss(start: &Coloring, iters: u64, rounds: usize, seed: u64) -> (i64, Vec<usize>) {
    let mut rng = DefaultRng::seed_from_u64(seed);
    let mut cur = start.clone();
    let mut best = (start.evaluate(), start.color.clone());
    for _ in 0..rounds {
        let mut st = State::new(cur, rng.random());
        tabucol()
            .runner(&mut st)
            .until(MaxIterations::new(iters).or(Target::new(0)))
            .run();
        let b = st.global_best();
        if b.cost() < best.0 {
            best = (b.cost(), b.color.clone());
        }
        if best.0 == 0 {
            break;
        }
        let mut st = State::new(PartialColoring::from_coloring(b, &mut rng), rng.random());
        partialcol()
            .runner(&mut st)
            .until(MaxIterations::new(iters).or(Target::new(0)))
            .run();
        cur = st.global_best().complete(&mut rng);
        let cost = cur.evaluate();
        if cost < best.0 {
            best = (cost, cur.color.clone());
        }
        if best.0 == 0 {
            break;
        }
    }
    best
}
