//! Vehicle routing pieces (PLAN-problems.md P7): the CVRPLIB `.vrp` reader
//! and the `Routes` model with its penalties.
//!
//! Distances follow CVRPLIB's conventions: `EUC_2D` rounded to the nearest
//! integer for the X, XL and XTW sets, real for CMT and Golden (chosen by
//! name unless forced), `EXPLICIT` matrices as given. The depot is node 0;
//! customers are 1..n, numbered as in a `.sol` file.

#![allow(dead_code)] // each example uses part of it

use mets::*;
use std::path::Path;

/// Route versions: a fresh number at every change of a route's sequence,
/// unique across all clones, so a cache keyed by it is never stale.
static VERSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
fn fresh_version() -> u64 {
    VERSION.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

#[path = "geni.rs"]
pub mod geni;
use geni::Geni;

/// Euclidean distances: rounded to the nearest integer, or real.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rounding {
    Nearest,
    Real,
}

pub struct Instance {
    pub name: String,
    /// Nodes, depot included (node 0).
    pub n: usize,
    pub capacity: f64,
    /// Maximum route duration (distance plus service times), if any.
    pub max_duration: Option<f64>,
    pub demand: Vec<f64>,
    pub service: Vec<f64>,
    /// Time windows (ready, due) per node, for VRPTW instances.
    pub windows: Option<Vec<(f64, f64)>>,
    /// Row-major n × n.
    pub dist: Vec<f64>,
    pub rounding: Option<Rounding>,
    /// Coordinates, when the file gives them (for the sweep start and the
    /// cluster perturbation's distances they are not needed: `dist` is).
    pub xy: Option<Vec<(f64, f64)>>,
}

impl Instance {
    #[inline]
    pub fn d(&self, a: usize, b: usize) -> f64 {
        self.dist[a * self.n + b]
    }

    /// Read a `.vrp` file. `rounding` forces the EUC_2D convention; by
    /// default real for names starting with CMT or Golden, rounded otherwise.
    pub fn read(path: &Path, rounding: Option<Rounding>) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mut spec = std::collections::HashMap::new();
        let mut sections: std::collections::HashMap<String, Vec<Vec<String>>> = Default::default();
        let mut current: Option<String> = None;
        for raw in text.lines() {
            let line = raw.trim();
            if line.is_empty() || line == "EOF" {
                continue;
            }
            let starts_numeric = line.starts_with(|c: char| c.is_ascii_digit() || c == '-');
            if !starts_numeric && line.contains(':') {
                let (k, v) = line.split_once(':').expect("':'");
                let k = k.trim().to_string();
                if k.ends_with("SECTION") {
                    sections.insert(k.clone(), Vec::new());
                    current = Some(k);
                } else {
                    spec.insert(k, v.trim().trim_matches('"').to_string());
                    current = None;
                }
                continue;
            }
            if line.ends_with("SECTION") {
                sections.insert(line.to_string(), Vec::new());
                current = Some(line.to_string());
                continue;
            }
            if let Some(c) = &current {
                let row = line.split_whitespace().map(str::to_string).collect();
                sections.get_mut(c).expect("open section").push(row);
            }
        }
        let get = |k: &str| spec.get(k).ok_or(format!("{}: no {k}", path.display()));
        let name = spec.get("NAME").cloned().unwrap_or_else(|| {
            path.file_stem()
                .map_or("vrp".into(), |s| s.to_string_lossy().into_owned())
        });
        let n: usize = get("DIMENSION")?
            .parse()
            .map_err(|e| format!("DIMENSION: {e}"))?;
        let num = |s: &str| s.parse::<f64>().map_err(|e| format!("{s}: {e}"));
        let capacity = num(get("CAPACITY")?)?;
        let max_duration = spec.get("DISTANCE").map(|s| num(s)).transpose()?;

        // node ids -> 0..n in id order; the depot must come first
        let section = |k: &str| sections.get(k).cloned().unwrap_or_default();
        let demand_rows = section("DEMAND_SECTION");
        let mut ids: Vec<i64> = demand_rows
            .iter()
            .map(|r| r[0].parse().unwrap_or(-1))
            .collect();
        ids.sort_unstable();
        if ids.len() != n {
            return Err(format!(
                "{}: {} demands for {n} nodes",
                path.display(),
                ids.len()
            ));
        }
        let index = |id: i64| {
            ids.binary_search(&id)
                .map_err(|_| format!("unknown node {id}"))
        };
        let depots: Vec<i64> = section("DEPOT_SECTION")
            .iter()
            .map(|r| r[0].parse().unwrap_or(-1))
            .filter(|&d| d > 0)
            .collect();
        if depots.len() != 1 || index(depots[0])? != 0 {
            return Err(format!("{}: one depot, the first node", path.display()));
        }
        let mut demand = vec![0.0; n];
        for r in &demand_rows {
            demand[index(r[0].parse().map_err(|e| format!("{e}"))?)?] = num(&r[1])?;
        }
        let mut service = vec![0.0; n];
        if let Some(s) = spec.get("SERVICE_TIME") {
            let t = num(s)?;
            service.iter_mut().skip(1).for_each(|x| *x = t);
        }
        for r in section("SERVICE_TIME_SECTION") {
            service[index(r[0].parse().map_err(|e| format!("{e}"))?)?] = num(&r[1])?;
        }
        let windows = sections.get("TIME_WINDOW_SECTION").map(|rows| {
            let mut w = vec![(0.0, f64::INFINITY); n];
            for r in rows {
                if let (Ok(id), Ok(a), Ok(b)) = (r[0].parse::<i64>(), num(&r[1]), num(&r[2])) {
                    if let Ok(i) = index(id) {
                        w[i] = (a, b);
                    }
                }
            }
            w
        });

        let xy = match sections.get("NODE_COORD_SECTION") {
            Some(rows) if rows.len() == n => {
                let mut xy = vec![(0.0, 0.0); n];
                for r in rows {
                    xy[index(r[0].parse().map_err(|e| format!("{e}"))?)?] =
                        (num(&r[1])?, num(&r[2])?);
                }
                Some(xy)
            }
            _ => None,
        };
        let kind = get("EDGE_WEIGHT_TYPE")?.as_str();
        let mut dist = vec![0.0; n * n];
        let mut used_rounding = None;
        match kind {
            "EUC_2D" => {
                let mode =
                    rounding.unwrap_or(if name.starts_with("CMT") || name.starts_with("Golden") {
                        Rounding::Real
                    } else {
                        Rounding::Nearest
                    });
                used_rounding = Some(mode);
                let xy = xy
                    .as_ref()
                    .ok_or(format!("{}: EUC_2D without coordinates", path.display()))?;
                for a in 0..n {
                    for b in 0..n {
                        let e = ((xy[a].0 - xy[b].0).powi(2) + (xy[a].1 - xy[b].1).powi(2)).sqrt();
                        dist[a * n + b] = match mode {
                            Rounding::Nearest => (e + 0.5).floor(),
                            Rounding::Real => e,
                        };
                    }
                }
            }
            "EXPLICIT" => {
                let fmt = spec
                    .get("EDGE_WEIGHT_FORMAT")
                    .map(String::as_str)
                    .unwrap_or("");
                let vals: Vec<f64> = section("EDGE_WEIGHT_SECTION")
                    .iter()
                    .flatten()
                    .map(|x| num(x))
                    .collect::<Result<_, _>>()?;
                let pairs: Vec<(usize, usize)> = match fmt {
                    "FULL_MATRIX" => (0..n).flat_map(|a| (0..n).map(move |b| (a, b))).collect(),
                    "LOWER_ROW" => (0..n).flat_map(|a| (0..a).map(move |b| (a, b))).collect(),
                    "LOWER_DIAG_ROW" => (0..n).flat_map(|a| (0..=a).map(move |b| (a, b))).collect(),
                    "UPPER_ROW" => (0..n)
                        .flat_map(|a| (a + 1..n).map(move |b| (a, b)))
                        .collect(),
                    "UPPER_DIAG_ROW" => (0..n).flat_map(|a| (a..n).map(move |b| (a, b))).collect(),
                    f => return Err(format!("{}: EDGE_WEIGHT_FORMAT {f:?}", path.display())),
                };
                if pairs.len() != vals.len() {
                    return Err(format!(
                        "{}: {} weights for {} pairs",
                        path.display(),
                        vals.len(),
                        pairs.len()
                    ));
                }
                for ((a, b), v) in pairs.into_iter().zip(vals) {
                    dist[a * n + b] = v;
                    if fmt != "FULL_MATRIX" {
                        dist[b * n + a] = v;
                    }
                }
            }
            k => return Err(format!("{}: EDGE_WEIGHT_TYPE {k:?}", path.display())),
        }
        Ok(Instance {
            name,
            n,
            capacity,
            max_duration,
            demand,
            service,
            windows,
            dist,
            rounding: used_rounding,
            xy,
        })
    }
}

/// Routes, stated cost, and the precision the cost is written with.
pub type Sol = (Vec<Vec<usize>>, Option<f64>, f64);

/// A `.sol` file: routes of customers (1..n, as in the file), the stated
/// cost, and the precision it is written with (half a unit of its last
/// digit: Golden's files round to one decimal, CMT's to two).
pub fn read_sol(path: &Path) -> Result<Sol, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut routes = Vec::new();
    let (mut cost, mut precision) = (None, 0.0);
    for line in text.lines().map(str::trim) {
        let lower = line.to_lowercase();
        if lower.starts_with("route") {
            let rest = line.split_once(':').map_or("", |(_, r)| r);
            routes.push(
                rest.split_whitespace()
                    .map(|x| x.parse().map_err(|e| format!("{x}: {e}")))
                    .collect::<Result<_, _>>()?,
            );
        } else if lower.starts_with("cost") {
            let text = line.split_whitespace().nth(1).unwrap_or("");
            cost = text.parse().ok();
            let decimals = text.split_once('.').map_or(0, |(_, d)| d.len());
            precision = 0.5 * 10f64.powi(-(decimals as i32));
        }
    }
    Ok((routes, cost, precision))
}

/// Penalty weights of the relaxed constraints (Cordeau, Laporte & Mercier's
/// unified tabu search): cost = distance + α·excess load + β·excess duration.
#[derive(Clone, Copy, Debug)]
pub struct Penalties {
    pub capacity: f64,
    pub duration: f64,
}

impl Default for Penalties {
    fn default() -> Self {
        Penalties {
            capacity: 1.0,
            duration: 1.0,
        }
    }
}

/// A solution with `m` routes (vehicles), some possibly empty, as linked
/// lists over flat arrays (no allocation per move; the layout a GPU block
/// would hold): customers are nodes 1..n−1, route k's depot is the sentinel
/// node n + k, and each route is the cycle sentinel → c₁ → … → sentinel.
/// Per route: load, length, service time and size. Relocating a customer is
/// O(1) to evaluate and to apply.
#[derive(Clone)]
pub struct Routes<'a> {
    pub inst: &'a Instance,
    pub m: usize,
    pub succ: Vec<u32>,
    pub pred: Vec<u32>,
    /// Route of each customer and sentinel.
    pub route: Vec<u32>,
    pub load: Vec<f64>,
    pub length: Vec<f64>,
    pub service: Vec<f64>,
    pub size: Vec<u32>,
    pub penalties: Penalties,
    /// (customer, from, to) of the last relocation, for the search's bookkeeping.
    pub last: Option<(usize, usize, usize)>,
    /// Per route, a number that changes whenever its sequence does.
    pub version: Vec<u64>,
}

impl<'a> Routes<'a> {
    /// `routes` are lists of customers (1..n−1); `m` ≥ their number
    /// vehicles, the rest empty.
    pub fn new(inst: &'a Instance, routes: Vec<Vec<usize>>, penalties: Penalties) -> Self {
        let m = routes.len();
        Self::with_vehicles(inst, routes, m, penalties)
    }

    pub fn with_vehicles(
        inst: &'a Instance,
        routes: Vec<Vec<usize>>,
        m: usize,
        penalties: Penalties,
    ) -> Self {
        assert!(
            m >= routes.len(),
            "{} routes for {m} vehicles",
            routes.len()
        );
        let n = inst.n;
        let total = n + m;
        let mut s = Routes {
            inst,
            m,
            succ: (0..total as u32).collect(),
            pred: (0..total as u32).collect(),
            route: vec![u32::MAX; total],
            load: vec![0.0; m],
            length: vec![0.0; m],
            service: vec![0.0; m],
            size: vec![0; m],
            penalties,
            last: None,
            version: (0..m).map(|_| fresh_version()).collect(),
        };
        for k in 0..m {
            s.route[n + k] = k as u32;
        }
        for (k, r) in routes.iter().enumerate() {
            let mut prev = n + k;
            for &c in r {
                s.link(prev, c);
                s.route[c] = k as u32;
                prev = c;
            }
            s.link(prev, n + k);
        }
        s.rebuild();
        s
    }

    #[inline]
    fn link(&mut self, a: usize, b: usize) {
        self.succ[a] = b as u32;
        self.pred[b] = a as u32;
    }

    /// The instance node of a list node (sentinels are the depot).
    #[inline]
    pub fn site(&self, x: usize) -> usize {
        if x >= self.inst.n {
            0
        } else {
            x
        }
    }

    #[inline]
    pub fn d(&self, x: usize, y: usize) -> f64 {
        self.inst.d(self.site(x), self.site(y))
    }

    /// Recompute the per-route tables from the lists.
    pub fn rebuild(&mut self) {
        let n = self.inst.n;
        for k in 0..self.m {
            let start = n + k;
            let (mut load, mut length, mut service, mut size) = (0.0, 0.0, 0.0, 0);
            let mut x = start;
            loop {
                let y = self.succ[x] as usize;
                length += self.d(x, y);
                if y == start {
                    break;
                }
                load += self.inst.demand[y];
                service += self.inst.service[y];
                size += 1;
                x = y;
            }
            (self.load[k], self.length[k], self.service[k], self.size[k]) =
                (load, length, service, size);
        }
    }

    /// The customers of route k, in order.
    pub fn customers(&self, k: usize) -> Vec<usize> {
        let start = self.inst.n + k;
        let mut out = Vec::with_capacity(self.size[k] as usize);
        let mut x = self.succ[start] as usize;
        while x != start {
            out.push(x);
            x = self.succ[x] as usize;
        }
        out
    }

    /// The non-empty routes, as lists of customers.
    pub fn routes(&self) -> Vec<Vec<usize>> {
        (0..self.m)
            .map(|k| self.customers(k))
            .filter(|r| !r.is_empty())
            .collect()
    }

    #[inline]
    pub fn load_excess(&self, load: f64) -> f64 {
        (load - self.inst.capacity).max(0.0)
    }

    #[inline]
    pub fn duration_excess(&self, length: f64, service: f64) -> f64 {
        self.inst
            .max_duration
            .map_or(0.0, |m| (length + service - m).max(0.0))
    }

    /// Total distance, without penalties.
    pub fn distance(&self) -> f64 {
        self.length.iter().sum()
    }

    pub fn excess_load(&self) -> f64 {
        self.load.iter().map(|&l| self.load_excess(l)).sum()
    }

    pub fn excess_duration(&self) -> f64 {
        (0..self.m)
            .map(|k| self.duration_excess(self.length[k], self.service[k]))
            .sum()
    }

    /// No constraint violated (every customer is on exactly one route by
    /// construction).
    pub fn is_feasible(&self) -> bool {
        self.excess_load() <= 1e-9 && self.excess_duration() <= 1e-9
    }

    /// Remove customer c from its route (reconnecting its neighbours).
    fn unlink(&mut self, c: usize) {
        let (p, s) = (self.pred[c] as usize, self.succ[c] as usize);
        self.link(p, s);
    }

    /// Insert customer c after node a (a customer or a route's sentinel).
    fn insert_after(&mut self, c: usize, a: usize) {
        let b = self.succ[a] as usize;
        self.link(a, c);
        self.link(c, b);
        self.route[c] = self.route[a];
    }

    /// Best position for customer c (not on any route) in route k, by distance:
    /// (node to insert after, added distance).
    pub fn best_insertion(&self, c: usize, k: usize) -> (usize, f64) {
        let start = self.inst.n + k;
        let (mut best, mut cost) = (start, f64::INFINITY);
        let mut a = start;
        loop {
            let b = self.succ[a] as usize;
            let add = self.d(a, c) + self.d(c, b) - self.d(a, b);
            if add < cost {
                (best, cost) = (a, add);
            }
            if b == start {
                break;
            }
            a = b;
        }
        (best, cost)
    }

    /// Take customer c off its route; route tables updated.
    pub fn remove(&mut self, c: usize) {
        let k = self.route[c] as usize;
        let (p, s) = (self.pred[c] as usize, self.succ[c] as usize);
        self.length[k] += self.d(p, s) - self.d(p, c) - self.d(c, s);
        self.load[k] -= self.inst.demand[c];
        self.service[k] -= self.inst.service[c];
        self.size[k] -= 1;
        self.version[k] = fresh_version();
        self.unlink(c);
        self.route[c] = u32::MAX;
        self.succ[c] = c as u32;
        self.pred[c] = c as u32;
    }

    /// Put customer c (off every route) after node a; route tables updated.
    pub fn put(&mut self, c: usize, a: usize) {
        let k = self.route[a] as usize;
        let b = self.succ[a] as usize;
        self.length[k] += self.d(a, c) + self.d(c, b) - self.d(a, b);
        self.load[k] += self.inst.demand[c];
        self.service[k] += self.inst.service[c];
        self.size[k] += 1;
        self.version[k] = fresh_version();
        self.insert_after(c, a);
    }

    /// Route k as a GENI tour: its sentinel, then its customers.
    pub fn tour(&self, k: usize) -> Vec<usize> {
        let mut t = Vec::with_capacity(self.size[k] as usize + 1);
        t.push(self.inst.n + k);
        t.extend(self.customers(k));
        t
    }

    /// Put customer c (off every route) into route k by the GENI move g,
    /// evaluated on `self.tour(k)`; route tables updated.
    pub fn put_geni(&mut self, c: usize, k: usize, g: &Geni) {
        let tour = self.tour(k);
        let cycle = geni::apply(&tour, c, g);
        debug_assert_eq!(cycle[0], self.inst.n + k);
        self.load[k] += self.inst.demand[c];
        self.service[k] += self.inst.service[c];
        self.size[k] += 1;
        self.route[c] = k as u32;
        self.relink(k, &cycle[1..]);
    }

    /// Intra-route refinement (the paper's every-200-iterations pass, in the
    /// plan's simpler variant): each route is improved by 2-opt (reverse a
    /// segment) and Or-opt (move a segment of 1–3 customers elsewhere in the
    /// route, either way round), first improvement, until neither shortens
    /// it. Distances are taken as symmetric (2-opt), as in every set read
    /// here.
    pub fn refine(&mut self) {
        for k in 0..self.m {
            if self.size[k] < 2 {
                continue;
            }
            let mut seq = self.customers(k);
            if self.improve_sequence(&mut seq) {
                self.relink(k, &seq);
            }
        }
    }

    /// 2-opt and Or-opt on a route's customer sequence (depot at both ends);
    /// true if it changed.
    fn improve_sequence(&self, seq: &mut Vec<usize>) -> bool {
        let at = |seq: &[usize], i: isize| -> usize {
            if i < 0 || i as usize >= seq.len() {
                0
            } else {
                seq[i as usize]
            }
        };
        let d = |a: usize, b: usize| self.inst.d(a, b);
        let mut changed = false;
        loop {
            let mut improved = false;
            let len = seq.len() as isize;
            // 2-opt: reverse seq[i..=j]
            'two: for i in 0..len {
                for j in i + 1..len {
                    let (a, b, c, e) = (at(seq, i - 1), at(seq, i), at(seq, j), at(seq, j + 1));
                    if d(a, c) + d(b, e) < d(a, b) + d(c, e) - 1e-9 {
                        seq[i as usize..=j as usize].reverse();
                        improved = true;
                        break 'two;
                    }
                }
            }
            if !improved {
                // Or-opt: move seq[i..i+l] between positions p-1 and p (outside it)
                'or: for l in 1..=3isize.min(len) {
                    for i in 0..=len - l {
                        let (before, first, last, after) = (
                            at(seq, i - 1),
                            at(seq, i),
                            at(seq, i + l - 1),
                            at(seq, i + l),
                        );
                        let cut = d(before, first) + d(last, after) - d(before, after);
                        for p in 0..=len {
                            if p >= i && p <= i + l {
                                continue;
                            }
                            let (x, y) = (at(seq, p - 1), at(seq, p));
                            let base = d(x, y);
                            let forward = d(x, first) + d(last, y) - base;
                            let backward = d(x, last) + d(first, y) - base;
                            let (gain, reversed) = if backward < forward {
                                (backward, true)
                            } else {
                                (forward, false)
                            };
                            if gain < cut - 1e-9 {
                                let mut segment: Vec<usize> =
                                    seq.drain(i as usize..(i + l) as usize).collect();
                                if reversed {
                                    segment.reverse();
                                }
                                let q = if p > i { p - l } else { p } as usize;
                                seq.splice(q..q, segment);
                                improved = true;
                                break 'or;
                            }
                        }
                    }
                }
            }
            if !improved {
                return changed;
            }
            changed = true;
        }
    }

    /// Replace route k's sequence (same customers, new order).
    fn relink(&mut self, k: usize, seq: &[usize]) {
        self.version[k] = fresh_version();
        let start = self.inst.n + k;
        let mut prev = start;
        for &c in seq {
            self.link(prev, c);
            prev = c;
        }
        self.link(prev, start);
        let mut length = 0.0;
        let mut x = start;
        loop {
            let y = self.succ[x] as usize;
            length += self.d(x, y);
            if y == start {
                break;
            }
            x = y;
        }
        self.length[k] = length;
    }
}

impl Model for Routes<'_> {
    type Cost = f64;
    /// f(s) = distance + α·excess load + β·excess duration, from the lists
    /// alone (not the cached per-route tables).
    fn evaluate(&self) -> f64 {
        let n = self.inst.n;
        let mut total = 0.0;
        let (mut q, mut t) = (0.0, 0.0);
        for k in 0..self.m {
            let start = n + k;
            let (mut load, mut length, mut service) = (0.0, 0.0, 0.0);
            let mut x = start;
            loop {
                let y = self.succ[x] as usize;
                length += self.d(x, y);
                if y == start {
                    break;
                }
                load += self.inst.demand[y];
                service += self.inst.service[y];
                x = y;
            }
            total += length;
            q += self.load_excess(load);
            t += self.duration_excess(length, service);
        }
        total + self.penalties.capacity * q + self.penalties.duration * t
    }
}

/// Move customer c to route `to`, after node `after` (a customer of `to`, or
/// its sentinel). The delta is O(1): three arcs and the two routes' load and
/// duration penalties.
///
/// With `geni`, the insertion into `to` is that GENI move instead (`after`
/// unused), on route `to`'s tour [sentinel, c₁, …] as it is when the move
/// is evaluated; removal is always by reconnection.
#[derive(Clone, Copy, Debug)]
pub struct Relocate {
    pub c: usize,
    pub to: usize,
    pub after: usize,
    pub geni: Option<Geni>,
}

impl<'a> Relocate {
    /// The change of f(s), for a customer moving to another route.
    #[inline]
    pub fn delta_of(&self, s: &Routes<'a>) -> f64 {
        let (c, a) = (self.c, self.after);
        let from = s.route[c] as usize;
        let (p, q) = (s.pred[c] as usize, s.succ[c] as usize);
        let removed = s.d(p, q) - s.d(p, c) - s.d(c, q);
        if from == self.to {
            // same route: remove, then insert (a is not c)
            let b = if s.succ[a] as usize == c {
                q
            } else {
                s.succ[a] as usize
            };
            if a == p {
                return 0.0;
            }
            let added = s.d(a, c) + s.d(c, b) - s.d(a, b);
            let (len, srv) = (s.length[from], s.service[from]);
            let pen = s.penalties.duration
                * (s.duration_excess(len + removed + added, srv) - s.duration_excess(len, srv));
            return removed + added + pen;
        }
        let added = match self.geni {
            Some(g) => g.added,
            None => {
                let b = s.succ[a] as usize;
                s.d(a, c) + s.d(c, b) - s.d(a, b)
            }
        };
        let (dq, ds) = (s.inst.demand[c], s.inst.service[c]);
        let (lf, lt) = (s.load[from], s.load[self.to]);
        let load_pen =
            s.load_excess(lf - dq) - s.load_excess(lf) + s.load_excess(lt + dq) - s.load_excess(lt);
        let (nf, nt) = (s.length[from], s.length[self.to]);
        let (sf, st) = (s.service[from], s.service[self.to]);
        let dur_pen = s.duration_excess(nf + removed, sf - ds) - s.duration_excess(nf, sf)
            + s.duration_excess(nt + added, st + ds)
            - s.duration_excess(nt, st);
        removed + added + s.penalties.capacity * load_pen + s.penalties.duration * dur_pen
    }
}

impl<'a> Move<Routes<'a>> for Relocate {
    fn apply(&self, s: &mut Routes<'a>) {
        let from = s.route[self.c] as usize;
        s.remove(self.c);
        match self.geni {
            Some(g) => s.put_geni(self.c, self.to, &g),
            None => s.put(self.c, self.after),
        }
        s.last = Some((self.c, from, self.to));
    }
    #[inline]
    fn delta(&self, s: &Evaluated<Routes<'a>>) -> Option<f64> {
        Some(self.delta_of(s))
    }
}

/// The paper's attribute rule: after customer c leaves route k, putting it
/// back in route k is tabu (key c·m + route).
impl<'a> TabuAttr<Routes<'a>> for Relocate {
    type Key = usize;
    #[inline]
    fn key(&self, s: &Routes<'a>) -> usize {
        self.c * s.m + self.to
    }
    #[inline]
    fn reverse_key(&self, before: &Routes<'a>) -> usize {
        self.c * before.m + before.route[self.c] as usize
    }
}
impl<'a> DenseAttr<Routes<'a>> for Relocate {
    fn key_count(s: &Routes<'a>) -> usize {
        s.inst.n * s.m
    }
}

/// Relocate each customer to the best position (by distance) of every other
/// route; empty routes are all the same, so only the first is tried.
/// With `geni_p` > 0 the position is the best GENI insertion (types I and
/// II, p-neighbourhoods of that size, plain insertion included). The best
/// GENI insertion of c into route k depends only on k's sequence, so it is
/// cached per (c, k) under the route's version: a move recomputes two
/// routes' entries, not m.
pub struct InterRoute {
    pub geni_p: usize,
    cache: Vec<(u64, Geni)>,
}

impl InterRoute {
    pub fn new(geni_p: usize) -> Self {
        InterRoute {
            geni_p,
            cache: Vec::new(),
        }
    }
}

impl<'a> Neighborhood<Routes<'a>> for InterRoute {
    type Move = Relocate;
    fn explore<V: Visitor<Relocate, f64>>(
        &mut self,
        s: &Evaluated<Routes<'a>>,
        vis: &mut V,
    ) -> std::ops::ControlFlow<()> {
        let (n, m) = (s.inst.n, s.m);
        let first_empty = (0..m).find(|&k| s.size[k] == 0);
        let d = |a: usize, b: usize| s.d(a, b);
        if self.geni_p > 0 {
            let stale = (0, geni::best(&[0], 0, 0, &d));
            self.cache.resize(n * m, stale);
            for k in 0..m {
                if s.size[k] == 0 && Some(k) != first_empty {
                    continue;
                }
                if (1..n)
                    .all(|c| s.route[c] as usize == k || self.cache[c * m + k].0 == s.version[k])
                {
                    continue;
                }
                let tour = s.tour(k);
                let near = geni::Near::new(&tour, self.geni_p, &d);
                for c in 1..n {
                    let e = &mut self.cache[c * m + k];
                    if s.route[c] as usize != k && e.0 != s.version[k] {
                        let mut g = geni::best_with(&tour, &near, c, self.geni_p, &d);
                        if g.kind == 0 {
                            // plain: keep the node to insert after in `i`
                            g.i = tour[g.i as usize] as u32;
                        }
                        *e = (s.version[k], g);
                    }
                }
            }
        }
        for c in 1..n {
            let from = s.route[c] as usize;
            for k in 0..m {
                if k == from || (s.size[k] == 0 && Some(k) != first_empty) {
                    continue;
                }
                let m = if self.geni_p > 0 {
                    let g = self.cache[c * s.m + k].1;
                    if g.kind == 0 {
                        Relocate {
                            c,
                            to: k,
                            after: g.i as usize,
                            geni: None,
                        }
                    } else {
                        Relocate {
                            c,
                            to: k,
                            after: c,
                            geni: Some(g),
                        }
                    }
                } else {
                    let (after, _) = s.best_insertion(c, k);
                    Relocate {
                        c,
                        to: k,
                        after,
                        geni: None,
                    }
                };
                vis.visit(m, m.delta_of(s))?;
            }
        }
        std::ops::ControlFlow::Continue(())
    }
}

/// The sweep start (the paper's initial solution): customers by angle
/// around the depot, each inserted at its best position in the current
/// route; a new route opens when capacity or duration would be exceeded
/// while vehicles remain, so the first m − 1 routes are feasible.
pub fn sweep<'a>(inst: &'a Instance, m: usize, rng: &mut impl mets::rand::Rng) -> Routes<'a> {
    let n = inst.n;
    let xy = inst.xy.as_ref().expect("the sweep needs coordinates");
    let offset: f64 = rng.random::<f64>() * std::f64::consts::TAU;
    let mut order: Vec<usize> = (1..n).collect();
    let angle = |c: usize| {
        let a = (xy[c].1 - xy[0].1).atan2(xy[c].0 - xy[0].0) - offset;
        a.rem_euclid(std::f64::consts::TAU)
    };
    order.sort_by(|&a, &b| angle(a).partial_cmp(&angle(b)).expect("finite angles"));
    let mut s = Routes::with_vehicles(inst, Vec::new(), m, Penalties::default());
    let mut k = 0;
    for c in order {
        let (a, add) = s.best_insertion(c, k);
        let over_q = s.load[k] + inst.demand[c] > inst.capacity + 1e-9;
        let over_t = s.duration_excess(s.length[k] + add, s.service[k] + inst.service[c]) > 1e-9;
        if (over_q || over_t) && k + 1 < m && s.size[k] > 0 {
            k += 1;
            let (a, _) = s.best_insertion(c, k);
            s.put(c, a);
        } else {
            s.put(c, a);
        }
    }
    s
}

/// The paper's perturbation (cluster removal, after Pisinger & Ropke): a
/// random seed customer and its π closest customers are removed and
/// reinserted one by one, in random order, at least distance over all
/// routes (the first empty route included).
pub fn cluster_perturbation(s: &mut Routes, pi: usize, rng: &mut impl mets::rand::Rng) {
    let n = s.inst.n;
    let seed = rng.random_range(1..n);
    let mut by: Vec<usize> = (1..n).collect();
    by.sort_by(|&a, &b| {
        s.inst
            .d(seed, a)
            .partial_cmp(&s.inst.d(seed, b))
            .expect("finite")
    });
    let mut cluster: Vec<usize> = by.into_iter().take(pi + 1).collect();
    for &c in &cluster {
        s.remove(c);
    }
    for i in (1..cluster.len()).rev() {
        cluster.swap(i, rng.random_range(0..=i));
    }
    for c in cluster {
        let first_empty = (0..s.m).find(|&k| s.size[k] == 0);
        let (k, a) = (0..s.m)
            .filter(|&k| s.size[k] > 0 || Some(k) == first_empty)
            .map(|k| {
                let (a, add) = s.best_insertion(c, k);
                (add, k, a)
            })
            .min_by(|x, y| x.0.partial_cmp(&y.0).expect("finite"))
            .map(|(_, k, a)| (k, a))
            .expect("a route");
        let _ = k;
        s.put(c, a);
    }
    s.last = None;
}
