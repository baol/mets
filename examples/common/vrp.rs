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
                let mut xy = vec![(0.0, 0.0); n];
                for r in section("NODE_COORD_SECTION") {
                    xy[index(r[0].parse().map_err(|e| format!("{e}"))?)?] =
                        (num(&r[1])?, num(&r[2])?);
                }
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
        })
    }
}

/// A `.sol` file: routes of customers (1..n, as in the file), the stated
/// cost, and the precision it is written with (half a unit of its last
/// digit: Golden's files round to one decimal, CMT's to two).
pub fn read_sol(path: &Path) -> Result<(Vec<Vec<usize>>, Option<f64>, f64), String> {
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

/// A solution: one route per vehicle (customers, depot implicit at both ends),
/// with each route's load, length and duration, and where each customer is.
#[derive(Clone)]
pub struct Routes<'a> {
    pub inst: &'a Instance,
    pub routes: Vec<Vec<usize>>,
    pub load: Vec<f64>,
    pub length: Vec<f64>,
    /// Length plus service times.
    pub duration: Vec<f64>,
    /// (route, position) of each customer; (usize::MAX, _) for the depot.
    pub at: Vec<(usize, usize)>,
    pub penalties: Penalties,
}

impl<'a> Routes<'a> {
    pub fn new(inst: &'a Instance, routes: Vec<Vec<usize>>, penalties: Penalties) -> Self {
        let mut s = Routes {
            inst,
            routes,
            load: Vec::new(),
            length: Vec::new(),
            duration: Vec::new(),
            at: Vec::new(),
            penalties,
        };
        s.rebuild();
        s
    }

    /// Recompute every per-route table and the position index.
    pub fn rebuild(&mut self) {
        let inst = self.inst;
        self.at = vec![(usize::MAX, 0); inst.n];
        self.load.clear();
        self.length.clear();
        self.duration.clear();
        for (k, r) in self.routes.iter().enumerate() {
            let mut prev = 0;
            let (mut load, mut length, mut service) = (0.0, 0.0, 0.0);
            for (p, &c) in r.iter().enumerate() {
                self.at[c] = (k, p);
                load += inst.demand[c];
                length += inst.d(prev, c);
                service += inst.service[c];
                prev = c;
            }
            length += inst.d(prev, 0);
            self.load.push(load);
            self.length.push(length);
            self.duration.push(length + service);
        }
    }

    /// Total distance, without penalties.
    pub fn distance(&self) -> f64 {
        self.length.iter().sum()
    }

    pub fn excess_load(&self) -> f64 {
        self.load
            .iter()
            .map(|&l| (l - self.inst.capacity).max(0.0))
            .sum()
    }

    pub fn excess_duration(&self) -> f64 {
        self.inst.max_duration.map_or(0.0, |m| {
            self.duration.iter().map(|&t| (t - m).max(0.0)).sum()
        })
    }

    /// Every customer exactly once, no constraint violated.
    pub fn is_feasible(&self) -> bool {
        let mut seen = vec![0u32; self.inst.n];
        self.routes.iter().flatten().for_each(|&c| seen[c] += 1);
        seen[0] == 0
            && seen[1..].iter().all(|&s| s == 1)
            && self.excess_load() == 0.0
            && self.excess_duration() <= 1e-9
    }
}

impl Model for Routes<'_> {
    type Cost = f64;
    /// Distance plus the penalised excess load and duration, from scratch.
    fn evaluate(&self) -> f64 {
        let inst = self.inst;
        let mut total = 0.0;
        let (mut load_excess, mut duration_excess) = (0.0, 0.0);
        for r in &self.routes {
            let mut prev = 0;
            let (mut load, mut length, mut service) = (0.0, 0.0, 0.0);
            for &c in r {
                load += inst.demand[c];
                length += inst.d(prev, c);
                service += inst.service[c];
                prev = c;
            }
            length += inst.d(prev, 0);
            total += length;
            load_excess += (load - inst.capacity).max(0.0);
            if let Some(m) = inst.max_duration {
                duration_excess += (length + service - m).max(0.0);
            }
        }
        total + self.penalties.capacity * load_excess + self.penalties.duration * duration_excess
    }
}
