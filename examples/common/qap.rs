//! Shared by the `qap` and `qap_advanced` examples: the QAP instance and
//! models, QAPLIB `.sln` reading, and the sweep driver that produces the
//! `QAPLIB.md` tables.
//!
//! Included with `#[path = "common/qap.rs"] mod common;` from each example.
#![allow(dead_code)]

use mets::perm::Permutation;
use mets::rand::seq::SliceRandom;
use mets::rand::SeedableRng;
use mets::*;
use std::collections::BTreeMap;
use std::path::Path;
use std::time::Instant;

pub type C = i64;

// ------------------------------------------------------------- instance

/// Row `i` of the n×n row-major matrix `m`.
#[inline]
fn row(m: &[C], i: usize, n: usize) -> &[C] {
    &m[i * n..][..n]
}

pub struct Instance {
    pub n: usize,
    pub a: Vec<C>, // first matrix, row-major n×n
    pub b: Vec<C>, // second matrix
    /// Both matrices symmetric with zero diagonal (most QAPLIB instances).
    /// Halves the arithmetic of the swap deltas and of Taillard's update.
    pub symmetric: bool,
    /// Transposes of `a` and `b`: column reads become row reads.
    pub at: Vec<C>,
    pub bt: Vec<C>,
}

impl Instance {
    pub fn new(n: usize, a: Vec<C>, b: Vec<C>) -> Self {
        let sym = |m: &[C]| {
            (0..n).all(|i| m[i * n + i] == 0 && (i + 1..n).all(|j| m[i * n + j] == m[j * n + i]))
        };
        let symmetric = sym(&a) && sym(&b);
        let t = |m: &[C]| (0..n * n).map(|k| m[(k % n) * n + k / n]).collect();
        let (at, bt) = (t(&a), t(&b));
        Instance {
            n,
            a,
            b,
            symmetric,
            at,
            bt,
        }
    }

    /// cost(p) = Σ a[i][j] · b[p[i]][p[j]]
    pub fn cost_of(&self, p: &[usize]) -> C {
        let n = self.n;
        let mut s = 0;
        for i in 0..n {
            for j in 0..n {
                s += self.a[i * n + j] * self.b[p[i] * n + p[j]];
            }
        }
        s
    }

    /// O(n) delta of swapping positions r and s.
    #[inline]
    pub fn swap_delta(&self, p: &[usize], r: usize, s: usize) -> C {
        if self.symmetric {
            self.swap_delta_sym(p, r, s)
        } else {
            self.swap_delta_gen(p, r, s)
        }
    }

    /// General (asymmetric) formula, 2 products per k:
    /// Σ (a[k][r] − a[k][s])(b[pk][ps] − b[pk][pr]) + (a[r][k] − a[s][k])(b[ps][pk] − b[pr][pk]).
    /// Columns are read as rows of the transposes, so every access is row-wise.
    fn swap_delta_gen(&self, p: &[usize], r: usize, s: usize) -> C {
        let n = self.n;
        let (pr, ps) = (p[r], p[s]);
        let (ar, as_, atr, ats) = (
            row(&self.a, r, n),
            row(&self.a, s, n),
            row(&self.at, r, n),
            row(&self.at, s, n),
        );
        let (br, bs, btr, bts) = (
            row(&self.b, pr, n),
            row(&self.b, ps, n),
            row(&self.bt, pr, n),
            row(&self.bt, ps, n),
        );
        let mut d = ar[r] * (bs[ps] - br[pr])
            + ar[s] * (bs[pr] - br[ps])
            + as_[r] * (br[ps] - bs[pr])
            + as_[s] * (br[pr] - bs[ps]);
        for (k, &pk) in p.iter().enumerate() {
            if k == r || k == s {
                continue;
            }
            d += (atr[k] - ats[k]) * (bts[pk] - btr[pk]) + (ar[k] - as_[k]) * (bs[pk] - br[pk]);
        }
        d
    }

    /// Symmetric, zero-diagonal formula: 2 Σ (a[r][k] − a[s][k]) (b[ps][pk] − b[pr][pk]).
    fn swap_delta_sym(&self, p: &[usize], r: usize, s: usize) -> C {
        let n = self.n;
        let (pr, ps) = (p[r], p[s]);
        let (ar, as_) = (&self.a[r * n..][..n], &self.a[s * n..][..n]);
        let (br, bs) = (&self.b[pr * n..][..n], &self.b[ps * n..][..n]);
        let mut d = 0;
        for (k, &pk) in p.iter().enumerate() {
            if k == r || k == s {
                continue;
            }
            d += (ar[k] - as_[k]) * (bs[pk] - br[pk]);
        }
        2 * d
    }

    /// Like Taillard's taiXXa: symmetric matrices, zero diagonal, entries
    /// uniform in 0..=99. (Not one of the published instances.)
    pub fn random(n: usize, seed: u64) -> Self {
        use mets::rand::Rng;
        let mut rng = DefaultRng::seed_from_u64(seed);
        let mut sym = || {
            let mut m = vec![0; n * n];
            for i in 0..n {
                for j in i + 1..n {
                    let v = rng.random_range(0..=99);
                    m[i * n + j] = v;
                    m[j * n + i] = v;
                }
            }
            m
        };
        let a = sym();
        let b = sym();
        Instance::new(n, a, b)
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let v: Vec<C> = text
            .split_whitespace()
            .map(|t| t.parse().map_err(|_| format!("bad number {t:?}")))
            .collect::<Result<_, _>>()?;
        let n = *v.first().ok_or("empty file")? as usize;
        if v.len() != 1 + 2 * n * n {
            return Err(format!(
                "expected {} numbers for n = {n}, found {}",
                1 + 2 * n * n,
                v.len()
            ));
        }
        Ok(Instance::new(
            n,
            v[1..1 + n * n].to_vec(),
            v[1 + n * n..].to_vec(),
        ))
    }

    pub fn read(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::parse(&text).map_err(|e| format!("{}: {e}", path.display()))
    }
}

/// The best-known value of an instance, from a QAPLIB `.sln` file.
///
/// QAPLIB solution files do not all use the same convention: most list the
/// permutation in the direction of `cost_of`, a few list its inverse, and a
/// few are inconsistent (kra32's header is above the cost of its own
/// permutation; tai40a's permutation is invalid). The permutation is checked
/// both ways, and the reference is the lower of the header and the verified
/// cost of the permutation.
pub struct BestKnown {
    pub value: C,
    pub note: Option<String>,
}

pub fn read_sln(path: &Path, inst: &Instance) -> Result<BestKnown, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Vec<i64> = text
        .replace(',', " ")
        .split_whitespace()
        .map(|t| t.parse().unwrap_or(-1))
        .collect();
    if v.len() < 2 || v[0] as usize != inst.n {
        return Err(format!(
            "{}: header does not match the instance",
            path.display()
        ));
    }
    let header = v[1];
    let n = inst.n;
    let p: Vec<usize> = v[2..].iter().map(|&x| (x - 1) as usize).collect();
    let mut seen = vec![false; n];
    let valid = p.len() == n
        && p.iter()
            .all(|&x| x < n && !std::mem::replace(&mut seen[x], true));
    if !valid {
        return Ok(BestKnown {
            value: header,
            note: Some("permutation in .sln is invalid; header value used".into()),
        });
    }
    let mut inv = vec![0; n];
    for (i, &x) in p.iter().enumerate() {
        inv[x] = i;
    }
    let (direct, inverse) = (inst.cost_of(&p), inst.cost_of(&inv));
    Ok(if direct == header || inverse == header {
        BestKnown {
            value: header,
            note: None,
        }
    } else {
        let best = direct.min(inverse);
        BestKnown {
            value: best.min(header),
            note: Some(format!(
                "header {header} does not match its permutation ({best})"
            )),
        }
    })
}

// ------------------------------------------------------------ plain model

/// O(n) swap delta, O(1) apply (like METSlib's qap_model.hpp).
#[derive(Clone)]
pub struct QapPlain<'a> {
    pub inst: &'a Instance,
    pub p: Vec<usize>,
}

impl Model for QapPlain<'_> {
    type Cost = C;
    fn evaluate(&self) -> C {
        self.inst.cost_of(&self.p)
    }
}

impl Permutation for QapPlain<'_> {
    fn as_slice(&self) -> &[usize] {
        &self.p
    }
    fn swap(&mut self, i: usize, j: usize) {
        self.p.swap(i, j);
    }
    fn swap_delta(&self, i: usize, j: usize) -> Option<C> {
        Some(self.inst.swap_delta(&self.p, i, j))
    }
}

// --------------------------------------------------------- Taillard model

/// Keeps Taillard's delta matrix: O(1) swap delta, O(n²) apply.
#[derive(Clone)]
pub struct QapTaillard<'a> {
    pub inst: &'a Instance,
    pub p: Vec<usize>,
    delta: Vec<C>, // delta[r*n+s] for r<s
    work: Vec<C>,  // scratch of `swap`
}

impl<'a> QapTaillard<'a> {
    pub fn new(inst: &'a Instance, p: Vec<usize>) -> Self {
        let n = inst.n;
        let mut delta = vec![0; n * n];
        for r in 0..n {
            for s in r + 1..n {
                delta[r * n + s] = inst.swap_delta(&p, r, s);
            }
        }
        QapTaillard {
            inst,
            p,
            delta,
            work: Vec::new(),
        }
    }
}

impl Model for QapTaillard<'_> {
    type Cost = C;
    fn evaluate(&self) -> C {
        self.inst.cost_of(&self.p)
    }
}

impl Permutation for QapTaillard<'_> {
    fn as_slice(&self) -> &[usize] {
        &self.p
    }
    #[inline]
    fn swap_delta(&self, i: usize, j: usize) -> Option<C> {
        Some(self.delta[i * self.inst.n + j]) // O(1)
    }
    fn swap(&mut self, u: usize, v: usize) {
        let (inst, n) = (self.inst, self.inst.n);
        let (a, b) = (&inst.a, &inst.b);
        self.p.swap(u, v);
        let p = &self.p;
        let (pu, pv) = (p[u], p[v]);
        // Taillard's O(1) update for pairs disjoint from {u, v} factors into
        // per-position differences, computed once per swap so that the inner
        // loop reads contiguous memory:
        //   fa(r, s) = ca[r] − ca[s],  ca[k] = a[k][u] − a[k][v]
        //   fb(r, s) = cb[s] − cb[r],  cb[k] = b[p[k]][pu] − b[p[k]][pv]
        // (and the transposed terms ra, rb for asymmetric instances).
        let w = &mut self.work;
        w.resize(4 * n, 0);
        let (ca, rest) = w.split_at_mut(n);
        let (cb, rest) = rest.split_at_mut(n);
        let (ra, rb) = rest.split_at_mut(n);
        let (at, bt) = (&inst.at, &inst.bt);
        let (atu, atv, au, av) = (row(at, u, n), row(at, v, n), row(a, u, n), row(a, v, n));
        let (btu, btv, bu, bv) = (row(bt, pu, n), row(bt, pv, n), row(b, pu, n), row(b, pv, n));
        for (k, &pk) in p.iter().enumerate() {
            ca[k] = atu[k] - atv[k]; // a[k][u] − a[k][v]
            cb[k] = btu[pk] - btv[pk]; // b[pk][pu] − b[pk][pv]
            ra[k] = au[k] - av[k];
            rb[k] = bu[pk] - bv[pk];
        }
        for r in 0..n {
            if r == u || r == v {
                continue;
            }
            let row = &mut self.delta[r * n..(r + 1) * n];
            let (car, cbr, rar, rbr) = (ca[r], cb[r], ra[r], rb[r]);
            if inst.symmetric {
                for s in r + 1..n {
                    row[s] += 2 * (car - ca[s]) * (cb[s] - cbr);
                }
            } else {
                for s in r + 1..n {
                    row[s] += (car - ca[s]) * (cb[s] - cbr) + (rar - ra[s]) * (rb[s] - rbr);
                }
            }
        }
        // pairs that touch u or v: O(n) each (overwrites the entries above)
        for k in 0..n {
            for x in [u, v] {
                if k != x {
                    let (r, s) = (k.min(x), k.max(x));
                    self.delta[r * n + s] = inst.swap_delta(p, r, s);
                }
            }
        }
    }
}

// --------------------------------------------------------------- helpers

pub fn random_start(n: usize, seed: u64) -> Vec<usize> {
    let mut p: Vec<usize> = (0..n).collect();
    p.shuffle(&mut DefaultRng::seed_from_u64(seed ^ 0x9e37_79b9_7f4a_7c15));
    p
}

pub fn gap(cost: C, best_known: C) -> f64 {
    (cost - best_known) as f64 * 100.0 / (best_known as f64).abs().max(1.0)
}

// ----------------------------------------------------------------- sweep

/// What a solver returns for one instance.
pub struct Solved {
    pub found: C,
    /// Seconds until `found` was first reached (if the solver tracks it).
    pub to_found: Option<f64>,
}

/// One instance of the sweep.
pub struct SweepRow {
    pub name: String,
    pub n: usize,
    pub found: C,
    pub best_known: Option<C>,
    pub note: Option<String>,
    pub secs: f64,
    pub to_found: Option<f64>,
}

pub struct SweepOptions {
    /// Description of the method, for the table header.
    pub setup: String,
    /// Instances solved at the same time (1 when the solver is parallel itself).
    pub instance_jobs: usize,
    pub markdown: bool,
}

/// Run `solve` on every `.dat` in `data` (largest first, `instance_jobs` at a
/// time), compare with the `.sln` in `soln`, and print the table.
pub fn sweep<F>(data: &Path, soln: &Path, opt: &SweepOptions, solve: F)
where
    F: Fn(&Instance, Option<C>) -> Solved + Sync,
{
    let mut names: Vec<String> = std::fs::read_dir(data)
        .expect("cannot read the data directory")
        .filter_map(|e| {
            e.ok()?
                .file_name()
                .to_str()?
                .strip_suffix(".dat")
                .map(String::from)
        })
        .collect();
    names.sort();
    let size =
        |name: &String| std::fs::metadata(data.join(format!("{name}.dat"))).map_or(0, |m| m.len());
    let mut order: Vec<usize> = (0..names.len()).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(size(&names[i])));

    let solve_one = |name: &str| -> Result<SweepRow, String> {
        let inst = Instance::read(&data.join(format!("{name}.dat")))?;
        let sln = soln.join(format!("{name}.sln"));
        let (best_known, note) = if sln.exists() {
            match read_sln(&sln, &inst) {
                Ok(b) => (Some(b.value), b.note),
                Err(e) => (None, Some(e)),
            }
        } else {
            (None, Some("no published solution".into()))
        };
        let t = Instant::now();
        let s = solve(&inst, best_known);
        Ok(SweepRow {
            name: name.into(),
            n: inst.n,
            found: s.found,
            best_known,
            note,
            secs: t.elapsed().as_secs_f64(),
            to_found: s.to_found,
        })
    };

    let next = std::sync::atomic::AtomicUsize::new(0);
    let results = std::sync::Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..opt.instance_jobs.max(1) {
            scope.spawn(|| loop {
                let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&i) = order.get(k) else { break };
                let name = &names[i];
                match solve_one(name) {
                    Ok(row) => {
                        eprintln!("{:<10} {:>12} {:>8.1}s", row.name, row.found, row.secs);
                        results.lock().unwrap().push((i, row));
                    }
                    Err(e) => eprintln!("{name}: skipped: {e}"),
                }
            });
        }
    });
    let mut rows = results.into_inner().unwrap();
    rows.sort_by_key(|(i, _)| *i);
    let rows: Vec<SweepRow> = rows.into_iter().map(|(_, r)| r).collect();
    print_sweep(&rows, opt);
}

fn print_sweep(rows: &[SweepRow], opt: &SweepOptions) {
    let has_to_found = rows.iter().any(|r| r.to_found.is_some());
    let fmt_to_found = |r: &SweepRow| r.to_found.map_or("—".to_string(), |t| format!("{t:.1}"));

    if !opt.markdown {
        println!("{}", opt.setup);
        println!(
            "{:<10} {:>4} {:>12} {:>12} {:>8} {:>8} {:>9}  note",
            "instance", "n", "found", "best known", "gap", "time", "to found"
        );
        for row in rows {
            let (bk, g) = match row.best_known {
                Some(b) => (b.to_string(), format!("{:.2}%", gap(row.found, b))),
                None => ("-".into(), "-".into()),
            };
            println!(
                "{:<10} {:>4} {:>12} {bk:>12} {g:>8} {:>7.1}s {:>9}  {}",
                row.name,
                row.n,
                row.found,
                row.secs,
                fmt_to_found(row),
                row.note.as_deref().unwrap_or("")
            );
        }
    }

    // summary by family (instances with a best-known value)
    let mut families: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    for r in rows {
        if let Some(b) = r.best_known {
            let family: String = r
                .name
                .chars()
                .take_while(|c| c.is_ascii_alphabetic())
                .collect();
            families.entry(family).or_default().push(gap(r.found, b));
        }
    }
    let with_bk = rows.iter().filter(|r| r.best_known.is_some()).count();
    let reached = rows
        .iter()
        .filter(|r| r.best_known.is_some_and(|b| r.found <= b))
        .count();
    let improved: Vec<&SweepRow> = rows
        .iter()
        .filter(|r| r.best_known.is_some_and(|b| r.found < b))
        .collect();
    let total_secs: f64 = rows.iter().map(|r| r.secs).sum();

    if opt.markdown {
        println!(
            "{}: {} instances, best-known value reached on {reached} of the {with_bk} that have one; {:.1} CPU-hours in total.\n",
            opt.setup,
            rows.len(),
            total_secs / 3600.0
        );
        if has_to_found {
            println!("| Instance | n | Found | Best known | Gap | Time (s) | To found (s) |");
            println!("|---|---:|---:|---:|---:|---:|---:|");
        } else {
            println!("| Instance | n | Found | Best known | Gap | Time (s) |");
            println!("|---|---:|---:|---:|---:|---:|");
        }
        for r in rows {
            let mark = if r.note.is_some() { " *" } else { "" };
            let (bk, g) = match r.best_known {
                Some(b) => (b.to_string(), format!("{:.2}%", gap(r.found, b))),
                None => ("—".into(), "—".into()),
            };
            let tail = if has_to_found {
                format!(" {} |", fmt_to_found(r))
            } else {
                String::new()
            };
            println!(
                "| {}{mark} | {} | {} | {bk} | {g} | {:.1} |{tail}",
                r.name, r.n, r.found, r.secs
            );
        }
        let notes: Vec<&SweepRow> = rows.iter().filter(|r| r.note.is_some()).collect();
        if !notes.is_empty() {
            println!("\n\\* Notes:\n");
            for r in notes {
                println!("- `{}`: {}", r.name, r.note.as_deref().unwrap_or(""));
            }
        }
        println!("\n| Family | Instances | Best known reached | Mean gap | Max gap |");
        println!("|---|---:|---:|---:|---:|");
    } else {
        println!(
            "\nbest-known value reached on {reached} of {with_bk} instances; mean gap by family:"
        );
    }
    for (family, gaps) in &families {
        let at_best = gaps.iter().filter(|&&g| g <= 0.0).count();
        let mean = gaps.iter().sum::<f64>() / gaps.len() as f64;
        let max = gaps.iter().cloned().fold(f64::MIN, f64::max);
        if opt.markdown {
            println!(
                "| {family} | {} | {at_best} | {mean:.2}% | {max:.2}% |",
                gaps.len()
            );
        } else {
            println!(
                "  {family:<5} {:>3} instances  mean {mean:>6.2}%  max {max:>6.2}%  best known reached on {at_best}",
                gaps.len()
            );
        }
    }
    if !improved.is_empty() {
        let list: Vec<String> = improved.iter().map(|r| r.name.clone()).collect();
        println!("\nbetter than the best-known value: {}", list.join(", "));
    }
}

/// Parses `--flag VALUE` options out of `args`, removing them.
pub struct Args(pub Vec<String>);

impl Args {
    pub fn from_env() -> Self {
        Args(std::env::args().skip(1).collect())
    }
    /// Removes `--flag` and returns true if it was present.
    pub fn flag(&mut self, flag: &str) -> bool {
        let before = self.0.len();
        self.0.retain(|a| a != flag);
        self.0.len() != before
    }
    /// Removes `--flag VALUE` and returns the parsed value.
    pub fn value<T: std::str::FromStr>(&mut self, flag: &str) -> Option<T> {
        let i = self.0.iter().position(|a| a == flag)?;
        let v = self.0.get(i + 1).cloned();
        self.0.drain(i..(i + 2).min(self.0.len()));
        Some(
            v?.parse()
                .ok()
                .unwrap_or_else(|| panic!("{flag} needs a value")),
        )
    }
}
