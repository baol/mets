//! Quadratic Assignment Problem: port of metslib-examples/qap (main_ts.cc and
//! main.cc), plus SA / LAHC / RRT and ILS on the same models.
//!
//! Two models of the same problem show where the delta trick lives:
//! * `QapPlain`: O(n) swap delta (like qap_model.hpp). Good for sampling
//!   methods that evaluate one move per iteration.
//! * `QapTaillard`: keeps Taillard's delta matrix, so a swap delta is O(1) and
//!   a full neighborhood scan costs O(n²) instead of O(n³). Each swap updates
//!   the matrix in O(n²). Good for tabu search on the full neighborhood.
//!
//! usage: qap [qaplib-file]                (default: random instance n=60)
//!        qap qaplib-file --bench ITERS    (same setup as the C++ benchmark)

use mets::perm::{perturb, shuffle, Permutation, SwapNeighborhood};
use mets::rand::seq::SliceRandom;
use mets::rand::{Rng, SeedableRng};
use mets::*;
use std::time::Instant;

type C = i64;

struct Instance {
    n: usize,
    a: Vec<C>, // flow, row-major n×n
    b: Vec<C>, // distance
}

impl Instance {
    fn cost_of(&self, p: &[usize]) -> C {
        let n = self.n;
        let mut s = 0;
        for i in 0..n {
            for j in 0..n {
                s += self.a[i * n + j] * self.b[p[i] * n + p[j]];
            }
        }
        s
    }

    /// O(n) delta of swapping positions r and s (valid for asymmetric instances).
    fn swap_delta(&self, p: &[usize], r: usize, s: usize) -> C {
        let n = self.n;
        let (pr, ps) = (p[r], p[s]);
        // row slices: bounds are checked once here, not in the loop
        let (ar, as_) = (&self.a[r * n..][..n], &self.a[s * n..][..n]);
        let (br, bs) = (&self.b[pr * n..][..n], &self.b[ps * n..][..n]);
        let mut d = ar[r] * (bs[ps] - br[pr])
            + ar[s] * (bs[pr] - br[ps])
            + as_[r] * (br[ps] - bs[pr])
            + as_[s] * (br[pr] - bs[ps]);
        for (k, &pk) in p.iter().enumerate() {
            if k == r || k == s {
                continue;
            }
            let bk = &self.b[pk * n..][..n];
            d += self.a[k * n + r] * (bk[ps] - bk[pr])
                + self.a[k * n + s] * (bk[pr] - bk[ps])
                + ar[k] * (bs[pk] - br[pk])
                + as_[k] * (br[pk] - bs[pk]);
        }
        d
    }

    fn random(n: usize, seed: u64) -> Self {
        let mut rng = DefaultRng::seed_from_u64(seed);
        let a = (0..n * n).map(|_| rng.random_range(0..100)).collect();
        let b = (0..n * n).map(|_| rng.random_range(0..100)).collect();
        Instance { n, a, b }
    }

    fn parse(text: &str) -> Self {
        let v: Vec<C> = text
            .split_whitespace()
            .map(|t| t.parse().expect("not a number"))
            .collect();
        let n = v[0] as usize;
        Instance {
            n,
            a: v[1..1 + n * n].to_vec(),
            b: v[1 + n * n..1 + 2 * n * n].to_vec(),
        }
    }
}

// ------------------------------------------------------------ plain model

#[derive(Clone)]
struct QapPlain<'a> {
    inst: &'a Instance,
    p: Vec<usize>,
}

impl Model for QapPlain<'_> {
    type Cost = C;
    fn evaluate(&self) -> C {
        self.inst.cost_of(&self.p)
    }
}

impl Permutation for QapPlain<'_> {
    fn size(&self) -> usize {
        self.p.len()
    }
    fn swap(&mut self, i: usize, j: usize) {
        self.p.swap(i, j);
    }
    fn swap_delta(&self, i: usize, j: usize) -> Option<C> {
        Some(self.inst.swap_delta(&self.p, i, j))
    }
}

// --------------------------------------------------------- Taillard model

#[derive(Clone)]
struct QapTaillard<'a> {
    inst: &'a Instance,
    p: Vec<usize>,
    delta: Vec<C>, // delta[r*n+s] for r<s
}

impl<'a> QapTaillard<'a> {
    fn new(inst: &'a Instance, p: Vec<usize>) -> Self {
        let n = inst.n;
        let mut delta = vec![0; n * n];
        for r in 0..n {
            for s in r + 1..n {
                delta[r * n + s] = inst.swap_delta(&p, r, s);
            }
        }
        QapTaillard { inst, p, delta }
    }
}

impl Model for QapTaillard<'_> {
    type Cost = C;
    fn evaluate(&self) -> C {
        self.inst.cost_of(&self.p)
    }
}

impl Permutation for QapTaillard<'_> {
    fn size(&self) -> usize {
        self.p.len()
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
        // Taillard's O(1) update for pairs disjoint from {u, v}, O(n) otherwise
        for r in 0..n {
            for s in r + 1..n {
                let d = &mut self.delta[r * n + s];
                if r == u || r == v || s == u || s == v {
                    *d = inst.swap_delta(p, r, s);
                } else {
                    let (pr, ps, pu, pv) = (p[r], p[s], p[u], p[v]);
                    *d += (a[r * n + u] - a[r * n + v] + a[s * n + v] - a[s * n + u])
                        * (b[ps * n + pu] - b[ps * n + pv] + b[pr * n + pv] - b[pr * n + pu])
                        + (a[u * n + r] - a[v * n + r] + a[v * n + s] - a[u * n + s])
                            * (b[pu * n + ps] - b[pv * n + ps] + b[pv * n + pr] - b[pu * n + pr]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- runs

fn report<S: Model<Cost = C>>(name: &str, st: &State<S>, t: Instant) {
    println!(
        "{name:<46} best {:>10}  {:>9} iters  {:>6.2}s",
        st.global_best().cost(),
        st.iteration(),
        t.elapsed().as_secs_f64()
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let inst = match args.first() {
        Some(f) => Instance::parse(&std::fs::read_to_string(f).expect("cannot read file")),
        None => Instance::random(60, 1),
    };
    if args.get(1).map(String::as_str) == Some("--bench") {
        let iters = args
            .get(2)
            .and_then(|s| s.parse().ok())
            .expect("--bench ITERS");
        return bench(&inst, iters);
    }
    let n = inst.n;

    // A random starting permutation, shared by every run.
    let mut st0 = State::new(
        QapPlain {
            inst: &inst,
            p: (0..n).collect(),
        },
        7,
    );
    st0.modify_current(shuffle);
    let start = st0.current().p.clone();
    println!("n = {n}, initial cost = {}", inst.cost_of(&start));

    // 1. tabu search, full swap neighborhood, Taillard model
    {
        let t = Instant::now();
        let mut st = State::new(QapTaillard::new(&inst, start.clone()), 1);
        let mut ts = TabuSearch::new(SwapNeighborhood, DenseTabu::new(Tenure::Fixed(n as u64)));
        ts.runner(&mut st).until(MaxIterations::new(5000)).run();
        report("tabu, full neighborhood, Taillard deltas", &st, t);
    }

    // 2. same search, plain model: same trajectory, O(n) slower per move
    {
        let t = Instant::now();
        let mut st = State::new(
            QapPlain {
                inst: &inst,
                p: start.clone(),
            },
            1,
        );
        let mut ts = TabuSearch::new(SwapNeighborhood, DenseTabu::new(Tenure::Fixed(n as u64)));
        ts.runner(&mut st).until(MaxIterations::new(5000)).run();
        report("tabu, full neighborhood, plain O(n) deltas", &st, t);
    }

    // 3. main_ts.cc: tabu search on n*sqrt(n) sampled swaps per iteration
    {
        let t = Instant::now();
        let k = (n as f64 * (n as f64).sqrt()) as usize;
        let mut st = State::new(
            QapPlain {
                inst: &inst,
                p: start.clone(),
            },
            1,
        );
        let mut ts = TabuSearch::new(
            Sampled::new(SwapNeighborhood, k),
            RecencyTabu::new(Tenure::Fixed(k as u64)),
        );
        ts.runner(&mut st).until(NoImprovement::new(1000)).run();
        report("tabu, sampled neighborhood (main_ts.cc)", &st, t);
    }

    // 4. main.cc: random restarts of an iterated tabu search with random tenure
    {
        let t = Instant::now();
        let ts = TabuSearch::new(
            SwapNeighborhood,
            DenseTabu::new(Tenure::Random(7..=7 * n as u64 / 4)),
        );
        let its = IteratedSearch::new(
            ts,
            NoImprovement::new(200),
            perturb_with(|s: &mut Evaluated<QapTaillard>, rng| {
                let k = rng.random_range(2..=n / 4);
                perturb(s, k, rng)
            }),
            Improving,
        );
        let mut restarts = IteratedSearch::new(
            its,
            NoImprovement::new(20),
            perturb_with(|s: &mut Evaluated<QapTaillard>, rng| {
                let mut p: Vec<usize> = (0..n).collect();
                p.shuffle(rng);
                s.modify(|q| *q = QapTaillard::new(q.inst, p));
            }),
            Always,
        );
        let mut st = State::new(QapTaillard::new(&inst, start.clone()), 1);
        let mut runs = 0;
        restarts
            .runner(&mut st)
            .until(MaxIterations::new(3))
            .on_event(|e, _| {
                if e == Event::Restart {
                    runs += 1
                }
            })
            .run();
        report(
            &format!("restarts(ITS(tabu)) (main.cc), {runs} runs"),
            &st,
            t,
        );
    }

    // 5. stochastic searches, chosen at runtime: only the acceptance rule differs
    let iters = 20_000_000 / n as u64;
    let t0 = inst.cost_of(&start) as f64 / 1000.0;
    let dev = inst.cost_of(&start) as f64 * 0.003;
    let algorithms: Vec<(&str, Box<dyn Search<QapPlain>>)> = vec![
        (
            "simulated annealing",
            Box::new(StochasticSearch::new(
                SwapNeighborhood,
                SimulatedAnnealing::new(t0, 1e-3, 0.95, iters / 300),
            )),
        ),
        (
            "late acceptance hill climbing",
            Box::new(StochasticSearch::new(
                SwapNeighborhood,
                LateAcceptance::new(5 * n),
            )),
        ),
        (
            "record-to-record travel",
            Box::new(StochasticSearch::new(
                SwapNeighborhood,
                RecordToRecord::new(dev),
            )),
        ),
    ];
    for (name, mut alg) in algorithms {
        let t = Instant::now();
        let mut st = State::new(
            QapPlain {
                inst: &inst,
                p: start.clone(),
            },
            1,
        );
        alg.runner(&mut st).until(MaxIterations::new(iters)).run();
        report(name, &st, t);
    }

    // 6. ILS around a steepest descent
    {
        let t = Instant::now();
        let mut st = State::new(QapTaillard::new(&inst, start.clone()), 1);
        let mut ils = IteratedSearch::new(
            LocalSearch::best_improvement(SwapNeighborhood),
            Unlimited, // each descent runs to its local optimum
            perturb_with(|s: &mut Evaluated<QapTaillard>, rng| perturb(s, n / 8, rng)),
            Improving,
        );
        ils.runner(&mut st).until(MaxIterations::new(2000)).run();
        report("ILS(steepest descent)", &st, t);
    }
}

/// Same setup as METSlib's tabu_search<swap_full_neighborhood> +
/// simple_tabu_list(n) + best_ever_criteria, starting from the identity.
fn bench(inst: &Instance, iters: u64) {
    fn go<S: Permutation<Cost = C>>(
        name: &str,
        s: S,
        tabu: impl TabuList<S, perm::Swap>,
        iters: u64,
    ) {
        let mut st = State::new(s, 1);
        let mut ts = TabuSearch::new(SwapNeighborhood, tabu);
        let t = Instant::now();
        ts.runner(&mut st).until(MaxIterations::new(iters)).run();
        let secs = t.elapsed().as_secs_f64();
        println!(
            "{name:<34} best {}  {iters} iters  {secs:.4} s  {:.2} us/iter",
            st.best().cost(),
            secs / iters as f64 * 1e6
        );
    }
    let n = inst.n;
    let id: Vec<usize> = (0..n).collect();
    let tenure = Tenure::Fixed(n as u64);
    go(
        "Rust plain    + RecencyTabu(hash)",
        QapPlain {
            inst,
            p: id.clone(),
        },
        RecencyTabu::new(tenure.clone()),
        iters,
    );
    go(
        "Rust plain    + DenseTabu",
        QapPlain {
            inst,
            p: id.clone(),
        },
        DenseTabu::new(tenure.clone()),
        iters,
    );
    go(
        "Rust Taillard + DenseTabu",
        QapTaillard::new(inst, id),
        DenseTabu::new(tenure),
        iters,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deltas_match_full_evaluation() {
        let inst = Instance::random(13, 5);
        let mut st = State::new(QapTaillard::new(&inst, (0..13).collect()), 1);
        for _ in 0..200 {
            st.modify_current(|s, rng| perturb(s, 1, rng));
            assert!(check::deltas(&mut SwapNeighborhood, st.current()).is_empty());
            let plain = Evaluated::new(QapPlain {
                inst: &inst,
                p: st.current().p.clone(),
            });
            assert!(check::deltas(&mut SwapNeighborhood, &plain).is_empty());
        }
    }
}
