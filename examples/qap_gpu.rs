//! QAP: a memetic algorithm whose children are improved in batches by
//! Taillard's robust tabu search, on the GPU (`--features gpu`, one CUDA block
//! per individual) or on the CPU threads (`--cpu`, the library's
//! `perm::RobustTabuSearch` on the delta-matrix model).
//!
//! A robust tabu iteration is n²/2 independent pieces of work (scan every
//! swap, then update every entry of the delta matrix), so the block's threads
//! share it: unlike TabuCol, one iteration is large enough to keep a GPU busy.
//!
//! Each generation, every member i has a child: a uniform crossover of i and
//! a random other member, improved by `--iters`·n tabu iterations. The child
//! replaces i if it is better (differential-evolution one-to-one selection,
//! as `Memetic::replace_parent`): every member keeps its own lineage.
//!
//! usage: qap_gpu FILE.dat [FILE.sln] [options]
//!   --population P   members (default 224 on the GPU, the core count with --cpu)
//!   --iters L        tabu iterations per child, in units of n (default 100)
//!   --max-seconds T  give up after T seconds (default 3600)
//!   --run-to-end     do not stop at the best-known value
//!   --stagnation G   a member whose children have not beaten it for G generations
//!                    restarts from a random permutation (default 0: never)
//!   --seed S         (default 1)
//!   --threads T      GPU threads per block (default 256)
//!   --cpu            improve on the CPU threads instead of the GPU
//!   --trace FILE     append one line per batch: time, cost spread, and a CPU
//!                    recount of a few returned permutations (mismatches = kernel bug)
//! The best permutation found is saved as NAME.best (cost, then the 1-based
//! facility of each location).
//!
//! Build and run as `coloring_gpu` (driver and NVRTC at run time):
//!   cargo run --release --features gpu --example qap_gpu -- tai100a.dat tai100a.sln --trace qap.trace

#[path = "common/qap.rs"]
mod common;

use common::*;
use mets::perm::UniformCrossover;
use mets::rand::{Rng, SeedableRng};
use mets::*;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

/// Improves a batch of permutations in place: each becomes the best one its
/// tabu run found. Returns their costs.
trait Improver {
    fn improve(&mut self, batch: &mut [Vec<usize>], iters: u64, seed: u64) -> Vec<C>;
    fn name(&self) -> String;
}

/// The library's robust tabu search on the CPU threads.
struct CpuRots<'a> {
    inst: &'a Instance,
    jobs: usize,
}

impl Improver for CpuRots<'_> {
    fn improve(&mut self, batch: &mut [Vec<usize>], iters: u64, seed: u64) -> Vec<C> {
        let mut costs = vec![0; batch.len()];
        let chunk = batch.len().div_ceil(self.jobs.max(1)).max(1);
        std::thread::scope(|scope| {
            for (c, (perms, out)) in batch
                .chunks_mut(chunk)
                .zip(costs.chunks_mut(chunk))
                .enumerate()
            {
                let inst = self.inst;
                scope.spawn(move || {
                    for (i, (p, cost)) in perms.iter_mut().zip(out).enumerate() {
                        let run_seed = seed.wrapping_mul(1 << 20) + (c * chunk + i) as u64;
                        let mut st = State::new(QapTaillard::new(inst, p.clone()), run_seed);
                        perm::RobustTabuSearch::new()
                            .runner(&mut st)
                            .until(MaxIterations::new(iters))
                            .run();
                        p.clone_from(&st.global_best().p);
                        *cost = st.global_best().cost();
                    }
                });
            }
        });
        costs
    }
    fn name(&self) -> String {
        format!("CPU, {} threads", self.jobs)
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::{Improver, Instance, C};
    use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
    use std::sync::Arc;

    /// Robust tabu search, one block per individual: the threads share the
    /// scan of the n(n−1)/2 swaps (a warp and a block minimum of packed keys
    /// (not aspired, delta, swap index): aspired moves first, then the best
    /// authorized one) and the update of the delta matrix. Taillard's O(1)
    /// update factors into per-position differences (ca, cb, ra, rb),
    /// computed once per swap in shared memory; pairs touching the swapped
    /// positions are recomputed in O(n).
    const KERNEL: &str = r#"
typedef long long ll;
typedef unsigned long long u64;
__device__ __forceinline__ unsigned int xs(unsigned int &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}

// delta of swapping positions r and s of p (general, asymmetric formula),
// summed by the 32 lanes of a warp; the result is valid on every lane.
// Columns of a and b are read as rows of their transposes at and bt.
__device__ ll warp_delta(const ll* __restrict__ a, const ll* __restrict__ b,
                         const ll* __restrict__ at, const ll* __restrict__ bt,
                         const int* p, int n, int r, int s, int lane) {
    const int pr = p[r], ps = p[s];
    ll d = 0;
    for (int k = lane; k < n; k += 32) {
        if (k == r || k == s) continue;
        const int pk = p[k];
        d += (at[r * n + k] - at[s * n + k]) * (bt[ps * n + pk] - bt[pr * n + pk])
           + (a[r * n + k] - a[s * n + k]) * (b[ps * n + pk] - b[pr * n + pk]);
    }
    for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
    return d + a[r * n + r] * (b[ps * n + ps] - b[pr * n + pr])
             + a[r * n + s] * (b[ps * n + pr] - b[pr * n + ps])
             + a[s * n + r] * (b[pr * n + ps] - b[ps * n + pr])
             + a[s * n + s] * (b[pr * n + pr] - b[ps * n + ps]);
}

extern "C" __global__ void rots(
    const ll* __restrict__ a, const ll* __restrict__ b,
    const ll* __restrict__ at, const ll* __restrict__ bt, int n,
    int* best_perm, ll* delta_all, int* tabu_all, ll* out_cost,
    long long iters, unsigned int seed)
{
    const int blk = blockIdx.x, t = threadIdx.x, T = blockDim.x;
    const int lane = t & 31, warp = t >> 5, warps = T >> 5;
    const int nn = n * n;
    extern __shared__ __align__(8) unsigned char smem[];
    ll* ca = (ll*)smem; ll* cb = ca + n; ll* ra = cb + n; ll* rb = ra + n;
    int* p = (int*)(rb + n);
    int* best = best_perm + (size_t)blk * n;
    ll* D = delta_all + (size_t)blk * nn;     // D[i*n + j], i < j
    int* tb = tabu_all + (size_t)blk * nn;    // tb[i*n + e]: element e may return to i after it
    __shared__ u64 sel;
    __shared__ ll cur, bestc;
    __shared__ int tenure;
    unsigned int rng = seed ^ (0x9E3779B9u * (unsigned)(blk + 1));
    if (rng == 0) rng = 1;
    for (int i = 0; i < 4; i++) xs(rng);
    const int lo = max(1, (int)(0.9 * n)), hi = max(lo, (int)(1.1 * n));
    const ll asp = 5LL * nn;  // long-term aspiration age

    for (int v = t; v < n; v += T) p[v] = best[v];
    for (int q = t; q < nn; q += T) tb[q] = -q;  // staggered "never"
    if (t == 0) cur = 0;
    __syncthreads();
    {
        ll local = 0;
        for (int q = t; q < nn; q += T) {
            const int i = q / n, j = q - i * n;
            local += a[q] * b[p[i] * n + p[j]];
        }
        atomicAdd((u64*)&cur, (u64)local);
        // the delta matrix, one pair per warp
        for (int q = warp; q < nn; q += warps) {
            const int i = q / n, j = q - i * n;
            if (i >= j) continue;
            const ll d = warp_delta(a, b, at, bt, p, n, i, j, lane);
            if (lane == 0) D[q] = d;
        }
    }
    __syncthreads();
    if (t == 0) { bestc = cur; tenure = lo + (int)(xs(rng) % (unsigned)(hi - lo + 1)); }
    __syncthreads();

    for (ll it = 1; it <= iters; it++) {
        if (t == 0) {
            sel = ~0ull;
            if (it % (2 * hi) == 0) tenure = lo + (int)(xs(rng) % (unsigned)(hi - lo + 1));
        }
        __syncthreads();
        u64 local = ~0ull;
        const ll c0 = cur, bc = bestc;
        for (int q = t; q < nn; q += T) {
            const int i = q / n, j = q - i * n;
            if (j <= i) continue;
            const ll d = D[q];
            const int ti = tb[i * n + p[j]], tj = tb[j * n + p[i]];  // p[j] back to i, p[i] back to j
            const bool aspired = ti < it - asp || tj < it - asp || c0 + d < bc;
            const bool authorized = ti < it || tj < it;
            if (!aspired && !authorized) continue;
            const u64 key = ((u64)(aspired ? 0 : 1) << 63)
                          | ((u64)(d + (1LL << 39)) << 23)
                          | (u64)q;
            if (key < local) local = key;
        }
        for (int o = 16; o > 0; o >>= 1) {
            const u64 x = __shfl_down_sync(0xffffffffu, local, o);
            if (x < local) local = x;
        }
        if ((t & 31) == 0) atomicMin(&sel, local);
        __syncthreads();
        const u64 s = sel;
        if (s == ~0ull) break;  // every move tabu and none aspired
        const int q = (int)(s & ((1u << 23) - 1)), u = q / n, v = q - u * n;
        if (t == 0) {
            tb[u * n + p[u]] = (int)(it + tenure);  // p[u] leaves u
            tb[v * n + p[v]] = (int)(it + tenure);  // p[v] leaves v
            const int x = p[u]; p[u] = p[v]; p[v] = x;
            cur += D[q];
        }
        __syncthreads();
        // per-position differences of the swap (p is already swapped)
        const int pu = p[u], pv = p[v];
        for (int k = t; k < n; k += T) {
            const int pk = p[k];
            ca[k] = a[k * n + u] - a[k * n + v];
            cb[k] = b[pk * n + pu] - b[pk * n + pv];
            ra[k] = a[u * n + k] - a[v * n + k];
            rb[k] = b[pu * n + pk] - b[pv * n + pk];
        }
        __syncthreads();
        // Taillard's O(1) update of the pairs disjoint from {u, v} ...
        for (int q2 = t; q2 < nn; q2 += T) {
            const int i = q2 / n, j = q2 - i * n;
            if (j <= i || i == u || i == v || j == u || j == v) continue;
            D[q2] += (ca[i] - ca[j]) * (cb[j] - cb[i]) + (ra[i] - ra[j]) * (rb[j] - rb[i]);
        }
        // ... and the 2n - 3 pairs touching u or v from scratch, one per warp:
        // slots m < n - 1 are the pairs (u, k), k != u; slots n - 1 .. 2n - 3
        // the pairs (v, k), k != v, skipping (v, u), already in the first part
        for (int m = warp; m < 2 * n - 2; m += warps) {
            int x = u, k = m;
            if (m >= n - 1) { x = v; k = m - (n - 1); }
            if (k >= x) k++;                // skip x itself
            if (x == v && k == u) continue;
            const int i = min(x, k), j = max(x, k);
            const ll d = warp_delta(a, b, at, bt, p, n, i, j, lane);
            if (lane == 0) D[i * n + j] = d;
        }
        __syncthreads();
        const bool improved = cur < bestc;
        __syncthreads();
        if (improved) {
            for (int w = t; w < n; w += T) best[w] = p[w];
            if (t == 0) bestc = cur;
        }
        __syncthreads();
    }
    if (t == 0) out_cost[blk] = bestc;
}
"#;

    pub struct GpuRots {
        stream: Arc<CudaStream>,
        f: CudaFunction,
        a: CudaSlice<i64>,
        b: CudaSlice<i64>,
        at: CudaSlice<i64>,
        bt: CudaSlice<i64>,
        best: CudaSlice<i32>,
        delta: CudaSlice<i64>,
        tabu: CudaSlice<i32>,
        cost: CudaSlice<i64>,
        n: usize,
        p: usize,
        threads: u32,
        shared_bytes: u32,
        resident: String,
        host: Vec<i32>,
        device: String,
    }

    type Error = Box<dyn std::error::Error>;

    impl GpuRots {
        /// Buffers for a batch of `p` permutations of `inst` on GPU 0.
        pub fn new(inst: &Instance, p: usize, threads: u32) -> Result<Self, Error> {
            let n = inst.n;
            assert!(n * n < 1 << 23, "swap index must fit 23 bits");
            assert!(threads.is_multiple_of(32), "whole warps");
            let ctx = cudarc::driver::CudaContext::new(0)?;
            let device = ctx.name()?;
            let stream = ctx.default_stream();
            let ptx = cudarc::nvrtc::compile_ptx(KERNEL)?;
            let f = ctx.load_module(ptx)?.load_function("rots")?;
            let shared_bytes = (36 * n) as u32; // ca, cb, ra, rb (i64) and p (i32)
            let blocks_per_sm = f.occupancy_max_active_blocks_per_multiprocessor(
                threads,
                shared_bytes as usize,
                None,
            )?;
            let sms = ctx.attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )?;
            Ok(GpuRots {
                a: stream.clone_htod(&inst.a)?,
                b: stream.clone_htod(&inst.b)?,
                at: stream.clone_htod(&inst.at)?,
                bt: stream.clone_htod(&inst.bt)?,
                best: stream.alloc_zeros(p * n)?,
                delta: stream.alloc_zeros(p * n * n)?,
                tabu: stream.alloc_zeros(p * n * n)?,
                cost: stream.alloc_zeros(p)?,
                stream,
                f,
                n,
                p,
                threads,
                shared_bytes,
                resident: format!("{blocks_per_sm} blocks per SM × {sms} SMs"),
                host: vec![0; p * n],
                device,
            })
        }

        fn run(
            &mut self,
            batch: &mut [Vec<usize>],
            iters: u64,
            seed: u64,
        ) -> Result<Vec<C>, Error> {
            assert_eq!(batch.len(), self.p);
            let n = self.n;
            for (i, perm) in batch.iter().enumerate() {
                for (h, &x) in self.host[i * n..][..n].iter_mut().zip(perm) {
                    *h = x as i32;
                }
            }
            self.stream.memcpy_htod(&self.host, &mut self.best)?;
            let (n32, iters, seed) = (n as i32, iters as i64, seed as u32);
            let mut launch = self.stream.launch_builder(&self.f);
            launch
                .arg(&self.a)
                .arg(&self.b)
                .arg(&self.at)
                .arg(&self.bt)
                .arg(&n32)
                .arg(&mut self.best)
                .arg(&mut self.delta)
                .arg(&mut self.tabu)
                .arg(&mut self.cost)
                .arg(&iters)
                .arg(&seed);
            let cfg = LaunchConfig {
                grid_dim: (self.p as u32, 1, 1),
                block_dim: (self.threads, 1, 1),
                shared_mem_bytes: self.shared_bytes,
            };
            unsafe { launch.launch(cfg) }?;
            self.stream.memcpy_dtoh(&self.best, &mut self.host)?;
            let costs = self.stream.clone_dtoh(&self.cost)?;
            for (i, perm) in batch.iter_mut().enumerate() {
                for (x, &h) in perm.iter_mut().zip(&self.host[i * n..][..n]) {
                    *x = h as usize;
                }
            }
            Ok(costs)
        }
    }

    impl Improver for GpuRots {
        fn improve(&mut self, batch: &mut [Vec<usize>], iters: u64, seed: u64) -> Vec<C> {
            self.run(batch, iters, seed)
                .expect("GPU robust tabu search")
        }
        fn name(&self) -> String {
            format!(
                "GPU {}, {} threads per block, {}",
                self.device, self.threads, self.resident
            )
        }
    }
}

/// Minimal traces for debugging runs on another machine (`--trace FILE`).
struct Trace(Option<std::fs::File>);

impl Trace {
    fn line(&mut self, text: std::fmt::Arguments) {
        if let Some(f) = &mut self.0 {
            let _ = writeln!(f, "{text}");
        }
    }

    /// One batch: time, cost spread, and a recount of up to 8 permutations.
    fn batch(
        &mut self,
        generation: u64,
        inst: &Instance,
        batch: &[Vec<usize>],
        costs: &[C],
        secs: f64,
    ) {
        if self.0.is_none() {
            return;
        }
        let mut sorted = costs.to_vec();
        sorted.sort_unstable();
        let (mut checked, mut mismatches, mut bad) = (0, 0, 0);
        for (p, &cost) in batch.iter().zip(costs).take(8) {
            checked += 1;
            if !is_permutation(p, inst.n) {
                bad += 1;
                continue;
            }
            let recount = inst.cost_of(p);
            if recount != cost {
                mismatches += 1;
                self.line(format_args!(
                    "  mismatch: reported {cost}, recount {recount}"
                ));
            }
        }
        self.line(format_args!(
            "gen={generation} improve_s={secs:.3} min={} med={} max={} checked={checked} mismatches={mismatches} bad_permutations={bad}",
            sorted[0],
            sorted[sorted.len() / 2],
            sorted[sorted.len() - 1]
        ));
    }
}

fn is_permutation(p: &[usize], n: usize) -> bool {
    let mut seen = vec![false; n];
    p.len() == n
        && p.iter()
            .all(|&x| x < n && !std::mem::replace(&mut seen[x], true))
}

fn main() {
    let mut args = Args::from_env();
    let cpu = args.flag("--cpu");
    let run_to_end = args.flag("--run-to-end");
    let jobs = std::thread::available_parallelism().map_or(1, |n| n.get());
    let population: usize = args
        .value("--population")
        .unwrap_or(if cpu { jobs } else { 224 });
    let iters_n: u64 = args.value("--iters").unwrap_or(100);
    let stagnation: u64 = args.value("--stagnation").unwrap_or(0);
    let max_seconds: f64 = args.value("--max-seconds").unwrap_or(3600.0);
    let seed: u64 = args.value("--seed").unwrap_or(1);
    let threads: u32 = args.value("--threads").unwrap_or(256);
    let trace_file: Option<String> = args.value("--trace");
    let mut trace = Trace(trace_file.map(|f| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&f)
            .unwrap_or_else(|e| panic!("{f}: {e}"))
    }));
    let usage = "usage: qap_gpu FILE.dat [FILE.sln] [options]";
    let file = args.0.first().expect(usage).clone();
    let inst = Instance::read(Path::new(&file)).unwrap_or_else(|e| panic!("{file}: {e}"));
    let best_known = args.0.get(1).map(|s| {
        read_sln(Path::new(s), &inst)
            .unwrap_or_else(|e| panic!("{s}: {e}"))
            .value
    });
    assert!(population >= 2, "a population needs two members");
    let n = inst.n;
    let iters = iters_n * n as u64;

    let setup = Instant::now();
    let mut improver: Box<dyn Improver> = if cpu {
        Box::new(CpuRots { inst: &inst, jobs })
    } else {
        gpu_improver(&inst, population, threads)
    };
    trace.line(format_args!(
        "start build={} file={file} n={n} population={population} iters={iters} threads={threads} seed={seed} best_known={best_known:?} improver=\"{}\" setup_s={:.3} (includes NVRTC compile)",
        env!("METS_GIT_HASH"),
        improver.name(),
        setup.elapsed().as_secs_f64(),
    ));
    println!(
        "{file}: n = {n}; population {population} × {iters_n}n robust tabu iterations per child; {}; build {}",
        improver.name(),
        env!("METS_GIT_HASH")
    );

    let t0 = Instant::now();
    let mut rng = DefaultRng::seed_from_u64(seed);
    let mut pop: Vec<Vec<usize>> = (0..population)
        .map(|i| random_start(n, seed.wrapping_mul(1000) + i as u64))
        .collect();
    let t = Instant::now();
    let mut costs = improver.improve(&mut pop, iters, rng.random());
    trace.batch(0, &inst, &pop, &costs, t.elapsed().as_secs_f64());
    let mut best = (C::MAX, Vec::new());
    let mut children = pop.clone();
    let mut cross = UniformCrossover;
    let mut since = vec![0u64; population]; // generations since member i improved
    let mut restarts = 0u64;
    for generation in 1.. {
        if let Some(i) = (0..population).min_by_key(|&i| costs[i]) {
            if costs[i] < best.0 {
                best = (costs[i], pop[i].clone());
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        let reached = best_known.is_some_and(|b| best.0 <= b);
        if generation % 10 == 1 || reached {
            let g = best_known.map_or(String::new(), |b| format!(" (gap {:.3}%)", gap(best.0, b)));
            println!(
                "  generation {generation:5}: best {}{g}, {secs:7.1} s, {restarts} restarts",
                best.0
            );
        }
        if (reached && !run_to_end) || secs > max_seconds {
            break;
        }
        // each member's child with a random other member
        for i in 0..population {
            if stagnation > 0 && since[i] >= stagnation {
                // stagnant: this member's next child is a fresh random start
                children[i] = random_start(n, rng.random());
                (costs[i], since[i]) = (C::MAX, 0);
                restarts += 1;
                continue;
            }
            let mut j = rng.random_range(0..population - 1);
            if j >= i {
                j += 1;
            }
            let (a, b) = (
                QapPlain {
                    inst: &inst,
                    p: pop[i].clone(),
                },
                QapPlain {
                    inst: &inst,
                    p: pop[j].clone(),
                },
            );
            let mut child = QapPlain {
                inst: &inst,
                p: pop[i].clone(),
            };
            cross.cross(&a, &b, &mut child, &mut rng);
            children[i] = child.p;
        }
        let t = Instant::now();
        let child_costs = improver.improve(&mut children, iters, rng.random());
        trace.batch(
            generation,
            &inst,
            &children,
            &child_costs,
            t.elapsed().as_secs_f64(),
        );
        // one-to-one selection: a child replaces its first parent if better
        for i in 0..population {
            if child_costs[i] < costs[i] {
                std::mem::swap(&mut pop[i], &mut children[i]);
                (costs[i], since[i]) = (child_costs[i], 0);
            } else {
                since[i] += 1;
            }
        }
    }
    trace.line(format_args!(
        "end best={} total_s={:.1}",
        best.0,
        t0.elapsed().as_secs_f64()
    ));
    assert!(
        is_permutation(&best.1, n) && inst.cost_of(&best.1) == best.0,
        "verified"
    );
    let name = Path::new(&file)
        .file_stem()
        .map_or("qap".into(), |s| s.to_string_lossy());
    let out = format!("{name}.best");
    let perm: Vec<String> = best.1.iter().map(|&x| (x + 1).to_string()).collect();
    std::fs::write(&out, format!("{}\n{}\n", best.0, perm.join(" ")))
        .unwrap_or_else(|e| panic!("{out}: {e}"));
    let g = best_known.map_or(String::new(), |b| {
        format!(", best known {b} (gap {:.3}%)", gap(best.0, b))
    });
    println!(
        "{file}: best {}{g} in {:.1} s, saved to {out}",
        best.0,
        t0.elapsed().as_secs_f64()
    );
}

#[cfg(feature = "gpu")]
fn gpu_improver<'a>(inst: &'a Instance, p: usize, threads: u32) -> Box<dyn Improver + 'a> {
    Box::new(gpu::GpuRots::new(inst, p, threads).unwrap_or_else(|e| panic!("GPU: {e}")))
}

#[cfg(not(feature = "gpu"))]
fn gpu_improver<'a>(_: &'a Instance, _: usize, _: u32) -> Box<dyn Improver + 'a> {
    panic!("built without the GPU: rebuild with --features gpu, or run with --cpu")
}
