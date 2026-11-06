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
//!   --crossover X    uniform | cohesive (Drezner: one parent's elements on a ball of
//!                    close positions; uniform without a distance matrix) (default uniform)
//!   --independent    Q6 arm B: no crossover; every member is an independent robust tabu
//!                    trajectory, each batch continuing where the last stopped, with its
//!                    tabu memory (on the GPU and on the CPU path)
//!   --perturb-stalled  Q6 arm C (implies --independent): after a batch that did not
//!                    improve a member's best, k random swaps on it, k uniform in [n/10, n/4]
//!   --tenure LO,HI   tabu tenure drawn in [LO·n, HI·n] (default 0.9,1.1)
//!   --aspiration X   long-term aspiration after X·n² iterations (default 5)
//!   --stagnation G   a member whose children have not beaten it for G generations
//!                    restarts from a random permutation (default 0: never)
//!   --seed S         (default 1)
//!   --threads T      GPU threads per block (default 256)
//!   --cpu            improve on the CPU threads instead of the GPU
//!   --trace FILE     append one line per batch: time, cost spread, and a CPU
//!                    recount of a few returned permutations (mismatches = kernel bug)
//!   --kopt K         at the end, polish the best permutation by exhaustive k-opt:
//!                    the best of all K! rearrangements of each of many groups of
//!                    K positions, applied while one improves (default 0: off; K <= 12)
//!   --kopt-groups G  groups per round (default 1024 on the GPU, 64 with --cpu)
//!   --kopt-rounds R  stop after R rounds without improvement (default 20)
//!   --polish-from FILE.best  skip the search: k-opt polish this permutation
//!                    (default K = 9), save the result as NAME.kopt.best
//! The best permutation found is saved as NAME.best (cost, then the 1-based
//! facility of each location), rewritten at every improvement.
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
/// Robust tabu search parameters (Taillard): the tenure is drawn in
/// [tenure.0·n, tenure.1·n] every 2·tenure.1·n iterations; a swap that puts
/// back an assignment unseen for aspiration·n² iterations is aspired.
#[derive(Clone, Copy, Debug)]
pub struct RotsParams {
    pub tenure: (f64, f64),
    pub aspiration: f64,
}

impl Default for RotsParams {
    fn default() -> Self {
        RotsParams {
            tenure: (0.9, 1.1),
            aspiration: 5.0,
        }
    }
}

/// Runs robust tabu search on a batch of permutations.
trait Improver {
    /// Continue each member: `iters` iterations from `cur[i]`, after which
    /// `cur[i]` is the final position; `best[i]` and `best_cost[i]` are
    /// replaced when the run beats `best_cost[i]`. With `resume`, a member
    /// keeps its tabu memory from the previous call, whose iterations ended
    /// at `it0` (iteration count and tabu table carry over, on the GPU and
    /// on the CPU path alike).
    #[allow(clippy::too_many_arguments)]
    fn advance(
        &mut self,
        cur: &mut [Vec<usize>],
        best: &mut [Vec<usize>],
        best_cost: &mut [C],
        iters: u64,
        it0: u64,
        resume: bool,
        seed: u64,
    );

    /// Improve each start in place: it becomes the best its run found.
    /// Returns the costs.
    fn improve(&mut self, batch: &mut [Vec<usize>], iters: u64, seed: u64) -> Vec<C> {
        let mut cur = batch.to_vec();
        let mut cost = vec![C::MAX; batch.len()];
        self.advance(&mut cur, batch, &mut cost, iters, 0, false, seed);
        cost
    }

    fn name(&self) -> String;
}

/// The library's robust tabu search on the CPU threads. Each member keeps
/// its `State` and its `RobustTabuSearch` between calls, so a resumed call
/// continues the same trajectory with the same tabu memory, as the kernel
/// does.
struct CpuRots<'a> {
    inst: &'a Instance,
    jobs: usize,
    params: RotsParams,
    members: Vec<Option<Member<'a>>>,
}

type Member<'a> = (State<QapTaillard<'a>>, perm::RobustTabuSearch);

impl<'a> CpuRots<'a> {
    fn new(inst: &'a Instance, jobs: usize, params: RotsParams) -> Self {
        CpuRots {
            inst,
            jobs,
            params,
            members: Vec::new(),
        }
    }
}

impl Improver for CpuRots<'_> {
    fn advance(
        &mut self,
        cur: &mut [Vec<usize>],
        best: &mut [Vec<usize>],
        best_cost: &mut [C],
        iters: u64,
        _it0: u64,
        resume: bool,
        seed: u64,
    ) {
        let chunk = cur.len().div_ceil(self.jobs.max(1)).max(1);
        let (inst, params) = (self.inst, self.params);
        self.members.resize_with(cur.len(), || None);
        std::thread::scope(|scope| {
            for (c, (((cs, bs), costs), ms)) in cur
                .chunks_mut(chunk)
                .zip(best.chunks_mut(chunk))
                .zip(best_cost.chunks_mut(chunk))
                .zip(self.members.chunks_mut(chunk))
                .enumerate()
            {
                scope.spawn(move || {
                    for (i, (((p, b), cost), m)) in
                        cs.iter_mut().zip(bs).zip(costs).zip(ms).enumerate()
                    {
                        let run_seed = seed.wrapping_mul(1 << 20) + (c * chunk + i) as u64;
                        let continuing = resume && m.is_some();
                        if !continuing {
                            let mut rots = perm::RobustTabuSearch::new();
                            (rots.tenure, rots.aspiration) = (params.tenure, params.aspiration);
                            *m = Some((
                                State::new(QapTaillard::new(inst, p.clone()), run_seed),
                                rots,
                            ));
                        }
                        let (st, rots) = m.as_mut().expect("member");
                        if continuing && st.current().p != *p {
                            // the host moved this member (a perturbation): same tabu memory
                            st.modify_current(|s, _| {
                                *s = Evaluated::new(QapTaillard::new(inst, p.clone()))
                            });
                        }
                        let runner = rots.runner(st).until(MaxIterations::new(iters));
                        if continuing {
                            runner.resume();
                        } else {
                            runner.run();
                        }
                        p.clone_from(&st.current().p);
                        if st.global_best().cost() < *cost {
                            *cost = st.global_best().cost();
                            b.clone_from(&st.global_best().p);
                        }
                    }
                });
            }
        });
    }
    fn name(&self) -> String {
        format!("CPU, {} threads", self.jobs)
    }
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::{unrank, Improver, Instance, KOpt, RotsParams, C};
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
                         const int* p, int n, int r, int s, int lane, int symmetric) {
    const int pr = p[r], ps = p[s];
    ll d = 0;
    if (symmetric) {
        // symmetric matrices, zero diagonal: 2 Σ (a[r][k] − a[s][k]) (b[ps][pk] − b[pr][pk])
        for (int k = lane; k < n; k += 32) {
            if (k == r || k == s) continue;
            const int pk = p[k];
            d += (a[r * n + k] - a[s * n + k]) * (b[ps * n + pk] - b[pr * n + pk]);
        }
        for (int o = 16; o > 0; o >>= 1) d += __shfl_xor_sync(0xffffffffu, d, o);
        return 2 * d;
    }
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
    int* best_perm, int* cur_perm, const ll* best_in, ll* delta_all, int* tabu_all,
    ll* out_cost, long long iters, long long it0, int resume, double tenure_lo,
    double tenure_hi, double aspiration, unsigned int seed, int use_shared, int symmetric)
{
    const int blk = blockIdx.x, t = threadIdx.x, T = blockDim.x;
    const int lane = t & 31, warp = t >> 5, warps = T >> 5;
    const int nn = n * n;
    extern __shared__ __align__(8) unsigned char smem[];
    // shared: ca, cb, ra, rb (n each), the delta matrix as a triangle of
    // n(n-1)/2 entries when it fits (use_shared), then p
    const int tri = n * (n - 1) / 2;
    ll* ca = (ll*)smem; ll* cb = ca + n; ll* ra = cb + n; ll* rb = ra + n;
    ll* Dt = rb + n;
    int* p = (int*)(Dt + (use_shared ? tri : 0));
    int* best = best_perm + (size_t)blk * n;
    // D[TRI(i, j)], i < j: the swap deltas, a triangle in shared or global memory
    ll* D = use_shared ? Dt : delta_all + (size_t)blk * tri;
#define TRI(i, j) ((i) * (2 * n - (i) - 1) / 2 + (j) - (i) - 1)
    int* tb = tabu_all + (size_t)blk * nn;    // tb[i*n + e]: element e may return to i after it
    __shared__ u64 sel;
    __shared__ ll cur, bestc;
    __shared__ int tenure;
    unsigned int rng = seed ^ (0x9E3779B9u * (unsigned)(blk + 1));
    if (rng == 0) rng = 1;
    for (int i = 0; i < 4; i++) xs(rng);
    const int lo = max(1, (int)(tenure_lo * n)), hi = max(lo, (int)(tenure_hi * n));
    const ll asp = (ll)(aspiration * nn);  // long-term aspiration age
    int* curp = cur_perm + (size_t)blk * n;

    // start from the member's current position; keep its tabu memory with resume
    for (int v = t; v < n; v += T) p[v] = curp[v];
    if (!resume)
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
            const ll d = warp_delta(a, b, at, bt, p, n, i, j, lane, symmetric);
            if (lane == 0) D[TRI(i, j)] = d;
        }
    }
    __syncthreads();
    // the member's best so far, or the start if it is better
    const bool start_best = cur < best_in[blk];
    if (start_best)
        for (int w = t; w < n; w += T) best[w] = p[w];
    if (t == 0) {
        bestc = start_best ? cur : best_in[blk];
        tenure = lo + (int)(xs(rng) % (unsigned)(hi - lo + 1));
    }
    __syncthreads();

    for (ll it = it0 + 1; it <= it0 + iters; it++) {
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
            const ll d = D[TRI(i, j)];
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
            cur += D[TRI(u, v)];
        }
        __syncthreads();
        // per-position differences of the swap (p is already swapped)
        const int pu = p[u], pv = p[v];
        for (int k = t; k < n; k += T) {
            const int pk = p[k];
            ca[k] = a[k * n + u] - a[k * n + v];
            cb[k] = b[pk * n + pu] - b[pk * n + pv];
            if (!symmetric) {
                ra[k] = a[u * n + k] - a[v * n + k];
                rb[k] = b[pu * n + pk] - b[pv * n + pk];
            }
        }
        __syncthreads();
        // Taillard's O(1) update of the pairs disjoint from {u, v} ...
        for (int q2 = t; q2 < nn; q2 += T) {
            const int i = q2 / n, j = q2 - i * n;
            if (j <= i || i == u || i == v || j == u || j == v) continue;
            const ll f = (ca[i] - ca[j]) * (cb[j] - cb[i]);
            D[TRI(i, j)] += symmetric ? 2 * f : f + (ra[i] - ra[j]) * (rb[j] - rb[i]);
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
            const ll d = warp_delta(a, b, at, bt, p, n, i, j, lane, symmetric);
            if (lane == 0) D[TRI(i, j)] = d;
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
    for (int v = t; v < n; v += T) curp[v] = p[v];
    if (t == 0) out_cost[blk] = bestc;
#undef TRI
}
"#;

    /// Exhaustive k-opt, one block per group of k positions: the threads
    /// compute the group's row terms (see `GroupTerms`), then share the k!
    /// rearrangement ranks; improving ones are packed as (delta + 2^34) above
    /// a 29-bit rank, so the block minimum is the best improvement.
    const KOPT_KERNEL: &str = r#"
typedef long long ll;
typedef unsigned long long u64;
extern "C" __global__ void kopt(
    const ll* __restrict__ a, const ll* __restrict__ b, int n, int k,
    const int* __restrict__ p, const int* __restrict__ groups, u64* out, long long total)
{
    const int g = blockIdx.x, t = threadIdx.x, T = blockDim.x;
    __shared__ int S[12], e[12];
    __shared__ ll R[144], A[144], B[144];
    __shared__ u64 sel;
    __shared__ ll base;
    if (t < k) { S[t] = groups[g * k + t]; e[t] = p[S[t]]; }
    if (t == 0) sel = ~0ull;
    __syncthreads();
    for (int q = t; q < k * k; q += T) {
        const int tt = q / k, x = q % k;
        ll sum = 0;
        for (int j = 0; j < n; j++) {
            bool inside = false;
            for (int m = 0; m < k; m++) inside |= (S[m] == j);
            if (inside) continue;
            sum += a[S[tt] * n + j] * b[e[x] * n + p[j]] + a[j * n + S[tt]] * b[p[j] * n + e[x]];
        }
        R[q] = sum;
        A[q] = a[S[tt] * n + S[x]];
        B[q] = b[e[tt] * n + e[x]];
    }
    __syncthreads();
    if (t == 0) {
        ll v = 0;
        for (int i = 0; i < k; i++) {
            v += R[i * k + i];
            for (int u = 0; u < k; u++) v += A[i * k + u] * B[i * k + u];
        }
        base = v;
    }
    __syncthreads();
    ll fact[13];
    fact[0] = 1;
    for (int i = 1; i <= 12; i++) fact[i] = fact[i - 1] * i;
    u64 local = ~0ull;
    for (ll r = t; r < total; r += T) {
        int q[12];
        unsigned used = 0;
        ll rr = r;
        for (int i = 0; i < k; i++) {
            const ll f = fact[k - 1 - i];
            int d = (int)(rr / f);
            rr %= f;
            int c = 0;
            for (int m = 0; m < k; m++) {
                if (!((used >> m) & 1u)) {
                    if (d == 0) { c = m; break; }
                    d--;
                }
            }
            q[i] = c;
            used |= 1u << c;
        }
        ll v = 0;
        for (int i = 0; i < k; i++) {
            v += R[i * k + q[i]];
            for (int u = 0; u < k; u++) v += A[i * k + u] * B[q[i] * k + q[u]];
        }
        const ll delta = v - base;
        if (delta < 0) {
            const u64 key = ((u64)(delta + (1LL << 34)) << 29) | (u64)r;
            if (key < local) local = key;
        }
    }
    for (int o = 16; o > 0; o >>= 1) {
        const u64 x = __shfl_down_sync(0xffffffffu, local, o);
        if (x < local) local = x;
    }
    if ((t & 31) == 0) atomicMin(&sel, local);
    __syncthreads();
    if (t == 0) out[g] = sel;
}
"#;

    pub struct GpuKOpt {
        stream: Arc<CudaStream>,
        f: CudaFunction,
        a: CudaSlice<i64>,
        b: CudaSlice<i64>,
        n: usize,
        threads: u32,
    }

    impl GpuKOpt {
        pub fn new(inst: &Instance, threads: u32) -> Result<Self, Error> {
            assert!(threads.is_multiple_of(32), "whole warps");
            // non-negative matrices: a rearrangement lowers the cost by at most
            // the cost itself, checked against 2^34 in `run`
            assert!(
                inst.a.iter().chain(&inst.b).all(|&x| x >= 0),
                "k-opt on the GPU needs non-negative matrices"
            );
            let ctx = cudarc::driver::CudaContext::new(0)?;
            let stream = ctx.default_stream();
            let ptx = cudarc::nvrtc::compile_ptx(KOPT_KERNEL)?;
            let f = ctx.load_module(ptx)?.load_function("kopt")?;
            Ok(GpuKOpt {
                a: stream.clone_htod(&inst.a)?,
                b: stream.clone_htod(&inst.b)?,
                stream,
                f,
                n: inst.n,
                threads,
            })
        }

        fn run(
            &mut self,
            inst: &Instance,
            p: &[usize],
            groups: &[Vec<usize>],
        ) -> Result<Vec<(C, Vec<usize>)>, Error> {
            assert!(inst.cost_of(p) < 1 << 34, "k-opt deltas must fit 35 bits");
            let k = groups[0].len();
            assert!((2..=12).contains(&k), "2 <= k <= 12");
            let p32: Vec<i32> = p.iter().map(|&x| x as i32).collect();
            let g32: Vec<i32> = groups.iter().flatten().map(|&x| x as i32).collect();
            let (dp, dg) = (self.stream.clone_htod(&p32)?, self.stream.clone_htod(&g32)?);
            let mut out = self.stream.alloc_zeros::<u64>(groups.len())?;
            let total: i64 = (1..=k as i64).product();
            let (n32, k32) = (self.n as i32, k as i32);
            let mut launch = self.stream.launch_builder(&self.f);
            launch
                .arg(&self.a)
                .arg(&self.b)
                .arg(&n32)
                .arg(&k32)
                .arg(&dp)
                .arg(&dg)
                .arg(&mut out)
                .arg(&total);
            let cfg = LaunchConfig {
                grid_dim: (groups.len() as u32, 1, 1),
                block_dim: (self.threads, 1, 1),
                shared_mem_bytes: 0,
            };
            unsafe { launch.launch(cfg) }?;
            let keys = self.stream.clone_dtoh(&out)?;
            Ok(keys
                .into_iter()
                .map(|key| {
                    if key == u64::MAX {
                        (0, (0..k).collect())
                    } else {
                        let delta = (key >> 29) as i64 - (1i64 << 34);
                        (delta, unrank(key & ((1 << 29) - 1), k))
                    }
                })
                .collect())
        }
    }

    impl KOpt for GpuKOpt {
        fn best(
            &mut self,
            inst: &Instance,
            p: &[usize],
            groups: &[Vec<usize>],
        ) -> Vec<(C, Vec<usize>)> {
            self.run(inst, p, groups).expect("GPU k-opt")
        }
    }

    pub struct GpuRots {
        stream: Arc<CudaStream>,
        f: CudaFunction,
        a: CudaSlice<i64>,
        b: CudaSlice<i64>,
        at: CudaSlice<i64>,
        bt: CudaSlice<i64>,
        best: CudaSlice<i32>,
        cur: CudaSlice<i32>,
        best_in: CudaSlice<i64>,
        delta: CudaSlice<i64>,
        tabu: CudaSlice<i32>,
        cost: CudaSlice<i64>,
        n: usize,
        p: usize,
        threads: u32,
        shared_bytes: u32,
        use_shared: bool,
        /// Symmetric matrices with zero diagonal: half the delta arithmetic.
        symmetric: bool,
        resident: String,
        params: RotsParams,
        host: Vec<i32>,
        device: String,
    }

    type Error = Box<dyn std::error::Error>;

    impl GpuRots {
        /// Buffers for a batch of `p` permutations of `inst` on GPU 0.
        pub fn new(
            inst: &Instance,
            p: usize,
            threads: u32,
            params: RotsParams,
        ) -> Result<Self, Error> {
            let n = inst.n;
            assert!(n * n < 1 << 23, "swap index must fit 23 bits");
            assert!(threads.is_multiple_of(32), "whole warps");
            let ctx = cudarc::driver::CudaContext::new(0)?;
            let device = ctx.name()?;
            let stream = ctx.default_stream();
            let ptx = cudarc::nvrtc::compile_ptx(KERNEL)?;
            let f = ctx.load_module(ptx)?.load_function("rots")?;
            // ca, cb, ra, rb (i64) and p (i32), plus the delta triangle if it fits
            let tri = n * (n - 1) / 2;
            let optin = ctx.attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )? as usize;
            let with_delta = 36 * n + 8 * tri;
            let use_shared = with_delta <= optin;
            let shared_bytes = if use_shared { with_delta } else { 36 * n } as u32;
            if shared_bytes > 48 * 1024 {
                f.set_attribute(
                    cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_bytes as i32,
                )?;
            }
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
                cur: stream.alloc_zeros(p * n)?,
                best_in: stream.alloc_zeros(p)?,
                delta: stream.alloc_zeros(if use_shared { 1 } else { p * tri })?,
                tabu: stream.alloc_zeros(p * n * n)?,
                cost: stream.alloc_zeros(p)?,
                stream,
                f,
                n,
                p,
                threads,
                shared_bytes,
                use_shared,
                symmetric: inst.symmetric,
                resident: format!(
                    "delta matrix in {} memory ({} KB shared per block), {} formulas, {blocks_per_sm} blocks per SM × {sms} SMs",
                    if use_shared { "shared" } else { "global" },
                    shared_bytes.div_ceil(1024),
                    if inst.symmetric { "symmetric" } else { "general" }
                ),
                params,
                host: vec![0; p * n],
                device,
            })
        }

        fn upload(&mut self, perms: &[Vec<usize>], to_cur: bool) -> Result<(), Error> {
            let n = self.n;
            for (i, perm) in perms.iter().enumerate() {
                for (h, &x) in self.host[i * n..][..n].iter_mut().zip(perm) {
                    *h = x as i32;
                }
            }
            let dst = if to_cur {
                &mut self.cur
            } else {
                &mut self.best
            };
            self.stream.memcpy_htod(&self.host, dst)?;
            Ok(())
        }

        fn download(&mut self, perms: &mut [Vec<usize>], from_cur: bool) -> Result<(), Error> {
            let n = self.n;
            let src = if from_cur { &self.cur } else { &self.best };
            self.stream.memcpy_dtoh(src, &mut self.host)?;
            for (i, perm) in perms.iter_mut().enumerate() {
                for (x, &h) in perm.iter_mut().zip(&self.host[i * n..][..n]) {
                    *x = h as usize;
                }
            }
            Ok(())
        }

        #[allow(clippy::too_many_arguments)]
        fn run_advance(
            &mut self,
            cur: &mut [Vec<usize>],
            best: &mut [Vec<usize>],
            best_cost: &mut [C],
            iters: u64,
            it0: u64,
            resume: bool,
            seed: u64,
        ) -> Result<(), Error> {
            assert_eq!(cur.len(), self.p);
            self.upload(cur, true)?;
            self.upload(best, false)?;
            self.stream.memcpy_htod(&*best_cost, &mut self.best_in)?;
            let (n32, iters, it0, seed) = (self.n as i32, iters as i64, it0 as i64, seed as u32);
            let resume = resume as i32;
            let (tlo, thi) = (self.params.tenure.0, self.params.tenure.1);
            let aspiration = self.params.aspiration;
            let use_shared = self.use_shared as i32;
            let symmetric = self.symmetric as i32;
            let mut launch = self.stream.launch_builder(&self.f);
            launch
                .arg(&self.a)
                .arg(&self.b)
                .arg(&self.at)
                .arg(&self.bt)
                .arg(&n32)
                .arg(&mut self.best)
                .arg(&mut self.cur)
                .arg(&self.best_in)
                .arg(&mut self.delta)
                .arg(&mut self.tabu)
                .arg(&mut self.cost)
                .arg(&iters)
                .arg(&it0)
                .arg(&resume)
                .arg(&tlo)
                .arg(&thi)
                .arg(&aspiration)
                .arg(&seed)
                .arg(&use_shared)
                .arg(&symmetric);
            let cfg = LaunchConfig {
                grid_dim: (self.p as u32, 1, 1),
                block_dim: (self.threads, 1, 1),
                shared_mem_bytes: self.shared_bytes,
            };
            unsafe { launch.launch(cfg) }?;
            self.download(cur, true)?;
            self.download(best, false)?;
            self.stream.memcpy_dtoh(&self.cost, best_cost)?;
            Ok(())
        }
    }

    impl Improver for GpuRots {
        fn advance(
            &mut self,
            cur: &mut [Vec<usize>],
            best: &mut [Vec<usize>],
            best_cost: &mut [C],
            iters: u64,
            it0: u64,
            resume: bool,
            seed: u64,
        ) {
            self.run_advance(cur, best, best_cost, iters, it0, resume, seed)
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

// ------------------------------------------------------------------ k-opt

/// For a group `s` of positions of `p` (elements `e[x] = p[s[x]]`), the change
/// of cost of putting `e[q[t]]` at `s[t]` splits into row terms and pairs
/// inside the group:
///   Δ(q) = Σ_t (R[t][q_t] − R[t][t]) + Σ_{t,u} A[t][u]·(B[q_t][q_u] − B[t][u])
/// with R[t][x] = Σ_{j ∉ s} a[s_t][j]·b[e_x][p_j] + a[j][s_t]·b[p_j][e_x],
/// A[t][u] = a[s_t][s_u] and B[x][y] = b[e_x][e_y].
struct GroupTerms {
    k: usize,
    r: Vec<C>,
    a: Vec<C>,
    b: Vec<C>,
    base: C,
}

impl GroupTerms {
    fn new(inst: &Instance, p: &[usize], s: &[usize]) -> Self {
        let (n, k) = (inst.n, s.len());
        let e: Vec<usize> = s.iter().map(|&i| p[i]).collect();
        let mut inside = vec![false; n];
        s.iter().for_each(|&i| inside[i] = true);
        let (a, b) = (&inst.a, &inst.b);
        let mut r = vec![0; k * k];
        for t in 0..k {
            for x in 0..k {
                r[t * k + x] = (0..n)
                    .filter(|&j| !inside[j])
                    .map(|j| {
                        a[s[t] * n + j] * b[e[x] * n + p[j]] + a[j * n + s[t]] * b[p[j] * n + e[x]]
                    })
                    .sum();
            }
        }
        let am: Vec<C> = (0..k * k).map(|q| a[s[q / k] * n + s[q % k]]).collect();
        let bm: Vec<C> = (0..k * k).map(|q| b[e[q / k] * n + e[q % k]]).collect();
        let mut terms = GroupTerms {
            k,
            r,
            a: am,
            b: bm,
            base: 0,
        };
        let identity: Vec<usize> = (0..k).collect();
        terms.base = terms.raw(&identity);
        terms
    }

    /// Σ_t R[t][q_t] + Σ_{t,u} A[t][u]·B[q_t][q_u]
    fn raw(&self, q: &[usize]) -> C {
        let k = self.k;
        let mut v = 0;
        for t in 0..k {
            v += self.r[t * k + q[t]];
            for u in 0..k {
                v += self.a[t * k + u] * self.b[q[t] * k + q[u]];
            }
        }
        v
    }

    fn delta(&self, q: &[usize]) -> C {
        self.raw(q) - self.base
    }
}

/// The rearrangement of rank `r` (factorial number system, 0 = identity).
#[cfg_attr(not(any(feature = "gpu", test)), allow(dead_code))]
fn unrank(mut r: u64, k: usize) -> Vec<usize> {
    let mut fact = vec![1u64; k + 1];
    for i in 1..=k {
        fact[i] = fact[i - 1] * i as u64;
    }
    let mut free: Vec<usize> = (0..k).collect();
    let mut q = Vec::with_capacity(k);
    for i in 0..k {
        let f = fact[k - 1 - i];
        let digit = (r / f) as usize;
        r %= f;
        q.push(free.remove(digit));
    }
    q
}

/// The best rearrangement of each group: (delta, q) with `q[t]` the index in
/// the group of the element to put at its t-th position.
trait KOpt {
    fn best(&mut self, inst: &Instance, p: &[usize], groups: &[Vec<usize>])
        -> Vec<(C, Vec<usize>)>;
}

/// All k! rearrangements by Heap's algorithm, on the CPU threads.
struct CpuKOpt {
    jobs: usize,
}

impl KOpt for CpuKOpt {
    fn best(
        &mut self,
        inst: &Instance,
        p: &[usize],
        groups: &[Vec<usize>],
    ) -> Vec<(C, Vec<usize>)> {
        let chunk = groups.len().div_ceil(self.jobs.max(1)).max(1);
        let mut out = vec![(0, Vec::new()); groups.len()];
        std::thread::scope(|scope| {
            for (gs, os) in groups.chunks(chunk).zip(out.chunks_mut(chunk)) {
                scope.spawn(move || {
                    for (g, o) in gs.iter().zip(os) {
                        let terms = GroupTerms::new(inst, p, g);
                        let k = g.len();
                        let mut q: Vec<usize> = (0..k).collect();
                        let mut best = (0, q.clone());
                        let mut c = vec![0usize; k];
                        let mut i = 1;
                        while i < k {
                            if c[i] < i {
                                if i % 2 == 0 {
                                    q.swap(0, i)
                                } else {
                                    q.swap(c[i], i)
                                }
                                let d = terms.delta(&q);
                                if d < best.0 {
                                    best = (d, q.clone());
                                }
                                c[i] += 1;
                                i = 1;
                            } else {
                                c[i] = 0;
                                i += 1;
                            }
                        }
                        *o = best;
                    }
                });
            }
        });
        out
    }
}

/// k random distinct positions, or, with a distance, a pivot and its k − 1
/// closest positions.
fn groups(
    n: usize,
    k: usize,
    count: usize,
    close: Option<&[C]>,
    rng: &mut DefaultRng,
) -> Vec<Vec<usize>> {
    (0..count)
        .map(|_| match close {
            Some(d) => {
                let pivot = rng.random_range(0..n);
                let mut by: Vec<usize> = (0..n).collect();
                // random tie-breaks, so equal distances give different groups
                let key: Vec<u32> = (0..n).map(|_| rng.random()).collect();
                by.sort_by_key(|&j| (j != pivot, d[pivot * n + j], key[j]));
                by.truncate(k);
                by
            }
            None => {
                let mut all: Vec<usize> = (0..n).collect();
                for i in 0..k {
                    let j = rng.random_range(i..n);
                    all.swap(i, j);
                }
                all.truncate(k);
                all
            }
        })
        .collect()
}

/// Exhaustive k-opt polish: each round, the best rearrangement over `count`
/// groups of k positions is applied if it improves; stops after `patience`
/// rounds without improvement. Returns the polished permutation and cost.
#[allow(clippy::too_many_arguments)]
fn kopt_polish(
    inst: &Instance,
    mut p: Vec<usize>,
    kopt: &mut dyn KOpt,
    k: usize,
    count: usize,
    patience: usize,
    rng: &mut DefaultRng,
    trace: &mut Trace,
) -> (C, Vec<usize>) {
    let n = inst.n;
    let close = inst
        .distance()
        .filter(|(over_elements, _)| !over_elements)
        .map(|(_, d)| d);
    let mut cost = inst.cost_of(&p);
    let (mut idle, mut round) = (0, 0);
    let t0 = Instant::now();
    while idle < patience {
        round += 1;
        let gs = groups(n, k.min(n), count, close, rng);
        let t = Instant::now();
        let results = kopt.best(inst, &p, &gs);
        let (g, (d, q)) = gs
            .iter()
            .zip(results)
            .min_by_key(|(_, (d, _))| *d)
            .expect("groups");
        if d < 0 {
            let e: Vec<usize> = g.iter().map(|&i| p[i]).collect();
            for (t, &i) in g.iter().enumerate() {
                p[i] = e[q[t]];
            }
            assert!(is_permutation(&p, n), "a rearrangement keeps a permutation");
            let recount = inst.cost_of(&p);
            assert_eq!(recount, cost + d, "k-opt delta matches the recount");
            (cost, idle) = (recount, 0);
            println!(
                "  k-opt round {round}: {cost} ({d}), {:.1} s",
                t0.elapsed().as_secs_f64()
            );
        } else {
            idle += 1;
        }
        trace.line(format_args!(
            "kopt round={round} k={k} groups={count} best_delta={d} cost={cost} round_s={:.3}",
            t.elapsed().as_secs_f64()
        ));
    }
    (cost, p)
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
    let crossover: String = args
        .value("--crossover")
        .unwrap_or_else(|| "uniform".into());
    let polish_from: Option<String> = args.value("--polish-from");
    let kopt_k: usize = args
        .value("--kopt")
        .unwrap_or(if polish_from.is_some() { 9 } else { 0 });
    let kopt_groups: usize = args
        .value("--kopt-groups")
        .unwrap_or(if cpu { 64 } else { 1024 });
    let kopt_rounds: usize = args.value("--kopt-rounds").unwrap_or(20);
    let perturb_stalled = args.flag("--perturb-stalled");
    let independent = args.flag("--independent") || perturb_stalled;
    let mut params = RotsParams::default();
    if let Some(t) = args.value::<String>("--tenure") {
        let v: Vec<f64> = t
            .split(',')
            .map(|x| x.parse().expect("--tenure LO,HI"))
            .collect();
        assert!(
            v.len() == 2 && 0.0 < v[0] && v[0] <= v[1],
            "--tenure LO,HI with 0 < LO <= HI"
        );
        params.tenure = (v[0], v[1]);
    }
    if let Some(a) = args.value::<f64>("--aspiration") {
        params.aspiration = a;
    }
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

    let name = Path::new(&file)
        .file_stem()
        .map_or("qap".into(), |s| s.to_string_lossy().into_owned());
    let mut kopt: Box<dyn KOpt> = if cpu || kopt_k == 0 {
        Box::new(CpuKOpt { jobs })
    } else {
        gpu_kopt(&inst, threads)
    };
    if let Some(from) = &polish_from {
        // k-opt polish of a saved permutation (NAME.best: cost, then 1-based facilities)
        let text = std::fs::read_to_string(from).unwrap_or_else(|e| panic!("{from}: {e}"));
        let p: Vec<usize> = text
            .lines()
            .nth(1)
            .expect("second line: the permutation")
            .split_whitespace()
            .map(|x| x.parse::<usize>().expect("a number") - 1)
            .collect();
        assert!(
            is_permutation(&p, n),
            "{from}: not a permutation of 1..={n}"
        );
        println!(
            "{from}: cost {}; k-opt polish, k = {kopt_k}, {kopt_groups} groups per round, until {kopt_rounds} rounds without improvement; build {}",
            inst.cost_of(&p),
            env!("METS_GIT_HASH")
        );
        let mut rng = DefaultRng::seed_from_u64(seed);
        let (cost, p) = kopt_polish(
            &inst,
            p,
            &mut *kopt,
            kopt_k,
            kopt_groups,
            kopt_rounds,
            &mut rng,
            &mut trace,
        );
        save(&format!("{name}.kopt.best"), cost, &p, best_known);
        return;
    }
    let setup = Instant::now();
    let mut improver: Box<dyn Improver> = if cpu {
        Box::new(CpuRots::new(&inst, jobs, params))
    } else {
        gpu_improver(&inst, population, threads, params)
    };
    trace.line(format_args!(
        "start build={} file={file} n={n} population={population} iters={iters} threads={threads} seed={seed} best_known={best_known:?} mode={} tenure={:?} aspiration={} improver=\"{}\" setup_s={:.3} (includes NVRTC compile)",
        env!("METS_GIT_HASH"),
        if perturb_stalled { "independent+perturb" } else if independent { "independent" } else { "memetic" },
        params.tenure,
        params.aspiration,
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
    let mut best = (C::MAX, Vec::new());
    // record a new best (on disk at once, so an interrupted run keeps it) and
    // report; true when the run should stop
    let progress =
        |generation: u64, best: &mut (C, Vec<usize>), cand: (C, &[usize]), extra: String| {
            if cand.0 < best.0 {
                *best = (cand.0, cand.1.to_vec());
                write_perm(&format!("{name}.best"), best.0, &best.1);
            }
            let secs = t0.elapsed().as_secs_f64();
            let reached = best_known.is_some_and(|b| best.0 <= b);
            if generation % 10 == 1 || reached {
                let g =
                    best_known.map_or(String::new(), |b| format!(" (gap {:.3}%)", gap(best.0, b)));
                println!(
                    "  generation {generation:5}: best {}{g}, {secs:7.1} s{extra}",
                    best.0
                );
            }
            (reached && !run_to_end) || secs > max_seconds
        };
    if independent {
        // Q6 arms B and C: independent robust tabu trajectories, each batch
        // continuing every member from where it stopped, with its tabu memory
        let mut cur = pop.clone();
        let mut bests = pop.clone();
        let mut costs = vec![C::MAX; population];
        let mut perturbations = 0u64;
        for generation in 1u64.. {
            let before = costs.clone();
            let t = Instant::now();
            let it0 = (generation - 1) * iters;
            improver.advance(
                &mut cur,
                &mut bests,
                &mut costs,
                iters,
                it0,
                generation > 1,
                rng.random(),
            );
            trace.batch(generation, &inst, &bests, &costs, t.elapsed().as_secs_f64());
            // a member's best can only improve, on either path (a kernel bug otherwise)
            assert!(
                costs.iter().zip(&before).all(|(c, b)| c <= b),
                "a member's best got worse across a resumed batch"
            );
            let i = (0..population)
                .min_by_key(|&i| costs[i])
                .expect("population");
            let extra = format!(", {perturbations} perturbations");
            if progress(generation, &mut best, (costs[i], &bests[i]), extra) {
                break;
            }
            if perturb_stalled {
                // arm C: a member whose best did not improve in this batch gets
                // k random swaps, k uniform in [n/10, n/4]
                let (lo, hi) = ((n / 10).max(1), (n / 4).max(1));
                for i in 0..population {
                    if costs[i] >= before[i] {
                        for _ in 0..rng.random_range(lo..=hi.max(lo)) {
                            let (x, y) = (rng.random_range(0..n), rng.random_range(0..n));
                            cur[i].swap(x, y);
                        }
                        perturbations += 1;
                    }
                }
            }
        }
    } else {
        let t = Instant::now();
        let mut costs = improver.improve(&mut pop, iters, rng.random());
        trace.batch(0, &inst, &pop, &costs, t.elapsed().as_secs_f64());
        let mut children = pop.clone();
        let mut cross: Box<dyn Crossover<QapPlain>> = match crossover.as_str() {
            "uniform" => Box::new(UniformCrossover),
            "cohesive" => match (inst.proximity(), inst.cohesive_crossover()) {
                (Some(px), Some(x)) => {
                    eprintln!("cohesive crossover on the {}", px.describe());
                    Box::new(x)
                }
                _ => {
                    eprintln!(
                        "note: no distance or similarity matrix in this instance; using uniform crossover"
                    );
                    Box::new(UniformCrossover)
                }
            },
            x => panic!("unknown crossover {x:?} (uniform, cohesive)"),
        };
        let mut since = vec![0u64; population]; // generations since member i improved
        let mut restarts = 0u64;
        for generation in 1u64.. {
            let i = (0..population)
                .min_by_key(|&i| costs[i])
                .expect("population");
            if progress(
                generation,
                &mut best,
                (costs[i], &pop[i]),
                format!(", {restarts} restarts"),
            ) {
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
    }
    if kopt_k > 0 {
        println!("{file}: k-opt polish of the best, k = {kopt_k}");
        best = kopt_polish(
            &inst,
            best.1,
            &mut *kopt,
            kopt_k,
            kopt_groups,
            kopt_rounds,
            &mut rng,
            &mut trace,
        );
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
    print!("{file}: in {:.1} s, ", t0.elapsed().as_secs_f64());
    save(&format!("{name}.best"), best.0, &best.1, best_known);
}

/// Write a permutation as `cost`, then the 1-based facilities.
fn write_perm(out: &str, cost: C, p: &[usize]) {
    let perm: Vec<String> = p.iter().map(|&x| (x + 1).to_string()).collect();
    std::fs::write(out, format!("{cost}\n{}\n", perm.join(" ")))
        .unwrap_or_else(|e| panic!("{out}: {e}"));
}

/// Save a permutation (see `write_perm`) and report it.
fn save(out: &str, cost: C, p: &[usize], best_known: Option<C>) {
    write_perm(out, cost, p);
    let g = best_known.map_or(String::new(), |b| {
        format!(", best known {b} (gap {:.3}%)", gap(cost, b))
    });
    println!("best {cost}{g}, saved to {out}");
}

#[cfg(feature = "gpu")]
fn gpu_improver<'a>(
    inst: &'a Instance,
    p: usize,
    threads: u32,
    params: RotsParams,
) -> Box<dyn Improver + 'a> {
    Box::new(gpu::GpuRots::new(inst, p, threads, params).unwrap_or_else(|e| panic!("GPU: {e}")))
}

#[cfg(feature = "gpu")]
fn gpu_kopt(inst: &Instance, threads: u32) -> Box<dyn KOpt> {
    Box::new(gpu::GpuKOpt::new(inst, threads).unwrap_or_else(|e| panic!("GPU: {e}")))
}

#[cfg(not(feature = "gpu"))]
fn gpu_kopt(_: &Instance, _: u32) -> Box<dyn KOpt> {
    panic!("built without the GPU: rebuild with --features gpu, or run with --cpu")
}

#[cfg(not(feature = "gpu"))]
fn gpu_improver<'a>(_: &'a Instance, _: usize, _: u32, _: RotsParams) -> Box<dyn Improver + 'a> {
    panic!("built without the GPU: rebuild with --features gpu, or run with --cpu")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A random asymmetric instance with a non-zero diagonal.
    fn asymmetric(n: usize, seed: u64) -> Instance {
        let mut rng = DefaultRng::seed_from_u64(seed);
        let mut m = || {
            (0..n * n)
                .map(|_| rng.random_range(0..20))
                .collect::<Vec<C>>()
        };
        let (a, b) = (m(), m());
        Instance::new(n, a, b)
    }

    fn all_rearrangements(k: usize) -> Vec<Vec<usize>> {
        (0..(1..=k as u64).product::<u64>())
            .map(|r| unrank(r, k))
            .collect()
    }

    #[test]
    fn group_deltas_match_recounts() {
        for inst in [Instance::random(12, 5), asymmetric(12, 6)] {
            let mut rng = DefaultRng::seed_from_u64(7);
            let p = random_start(12, 8);
            for g in groups(12, 5, 10, None, &mut rng) {
                let terms = GroupTerms::new(&inst, &p, &g);
                for q in all_rearrangements(5).into_iter().step_by(7) {
                    let mut after = p.clone();
                    for (t, &i) in g.iter().enumerate() {
                        after[i] = p[g[q[t]]];
                    }
                    assert_eq!(terms.delta(&q), inst.cost_of(&after) - inst.cost_of(&p));
                }
            }
        }
    }

    #[test]
    fn unrank_is_a_bijection() {
        let all = all_rearrangements(5);
        assert_eq!(all[0], vec![0, 1, 2, 3, 4]);
        let mut sorted = all.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 120);
        assert!(all.iter().all(|q| is_permutation(q, 5)));
    }

    #[test]
    fn cpu_kopt_finds_the_best_rearrangement() {
        let inst = asymmetric(10, 9);
        let mut rng = DefaultRng::seed_from_u64(10);
        let p = random_start(10, 11);
        let gs = groups(10, 5, 6, None, &mut rng);
        let found = CpuKOpt { jobs: 2 }.best(&inst, &p, &gs);
        for (g, (d, q)) in gs.iter().zip(found) {
            let terms = GroupTerms::new(&inst, &p, g);
            let brute = all_rearrangements(5)
                .iter()
                .map(|q| terms.delta(q))
                .min()
                .unwrap();
            assert_eq!(d, brute.min(0), "best (or no) improvement");
            assert_eq!(terms.delta(&q), d);
        }
    }

    #[test]
    fn kopt_polish_improves_and_stays_exact() {
        let inst = Instance::random(14, 12);
        let p = random_start(14, 13);
        let start = inst.cost_of(&p);
        let mut rng = DefaultRng::seed_from_u64(14);
        let (cost, q) = kopt_polish(
            &inst,
            p,
            &mut CpuKOpt { jobs: 2 },
            4,
            16,
            3,
            &mut rng,
            &mut Trace(None),
        );
        assert!(is_permutation(&q, 14) && inst.cost_of(&q) == cost && cost < start);
    }
}
