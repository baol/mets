//! Graph colouring with HEAD (Moalic & Gondran 2018) run as many independent
//! pairs at once, the TabuCol improvements of a whole generation in one batch:
//! on the GPU (`--features gpu`, one CUDA block per individual) or on the CPU
//! threads (`--cpu`, the library's `TabuSearch`, for comparison and testing).
//!
//! Each pair (a, b) evolves as in HEAD: the children GPX(a, b) and GPX(b, a)
//! are improved by TabuCol and always replace their first parent; every 10
//! generations the best child of the previous cycle replaces a random member
//! of its pair. A pair whose members coincide re-randomises one of them.
//!
//! usage: coloring_gpu FILE.col --k K [options]
//!   --pairs P        independent HEAD pairs (default 1024 on the GPU, 4 with --cpu)
//!   --iters L        TabuCol iterations per child, in units of n (default 60)
//!   --max-seconds T  give up after T seconds (default 3600)
//!   --seed S         (default 1)
//!   --threads T      GPU threads per block (default 128)
//!   --cpu            improve on the CPU threads instead of the GPU
//! A legal colouring is verified and saved as NAME.K.sol ("vertex colour", 1-based).
//!   --trace FILE     append one line per batch: time, cost spread, and a CPU
//!                    recount of a few returned colourings (mismatches = kernel bug)
//!
//! Build and run on a machine with an NVIDIA GPU. Building needs nothing
//! from CUDA; running needs the driver (libcuda) and NVRTC (libnvrtc, from
//! the CUDA 12 toolkit or `pip install nvidia-cuda-nvrtc-cu12`; put its
//! directory in LD_LIBRARY_PATH), which compiles the kernel at start-up:
//!   cargo run --release --features gpu --example coloring_gpu -- \
//!       DSJC500.5.col --k 47 --trace gpu.trace
//! Each block keeps its colouring and a 16-bit conflict table (4n + 2nk
//! bytes; 48 KB for DSJC500.5 at k = 46) in shared memory when they fit;
//! the tabu table, 8·pairs·n·k bytes of device memory, stays global
//! (DSJC500.5, k = 46, 2048 pairs: 750 MB). Lower --pairs on large graphs.

#[path = "common/qap.rs"]
mod common; // only for its argument parser
#[path = "common/coloring.rs"]
mod graph_coloring;

use common::Args;
use graph_coloring::*;
use mets::rand::{Rng, SeedableRng};
use mets::*;
use std::io::Write;
use std::time::Instant;

/// Improves a batch of k-colourings in place: each becomes the best colouring
/// its TabuCol run found. Returns their conflicting edges.
trait Improver {
    fn improve(&mut self, batch: &mut [Vec<usize>], iters: u64, seed: u64) -> Vec<i64>;
    fn name(&self) -> String;
}

/// The library's TabuCol on the CPU threads.
struct CpuTabuCol<'g> {
    g: &'g Graph,
    k: usize,
    jobs: usize,
}

impl Improver for CpuTabuCol<'_> {
    fn improve(&mut self, batch: &mut [Vec<usize>], iters: u64, seed: u64) -> Vec<i64> {
        let mut costs = vec![0; batch.len()];
        let chunk = batch.len().div_ceil(self.jobs.max(1)).max(1);
        std::thread::scope(|scope| {
            for (c, (cols, out)) in batch
                .chunks_mut(chunk)
                .zip(costs.chunks_mut(chunk))
                .enumerate()
            {
                let (g, k) = (self.g, self.k);
                scope.spawn(move || {
                    for (i, (col, cost)) in cols.iter_mut().zip(out).enumerate() {
                        let run_seed = seed.wrapping_mul(1 << 20) + (c * chunk + i) as u64;
                        let mut st = State::new(Coloring::new(g, k, col.clone()), run_seed);
                        tabucol()
                            .runner(&mut st)
                            .until(MaxIterations::new(iters).or(Target::new(0)))
                            .run();
                        col.clone_from(&st.global_best().color);
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
    use super::{Graph, Improver};
    use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};
    use std::sync::Arc;

    /// TabuCol, one block per individual. The block's threads share the scan
    /// for the best move (each takes some vertices, then a warp and a block
    /// minimum) and the update of the moved vertex's neighbours. A move's key
    /// packs (delta, random, vertex·k + colour), so the minimum key is the
    /// best admissible move with ties broken at random.
    const KERNEL: &str = r#"
typedef unsigned long long u64;
__device__ __forceinline__ unsigned int xs(unsigned int &s) {
    s ^= s << 13; s ^= s >> 17; s ^= s << 5; return s;
}

extern "C" __global__ void tabucol(
    const int* __restrict__ start, const int* __restrict__ adj, int n, int k,
    int* best_color, int* cur_color, unsigned short* gamma, int* tabu, long long* out_cost,
    long long iters, unsigned int seed, int use_shared)
{
    const int b = blockIdx.x, t = threadIdx.x, T = blockDim.x;
    const size_t nk = (size_t)n * k;
    // the current colouring and the conflict table live in shared memory
    // when they fit (4n + 2nk bytes), else in global memory
    extern __shared__ __align__(8) unsigned char smem[];
    int* best = best_color + (size_t)b * n;
    int* col = use_shared ? (int*)smem : cur_color + (size_t)b * n;
    unsigned short* g = use_shared ? (unsigned short*)(smem + 4 * (size_t)n) : gamma + (size_t)b * nk;
    int* tb = tabu + (size_t)b * nk;
    __shared__ u64 sel;
    __shared__ int f, bestf, nconf;
    unsigned int rng = seed ^ (0x9E3779B9u * (unsigned)(b * T + t + 1));
    if (rng == 0) rng = 1;
    for (int i = 0; i < 4; i++) xs(rng);

    // conflict table, cost and conflicting vertices of the start colouring
    for (int v = t; v < n; v += T) col[v] = best[v];
    for (size_t i = t; i < nk; i += T) { g[i] = 0; tb[i] = 0; }
    if (t == 0) { f = 0; nconf = 0; }
    __syncthreads();
    for (int v = t; v < n; v += T)
        for (int e = start[v]; e < start[v + 1]; e++) g[(size_t)v * k + col[adj[e]]]++;
    __syncthreads();
    {
        int lf = 0, lc = 0;
        for (int v = t; v < n; v += T) {
            int own = g[(size_t)v * k + col[v]];
            lf += own; lc += own > 0;
        }
        atomicAdd(&f, lf); atomicAdd(&nconf, lc);
    }
    __syncthreads();
    if (t == 0) { f /= 2; bestf = f; }
    __syncthreads();

    for (long long it = 1; it <= iters && f > 0; it++) {
        if (t == 0) sel = ~0ull;
        __syncthreads();
        // best admissible recolouring of a conflicting vertex
        u64 local = ~0ull;
        int local_d = 0x7fffffff;  // delta of `local`: worse moves need no tabu read
        const int fv = f, bf = bestf;
        for (int v = t; v < n; v += T) {
            const int cv = col[v];
            const unsigned short* gv = g + (size_t)v * k;
            const int own = gv[cv];
            if (own == 0) continue;
            const int* tv = tb + (size_t)v * k;
            for (int c = 0; c < k; c++) {
                if (c == cv) continue;
                const int d = (int)gv[c] - own;
                if (d > local_d) continue;
                if (tv[c] < it || fv + d < bf) {  // not tabu, or aspired
                    u64 key = ((u64)(unsigned)(d + (1 << 20)) << 40)
                            | ((u64)(xs(rng) & 0xFFFFFu) << 20)
                            | (u64)(v * k + c);
                    if (key < local) { local = key; local_d = d; }
                }
            }
        }
        for (int o = 16; o > 0; o >>= 1) {
            u64 x = __shfl_down_sync(0xffffffffu, local, o);
            if (x < local) local = x;
        }
        if ((t & 31) == 0) atomicMin(&sel, local);
        __syncthreads();
        const u64 s = sel;
        if (s == ~0ull) break;  // every move tabu and none aspired
        const int idx = (int)(s & 0xFFFFFu), v = idx / k, c = idx % k;
        const int d = (int)((s >> 40) & 0xFFFFFFu) - (1 << 20);
        const int old = col[v];

        // the neighbours of v: conflict table and conflicting-vertex count
        int lc = 0;
        for (int e = start[v] + t; e < start[v + 1]; e += T) {
            const int u = adj[e], cu = col[u];
            unsigned short* gu = g + (size_t)u * k;
            if (cu == old) lc -= (gu[old] == 1);
            else if (cu == c) lc += (gu[c] == 0);
            gu[old]--; gu[c]++;
        }
        if (lc) atomicAdd(&nconf, lc);
        __syncthreads();
        if (t == 0) {
            const unsigned short* gv = g + (size_t)v * k;
            nconf += (gv[c] > 0) - (gv[old] > 0);
            col[v] = c;
            f += d;
            // Galinier & Hao's tenure: random(0..10) + 0.6 * conflicting vertices
            tb[(size_t)v * k + old] = (int)it + (int)(xs(rng) % 10u) + (int)(0.6f * nconf);
        }
        __syncthreads();
        const bool improved = f < bestf;
        __syncthreads();
        if (improved) {
            for (int w = t; w < n; w += T) best[w] = col[w];
            if (t == 0) bestf = f;
        }
        __syncthreads();
    }
    if (t == 0) out_cost[b] = bestf;
}
"#;

    pub struct GpuTabuCol {
        stream: Arc<CudaStream>,
        f: CudaFunction,
        start: CudaSlice<i32>,
        adj: CudaSlice<i32>,
        best: CudaSlice<i32>,
        cur: CudaSlice<i32>,
        gamma: CudaSlice<u16>,
        tabu: CudaSlice<i32>,
        cost: CudaSlice<i64>,
        n: usize,
        k: usize,
        p: usize,
        threads: u32,
        /// Dynamic shared memory per block: 0 when the tables stay in global memory.
        shared_bytes: u32,
        host: Vec<i32>,
        device: String,
    }

    type Error = Box<dyn std::error::Error>;

    impl GpuTabuCol {
        /// Buffers for a batch of `p` k-colourings of `g` on GPU 0.
        pub fn new(g: &Graph, k: usize, p: usize, threads: u32) -> Result<Self, Error> {
            let n = g.n();
            assert!(n * k < 1 << 20, "move index must fit 20 bits");
            assert!(threads.is_multiple_of(32), "whole warps");
            let ctx = cudarc::driver::CudaContext::new(0)?;
            let device = ctx.name()?;
            let stream = ctx.default_stream();
            let ptx = cudarc::nvrtc::compile_ptx(KERNEL)?;
            let f = ctx.load_module(ptx)?.load_function("tabucol")?;
            assert!(
                g.adj.iter().all(|nb| nb.len() < 1 << 16),
                "16-bit conflict counts"
            );
            // the colouring and the 16-bit conflict table in shared memory, if they fit
            let need = 4 * n + 2 * n * k;
            let optin = ctx.attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )? as usize;
            let shared_bytes = if need <= optin {
                if need > 48 * 1024 {
                    f.set_attribute(
                        cudarc::driver::sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        need as i32,
                    )?;
                }
                need as u32
            } else {
                0
            };
            let global_table = if shared_bytes == 0 { p * n * k } else { 1 };
            let mut start = vec![0i32];
            let mut adj = Vec::with_capacity(2 * g.edges);
            for nb in &g.adj {
                adj.extend(nb.iter().map(|&u| u as i32));
                start.push(adj.len() as i32);
            }
            Ok(GpuTabuCol {
                start: stream.clone_htod(&start)?,
                adj: stream.clone_htod(&adj)?,
                best: stream.alloc_zeros(p * n)?,
                cur: stream.alloc_zeros(p * n)?,
                gamma: stream.alloc_zeros(global_table)?,
                tabu: stream.alloc_zeros(p * n * k)?,
                cost: stream.alloc_zeros(p)?,
                stream,
                f,
                n,
                k,
                p,
                threads,
                shared_bytes,
                host: vec![0; p * n],
                device,
            })
        }

        fn run(
            &mut self,
            batch: &mut [Vec<usize>],
            iters: u64,
            seed: u64,
        ) -> Result<Vec<i64>, Error> {
            assert_eq!(batch.len(), self.p);
            let n = self.n;
            for (i, col) in batch.iter().enumerate() {
                for (h, &c) in self.host[i * n..][..n].iter_mut().zip(col) {
                    *h = c as i32;
                }
            }
            self.stream.memcpy_htod(&self.host, &mut self.best)?;
            let (n32, k32, iters, seed) = (n as i32, self.k as i32, iters as i64, seed as u32);
            let use_shared = (self.shared_bytes > 0) as i32;
            let mut launch = self.stream.launch_builder(&self.f);
            launch
                .arg(&self.start)
                .arg(&self.adj)
                .arg(&n32)
                .arg(&k32)
                .arg(&mut self.best)
                .arg(&mut self.cur)
                .arg(&mut self.gamma)
                .arg(&mut self.tabu)
                .arg(&mut self.cost)
                .arg(&iters)
                .arg(&seed)
                .arg(&use_shared);
            let cfg = LaunchConfig {
                grid_dim: (self.p as u32, 1, 1),
                block_dim: (self.threads, 1, 1),
                shared_mem_bytes: self.shared_bytes,
            };
            unsafe { launch.launch(cfg) }?;
            self.stream.memcpy_dtoh(&self.best, &mut self.host)?;
            let costs = self.stream.clone_dtoh(&self.cost)?;
            for (i, col) in batch.iter_mut().enumerate() {
                for (c, &h) in col.iter_mut().zip(&self.host[i * n..][..n]) {
                    *c = h as usize;
                }
            }
            Ok(costs)
        }
    }

    impl Improver for GpuTabuCol {
        fn improve(&mut self, batch: &mut [Vec<usize>], iters: u64, seed: u64) -> Vec<i64> {
            self.run(batch, iters, seed).expect("GPU TabuCol")
        }
        fn name(&self) -> String {
            let tables = match self.shared_bytes {
                0 => "tables in global memory".to_string(),
                b => format!(
                    "colouring and conflict table in {} KB of shared memory",
                    b.div_ceil(1024)
                ),
            };
            format!(
                "GPU {}, {} threads per block, {tables}",
                self.device, self.threads
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

    /// One batch: time, cost spread, and a recount of up to 8 colourings.
    fn batch(
        &mut self,
        generation: u64,
        g: &Graph,
        k: usize,
        batch: &[Vec<usize>],
        costs: &[i64],
        secs: f64,
    ) {
        if self.0.is_none() {
            return;
        }
        let mut sorted = costs.to_vec();
        sorted.sort_unstable();
        let (mut checked, mut mismatches, mut bad) = (0, 0, 0);
        for (col, &cost) in batch.iter().zip(costs).take(8) {
            checked += 1;
            if col.len() != g.n() || col.iter().any(|&c| c >= k) {
                bad += 1;
                continue;
            }
            let recount = Coloring::new(g, k, col.clone()).evaluate();
            if recount != cost {
                mismatches += 1;
                self.line(format_args!(
                    "  mismatch: reported {cost}, recount {recount}"
                ));
            }
        }
        self.line(format_args!(
            "gen={generation} improve_s={secs:.3} min={} med={} max={} checked={checked} mismatches={mismatches} bad_colourings={bad}",
            sorted[0],
            sorted[sorted.len() / 2],
            sorted[sorted.len() - 1]
        ));
    }
}

/// The best child of the current and of the previous cycle of a pair.
#[derive(Default)]
struct Elites {
    current: Option<(i64, Vec<usize>)>,
    previous: Option<(i64, Vec<usize>)>,
}

fn main() {
    let mut args = Args::from_env();
    let k: usize = args.value("--k").expect("--k K");
    let cpu = args.flag("--cpu");
    let jobs = std::thread::available_parallelism().map_or(1, |n| n.get());
    let pairs: usize = args
        .value("--pairs")
        .unwrap_or(if cpu { jobs } else { 1024 });
    let iters_n: u64 = args.value("--iters").unwrap_or(60);
    let max_seconds: f64 = args.value("--max-seconds").unwrap_or(3600.0);
    let seed: u64 = args.value("--seed").unwrap_or(1);
    let threads: u32 = args.value("--threads").unwrap_or(128);
    let trace_file: Option<String> = args.value("--trace");
    let mut trace = Trace(trace_file.map(|f| {
        std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&f)
            .unwrap_or_else(|e| panic!("{f}: {e}"))
    }));
    let file = args
        .0
        .first()
        .expect("usage: coloring_gpu FILE.col --k K [options]");
    let text = std::fs::read_to_string(file).unwrap_or_else(|e| panic!("{file}: {e}"));
    let g = Graph::parse(&text).unwrap_or_else(|e| panic!("{file}: {e}"));
    let (n, p) = (g.n(), 2 * pairs);
    let iters = iters_n * n as u64;

    let setup = Instant::now();
    let mut improver: Box<dyn Improver> = if cpu {
        Box::new(CpuTabuCol { g: &g, k, jobs })
    } else {
        gpu_improver(&g, k, p, threads)
    };
    trace.line(format_args!(
        "start file={file} n={n} edges={} k={k} pairs={pairs} iters={iters} threads={threads} seed={seed} improver=\"{}\" setup_s={:.3} (includes NVRTC compile)",
        g.edges,
        improver.name(),
        setup.elapsed().as_secs_f64(),
    ));
    println!(
        "{file}: {n} vertices, {} edges; {k} colours; {pairs} HEAD pairs × {iters_n}n TabuCol iterations; {}",
        g.edges,
        improver.name()
    );

    let t0 = Instant::now();
    let mut rng = DefaultRng::seed_from_u64(seed);
    let random = |rng: &mut DefaultRng| (0..n).map(|_| rng.random_range(0..k)).collect::<Vec<_>>();
    let mut pop: Vec<Vec<usize>> = (0..p).map(|_| random(&mut rng)).collect();
    let t = Instant::now();
    let mut costs = improver.improve(&mut pop, iters, rng.random());
    trace.batch(0, &g, k, &pop, &costs, t.elapsed().as_secs_f64());
    let mut elites: Vec<Elites> = (0..pairs).map(|_| Elites::default()).collect();
    let mut best = (i64::MAX, Vec::new());
    let mut children = pop.clone();
    for generation in 1.. {
        // record the best, and stop on a legal colouring or out of time
        if let Some(i) = (0..p).min_by_key(|&i| costs[i]) {
            if costs[i] < best.0 {
                best = (costs[i], pop[i].clone());
            }
        }
        let secs = t0.elapsed().as_secs_f64();
        if generation % 10 == 1 || best.0 == 0 {
            let total = (generation as f64) * p as f64 * iters as f64;
            println!(
                "  generation {generation:5}: best {} conflicts, {secs:7.1} s, at most {:.2e} TabuCol it/s",
                best.0,
                total / secs
            );
        }
        if best.0 == 0 || secs > max_seconds {
            break;
        }
        // HEAD: each member's child with its partner always replaces it
        for i in 0..p {
            gpx(&pop[i], &pop[i ^ 1], k, &mut children[i], &mut rng);
        }
        let t = Instant::now();
        costs = improver.improve(&mut children, iters, rng.random());
        trace.batch(
            generation,
            &g,
            k,
            &children,
            &costs,
            t.elapsed().as_secs_f64(),
        );
        std::mem::swap(&mut pop, &mut children);
        for (pair, e) in elites.iter_mut().enumerate() {
            let i = (2 * pair..2 * pair + 2).min_by_key(|&i| costs[i]).unwrap();
            if e.current.as_ref().is_none_or(|(c, _)| costs[i] < *c) {
                e.current = Some((costs[i], pop[i].clone()));
            }
            if generation % 10 == 0 {
                if let Some((c, col)) = e.previous.take() {
                    let j = 2 * pair + rng.random_range(0..2);
                    (costs[j], pop[j]) = (c, col);
                }
                e.previous = e.current.take();
            }
            if partition_distance_of(&pop[2 * pair], &pop[2 * pair + 1], k) == 0 {
                pop[2 * pair + 1] = random(&mut rng);
                costs[2 * pair + 1] = i64::MAX; // improved with the next children
            }
        }
    }
    trace.line(format_args!(
        "end best={} total_s={:.1}",
        best.0,
        t0.elapsed().as_secs_f64()
    ));
    if best.0 == 0 {
        assert!(
            is_legal(&g, &best.1) && best.1.iter().all(|&c| c < k),
            "verified legal"
        );
        // one line per vertex, 1-based: "vertex colour"
        let name = std::path::Path::new(file)
            .file_stem()
            .map_or("graph".into(), |s| s.to_string_lossy());
        let out = format!("{name}.{k}.sol");
        let text: String = best
            .1
            .iter()
            .enumerate()
            .map(|(v, c)| format!("{} {}\n", v + 1, c + 1))
            .collect();
        std::fs::write(&out, text).unwrap_or_else(|e| panic!("{out}: {e}"));
        println!(
            "{file}: legal {k}-colouring in {:.1} s, saved to {out}",
            t0.elapsed().as_secs_f64()
        );
    } else {
        println!(
            "{file}: no legal {k}-colouring in {:.1} s (best {} conflicts)",
            t0.elapsed().as_secs_f64(),
            best.0
        );
    }
}

#[cfg(feature = "gpu")]
fn gpu_improver<'g>(g: &'g Graph, k: usize, p: usize, threads: u32) -> Box<dyn Improver + 'g> {
    Box::new(gpu::GpuTabuCol::new(g, k, p, threads).unwrap_or_else(|e| panic!("GPU: {e}")))
}

#[cfg(not(feature = "gpu"))]
fn gpu_improver<'g>(_: &'g Graph, _: usize, _: usize, _: u32) -> Box<dyn Improver + 'g> {
    panic!("built without the GPU: rebuild with --features gpu, or run with --cpu")
}
