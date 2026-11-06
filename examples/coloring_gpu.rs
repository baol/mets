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
//!   --threads T      GPU threads per block (default: 1024 when the tables in shared
//!                    memory leave room for one block per SM, else 128)
//!   --table T        conflict table: auto | 8 | 16 bits (default auto: 16 bits in shared
//!                    memory if they fit, else 8 bits if those fit, else 16 in global memory)
//!   --cpu            improve on the CPU threads instead of the GPU
//! A legal colouring is verified and saved as NAME.K.sol ("vertex colour", 1-based),
//! or the first free NAME.K.2.sol, … if that file exists.
//!   --polish-every G  every G generations, if the best colouring has 1 or 2
//!                    conflicts, improve copies of it in one whole batch, each with
//!                    its own seed, for --polish-iters L·n iterations (default 0: never)
//!   --polish-iters L  (default 1000, run as launches of at most 200n)
//!   --vss-rounds R   after each polish that leaves conflicts, continue the best
//!                    colouring on every CPU thread (own seed each) with R rounds of
//!                    Variable Space Search: TabuCol and PartialCol alternating
//!                    (default 0: off)
//!   --vss-iters L    iterations per space and round, in units of n (default 100)
//!   --trace FILE     append one line per batch: time, cost spread, and a CPU
//!                    recount of a few returned colourings (mismatches = kernel bug)
//!
//! Build and run on a machine with an NVIDIA GPU. Building needs nothing
//! from CUDA; running needs the driver (libcuda) and NVRTC (libnvrtc, from
//! the CUDA 12 toolkit or `pip install nvidia-cuda-nvrtc-cu12`; put its
//! directory in LD_LIBRARY_PATH), which compiles the kernel at start-up:
//!   cargo run --release --features gpu --example coloring_gpu -- \
//!       DSJC500.5.col --k 47 --trace gpu.trace
//! Each block keeps its conflict table (16 bits), its list of conflicting
//! vertices and its colouring (2nk + 3n bytes; 47 KB for DSJC500.5 at
//! k = 46) in shared memory when they fit; the tabu table, 8·pairs·n·k
//! bytes of device memory, stays global (DSJC500.5, k = 46, 2048 pairs:
//! 750 MB). The trace's start line gives the build (git commit) and how many
//! blocks run at once. Lower --pairs on large graphs.

#[path = "common/qap.rs"]
mod common; // only for its argument parser
#[path = "common/coloring.rs"]
mod graph_coloring;

use common::{fresh_path, Args};
use graph_coloring::*;
use mets::rand::{Rng, SeedableRng};
use mets::*;
use std::io::Write;
use std::time::Instant;

/// The conflict table's counters on the GPU: 16 bits, or 8 bits (saturating
/// at 255, which fails the run) so that larger graphs fit shared memory.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Table {
    /// 16 bits in shared memory if they fit, else 8 bits if those fit, else
    /// 16 bits in global memory.
    Auto,
    Bits8,
    Bits16,
}

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
    use super::{Graph, Improver, Table};
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

// one more neighbour of a colour; an 8-bit counter saturates at 255 and
// raises the block's overflow flag (the host fails the run)
template <typename G>
__device__ __forceinline__ void bump(G* x, int* overflow) {
    if (sizeof(G) == 1 && *x == (G)255) *overflow = 1;
    else (*x)++;
}

template <typename G>
__device__ __forceinline__ void tabucol_body(
    const int* __restrict__ start, const int* __restrict__ adj, int n, int k,
    int* best_color, unsigned char* cur_color, G* gamma, unsigned short* clist,
    int* tabu, long long* out_cost, long long iters, unsigned int seed, int use_shared,
    int* overflow_all)
{
    const int b = blockIdx.x, t = threadIdx.x, T = blockDim.x;
    const size_t nk = (size_t)n * k;
    // the conflict table (8 or 16 bits), the list of conflicting vertices
    // (16 bits) and the current colouring (8 bits) live in shared memory when
    // they fit, else in global memory; the table is padded to 8 bytes
    extern __shared__ __align__(8) unsigned char smem[];
    const size_t tbytes = (sizeof(G) * nk + 7) & ~(size_t)7;
    int* best = best_color + (size_t)b * n;
    int* overflow = overflow_all + b;
    G* g = use_shared ? (G*)smem : gamma + (size_t)b * nk;
    unsigned short* list = use_shared ? (unsigned short*)(smem + tbytes) : clist + (size_t)b * n;
    unsigned char* col = use_shared ? smem + tbytes + 2 * (size_t)n : cur_color + (size_t)b * n;
    int* tb = tabu + (size_t)b * nk;
    __shared__ u64 sel;
    __shared__ int f, bestf, nconf, nlist;
    unsigned int rng = seed ^ (0x9E3779B9u * (unsigned)(b * T + t + 1));
    if (rng == 0) rng = 1;
    for (int i = 0; i < 4; i++) xs(rng);

    // conflict table, cost and conflicting vertices of the start colouring
    for (int v = t; v < n; v += T) col[v] = (unsigned char)best[v];
    for (size_t i = t; i < nk; i += T) { g[i] = 0; tb[i] = 0; }
    if (t == 0) { f = 0; nconf = 0; }
    __syncthreads();
    for (int v = t; v < n; v += T)
        for (int e = start[v]; e < start[v + 1]; e++) bump(&g[(size_t)v * k + col[adj[e]]], overflow);
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
        if (t == 0) { sel = ~0ull; nlist = 0; }
        __syncthreads();
        // list the conflicting vertices, so that all threads share their moves
        for (int v = t; v < n; v += T)
            if (g[(size_t)v * k + col[v]] > 0) list[atomicAdd(&nlist, 1)] = (unsigned short)v;
        __syncthreads();
        // best admissible recolouring of a conflicting vertex: thread t takes
        // the moves q = t, t + T, ... of the (listed vertex, colour) pairs
        u64 local = ~0ull;
        int local_d = 0x7fffffff;  // delta of `local`: worse moves need no tabu read
        const int fv = f, bf = bestf, moves = nlist * k;
        for (int q = t; q < moves; q += T) {
            const int v = list[q / k], c = q % k;
            const int cv = col[v];
            if (c == cv) continue;
            const G* gv = g + (size_t)v * k;
            const int d = (int)gv[c] - (int)gv[cv];
            if (d > local_d) continue;
            if (tb[(size_t)v * k + c] < it || fv + d < bf) {  // not tabu, or aspired
                // (delta + 2^23) | 16 random bits | 24-bit move index
                u64 key = ((u64)(unsigned)(d + (1 << 23)) << 40)
                        | ((u64)(xs(rng) & 0xFFFFu) << 24)
                        | (u64)(v * k + c);
                if (key < local) { local = key; local_d = d; }
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
        const int idx = (int)(s & 0xFFFFFFu), v = idx / k, c = idx % k;
        const int d = (int)((s >> 40) & 0xFFFFFFu) - (1 << 23);
        const int old = col[v];

        // the neighbours of v: conflict table and conflicting-vertex count
        int lc = 0;
        for (int e = start[v] + t; e < start[v + 1]; e += T) {
            const int u = adj[e], cu = col[u];
            G* gu = g + (size_t)u * k;
            if (cu == old) lc -= (gu[old] == 1);
            else if (cu == c) lc += (gu[c] == 0);
            gu[old]--; bump(&gu[c], overflow);
        }
        if (lc) atomicAdd(&nconf, lc);
        __syncthreads();
        if (t == 0) {
            const G* gv = g + (size_t)v * k;
            nconf += (gv[c] > 0) - (gv[old] > 0);
            col[v] = (unsigned char)c;
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

extern "C" __global__ void tabucol16(
    const int* __restrict__ start, const int* __restrict__ adj, int n, int k,
    int* best_color, unsigned char* cur_color, unsigned short* gamma, unsigned short* clist,
    int* tabu, long long* out_cost, long long iters, unsigned int seed, int use_shared,
    int* overflow)
{
    tabucol_body<unsigned short>(start, adj, n, k, best_color, cur_color, gamma, clist, tabu,
                                 out_cost, iters, seed, use_shared, overflow);
}

extern "C" __global__ void tabucol8(
    const int* __restrict__ start, const int* __restrict__ adj, int n, int k,
    int* best_color, unsigned char* cur_color, unsigned char* gamma, unsigned short* clist,
    int* tabu, long long* out_cost, long long iters, unsigned int seed, int use_shared,
    int* overflow)
{
    tabucol_body<unsigned char>(start, adj, n, k, best_color, cur_color, gamma, clist, tabu,
                                out_cost, iters, seed, use_shared, overflow);
}
"#;

    pub struct GpuTabuCol {
        stream: Arc<CudaStream>,
        f: CudaFunction,
        start: CudaSlice<i32>,
        adj: CudaSlice<i32>,
        best: CudaSlice<i32>,
        cur: CudaSlice<u8>,
        list: CudaSlice<u16>,
        gamma: CudaSlice<u16>,
        /// Placeholder for the 8-bit kernel's (unused) global table argument.
        gamma8: CudaSlice<u8>,
        overflow: CudaSlice<i32>,
        /// 8-bit conflict table (`tabucol8`) instead of 16-bit.
        eight: bool,
        tabu: CudaSlice<i32>,
        cost: CudaSlice<i64>,
        n: usize,
        k: usize,
        p: usize,
        threads: u32,
        /// Dynamic shared memory per block: 0 when the tables stay in global memory.
        shared_bytes: u32,
        /// Blocks resident at once (occupancy), for the trace.
        resident: String,
        host: Vec<i32>,
        device: String,
    }

    type Error = Box<dyn std::error::Error>;

    impl GpuTabuCol {
        /// Buffers for a batch of `p` k-colourings of `g` on GPU 0.
        pub fn new(
            g: &Graph,
            k: usize,
            p: usize,
            threads: Option<u32>,
            table: Table,
        ) -> Result<Self, Error> {
            let n = g.n();
            assert!(n * k < 1 << 24, "move index must fit 24 bits");
            let ctx = cudarc::driver::CudaContext::new(0)?;
            let device = ctx.name()?;
            let stream = ctx.default_stream();
            let ptx = cudarc::nvrtc::compile_ptx(KERNEL)?;
            let module = ctx.load_module(ptx)?;
            assert!(
                g.adj.iter().all(|nb| nb.len() < 1 << 16),
                "16-bit conflict counts"
            );
            // the colouring and the 16-bit conflict table in shared memory, if they fit
            assert!(
                k < 256 && n < 1 << 16,
                "8-bit colours and 16-bit vertex numbers"
            );
            // shared bytes with a 16- or an 8-bit table (padded to 8), the list and the colouring
            let need16 = ((2 * n * k + 7) & !7) + 3 * n;
            let need8 = ((n * k + 7) & !7) + 3 * n;
            let optin = ctx.attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
            )? as usize;
            let eight = match table {
                Table::Bits16 => false,
                Table::Bits8 => {
                    assert!(
                        need8 <= optin,
                        "an 8-bit table needs {need8} bytes, the device allows {optin}"
                    );
                    true
                }
                Table::Auto => need16 > optin && need8 <= optin,
            };
            let f = module.load_function(if eight { "tabucol8" } else { "tabucol16" })?;
            let need = if eight { need8 } else { need16 };
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
            let global = |len: usize| if shared_bytes == 0 { len } else { 1 };
            let occupancy = |t: u32| {
                f.occupancy_max_active_blocks_per_multiprocessor(t, shared_bytes as usize, None)
            };
            // with room for one block per SM, the block must be as wide as the
            // SM holds to keep it busy (A2.6: 1,024 threads ran 8× faster than
            // 128 with the 8-bit table on DSJC1000.5)
            let threads = match threads {
                Some(t) => t,
                None if shared_bytes > 0 && occupancy(128)? == 1 => {
                    let mut t = 1024;
                    while t > 128 && occupancy(t)? == 0 {
                        t /= 2;
                    }
                    t
                }
                None => 128,
            };
            assert!(threads.is_multiple_of(32), "whole warps");
            let blocks_per_sm = occupancy(threads)?;
            let sms = ctx.attribute(
                cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
            )?;
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
                cur: stream.alloc_zeros(global(p * n))?,
                list: stream.alloc_zeros(global(p * n))?,
                gamma: stream.alloc_zeros(global(p * n * k))?,
                gamma8: stream.alloc_zeros(1)?,
                overflow: stream.alloc_zeros(p)?,
                eight,
                tabu: stream.alloc_zeros(p * n * k)?,
                cost: stream.alloc_zeros(p)?,
                stream,
                f,
                n,
                k,
                p,
                threads,
                shared_bytes,
                resident: {
                    let table = if shared_bytes == 0 {
                        3 * p * n + 2 * p * n * k
                    } else {
                        0
                    };
                    let bytes =
                        4 * p * n * k + 4 * p * n + 8 * p + 4 * (n + 1 + 2 * g.edges) + table;
                    format!(
                        "{}-bit conflict table, {blocks_per_sm} blocks per SM × {sms} SMs, {} MB of device memory",
                        if eight { 8 } else { 16 },
                        bytes.div_ceil(1 << 20)
                    )
                },
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
            self.stream
                .memcpy_htod(&vec![0i32; self.p], &mut self.overflow)?;
            let cfg = LaunchConfig {
                grid_dim: (self.p as u32, 1, 1),
                block_dim: (self.threads, 1, 1),
                shared_mem_bytes: self.shared_bytes,
            };
            let mut launch = self.stream.launch_builder(&self.f);
            launch
                .arg(&self.start)
                .arg(&self.adj)
                .arg(&n32)
                .arg(&k32)
                .arg(&mut self.best)
                .arg(&mut self.cur);
            if self.eight {
                launch.arg(&mut self.gamma8);
            } else {
                launch.arg(&mut self.gamma);
            }
            launch
                .arg(&mut self.list)
                .arg(&mut self.tabu)
                .arg(&mut self.cost)
                .arg(&iters)
                .arg(&seed)
                .arg(&use_shared)
                .arg(&mut self.overflow);
            unsafe { launch.launch(cfg) }?;
            let saturated = self
                .stream
                .clone_dtoh(&self.overflow)?
                .iter()
                .filter(|&&o| o != 0)
                .count();
            // a saturated counter makes that block's table wrong from then on
            // (a 256 counted as 255 later decrements to 254): fail the run
            // rather than let a drifted cost reach a result
            assert!(
                saturated == 0,
                "the 8-bit conflict table saturated in {saturated} blocks this batch; \
                 rerun with --table 16"
            );
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
                b => format!("tables in {} KB of shared memory", b.div_ceil(1024)),
            };
            format!(
                "GPU {}, {} threads per block, {tables}, {}",
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
    let polish_every: u64 = args.value("--polish-every").unwrap_or(0);
    let polish_iters: u64 = args.value("--polish-iters").unwrap_or(1000);
    let vss_rounds: usize = args.value("--vss-rounds").unwrap_or(0);
    let vss_iters: u64 = args.value("--vss-iters").unwrap_or(100);
    let seed: u64 = args.value("--seed").unwrap_or(1);
    let threads: Option<u32> = args.value("--threads");
    let table = match args.value::<String>("--table").as_deref() {
        None | Some("auto") => Table::Auto,
        Some("8") => Table::Bits8,
        Some("16") => Table::Bits16,
        Some(x) => panic!("--table auto|8|16, not {x:?}"),
    };
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
        gpu_improver(&g, k, p, threads, table)
    };
    trace.line(format_args!(
        "start build={} file={file} n={n} edges={} k={k} pairs={pairs} iters={iters} threads={} seed={seed} improver=\"{}\" setup_s={:.3} (includes NVRTC compile)",
        env!("METS_GIT_HASH"),
        g.edges,
        threads.map_or("auto".to_string(), |t| t.to_string()),
        improver.name(),
        setup.elapsed().as_secs_f64(),
    ));
    println!(
        "{file}: {n} vertices, {} edges; {k} colours; {pairs} HEAD pairs × {iters_n}n TabuCol iterations; {}; build {}",
        g.edges,
        improver.name(),
        env!("METS_GIT_HASH")
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
        // polish a near-solution: the whole batch continues the best colouring
        if polish_every > 0 && generation % polish_every == 0 && best.0 <= 2 {
            let t = Instant::now();
            let mut copies = vec![best.1.clone(); p];
            let mut done = 0;
            while done < polish_iters && best.0 > 0 {
                let chunk = (polish_iters - done).min(200);
                let c = improver.improve(&mut copies, chunk * n as u64, rng.random());
                if let Some(i) = (0..p).min_by_key(|&i| c[i]) {
                    if c[i] < best.0 {
                        best = (c[i], copies[i].clone());
                    }
                }
                done += chunk;
            }
            // then the other search space, on the CPU threads
            if vss_rounds > 0 && best.0 > 0 {
                let start = Coloring::new(&g, k, best.1.clone());
                let seed0: u64 = rng.random();
                let results = mets::parallel::run_many(jobs as u64, jobs, |r, cancel| {
                    let (c, col) = vss(&start, vss_iters * n as u64, vss_rounds, seed0 + r);
                    if c == 0 {
                        cancel.cancel();
                    }
                    (c, col)
                });
                if let Some((_, (c, col))) = results.into_iter().min_by_key(|(_, (c, _))| *c) {
                    if c < best.0 {
                        best = (c, col);
                    }
                }
            }
            let secs = t.elapsed().as_secs_f64();
            println!(
                "  polish after generation {generation}: best {} conflicts, {secs:.1} s",
                best.0
            );
            trace.line(format_args!(
                "polish gen={generation} iters={} best={} polish_s={secs:.3}",
                polish_iters * n as u64,
                best.0
            ));
            if best.0 == 0 {
                break;
            }
        }
        // HEAD: each member's child with its partner always replaces it
        // GPX on the CPU threads, one seeded generator per chunk (reproducible)
        let chunk = p.div_ceil(jobs).max(2) & !1; // whole pairs per chunk
        let seeds: Vec<u64> = (0..p.div_ceil(chunk)).map(|_| rng.random()).collect();
        std::thread::scope(|scope| {
            for ((c, kids), &seed) in children.chunks_mut(chunk).enumerate().zip(&seeds) {
                let pop = &pop;
                scope.spawn(move || {
                    let mut rng = DefaultRng::seed_from_u64(seed);
                    for (j, child) in kids.iter_mut().enumerate() {
                        let i = c * chunk + j;
                        gpx(&pop[i], &pop[i ^ 1], k, child, &mut rng);
                    }
                });
            }
        });
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
        let out = fresh_path(&format!("{name}.{k}"), "sol"); // never overwrite
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
fn gpu_improver<'g>(
    g: &'g Graph,
    k: usize,
    p: usize,
    threads: Option<u32>,
    table: Table,
) -> Box<dyn Improver + 'g> {
    Box::new(gpu::GpuTabuCol::new(g, k, p, threads, table).unwrap_or_else(|e| panic!("GPU: {e}")))
}

#[cfg(not(feature = "gpu"))]
fn gpu_improver<'g>(
    _: &'g Graph,
    _: usize,
    _: usize,
    _: Option<u32>,
    _: Table,
) -> Box<dyn Improver + 'g> {
    panic!("built without the GPU: rebuild with --features gpu, or run with --cpu")
}
