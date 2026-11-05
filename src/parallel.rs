//! Several searches on several threads.
//!
//! * [`run_many`]: independent runs. Each builds its own `State` from its run
//!   index (use it as the seed), so results do not depend on the number of
//!   threads.
//! * [`islands`]: an island model. Each island runs on its own thread and,
//!   at every [`Island::exchange`], sends a solution to the next island of a
//!   ring and receives one from the previous. Exchanges are synchronised, so
//!   results depend only on the seeds. [`Island::post`] / [`Island::take`]
//!   migrate asynchronously instead (no waiting, timing-dependent results).
//!
//! A shared [`Cancel`] token stops every search early, e.g. once a target cost
//! is reached.

use crate::{Evaluated, Model, State, Termination};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Barrier, Mutex};

/// A cancellation flag shared by parallel runs. `&Cancel` is a
/// [`Termination`] criterion, so `NoImprovement::new(k).or(&cancel)` stops
/// a search (inner or outer) as soon as another thread cancels.
#[derive(Debug, Default)]
pub struct Cancel(AtomicBool);

impl Cancel {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed)
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
}

impl<S: Model, R> Termination<S, R> for &Cancel {
    #[inline]
    fn stop(&mut self, _: &State<S, R>) -> bool {
        self.is_cancelled()
    }
}

/// Run `f(run, &cancel)` for `run` in `0..runs` on `jobs` threads.
///
/// Returns the results of the runs that were started, in run order. Once
/// `cancel` is set, runs not yet started are skipped (running ones should
/// observe the token through their termination criteria).
pub fn run_many<T, F>(runs: u64, jobs: usize, f: F) -> Vec<(u64, T)>
where
    T: Send,
    F: Fn(u64, &Cancel) -> T + Sync,
{
    let cancel = Cancel::new();
    let next = AtomicU64::new(0);
    let results = Mutex::new(Vec::new());
    std::thread::scope(|scope| {
        for _ in 0..jobs.max(1) {
            scope.spawn(|| loop {
                let run = next.fetch_add(1, Ordering::Relaxed);
                if run >= runs || cancel.is_cancelled() {
                    break;
                }
                let r = f(run, &cancel);
                results.lock().unwrap().push((run, r));
            });
        }
    });
    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|(run, _)| *run);
    results
}

/// One island of [`islands`]: its index and its link to the ring.
pub struct Island<'a, S: Model> {
    index: usize,
    slots: &'a [Mutex<Option<Evaluated<S>>>],
    barrier: &'a Barrier,
    /// Shared by all islands: cancel to stop them all.
    pub cancel: &'a Cancel,
}

impl<S: Model> Island<'_, S> {
    /// This island's index, `0..n` (use it to seed the island's search).
    pub fn index(&self) -> usize {
        self.index
    }

    /// Send `outgoing` to the next island and receive what the previous one
    /// sent. Every island must call this the same number of times (it waits
    /// for all of them), so run a fixed number of epochs and let a cancelled
    /// search finish its remaining epochs immediately.
    /// Asynchronous migration: leave `outgoing` in the next island's mailbox
    /// (replacing an unread one), without waiting. Faster islands don't wait
    /// for slower ones, but results then depend on thread timing.
    pub fn post(&self, outgoing: Evaluated<S>) {
        let n = self.slots.len();
        *self.slots[(self.index + 1) % n].lock().unwrap() = Some(outgoing);
    }

    /// Asynchronous migration: take what the previous island last posted, if
    /// anything. Never waits.
    pub fn take(&self) -> Option<Evaluated<S>> {
        self.slots[self.index].lock().unwrap().take()
    }

    pub fn exchange(&self, outgoing: Option<Evaluated<S>>) -> Option<Evaluated<S>> {
        let n = self.slots.len();
        *self.slots[(self.index + 1) % n].lock().unwrap() = outgoing;
        self.barrier.wait();
        let received = self.slots[self.index].lock().unwrap().take();
        self.barrier.wait();
        received
    }
}

/// Run `n` islands, one thread each, connected in a ring. Returns each
/// island's result, in island order.
///
/// ```ignore
/// parallel::islands(4, |island| {
///     let mut st = State::new(start(island.index()), island.index() as u64);
///     let mut memetic = Memetic::new(..);
///     for epoch in 0..epochs {
///         let run = memetic.runner(&mut st).until(MaxIterations::new(50).or(island.cancel));
///         if epoch == 0 { run.run() } else { run.resume() };
///         if let Some(m) = island.exchange(memetic.best().cloned()) { memetic.immigrate(m); }
///     }
///     st.global_best().cost()
/// })
/// ```
pub fn islands<S, T, F>(n: usize, f: F) -> Vec<T>
where
    S: Model + Send,
    S::Cost: Send,
    T: Send,
    F: Fn(&Island<S>) -> T + Sync,
{
    let n = n.max(1);
    let cancel = Cancel::new();
    let barrier = Barrier::new(n);
    let slots: Vec<Mutex<Option<Evaluated<S>>>> = (0..n).map(|_| Mutex::new(None)).collect();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..n)
            .map(|index| {
                let island = Island {
                    index,
                    slots: &slots,
                    barrier: &barrier,
                    cancel: &cancel,
                };
                let f = &f;
                scope.spawn(move || f(&island))
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("island panicked"))
            .collect()
    })
}
