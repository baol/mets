//! Independent runs on several threads.
//!
//! Each run builds its own `State` from its run index (use it as the seed),
//! so results do not depend on the number of threads. A shared [`Cancel`]
//! token stops every run early, e.g. once a target cost is reached.

use crate::{Model, State, Termination};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;

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
