//! # mets — neighborhood-based metaheuristics
//!
//! A Rust port of METSlib. You write:
//!
//! * a [`Model`]: your solution and its objective, computed from scratch;
//! * [`Move`]s: how to apply them, and optionally a delta formula;
//! * a [`Neighborhood`] (or a [`MoveSource`], or a [`Sampler`]) generating moves.
//!
//! The library wraps the solution in [`Evaluated`], which caches the cost and
//! keeps it correct, and provides the algorithms: [`LocalSearch`],
//! [`TabuSearch`], [`StochasticSearch`] with SA / late acceptance /
//! record-to-record travel / threshold accepting / great deluge, and
//! [`IteratedSearch`] for ILS, iterated tabu search and random restarts,
//! [`LargeNeighborhoodSearch`] (destroy and repair) and [`Memetic`]
//! (a population with crossover). [`perm`] has ready-made pieces for
//! permutation problems, and [`parallel`] runs independent searches on
//! several threads. Every algorithm is a [`Search`]: write one `step`
//! function to add yours.
//!
//! ```
//! use mets::perm::{Permutation, SwapNeighborhood};
//! use mets::*;
//!
//! /// Sort a permutation: the cost is the total displacement.
//! #[derive(Clone)]
//! struct Sort(Vec<usize>);
//! impl Model for Sort {
//!     type Cost = i64;
//!     fn evaluate(&self) -> i64 {
//!         self.0.iter().enumerate().map(|(i, &v)| (i as i64 - v as i64).abs()).sum()
//!     }
//! }
//! impl Permutation for Sort {
//!     fn as_slice(&self) -> &[usize] { &self.0 }
//!     fn swap(&mut self, i: usize, j: usize) { self.0.swap(i, j) }
//! }
//!
//! let mut st = State::new(Sort((0..10).rev().collect()), 1);
//! let mut ls = LocalSearch::best_improvement(SwapNeighborhood);
//! let outcome = ls.runner(&mut st).run();
//! assert_eq!(outcome.stop, Stop::Stuck); // local optimum
//! assert_eq!(st.best().cost(), 0);
//! ```

mod ils;
mod lns;
mod local_search;
mod model;
mod neighborhood;
mod population;
mod search;
mod state;
mod stochastic;
mod tabu;
mod termination;

pub mod check;
pub mod parallel;
pub mod perm;

pub use ils::{perturb_with, Adaptive, IteratedSearch, PerturbWith, Perturbation};
pub use lns::{rebuild_with, LargeNeighborhoodSearch, Rebuild, RebuildWith};
pub use local_search::LocalSearch;
pub use model::{Cost, Evaluated, Model, Move};
pub use neighborhood::{
    collect_moves, moves_fn, BestMove, FullEvaluation, MoveSource, MovesFn, Neighborhood, Sampled,
    Sampler, Visitor, WithDeltas,
};
pub use population::{crossover_with, Crossover, CrossoverWith, Memetic};
pub use search::{Outcome, Runner, Search, SearchExt, Step, Stop};
pub use state::{
    log_improvements, on_event, on_global_improvement, on_improvement, on_move, Event,
    LogImprovements, Observer, OnEvent, OnGlobalImprovement, OnImprovement, OnMove, Scope, State,
};
pub use stochastic::{
    Acceptance, Always, GreatDeluge, Improving, LateAcceptance, NonWorsening, RecordToRecord,
    SimulatedAnnealing, StochasticSearch, ThresholdAccepting,
};
pub use tabu::{
    Aspiration, BestEver, DenseAttr, DenseTabu, DynamicTabu, NoAspiration, RecencyTabu, TabuAttr,
    TabuList, TabuSearch, Tenure,
};
pub use termination::{
    stop_when, And, MaxIterations, NoImprovement, Or, StopWhen, Target, Termination,
    TerminationExt, TimeLimit, Unlimited,
};

pub use rand;

/// The default RNG: same output on every platform.
pub type DefaultRng = rand_xoshiro::Xoshiro256PlusPlus;

/// The README's example, compiled and run as a doctest.
#[cfg(doctest)]
#[doc = include_str!("../README.md")]
struct ReadmeDoctest;
