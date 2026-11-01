//! Test helpers for models.

use crate::model::approx_eq;
use crate::{collect_moves, Evaluated, Model, Move, Neighborhood};
use std::fmt::Debug;

/// A move whose delta does not match full evaluations.
#[derive(Debug)]
pub struct DeltaMismatch<M, C> {
    pub mv: M,
    pub reported: C,
    pub evaluated: C,
}

/// Compare the delta of every move of `n` with `evaluate(after) - evaluate(before)`.
/// Nothing is applied to `s`. Returns every mismatch (empty when all agree).
pub fn deltas<S, N>(n: &mut N, s: &Evaluated<S>) -> Vec<DeltaMismatch<N::Move, S::Cost>>
where
    S: Model,
    N: Neighborhood<S>,
    N::Move: Debug,
{
    let before = s.evaluate();
    collect_moves(n, s)
        .into_iter()
        .filter_map(|(m, reported)| {
            let mut t = (**s).clone();
            m.apply(&mut t);
            let evaluated = t.evaluate() - before;
            (!approx_eq(reported, evaluated, before)).then_some(DeltaMismatch {
                mv: m,
                reported,
                evaluated,
            })
        })
        .collect()
}
