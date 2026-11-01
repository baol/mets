//! The model: your solution, its objective, its moves, and the cost cache.

use num_traits::{Bounded, Signed, ToPrimitive};
use std::fmt::Debug;
use std::ops::Deref;

/// An objective value. Blanket-implemented for every type with these bounds:
/// `i32`, `i64`, `i128`, `isize`, `f32`, `f64`, and fixed-point, decimal or
/// user types that implement them.
///
/// Integer costs are recommended when the objective is integral: comparisons
/// are exact and no tolerance is needed.
pub trait Cost: Copy + PartialOrd + Debug + Signed + Bounded + ToPrimitive {}
impl<T: Copy + PartialOrd + Debug + Signed + Bounded + ToPrimitive> Cost for T {}

/// `c` as `f64` (for SA-style acceptance rules).
#[inline]
pub(crate) fn to_f64<C: Cost>(c: C) -> f64 {
    c.to_f64().unwrap_or(f64::NAN)
}

/// `true` for integer-like cost types (where `1 / 2 == 0`).
#[inline]
fn is_integral<C: Cost>() -> bool {
    C::one() / (C::one() + C::one()) == C::zero()
}

/// Numerical equality used by the delta checks: exact for integers, within a
/// relative 1e-9 of `scale` for floats.
pub(crate) fn approx_eq<C: Cost>(a: C, b: C, scale: C) -> bool {
    if a == b {
        return true;
    }
    if is_integral::<C>() {
        return false;
    }
    let s = 1f64.max(to_f64(scale).abs());
    (to_f64(a) - to_f64(b)).abs() <= 1e-9 * s
}

/// A solution, with its objective computed from scratch.
///
/// Keep immutable instance data out of the solution (borrow it with
/// `&'a Instance`) so that cloning a solution only copies the state.
pub trait Model: Clone {
    type Cost: Cost;
    /// The objective, computed from scratch. It gives the starting cost and is
    /// the reference every delta is checked against in debug builds.
    fn evaluate(&self) -> Self::Cost;
}

/// A transformation of a solution into a neighbor. Moves are plain values,
/// usually `Copy`: no allocation, no virtual call.
pub trait Move<S: Model>: Clone {
    fn apply(&self, s: &mut S);

    /// `cost(after) - cost(before)` without applying the move, if you have a
    /// formula. `None` (the default) means "evaluate the neighbor in full".
    #[inline]
    fn delta(&self, _s: &Evaluated<S>) -> Option<S::Cost> {
        None
    }

    /// A move that undoes `self`, computed on the solution *before* `self` is
    /// applied (a swap is its own inverse). Optional; it lets
    /// [`FullEvaluation`](crate::FullEvaluation) use apply/evaluate/undo
    /// instead of cloning.
    #[inline]
    fn inverse(&self, _before: &S) -> Option<Self> {
        None
    }
}

/// Your solution plus its cached cost. The cache is always valid: the only
/// ways to change the solution are [`apply`](Self::apply) and
/// [`modify`](Self::modify).
///
/// It implements `Deref<Target = S>` so you read your fields directly
/// (`s.picked[i]`). This is a deliberate smart-pointer-style exception: the
/// wrapper owns and guards one value, and there is no `DerefMut`.
pub struct Evaluated<S: Model> {
    inner: S,
    cost: S::Cost,
}

impl<S: Model> Evaluated<S> {
    /// Wraps `s`, evaluating it once.
    pub fn new(s: S) -> Self {
        let cost = s.evaluate();
        Evaluated { inner: s, cost }
    }

    /// The cached cost, O(1).
    #[inline]
    pub fn cost(&self) -> S::Cost {
        self.cost
    }

    /// The move's delta, or `evaluate(after) - cost()` (clone + apply) if the
    /// move has no formula.
    #[inline]
    pub fn delta_of<M: Move<S>>(&self, m: &M) -> S::Cost {
        match m.delta(self) {
            Some(d) => d,
            None => {
                let mut t = self.inner.clone();
                m.apply(&mut t);
                t.evaluate() - self.cost
            }
        }
    }

    /// Apply one move, updating the cache with the move's delta, or by
    /// re-evaluating when the move has no formula. Apply *sequences* of moves
    /// without formula through [`modify`](Self::modify) (one evaluation).
    pub fn apply<M: Move<S>>(&mut self, m: &M) {
        match m.delta(self) {
            Some(d) => {
                m.apply(&mut self.inner);
                self.cost = self.cost + d;
            }
            None => {
                m.apply(&mut self.inner);
                self.cost = self.inner.evaluate();
            }
        }
    }

    /// Apply a move whose delta is already known (no re-evaluation).
    #[inline]
    pub(crate) fn apply_known<M: Move<S>>(&mut self, m: &M, delta: S::Cost) {
        m.apply(&mut self.inner);
        self.cost = self.cost + delta;
    }

    /// Arbitrary change (re-initialisation, repair, several moves at once…).
    /// Re-evaluates once afterwards.
    pub fn modify<T>(&mut self, f: impl FnOnce(&mut S) -> T) -> T {
        let r = f(&mut self.inner);
        self.cost = self.inner.evaluate();
        r
    }

    /// Re-evaluate and reset the cache. Returns the drift that was corrected
    /// (`cached - evaluated`), always zero for integer costs.
    pub fn resync(&mut self) -> S::Cost {
        let fresh = self.inner.evaluate();
        let drift = self.cost - fresh;
        self.cost = fresh;
        drift
    }

    pub fn into_inner(self) -> S {
        self.inner
    }
}

impl<S: Model> Deref for Evaluated<S> {
    type Target = S;
    #[inline]
    fn deref(&self) -> &S {
        &self.inner
    }
}

impl<S: Model> Clone for Evaluated<S> {
    fn clone(&self) -> Self {
        Evaluated {
            inner: self.inner.clone(),
            cost: self.cost,
        }
    }
    /// Reuses the destination's buffers: recording a best solution does not
    /// allocate once warmed up.
    fn clone_from(&mut self, source: &Self) {
        self.inner.clone_from(&source.inner);
        self.cost = source.cost;
    }
}

impl<S: Model + Debug> Debug for Evaluated<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Evaluated")
            .field("cost", &self.cost)
            .field("solution", &self.inner)
            .finish()
    }
}
