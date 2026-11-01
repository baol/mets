//! Ready-made pieces for permutation problems (QAP, TSP, scheduling…): the
//! counterpart of METSlib's `permutation_problem`, `swap_elements`,
//! `swap_full_neighborhood` and `swap_neighborhood`.

use crate::{DenseAttr, Evaluated, Model, Move, Neighborhood, Sampler, TabuAttr, Visitor};
use rand::Rng;
use std::ops::ControlFlow;

/// A solution encoded as a permutation.
pub trait Permutation: Model {
    fn size(&self) -> usize;
    /// Exchange the elements at positions `i` and `j`.
    fn swap(&mut self, i: usize, j: usize);
    /// `cost(after swap) - cost(before)`, if you have a formula. Default:
    /// none, the library evaluates the neighbor in full.
    #[inline]
    fn swap_delta(&self, _i: usize, _j: usize) -> Option<Self::Cost> {
        None
    }
}

/// Exchange the elements at positions `i < j`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Swap {
    pub i: usize,
    pub j: usize,
}

impl Swap {
    /// Normalises the order so that `i < j`.
    pub fn new(a: usize, b: usize) -> Self {
        Swap {
            i: a.min(b),
            j: a.max(b),
        }
    }
}

impl<S: Permutation> Move<S> for Swap {
    #[inline]
    fn apply(&self, s: &mut S) {
        s.swap(self.i, self.j)
    }
    #[inline]
    fn delta(&self, s: &Evaluated<S>) -> Option<S::Cost> {
        s.swap_delta(self.i, self.j)
    }
    #[inline]
    fn inverse(&self, _: &S) -> Option<Self> {
        Some(*self)
    }
}

/// Swaps are their own inverse: forbid the same pair. Dense key `i * n + j`.
impl<S: Permutation> TabuAttr<S> for Swap {
    type Key = usize;
    #[inline]
    fn key(&self, s: &S) -> usize {
        self.i * s.size() + self.j
    }
}
impl<S: Permutation> DenseAttr<S> for Swap {
    fn key_count(s: &S) -> usize {
        s.size() * s.size()
    }
}

/// All n(n-1)/2 swaps, generated on the fly; also a uniform O(1) [`Sampler`].
#[derive(Clone, Copy, Debug, Default)]
pub struct SwapNeighborhood;

impl<S: Permutation> Neighborhood<S> for SwapNeighborhood {
    type Move = Swap;
    #[inline]
    fn explore<V: Visitor<Swap, S::Cost>>(
        &mut self,
        s: &Evaluated<S>,
        v: &mut V,
    ) -> ControlFlow<()> {
        let n = s.size();
        for i in 0..n {
            for j in i + 1..n {
                let m = Swap { i, j };
                v.visit(m, s.delta_of(&m))?;
            }
        }
        ControlFlow::Continue(())
    }
}

impl<S: Permutation> Sampler<S> for SwapNeighborhood {
    type Move = Swap;
    #[inline]
    fn sample<R: Rng + ?Sized>(&mut self, s: &Evaluated<S>, rng: &mut R) -> Option<Swap> {
        random_swap(s.size(), rng)
    }
}

/// A uniformly random swap of two distinct positions.
#[inline]
pub fn random_swap<R: Rng + ?Sized>(n: usize, rng: &mut R) -> Option<Swap> {
    if n < 2 {
        return None;
    }
    let i = rng.random_range(0..n);
    let mut j = rng.random_range(0..n - 1);
    if j >= i {
        j += 1;
    }
    Some(Swap::new(i, j))
}

/// Apply `k` random swaps (METSlib's `mets::perturbate`), re-evaluating once.
pub fn perturb<S: Permutation, R: Rng + ?Sized>(s: &mut Evaluated<S>, k: usize, rng: &mut R) {
    s.modify(|p| {
        for _ in 0..k {
            if let Some(m) = random_swap(p.size(), rng) {
                p.swap(m.i, m.j);
            }
        }
    })
}

/// Shuffle uniformly (Fisher–Yates), re-evaluating once.
pub fn shuffle<S: Permutation, R: Rng + ?Sized>(s: &mut Evaluated<S>, rng: &mut R) {
    s.modify(|p| {
        for i in (1..p.size()).rev() {
            let j = rng.random_range(0..=i);
            if i != j {
                p.swap(j, i);
            }
        }
    })
}
