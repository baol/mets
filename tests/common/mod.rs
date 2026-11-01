//! Toy permutation problems shared by the tests: minimise the weighted
//! displacement sum |p[i] - i| * w(i). The optimum is the identity (cost 0).
#![allow(dead_code)]

use mets::perm::Permutation;
use mets::Model;

pub fn term(i: usize, v: usize) -> i64 {
    (i as i64 - v as i64).abs() * (1 + i as i64 % 3)
}

fn cost(p: &[usize]) -> i64 {
    p.iter().enumerate().map(|(i, &v)| term(i, v)).sum()
}

/// With a delta formula.
#[derive(Clone, Debug)]
pub struct Sort(pub Vec<usize>);

impl Model for Sort {
    type Cost = i64;
    fn evaluate(&self) -> i64 {
        cost(&self.0)
    }
}
impl Permutation for Sort {
    fn size(&self) -> usize {
        self.0.len()
    }
    fn swap(&mut self, i: usize, j: usize) {
        self.0.swap(i, j)
    }
    fn swap_delta(&self, i: usize, j: usize) -> Option<i64> {
        let (a, b) = (self.0[i], self.0[j]);
        Some(term(i, b) + term(j, a) - term(i, a) - term(j, b))
    }
}

/// The same problem with full evaluation only.
#[derive(Clone, Debug)]
pub struct SortFull(pub Vec<usize>);

impl Model for SortFull {
    type Cost = i64;
    fn evaluate(&self) -> i64 {
        cost(&self.0)
    }
}
impl Permutation for SortFull {
    fn size(&self) -> usize {
        self.0.len()
    }
    fn swap(&mut self, i: usize, j: usize) {
        self.0.swap(i, j)
    }
}

/// A delta formula with a bug (forgets the weights).
#[derive(Clone, Debug)]
pub struct SortBuggy(pub Vec<usize>);

impl Model for SortBuggy {
    type Cost = i64;
    fn evaluate(&self) -> i64 {
        cost(&self.0)
    }
}
impl Permutation for SortBuggy {
    fn size(&self) -> usize {
        self.0.len()
    }
    fn swap(&mut self, i: usize, j: usize) {
        self.0.swap(i, j)
    }
    fn swap_delta(&self, i: usize, j: usize) -> Option<i64> {
        let (a, b) = (self.0[i] as i64, self.0[j] as i64);
        let (i, j) = (i as i64, j as i64);
        Some((i - b).abs() + (j - a).abs() - (i - a).abs() - (j - b).abs())
    }
}

pub fn reversed(n: usize) -> Vec<usize> {
    (0..n).rev().collect()
}
