//! Port of metslib-examples/tutorial: subset sum with tabu search.
//!
//! Pick a subset of the numbers whose sum is as close as possible to the
//! target; overshooting is penalised 100×.

use mets::*;
use std::ops::ControlFlow;

#[derive(Clone)]
struct Subset<'a> {
    set: &'a [i64], // instance data is borrowed, never copied
    target: i64,
    picked: Vec<bool>,
    sum: i64, // running sum, kept by `apply` so the delta is O(1)
}

fn penalty(diff: i64) -> i64 {
    if diff < 0 {
        -100 * diff
    } else {
        diff
    }
}

impl Model for Subset<'_> {
    type Cost = i64;
    /// From scratch: the reference the delta is checked against.
    fn evaluate(&self) -> i64 {
        let sum: i64 = (0..self.set.len())
            .filter(|&i| self.picked[i])
            .map(|i| self.set[i])
            .sum();
        penalty(self.target - sum)
    }
}

/// Toggle element i in/out of the subset.
#[derive(Clone, Copy)]
struct Toggle(usize);

impl<'a> Move<Subset<'a>> for Toggle {
    fn apply(&self, s: &mut Subset<'a>) {
        let v = s.set[self.0];
        s.sum += if s.picked[self.0] { -v } else { v };
        s.picked[self.0] ^= true;
    }
    fn delta(&self, s: &Evaluated<Subset<'a>>) -> Option<i64> {
        let v = s.set[self.0];
        let sum = if s.picked[self.0] {
            s.sum - v
        } else {
            s.sum + v
        };
        Some(penalty(s.target - sum) - s.cost())
    }
}

impl<'a> TabuAttr<Subset<'a>> for Toggle {
    type Key = usize;
    fn key(&self, _: &Subset<'a>) -> usize {
        self.0
    }
}
impl<'a> DenseAttr<Subset<'a>> for Toggle {
    fn key_count(s: &Subset<'a>) -> usize {
        s.set.len()
    }
}

struct AllToggles;

impl<'a> Neighborhood<Subset<'a>> for AllToggles {
    type Move = Toggle;
    fn explore<V: Visitor<Toggle, i64>>(
        &mut self,
        s: &Evaluated<Subset<'a>>,
        v: &mut V,
    ) -> ControlFlow<()> {
        for i in 0..s.set.len() {
            let m = Toggle(i);
            v.visit(m, s.delta_of(&m))?;
        }
        ControlFlow::Continue(())
    }
}

fn main() {
    let numbers = [
        475, 382, -202, 351, 296, -362, 336, 117, -319, 416, -304, 364, -386, -9, 391, 389, -457,
        261, -323, -498, 407, -81, 445, -308, 258, -274, 156,
    ];
    let start = Subset {
        set: &numbers,
        target: 109,
        picked: vec![false; numbers.len()],
        sum: 0,
    };

    let mut st = State::new(start, 42);
    let mut ts = TabuSearch::new(AllToggles, DenseTabu::new(Tenure::Fixed(7)));
    let outcome = ts
        .runner(&mut st)
        .until(Target::new(0).or(NoImprovement::new(200)))
        .on_move(|st| eprintln!("{}: {}", st.iteration(), st.current().cost()))
        .run();

    let picked: Vec<i64> = (0..numbers.len())
        .filter(|&i| st.best().picked[i])
        .map(|i| numbers[i])
        .collect();
    println!(
        "Best solution: {} ({:?} after {} iterations)",
        st.best().cost(),
        outcome.stop,
        outcome.iterations
    );
    println!("{picked:?} (sum = {})", st.best().sum);
}
