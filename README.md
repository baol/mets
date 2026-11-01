# mets

Neighborhood-based metaheuristics in Rust: local search, tabu search,
simulated annealing, late acceptance, record-to-record travel, threshold
accepting, great deluge, and iterated searches (ILS, iterated tabu search,
random restarts). A Rust port of [METSlib](https://github.com/baol/metslib).

You write the objective and the moves; deltas are optional, and the library
keeps the cached cost correct.

```rust
use mets::perm::{Permutation, SwapNeighborhood};
use mets::*;

/// Sort a permutation: the cost is the total displacement.
#[derive(Clone)]
struct Sort(Vec<usize>);

impl Model for Sort {
    type Cost = i64;
    fn evaluate(&self) -> i64 {
        self.0.iter().enumerate().map(|(i, &v)| (i as i64 - v as i64).abs()).sum()
    }
}

impl Permutation for Sort {
    fn size(&self) -> usize { self.0.len() }
    fn swap(&mut self, i: usize, j: usize) { self.0.swap(i, j) }
}

fn main() {
    let mut st = State::new(Sort((0..10).rev().collect()), 1);
    TabuSearch::new(SwapNeighborhood, DenseTabu::new(Tenure::Fixed(5)))
        .runner(&mut st)
        .until(Target::new(0).or(MaxIterations::new(1000)))
        .on_improvement(|st| println!("{}: {}", st.iteration(), st.best().cost()))
        .run();
    assert_eq!(st.best().cost(), 0);
}
```

Add a delta formula when you have one (`Permutation::swap_delta`, or
`Move::delta` for your own moves): nothing else changes, and debug builds check
it against the full evaluation.

## Examples

```bash
cargo run --release --example subset_sum   # tabu search (METSlib tutorial)
cargo run --release --example qap          # QAP: tabu, iterated tabu, SA, LAHC, RRT, ILS
cargo run --release --example vcp          # graph colouring, TabuCol-like iterated tabu
```

## Documentation

[`DESIGN.md`](DESIGN.md) explains the API and the design choices, compares it
with METSlib, and has benchmarks against the C++ library. `cargo doc --open`
for the API reference.

## License

GPL-3.0-or-later or CPL-1.0, at your option, like METSlib: see
[`COPYING`](COPYING) and [`COPYING-CPL`](COPYING-CPL).
