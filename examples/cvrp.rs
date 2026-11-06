//! Capacitated vehicle routing (PLAN-problems.md P7). The port of the
//! Cordeau & Maischberger parallel iterated tabu search comes in steps; for
//! now this example checks the reader and the `Routes` model against
//! CVRPLIB solutions.
//!
//! usage:
//!   cvrp --check VRP_DIR SOL_DIR [--real | --round]
//!       read every FILE.vrp and its FILE.sol, build the routes, and compare
//!       the distance and feasibility with the stated cost
//!
//! `scripts/verify_vrp.py` does the same with none of this code: the two
//! must agree.

#[path = "common/qap.rs"]
mod common; // only for its argument parser
#[path = "common/vrp.rs"]
mod vrp;

use common::Args;
use mets::Model;
use std::path::Path;
use vrp::*;

fn main() {
    let mut args = Args::from_env();
    let rounding = if args.flag("--real") {
        Some(Rounding::Real)
    } else if args.flag("--round") {
        Some(Rounding::Nearest)
    } else {
        None
    };
    let check = args.flag("--check");
    assert!(
        check,
        "usage: cvrp --check VRP_DIR SOL_DIR [--real | --round]"
    );
    let (vrp_dir, sol_dir) = (
        Path::new(args.0.first().expect("VRP_DIR")).to_path_buf(),
        Path::new(args.0.get(1).expect("SOL_DIR")).to_path_buf(),
    );
    let mut files: Vec<_> = std::fs::read_dir(&vrp_dir)
        .expect("VRP_DIR")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "vrp"))
        .collect();
    files.sort();
    let (mut ok, mut all) = (0, 0);
    for f in files {
        all += 1;
        let sol = sol_dir.join(f.with_extension("sol").file_name().expect("name"));
        let inst = match Instance::read(&f, rounding) {
            Ok(i) => i,
            Err(e) => {
                println!("FAIL {e}");
                continue;
            }
        };
        let (routes, stated, precision) = read_sol(&sol).unwrap_or_else(|e| panic!("{e}"));
        let s = Routes::new(&inst, routes, Penalties::default());
        let (dist, eval) = (s.distance(), s.evaluate());
        let tol = if inst.rounding == Some(Rounding::Real) {
            // equal at the precision the cost is written with, but never stricter
            // than 1e-3 (six-decimal files carry the writer's float error)
            precision.max(1e-3)
        } else {
            1e-6
        };
        let good = s.is_feasible()
            && stated.is_some_and(|c| (dist - c).abs() <= tol)
            && (eval - dist).abs() <= 1e-9;
        ok += good as usize;
        println!(
            "{} {}: {} routes, distance {dist:.4}, stated {stated:?}, feasible {}",
            if good { "OK" } else { "FAIL" },
            inst.name,
            s.routes.len(),
            s.is_feasible()
        );
    }
    println!("{ok} / {all} OK");
}
