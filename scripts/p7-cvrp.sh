#!/usr/bin/env bash
# P7 acceptance and diagnostics runs of examples/cvrp.rs (PLAN-problems.md P7).
# Restartable: a run whose result.txt exists is skipped.
#
# env: SET (CMT|Golden), INSTANCES (default: all in vrp/SET), ETA (100000),
#      SEEDS ("1 2 ... 10"), TAG (default: SET-ETA), EXTRA (more cvrp flags,
#      e.g. "--no-spare"), JOBS (nproc - 1), VRP (vrp),
#      FLEET (unset: the solver's default, reference + 1; "paper": --vehicles m
#      per instance from Cordeau & Maischberger 2012 Table 9, scripts/paper_table9.py)
# output: results/p7/TAG/INSTANCE/seedK/{log,NAME.sol,result.txt}
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
SET="${SET:-CMT}"; ETA="${ETA:-100000}"; SEEDS="${SEEDS:-$(seq -s ' ' 1 10)}"
TAG="${TAG:-$SET-$ETA}"; EXTRA="${EXTRA:-}"; VRP="${VRP:-$here/vrp}"; FLEET="${FLEET:-}"
case "$FLEET" in ""|paper) ;; *) echo "FLEET must be unset or paper, not $FLEET" >&2; exit 1 ;; esac
JOBS="${JOBS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}"
INSTANCES="${INSTANCES:-$(ls "$VRP/$SET"/*.vrp | xargs -n1 basename | sed 's/\.vrp$//' | sort -V | tr '\n' ' ')}"
cargo build --release --example cvrp --manifest-path "$here/Cargo.toml" >/dev/null 2>&1
bin="$here/target/release/examples/cvrp"
jobs_file="$(mktemp)"
for inst in $INSTANCES; do for s in $SEEDS; do echo "$inst $s"; done; done > "$jobs_file"
run() {
    inst="$1"; s="$2"
    dir="$here/results/p7/$TAG/$inst/seed$s"
    [ -f "$dir/result.txt" ] && return 0
    fleet=""
    if [ "$FLEET" = paper ]; then fleet="--vehicles $(python3 "$here/scripts/paper_table9.py" m "$inst")"; fi
    mkdir -p "$dir"; cd "$dir"; rm -f -- *.sol log
    "$bin" "$VRP/$SET/$inst.vrp" "$VRP/$SET/$inst.sol" --eta "$ETA" --seed "$s" $fleet $EXTRA > log 2>&1
    tail -n 1 log > result.txt
    echo "$TAG $inst seed$s: $(cat result.txt)"
}
export -f run; export here TAG VRP SET ETA EXTRA FLEET bin
xargs -P "$JOBS" -L 1 bash -c 'run $0 $1' < "$jobs_file"
rm -f "$jobs_file"
