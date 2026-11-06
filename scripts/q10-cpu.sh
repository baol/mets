#!/usr/bin/env bash
# Q10 (PLAN-problems.md): the aspiration effect on the CPU. qap_advanced on
# tai50a, single-threaded, 1 hour per run, seeds 1-5 per arm:
#   memetic at aspiration 5n^2 (the Q9 baseline), 20n^2 and off (1000n^2)
#   restarts at 5n^2 and 20n^2
# All arms on the same machine, so their times compare. JOBS runs at a time
# (each uses one core; leave one core for the GPU host when it runs).
# Every run has its own directory results/q10/<method>-asp<X>/seed<k>/ with
# its log and its saved .best; a finished run is skipped on a restart.
#
# Environment: QAPDATA [qapdata], JOBS [number of cores - 1], SEEDS [1 2 3 4 5],
#              MINUTES [60], ARMS [memetic:5 memetic:20 memetic:1000 restarts:5 restarts:20]
# Summary at any time: python3 scripts/q10-summary.py results/q10
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
data="$(cd "${QAPDATA:-qapdata}" && pwd)"
jobs="${JOBS:-$(( $(nproc) > 1 ? $(nproc) - 1 : 1 ))}"
seeds="${SEEDS:-1 2 3 4 5}"
seconds=$(( ${MINUTES:-60} * 60 ))
arms="${ARMS:-memetic:5 memetic:20 memetic:1000 restarts:5 restarts:20}"

cargo build --release --example qap_advanced
bin="$root/target/release/examples/qap_advanced"
build="$(git log -1 --format=%h)"

list=""
for arm in $arms; do for s in $seeds; do list+="${arm%%:*} ${arm##*:} $s"$'\n'; done; done
export bin data seconds build root
printf '%s' "$list" | xargs -P "$jobs" -L 1 bash -c '
    method=$0 asp=$1 seed=$2
    dir="$root/results/q10/$method-asp$asp/seed$seed"
    mkdir -p "$dir"
    if [[ -f "$dir/run.log" ]] && grep -q "saved to" "$dir/run.log"; then
        echo "skip $method asp$asp seed $seed (done)"; exit 0
    fi
    echo "run  $method asp$asp seed $seed ($(date -u +%FT%TZ), build $build)"
    (cd "$dir" && "$bin" "$data/tai50a.dat" "$data/tai50a.sln" --method "$method" \
        --aspiration "$asp" --steps 100000000 --runs 1 --jobs 1 --first-seed "$seed" \
        --max-seconds "$seconds" > run.log 2>&1) || echo "  failed: see $dir/run.log"
    tail -n 2 "$dir/run.log"
'
echo "done. Summary: python3 scripts/q10-summary.py results/q10"
