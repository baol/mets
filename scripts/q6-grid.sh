#!/usr/bin/env bash
# Q6 (PLAN-problems.md): tai-a on the GPU, populations against independent
# robust tabu trajectories.
#
#   arm A  memetic: population 112, 200n iterations per child, crossover
#   arm B  --independent: 112 independent trajectories, batches of 200n
#   arm C  --perturb-stalled: B plus k random swaps (k in [n/10, n/4]) on a
#          member whose best did not improve in a batch
#
# One run at a time (each needs the whole GPU). Every run has its own
# directory, results/q6/<instance>/<arm>/seed<k>/, with the log, the trace and
# the .best file. A run whose log ends with "saved to" is skipped, so the
# grid can be stopped and started again.
#
# Environment (defaults in brackets):
#   QAPDATA    directory with the .dat and .sln files [qapdata]
#   INSTANCES  [tai50a tai60a tai80a]
#   ARMS       [A B C]
#   SEEDS      [1 2 3 4 5]
#   MINUTES    budget per run [60]
#   SWEEP=1    instead of the grid: the aspiration sweep on tai50a, arm B,
#              aspiration 2, 5, 10 (n^2), SEEDS seeds each
#   EXTRA      more qap_gpu flags for every run (e.g. "--threads 256")
#   CPU=1      dry run without a GPU: build without the gpu feature, add --cpu
#
# Total for the default grid: 3 instances x 3 arms x 5 seeds x 1 h = 45 GPU-hours.
# Summarise at any time with:  python3 scripts/q6-summary.py results/q6
# results/ stays out of git (PLAN-problems.md Q6): the summary goes in a
# report in reports/.
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"
data="$(cd "${QAPDATA:-qapdata}" && pwd)"
instances=(${INSTANCES:-tai50a tai60a tai80a})
arms=(${ARMS:-A B C})
seeds=(${SEEDS:-1 2 3 4 5})
seconds=$(( ${MINUTES:-60} * 60 ))
extra=(${EXTRA:-})

if [[ "${CPU:-0}" == 1 ]]; then
    cargo build --release --example qap_gpu
    extra+=(--cpu)
else
    cargo build --release --features gpu --example qap_gpu
fi
bin="$root/target/release/examples/qap_gpu"
build="$(git log -1 --format=%h)"

run() { # instance arm-label seed flags...
    local inst=$1 label=$2 seed=$3; shift 3
    local dir="$root/results/q6/$inst/$label/seed$seed"
    mkdir -p "$dir"
    if [[ -f "$dir/run.log" ]] && grep -q "saved to" "$dir/run.log"; then
        echo "skip $inst $label seed $seed (done)"; return
    fi
    echo "run  $inst $label seed $seed ($(date -u +%FT%TZ), build $build)"
    (cd "$dir" && rm -f run.trace && \
     "$bin" "$data/$inst.dat" "$data/$inst.sln" --population 112 --iters 200 \
         --seed "$seed" --max-seconds "$seconds" "$@" "${extra[@]}" \
         --trace run.trace > run.log 2>&1) || echo "  failed: see $dir/run.log"
    tail -n 1 "$dir/run.log"
}

if [[ "${SWEEP:-0}" == 1 ]]; then
    for asp in 2 5 10; do
        for s in "${seeds[@]}"; do
            run tai50a "B-asp$asp" "$s" --independent --aspiration "$asp"
        done
    done
    exit 0
fi

for inst in "${instances[@]}"; do
    for arm in "${arms[@]}"; do
        case $arm in
            A) flags=() ;;
            B) flags=(--independent) ;;
            C) flags=(--perturb-stalled) ;;
            *) echo "unknown arm $arm"; exit 1 ;;
        esac
        for s in "${seeds[@]}"; do
            run "$inst" "$arm" "$s" "${flags[@]}"
        done
    done
done

echo "done. Summary: python3 scripts/q6-summary.py results/q6"
echo "results/ stays out of git: put the summary in a report under reports/"
