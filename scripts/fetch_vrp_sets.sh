#!/usr/bin/env bash
# Download the CMT (Christofides, Mingozzi & Toth 1979; 14 instances) and
# Golden (Golden, Wasil, Kelly & Chao 1998; 20 instances) CVRP sets with their
# best-known solutions, in CVRPLIB's .vrp/.sol format, into vrp/CMT and
# vrp/Golden, then recount every solution with scripts/verify_vrp.py.
#
# These are the sets of the Cordeau & Maischberger (2012) tables, the
# acceptance test of the CVRP port (PLAN-problems.md P7). CVRPLIB itself
# (vrp.galgos.inf.puc-rio.br) is the reference; this script uses GitHub
# mirrors in the same format:
#   CMT     github.com/zhu-he/cvrp-data, CMT/   (CMT1 matches vidalt/HGS-CVRP's copy)
#   Golden  github.com/vidalt/CVRPLIB, Golden/Instances and Golden/Solutions
#
# usage: scripts/fetch_vrp_sets.sh [DEST]   (default: vrp)
set -euo pipefail
dest="${1:-vrp}"
mkdir -p "$dest/CMT" "$dest/Golden"

fetch() { # repo path out
    for branch in master main; do
        if curl -sSfL --max-time 60 -o "$3" "https://raw.githubusercontent.com/$1/$branch/$2"; then
            return 0
        fi
    done
    echo "failed: $1 $2" >&2
    return 1
}

for i in $(seq 1 14); do
    fetch zhu-he/cvrp-data "CMT/CMT$i.vrp" "$dest/CMT/CMT$i.vrp"
    fetch zhu-he/cvrp-data "CMT/CMT$i.sol" "$dest/CMT/CMT$i.sol"
done
for i in $(seq 1 20); do
    fetch vidalt/CVRPLIB "Golden/Instances/Golden_$i.vrp" "$dest/Golden/Golden_$i.vrp"
    fetch vidalt/CVRPLIB "Golden/Solutions/Golden_$i.sol" "$dest/Golden/Golden_$i.sol"
done
echo "downloaded: $(ls "$dest"/CMT/*.vrp | wc -l) CMT and $(ls "$dest"/Golden/*.vrp | wc -l) Golden instances"

here="$(cd "$(dirname "$0")" && pwd)"
python3 "$here/verify_vrp.py" --dirs "$dest/CMT" "$dest/CMT" | tail -n 1
python3 "$here/verify_vrp.py" --dirs "$dest/Golden" "$dest/Golden" | tail -n 1
