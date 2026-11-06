#!/usr/bin/env python3
"""Recompute the cost and check the feasibility of a CVRP or VRPTW solution
from the instance, with none of the solver's code: an independent check
before any result is reported (PLAN-problems.md P7, step 7).

usage:
  python3 scripts/verify_vrp.py FILE.vrp FILE.sol [--real | --round]
  python3 scripts/verify_vrp.py --dirs VRP_DIR SOL_DIR [--real | --round]

Distances: EXPLICIT matrices as given (LOWER_ROW, UPPER_ROW, FULL_MATRIX,
LOWER_DIAG_ROW, UPPER_DIAG_ROW); EUC_2D rounded to the nearest integer
(--round, CVRPLIB's convention for the X, XL and XTW sets) or real
(--real, for CMT and Golden). Without either flag: real for names starting
with CMT or Golden, rounded otherwise.

A .sol lists routes as `Route #i: c c c ...` with customers numbered
1..N-1 (node id minus one when the depot is node 1, the CVRPLIB layout),
then `Cost C`. Checks: every customer exactly once, each route's load within
CAPACITY, route duration within DISTANCE when given (with service times),
time windows for VRPTW (arrive, wait until ready, serve, leave; back at the
depot by its due time), and the recomputed cost against `Cost`.
"""
import math
import sys
from pathlib import Path


def read_vrp(path):
    spec, sections, current = {}, {}, None
    for raw in Path(path).read_text().splitlines():
        line = raw.strip()
        if not line or line == "EOF":
            continue
        if ":" in line and not line[0].isdigit() and not line[0] == "-":
            key, _, value = line.partition(":")
            key, value = key.strip(), value.strip()
            if key.endswith("SECTION"):  # e.g. "NODE_COORD_SECTION :" variants
                current = key
                sections[current] = []
            else:
                spec[key] = value.strip('"')
                current = None
            continue
        if line.endswith("SECTION"):
            current = line
            sections[current] = []
            continue
        if current:
            sections[current].append(line.split())
    return spec, sections


def distance_matrix(spec, sections, real):
    n = int(spec["DIMENSION"])
    kind = spec.get("EDGE_WEIGHT_TYPE", "")
    if kind == "EUC_2D":
        xy = {}
        for row in sections["NODE_COORD_SECTION"]:
            xy[int(row[0])] = (float(row[1]), float(row[2]))
        ids = sorted(xy)
        assert len(ids) == n
        d = [[0.0] * n for _ in range(n)]
        for a in range(n):
            for b in range(n):
                (x1, y1), (x2, y2) = xy[ids[a]], xy[ids[b]]
                e = math.hypot(x1 - x2, y1 - y2)
                d[a][b] = e if real else float(math.floor(e + 0.5))
        return d
    if kind == "EXPLICIT":
        fmt = spec.get("EDGE_WEIGHT_FORMAT", "")
        vals = [float(x) for row in sections["EDGE_WEIGHT_SECTION"] for x in row]
        d = [[0.0] * n for _ in range(n)]
        it = iter(vals)
        if fmt == "FULL_MATRIX":
            for a in range(n):
                for b in range(n):
                    d[a][b] = next(it)
        else:
            pairs = {
                "LOWER_ROW": [(a, b) for a in range(n) for b in range(a)],
                "LOWER_DIAG_ROW": [(a, b) for a in range(n) for b in range(a + 1)],
                "UPPER_ROW": [(a, b) for a in range(n) for b in range(a + 1, n)],
                "UPPER_DIAG_ROW": [(a, b) for a in range(n) for b in range(a, n)],
            }[fmt]
            for a, b in pairs:
                d[a][b] = d[b][a] = next(it)
        rest = sum(1 for _ in it)
        assert rest == 0, f"{rest} extra edge weights"
        return d
    raise ValueError(f"unsupported EDGE_WEIGHT_TYPE {kind!r}")


def read_sol(path):
    routes, cost = [], None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line.lower().startswith("route"):
            _, _, rest = line.partition(":")
            routes.append([int(x) for x in rest.split()])
        elif line.lower().startswith("cost"):
            cost = float(line.split()[1])
    return routes, cost


def verify(vrp, sol, real=None):
    spec, sections = read_vrp(vrp)
    name = spec.get("NAME", Path(vrp).stem)
    if real is None:
        real = name.startswith(("CMT", "Golden"))
    n = int(spec["DIMENSION"])
    d = distance_matrix(spec, sections, real)
    ids = sorted(int(r[0]) for r in sections["DEMAND_SECTION"])
    index = {node: i for i, node in enumerate(ids)}  # node id -> 0..n-1
    demand = {index[int(r[0])]: float(r[1]) for r in sections["DEMAND_SECTION"]}
    depots = [int(r[0]) for r in sections.get("DEPOT_SECTION", [["1"]]) if int(r[0]) > 0]
    assert len(depots) == 1, "one depot"
    depot = index[depots[0]]
    assert depot == 0, "the .sol numbering assumes the depot is the first node"
    cap = float(spec["CAPACITY"])
    max_len = float(spec["DISTANCE"]) if "DISTANCE" in spec else None
    service = {}
    if "SERVICE_TIME" in spec:  # one service time for every customer (CMT)
        service = {i: float(spec["SERVICE_TIME"]) for i in range(1, n)}
    service.update({index[int(r[0])]: float(r[1]) for r in sections.get("SERVICE_TIME_SECTION", [])})
    windows = {index[int(r[0])]: (float(r[1]), float(r[2])) for r in sections.get("TIME_WINDOW_SECTION", [])}
    routes, claimed = read_sol(sol)
    problems = []
    seen = [0] * n
    total = 0.0
    for k, route in enumerate(routes, 1):
        nodes = [c for c in route]  # customer c is node index c (depot at 0)
        for c in nodes:
            if not 1 <= c < n:
                problems.append(f"route {k}: customer {c} out of range")
                continue
            seen[c] += 1
        if problems:
            continue
        load = sum(demand[c] for c in nodes)
        if load > cap + 1e-9:
            problems.append(f"route {k}: load {load:g} > capacity {cap:g}")
        path = [depot] + nodes + [depot]
        length = sum(d[a][b] for a, b in zip(path, path[1:]))
        total += length
        duration = length + sum(service.get(c, 0.0) for c in nodes)
        if max_len is not None and duration > max_len + 1e-6:
            problems.append(f"route {k}: duration {duration:g} > {max_len:g}")
        if windows:
            t = windows.get(depot, (0.0, math.inf))[0]
            for a, b in zip(path, path[1:]):
                t += service.get(a, 0.0) if a != depot else 0.0
                t += d[a][b]
                ready, due = windows.get(b, (0.0, math.inf))
                if t > due + 1e-6:
                    problems.append(f"route {k}: node {b} reached at {t:g} > due {due:g}")
                    break
                t = max(t, ready)
    missing = [c for c in range(1, n) if seen[c] == 0]
    twice = [c for c in range(1, n) if seen[c] > 1]
    if missing:
        problems.append(f"{len(missing)} customers not visited (e.g. {missing[:5]})")
    if twice:
        problems.append(f"{len(twice)} customers visited more than once (e.g. {twice[:5]})")
    # rounded distances: exact; real distances: the stated cost carries a few decimals
    cost_ok = claimed is not None and abs(total - claimed) <= (1e-6 if not real else 0.01)
    status = "OK" if cost_ok and not problems else "FAIL"
    detail = f"{name}: {len(routes)} routes, cost recomputed {total:.4f}, stated {claimed}, {'real' if real else 'rounded'} distances"
    return status, detail, problems


def main():
    args = sys.argv[1:]
    real = True if "--real" in args else False if "--round" in args else None
    args = [a for a in args if a not in ("--real", "--round")]
    if args and args[0] == "--dirs":
        vrp_dir, sol_dir = Path(args[1]), Path(args[2])
        pairs = [(v, sol_dir / (v.stem + ".sol")) for v in sorted(vrp_dir.glob("*.vrp"))]
    else:
        pairs = [(Path(args[0]), Path(args[1]))]
    failed = 0
    for vrp, sol in pairs:
        if not sol.exists():
            print(f"MISSING {sol}")
            failed += 1
            continue
        status, detail, problems = verify(vrp, sol, real)
        print(f"{status} {detail}")
        for p in problems:
            print(f"    {p}")
        failed += status != "OK"
    if len(pairs) > 1:
        print(f"{len(pairs) - failed} / {len(pairs)} OK")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
