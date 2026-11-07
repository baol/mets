#!/usr/bin/env python3
"""Summarise results/p7/TAG: per instance, the recounted gap of each seed's
saved .sol (scripts/verify_vrp.py, none of the solver's code) to the
reference .sol cost (today's best known), the gap of the mean to the paper's
2011 reference cost (Cordeau & Maischberger 2012, Table 9 "Prev. best",
scripts/paper_table9.py), and the mean gaps of the set.

usage: python3 scripts/p7-summary.py results/p7/TAG [VRP_DIR]  (default vrp/SET)
"""
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_table9 import REF as PAPER_REF  # noqa: E402
from verify_vrp import read_sol, verify  # noqa: E402

root = Path(sys.argv[1])
set_name = root.name.split("-")[0]
vrp_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(__file__).resolve().parent.parent / "vrp" / set_name
means, paper_means, fails = [], [], 0
print(f"{'instance':12} {'runs':>4} {'mean %':>7} {'best %':>7} {'worst %':>7} {'s/run':>6} {'gap to paper ref %':>18}")
for inst_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: [int(t) if t.isdigit() else t for t in p.name.replace("_", " ").split()] if False else (len(p.name), p.name)):
    vrp = vrp_dir / f"{inst_dir.name}.vrp"
    ref = read_sol(vrp_dir / f"{inst_dir.name}.sol")[1]
    gaps, paper_gaps, secs = [], [], []
    paper_ref = PAPER_REF.get(inst_dir.name)
    for run in sorted(inst_dir.glob("seed*")):
        sols = sorted(run.glob("*.sol"))
        if not (run / "result.txt").exists() or not sols:
            continue
        status, detail, problems = verify(vrp, sols[-1])
        if status != "OK":
            fails += 1
            print(f"  FAIL {run}: {detail} {problems}")
            continue
        cost = read_sol(sols[-1])[1]
        gaps.append((cost - ref) / ref * 100)
        if paper_ref is not None:
            paper_gaps.append((cost - paper_ref) / paper_ref * 100)
        text = (run / "result.txt").read_text()
        if " in " in text:
            secs.append(float(text.split(" in ")[-1].split()[0]))
    if gaps:
        means.append(statistics.mean(gaps))
        paper = "-"
        if paper_gaps:
            paper_means.append(statistics.mean(paper_gaps))
            paper = f"{paper_means[-1]:.2f}"
        print(f"{inst_dir.name:12} {len(gaps):4} {means[-1]:7.2f} {min(gaps):7.2f} {max(gaps):7.2f} {statistics.mean(secs) if secs else 0:6.1f} {paper:>18}")
if means:
    print(f"mean gap over {len(means)} instances: {statistics.mean(means):.2f}% to today's best known", end="")
    if paper_means:
        print(f", {statistics.mean(paper_means):.2f}% to the paper's reference ({len(paper_means)} instances)", end="")
    print(f"   ({fails} failed recounts)")
