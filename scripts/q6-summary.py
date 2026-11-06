#!/usr/bin/env python3
"""Summarise the Q6 grid: best value at 5, 15, 30, 60 minutes and the time
to the best known, per instance and arm, from results/q6/<inst>/<arm>/seed*/run.log.

usage: python3 scripts/q6-summary.py [results/q6]
"""
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/q6")
marks = [5, 15, 30, 60]  # minutes
line_re = re.compile(r"generation\s+\d+: best (\d+).*?,\s+([\d.]+) s")
known_re = re.compile(r"best known (\d+)")


def parse(log):
    """[(seconds, best)], best known, finished?"""
    points, known, done = [], None, False
    for line in log.read_text(errors="replace").splitlines():
        if m := line_re.search(line):
            points.append((float(m.group(2)), int(m.group(1))))
        if "saved to" in line:
            done = True
            if m := known_re.search(line):
                known = int(m.group(1))
            if m := re.search(r"in ([\d.]+) s, best (\d+)", line):
                points.append((float(m.group(1)), int(m.group(2))))
    return points, known, done


def best_at(points, seconds):
    vals = [b for t, b in points if t <= seconds]
    return min(vals) if vals else None


def fmt(x):
    return "—" if x is None else f"{x:,}"


print("| instance | arm | seeds (done) | " + " | ".join(f"best @ {m} min (median)" for m in marks)
      + " | reached | time to best known (median, max) |")
print("|---|---|---|" + "---|" * len(marks) + "---|---|")
for inst_dir in sorted(p for p in root.iterdir() if p.is_dir()):
    for arm_dir in sorted(p for p in inst_dir.iterdir() if p.is_dir()):
        runs = [parse(log) for log in sorted(arm_dir.glob("seed*/run.log"))]
        if not runs:
            continue
        known = next((k for _, k, _ in runs if k is not None), None)
        cells = []
        for m in marks:
            vals = [best_at(p, m * 60) for p, _, _ in runs]
            vals = [v for v in vals if v is not None]
            cells.append(fmt(int(statistics.median(vals))) if vals else "—")
        reached = []
        for p, _, _ in runs:
            hit = [t for t, b in p if known is not None and b <= known]
            if hit:
                reached.append(min(hit))
        ttt = (f"{statistics.median(reached):.0f} s, {max(reached):.0f} s" if reached else "—")
        done = sum(1 for _, _, d in runs if d)
        print(f"| {inst_dir.name} | {arm_dir.name} | {len(runs)} ({done}) | " + " | ".join(cells)
              + f" | {len(reached)}/{len(runs)} | {ttt} |")
print("\nBest known: the .sln value printed in each log; a tie is a value equal to it.")
