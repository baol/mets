#!/usr/bin/env python3
"""Summarise Q10: per arm, seeds that reached the best known and the time
to it (each run stops there), from results/q10/<arm>/seed*/run.log.

usage: python3 scripts/q10-summary.py [results/q10]
"""
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1] if len(sys.argv) > 1 else "results/q10")
found_re = re.compile(r"found (\d+) in ([\d.]+) s, best known (\d+)")
print("| arm | seeds (done) | reached | time to best known by seed (s) | median, max (reached) | build |")
print("|---|---|---|---|---|---|")
for arm in sorted(p for p in root.iterdir() if p.is_dir()):
    cells, reached, done, builds = [], [], 0, set()
    for log in sorted(arm.glob("seed*/run.log")):
        text = log.read_text(errors="replace")
        if m := re.search(r"build ([0-9a-f]+(?:-dirty)?)", text):
            builds.add(m.group(1))
        m = found_re.search(text)
        if not m or "saved to" not in text:
            cells.append("…")
            continue
        done += 1
        found, secs, known = int(m.group(1)), float(m.group(2)), int(m.group(3))
        if found <= known:
            reached.append(secs)
            cells.append(f"{secs:.0f}")
        else:
            cells.append(f"— ({found:,})")
    seeds = len(cells)
    stats = f"{statistics.median(reached):.0f}, {max(reached):.0f}" if reached else "—"
    print(f"| {arm.name} | {seeds} ({done}) | {len(reached)}/{done} | {', '.join(cells)} | {stats} | {', '.join(sorted(builds))} |")
