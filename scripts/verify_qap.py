#!/usr/bin/env python3
"""Recompute the cost of a QAP solution from the instance, with none of the
solver's code: an independent check before any result is reported.

usage: python3 scripts/verify_qap.py FILE.dat SOLUTION [FILE.sln]

SOLUTION is a NAME.best file (cost on the first line, then the 1-based
element at each position). With a .sln, also compares against the published
permutation, which QAPLIB lists in the other direction (facility -> location).
"""
import sys


def numbers(path):
    return [int(x) for x in open(path).read().split()]


dat, sol = sys.argv[1], sys.argv[2]
v = numbers(dat)
n, a, b = v[0], v[1:1 + v[0] ** 2], v[1 + v[0] ** 2:1 + 2 * v[0] ** 2]
lines = open(sol).read().split("\n")
claimed = int(lines[0])
p = [int(x) - 1 for x in lines[1].split()]
assert sorted(p) == list(range(n)), "not a permutation of 1..n"
cost = sum(a[i * n + j] * b[p[i] * n + p[j]] for i in range(n) for j in range(n))
print(f"n = {n}; claimed {claimed}; recomputed {cost}: {'OK' if cost == claimed else 'MISMATCH'}")
if len(sys.argv) > 3:
    s = numbers(sys.argv[3])
    known, q = s[1], [x - 1 for x in s[2:2 + n]]
    inverse = [0] * n
    for i, x in enumerate(q):
        inverse[x] = i
    same = "the published assignment" if p == inverse else (
        "the published permutation as listed" if p == q else "a different assignment")
    print(f".sln best known {known}; this solution is {same}; "
          f"{'tie' if cost == known else 'below' if cost < known else 'above'} the .sln value")
sys.exit(0 if cost == claimed else 1)
