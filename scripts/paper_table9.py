"""Cordeau & Maischberger 2012, Table 9: the fleet size m used per instance
and the 2011 reference costs ("Prev. best") the paper's gaps are taken
against. Shared by scripts/p7-cvrp.sh (FLEET=paper) and scripts/p7-summary.py.

usage: python3 scripts/paper_table9.py m INSTANCE   (prints m, e.g. for Golden_3)
"""
import sys

GOLDEN_M = [11, 11, 11, 11, 6, 8, 10, 12, 15, 17, 19, 20, 27, 31, 34, 38, 23, 28, 34, 39]
CMT_M = [5, 12, 10, 14, 20, 7, 13, 10, 16, 20, 8, 10, 13, 11]
GOLDEN_REF = [5626.81, 8431.66, 11036.22, 13592.88, 6460.98, 8404.26, 10156.58,
              11643.90, 580.42, 738.49, 914.72, 1106.76, 857.19, 1080.55,
              1342.53, 1620.85, 707.76, 995.13, 1365.97, 1820.02]
CMT_REF = [524.61, 835.26, 826.14, 1028.42, 1291.29, 555.43, 909.68, 865.94,
           1162.55, 1395.85, 1042.11, 819.56, 1541.14, 866.37]

M = {f"Golden_{i + 1}": m for i, m in enumerate(GOLDEN_M)}
M.update({f"CMT{i + 1}": m for i, m in enumerate(CMT_M)})
REF = {f"Golden_{i + 1}": c for i, c in enumerate(GOLDEN_REF)}
REF.update({f"CMT{i + 1}": c for i, c in enumerate(CMT_REF)})

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("m", "ref") or sys.argv[2] not in M:
        sys.exit(f"usage: {sys.argv[0]} m|ref INSTANCE (one of Golden_1..20, CMT1..14)")
    print((M if sys.argv[1] == "m" else REF)[sys.argv[2]])
