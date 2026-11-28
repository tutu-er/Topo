"""N2b, the price of a line chosen among many (post hoc, from the rows of N2; no new solves).

The node MIQP picks the best of 2^d - d - 2 unions at a junction. Theorem 1(b) bounds what one redundant line gains;
the best of M redundant candidates gains about ln M more (a union bound), so a line found by searching M unions needs a
higher price to keep false additions as rare as for one line. The extended BIC (Chen and Chen, 2008) adds
gamma * ln M to the price of such a line (half-BIC units). This script recounts N2's additions with that price:
a union is added only when its gain exceeds gamma * ln(2^d - d - 2), for gamma = 0, 0.5 and 1.

Run from topology_study:  python -m ptid.runs.node_price [--out experiments/node_miqp/recovery.jsonl]
"""
from __future__ import annotations

import argparse
import math

from ptid.runs.node_recovery import OUT, band, BANDS
from ptid.runs.node_study import read_rows


def extra_price(d, gamma):
    return gamma*math.log(2**d-d-2)


def adds(row, gamma):
    n, d = row.get("node") or row["miqp"], row["feeder"]["blocks"]
    if n["gain"] is None or n["gain"] <= extra_price(d, gamma):
        return "none"
    t = row["target"]
    return "right" if t is not None and tuple(n["line"]) == tuple(t) else "wrong"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--exact", default=str(OUT.parent/"exact.jsonl"), help="N1 rows, for the check on other seeds")
    summary(p.parse_args())


def summary(args):
    rows = read_rows(args.out)
    print(f"{len(rows)} rows of N2. Added = gain above gamma * ln(2^d - d - 2) (d = 10, 14, 18: "
          f"{', '.join(f'{math.log(2**d-d-2):.1f}' for d in (10, 14, 18))} score units at gamma = 1)\n")
    print("| true evidence | feeders | right added, gamma 0 / 0.5 / 1 | wrong added, gamma 0 / 0.5 / 1 |")
    print("| --- | ---: | --- | --- |")
    for b in BANDS:
        g = [r for r in rows if band(r["true_evidence"]) == b]
        if not g:
            continue
        right = " / ".join(str(sum(adds(r, x) == "right" for r in g)) for x in (0, .5, 1))
        wrong = " / ".join(str(sum(adds(r, x) == "wrong" for r in g)) for x in (0, .5, 1))
        print(f"| {b} | {len(g)} | {right} | {wrong} |")
    totals(rows)
    exact = read_rows(args.exact)
    if exact:
        print(f"\nCheck on other seeds: the {len(exact)} flat-junction scans of N1 (d = 6 to 24, seeds 1 to 3)")
        totals(exact)


def totals(rows):
    seg = [r for r in rows if r["target"] is not None]
    for x in (0, .5, 1):
        print(f"gamma {x}: right additions {sum(adds(r, x) == 'right' for r in seg)} of {len(seg)} segments, "
              f"wrong additions {sum(adds(r, x) == 'wrong' for r in rows)} of {len(rows)} feeders "
              f"({sum(adds(r, x) == 'wrong' for r in rows if r['target'] is None)} in the "
              f"{sum(r['target'] is None for r in rows)} controls)")


if __name__ == "__main__":
    main()
