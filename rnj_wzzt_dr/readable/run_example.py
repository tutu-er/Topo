"""Run the readable WZZT-C on study inputs.

    # watch every step on one input (a development input of the study)
    python -m rnj_wzzt_dr.readable.run_example trace eulv eulv_td400858_feeder59_s0

    # the development inputs: compare with WZZT-R and the production WZZT-C (stored runs)
    python -m rnj_wzzt_dr.readable.run_example dev --count 20

    # try another setting, e.g. a stricter block rule
    python -m rnj_wzzt_dr.readable.run_example dev --count 20 --m 10

Set PYTHONPATH to the repository root and use the Topo310 environment (Gurobi licence).
"""
import argparse
import json
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import capture
from rnj_wzzt_dr.experiments.wzzt_evidence_20261124 import run_phase2 as p2
from rnj_wzzt_dr.readable.settings import Settings
from rnj_wzzt_dr.readable.wzzt_c import run


def input_arrays(cond, name):
    """p, q, y, v0sq (training and validation slots joined) and the true internal lines of a study input."""
    inp = capture.load_input(cond, name)
    train, validation = p2.series(inp)
    data = train.join(validation)
    truth = set()
    for line in inp.truth_key:
        if 1 < len(line) < inp.n:
            truth.add(tuple(sorted(line)))
    return data.p, data.q, data.y, data.v0sq, truth


def trace(cond, name, settings):
    p, q, y, v0sq, truth = input_arrays(cond, name)
    print(f"Input {cond} {name}: {p.shape[1]} meters, {len(truth)} true internal lines\n")
    result = run(p, q, y, v0sq, settings, trace=True)
    found = set(result["tree"])
    print(f"\nTrue lines found {len(found & truth)} of {len(truth)}; lines output {len(found)}, of which "
          f"{len(found & truth)} true; exact tree: {found == truth}")


def dev(settings, count):
    P = capture.OUT
    stored_r = {}
    for line in (P/"phase2"/"ladder.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["variant"] == "R":
            stored_r[(r["cond"], r["name"])] = r
    stored_c = {}
    path_c = P/"combo"/"ev_pairs.jsonl"
    if path_c.exists():
        for line in path_c.read_text().splitlines():
            r = json.loads(line)
            if "error" not in r:
                stored_c[(r["cond"], r["name"])] = r
    keys = sorted(stored_r)[:count] if count else sorted(stored_r)
    totals = {"readable": [0, 0, 0, 0], "production WZZT-C": [0, 0, 0, 0], "WZZT-R": [0, 0, 0, 0]}
    same_as_production = 0
    for cond, name in keys:
        p, q, y, v0sq, truth = input_arrays(cond, name)
        n = p.shape[1]
        result = run(p, q, y, v0sq, settings)
        trees = {"readable": set(result["tree"]),
                 "WZZT-R": {tuple(sorted(C)) for C in stored_r[(cond, name)]["selected"] if 1 < len(C) < n}}
        if (cond, name) in stored_c:
            trees["production WZZT-C"] = set(map(tuple, stored_c[(cond, name)]["combo"]))
            same_as_production += trees["readable"] == trees["production WZZT-C"]
        for label, tree in trees.items():
            t = totals[label]
            t[0] += tree == truth
            t[1] += len(tree & truth)
            t[2] += len(tree)
            t[3] += len(truth)
        print(f"{name:32s} n={n:2d} readable: {len(trees['readable'] & truth)}/{len(truth)} found, "
              f"{len(trees['readable'])} output, certified {result['certified']}, {result['seconds']:.1f} s")
    print("\nmethod              exact trees   true lines found   precision")
    for label, t in totals.items():
        if t[3]:
            print(f"{label:20s} {t[0]:5d}        {t[1]:4d} / {t[3]:4d}        {100*t[1]/max(t[2], 1):5.1f}%")
    print(f"\nreadable tree identical to the production WZZT-C (stored, frozen settings): {same_as_production} of {len(keys)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("trace", "dev"))
    ap.add_argument("cond", nargs="?", default="eulv")
    ap.add_argument("name", nargs="?", default="eulv_td400858_feeder59_s0")
    ap.add_argument("--count", type=int, default=20, help="number of development inputs (0 = all 130)")
    ap.add_argument("--m", type=float, default=None, help="evidence margin for blocks")
    ap.add_argument("--sources", default=None, help="comma-separated sources")
    args = ap.parse_args()
    settings = Settings()
    if args.m is not None:
        settings.m = args.m
    if args.sources:
        settings.sources = tuple(args.sources.split(","))
    if args.what == "trace":
        trace(args.cond, args.name, settings)
    else:
        dev(settings, args.count)


if __name__ == "__main__":
    main()
