"""Simulated probing days (PROTOCOL.md, section 5): Algorithm 1 on a development case's 8-day history, then Algorithm 2
on day 9 with the case's world (ptid/cases/world.py). Development cases only.

    python -m ptid.runs.probe_cases --names dev_small_003_head --variant T0,T2 [--quick] [--psi 50] [--out file.jsonl]

psi is a PLACEHOLDER until the knee of the development cost-coverage curve is found (PLAN.md).
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
from dataclasses import replace

from ptid.cases import library
from ptid.cases.world import World
from ptid.method.identify import Settings, identify
from ptid.method.labels import DECIDED
from ptid.method.probe import ProbeSettings, probe_day
from ptid.runs.identify_cases import f1

QUICK = dict(short_seconds=5, long_seconds=5, start_seconds=2, node_seconds=5, rounds=1)


def run_case(name, variant, st, ps, say=print):
    rec = library.open_case(name)
    data = library.readings(rec)
    first = identify(**data, st=st)
    day = probe_day(data, first, World(rec), st, replace(ps, variant=variant), say)
    truth = library.truth(rec)
    tree = {tuple(g) for g in day["tree"]}
    targets = []
    for M in day["targets"]:                                    # a meter grouping, or "place"
        label = day["gates"][-1]["targets"][str(M)][0]
        is_grouping = M != "place"
        targets.append(dict(target=str(M), label=label, inside=is_grouping and M in tree,
                            true=is_grouping and M in truth,
                            wrong=is_grouping and label == DECIDED and (M in tree) != (M in truth)))
    return dict(name=name, n=rec["n"], load=library.load_where(rec), variant=variant,
                f1_before=f1({tuple(g) for g in first["tree"]}, truth)["f1"], f1_after=f1(tree, truth)["f1"],
                cost=day["cost"], blocks=sum(g["block"] is not None for g in day["gates"]), gates=len(day["gates"]),
                stop=day["stop"], targets=targets, seconds=[g["seconds"] for g in day["gates"]], day=day["gates"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--names", required=True, help="development cases, comma-separated")
    ap.add_argument("--variant", default="T0", help="T0, T1, T2, or several, comma-separated")
    ap.add_argument("--quick", action="store_true", help="short time limits (smoke test)")
    ap.add_argument("--hours", type=int, default=24)
    ap.add_argument("--psi", type=float, default=ProbeSettings.psi, help="CNY per half-BIC unit")
    ap.add_argument("--out", default=None, help="jsonl file to append to")
    args = ap.parse_args()
    st = Settings(**QUICK) if args.quick else Settings()
    for name in args.names.split(","):
        for variant in args.variant.split(","):
            row = run_case(name, variant, st, ProbeSettings(hours=args.hours, psi=args.psi))
            print(f"{name} {variant}: F1 {row['f1_before']:.3f} -> {row['f1_after']:.3f}, {row['blocks']} blocks, "
                  f"{row['cost']:.1f} CNY, stop: {row['stop']}; "
                  f"targets {[(t['target'], t['label']) for t in row['targets']]}", flush=True)
            if args.out:
                with open(args.out, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(row, default=str)+"\n")


if __name__ == "__main__":
    main()
