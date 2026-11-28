"""N1, exactness and scaling of the node MIQP at one fixed junction (experiments/node_miqp/README.md).

For junctions of d = 6 to 24 children, the node MIQP and its flip test (best union excluded) are compared with
exhaustive enumeration of all 2^d - d - 2 unions, each refitted, up to d = 16 (2^16 unions per scan); beyond that only
the node MIQP runs, and its bound shows whether it finished. The tree is the true tree with the hidden segment
contracted (the flat junction), so the node's best union is the hidden line when the data support it.

Run from topology_study:
    python -m ptid.runs.node_exact run --workers 8        # writes experiments/node_miqp/exact.jsonl (resumable)
    python -m ptid.runs.node_exact summary                # the table of README section N1
    python -m ptid.runs.node_exact run --widths 6,8 --seeds 1   # a quick check (about a minute)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import statistics
import time

from ptid.runs.node_study import (Feeder, Simulator, draw_shared, enumerate_node, feeder_dict, model_of, node_scan,
                                  read_rows, score, truth, write_row)

OUT = Path(__file__).resolve().parents[2]/"experiments"/"node_miqp"/"exact.jsonl"
WIDTHS = (6, 8, 10, 12, 14, 16, 20, 24)
R_LEVELS = (0., .004, .008)       # no segment, a weak one, a clear one (ohm)


def instances(widths, seeds):
    for d in widths:
        for seed in seeds:
            for r in R_LEVELS:
                shared = draw_shared(d, seed) if r > 0 else ()
                yield Feeder(blocks=d, per=2, shared=shared, r_shared=r, excitation=.32, hours=192, seed=seed)


def one(f, max_enum, seconds, warm=False):
    t0 = time.perf_counter()
    data = Simulator(f).history()
    model, load = model_of(data)
    true, flat, target = truth(f)
    row = dict(feeder=feeder_dict(f), load_modelled=load, p_s=model.p_s, N=model.N, beta=model.alpha*model.N/2,
               target=target, true_evidence=None if target is None else score(model, flat)-score(model, true))
    row["miqp"] = node_scan(model, flat, seconds, warm=warm)
    row["warm"] = warm
    row["enum"] = enumerate_node(model, flat) if f.blocks <= max_enum else None
    row["seconds"] = time.perf_counter()-t0
    return row


def run(args):
    out = Path(args.out)
    done = {r["feeder"]["name"] for r in read_rows(out)}
    todo = [f for f in instances(args.widths, args.seeds) if f.name not in done]
    print(f"{len(todo)} instances to run ({len(done)} done) -> {out}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        jobs = {pool.submit(one, f, args.max_enum, args.seconds, args.warm): f for f in todo}
        for job in as_completed(jobs):
            row = job.result()
            write_row(out, row)
            m, e = row["miqp"], row["enum"]
            print(f"{row['feeder']['name']}: miqp {m['seconds']:.1f}s status {m['status']}"
                  + ("" if e is None else f", enum {e['seconds']:.1f}s same {m['line'] == e['line']}"), flush=True)


def tolerance(row):
    """The node MIQP stops at a relative gap of 1e-4 in crit; in score units that is 1e-4 crit N/2 (crit ~ 1)."""
    return 1e-4*row["N"]/2*1.5


def summary(args):
    rows = read_rows(args.out)
    print(f"{len(rows)} rows from {args.out}\n")
    print("| children d | unions 2^d-d-2 | instances | node MIQP s (median / max) | proved optimal | "
          "enumeration s (median) | same best union | same runner-up | max objective difference |")
    print("| ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |")
    for d in sorted({r["feeder"]["blocks"] for r in rows}):
        g = [r for r in rows if r["feeder"]["blocks"] == d]
        ms = [r["miqp"]["seconds"] for r in g]
        opt = sum(r["miqp"]["status"] == 2 and r["miqp"]["runner_status"] == 2 for r in g)
        e = [r for r in g if r["enum"] is not None]
        if e:
            same = sum(r["miqp"]["line"] == r["enum"]["line"] or
                       abs(r["miqp"]["gain"]-r["enum"]["gain"]) <= tolerance(r) for r in e)
            same2 = sum(r["miqp"]["runner"] == r["enum"]["runner"] or
                        abs(r["miqp"]["gain"]-r["miqp"]["runner_gap"]-r["enum"]["gain"]+r["enum"]["runner_gap"])
                        <= tolerance(r) for r in e)
            diff = max(abs(r["miqp"]["gain"]-r["enum"]["gain"]) for r in e)
            es = f"{statistics.median(r['enum']['seconds'] for r in e):.1f}"
            cols = f"{es} | {same}/{len(e)} | {same2}/{len(e)} | {diff:.2g}"
        else:
            cols = "not run | - | - | -"
        print(f"| {d} | {2**d-d-2:,} | {len(g)} | {statistics.median(ms):.1f} / {max(ms):.1f} | {opt}/{len(g)} | {cols} |")
    print("\nRecovery of the hidden segment from the flat junction (node decision at margin 4.6):")
    print("| children d | segment r (ohm) | instances | true evidence (median) | decided present, right union | "
          "decided present, wrong union | open | absent |")
    print("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for d in sorted({r["feeder"]["blocks"] for r in rows}):
        for rr in R_LEVELS:
            g = [r for r in rows if r["feeder"]["blocks"] == d and r["feeder"]["r_shared"] == rr]
            if not g:
                continue
            right = sum(r["miqp"]["decision"] == "present" and r["miqp"]["line"] == r["target"] for r in g)
            wrong = sum(r["miqp"]["decision"] == "present" and r["miqp"]["line"] != r["target"] for r in g)
            ev = [r["true_evidence"] for r in g if r["true_evidence"] is not None]
            print(f"| {d} | {rr:g} | {len(g)} | {statistics.median(ev):.1f} | {right} | {wrong} | "
                  f"{sum(r['miqp']['decision'] == 'open' for r in g)} | "
                  f"{sum(r['miqp']['decision'] == 'absent' for r in g)} |" if ev else
                  f"| {d} | none | {len(g)} | - | - | {wrong} | {sum(r['miqp']['decision'] == 'open' for r in g)} | "
                  f"{sum(r['miqp']['decision'] == 'absent' for r in g)} |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=("run", "summary"))
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--widths", type=lambda s: tuple(int(x) for x in s.split(",")), default=WIDTHS)
    p.add_argument("--seeds", type=lambda s: tuple(int(x) for x in s.split(",")), default=(1, 2, 3))
    p.add_argument("--max-enum", type=int, default=16, help="enumerate nodes up to this many children")
    p.add_argument("--seconds", type=float, default=300., help="node MIQP time limit per solve")
    p.add_argument("--warm", action="store_true",
                   help="start each node MIQP from a quick local search over unions (speed only)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    run(args) if args.command == "run" else summary(args)


if __name__ == "__main__":
    main()
