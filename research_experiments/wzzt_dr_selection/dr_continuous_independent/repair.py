"""Use dyadic quadrature for the same frozen inversion and two price scales.

Only the frequency quadrature is changed. Completed run_v1 simulations are
referenced, never rerun. The original runner and its source snapshot stay fixed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import traceback
import zipfile

import numpy as np

import run as bounded
from continuous import IndependentResiduals
from gauss_frequency import gauss_probability


class GaussResiduals(IndependentResiduals):
    def probability(self, threshold, **options):
        options.pop("step", None)
        options.pop("max_cutoff", None)
        return gauss_probability(self, threshold, **options)


def main(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/"complete.json").exists():
        print("Already complete", flush=True)
        return
    lock = out/"running.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        if not (args.previous/"complete.json").exists():
            raise ValueError("Previous bounded run is incomplete")
        previous = bounded.read(args.previous/"results.json")
        old_frozen = previous["frozen"]
        problem = bounded.build_problem(args.archive.resolve(), args.public.resolve())
        point = bounded.nominal(problem, bounded.POLICIES["frozen"])
        baseline = IndependentResiduals(point["a_C"], point["a_F"], problem["sigma"].ravel())
        problem["normalization"] = (baseline.sum_mean, baseline.sum_sd)
        if not np.array_equal(problem["normalization"], old_frozen["normalization"]):
            raise ArithmeticError("Normalization differs from frozen Simpson check")
        for path in (Path(__file__), bounded.HERE/"gauss_frequency.py", bounded.HERE/"PLAN.md"):
            problem["sources"][str(path.relative_to(bounded.TASK))] = bounded.sha(path)
        request = dict(previous=bounded.reference(args.previous/"results.json"),
                       previous_complete=bounded.reference(args.previous/"complete.json"),
                       sources=problem["sources"], inputs=problem["inputs"],
                       normalization=problem["normalization"], cutoff=old_frozen["cutoff"],
                       initial_order=16,refinements=2,tolerance=1e-7,gradient_tolerance=1e-5,
                       search_order=32,search_refinements=0,maxiter=15,seconds=120,
                       no_new_MC=True,no_new_AC=True,
                       scope="conditional_voltage_probability; independent primitive MC only reused")
        if (out/"request.json").exists() and bounded.read(out/"request.json")!=bounded.plain(request):
            raise ValueError("Changed inputs/source in existing repair stage")
        bounded.write(out/"request.json",request)
        if not (out/"source_snapshot.zip").exists():
            with zipfile.ZipFile(out/"source_snapshot.zip","x",zipfile.ZIP_DEFLATED) as snapshot:
                for filename in problem["sources"]:
                    snapshot.write(bounded.TASK/filename,filename)
        # Preserve the original data preparation, feasible direction mapping,
        # gradient checks and optimizer. Only the probability method is swapped.
        bounded.IndependentResiduals = GaussResiduals
        options = dict(cutoff=old_frozen["cutoff"],order=16,refinements=2,
                       tolerance=1e-7,gradient_tolerance=1e-5)
        fixed = dict(cutoff=old_frozen["cutoff"],order=32,refinements=0,
                     tolerance=1e-7,gradient_tolerance=1e-5)
        results = bounded.read(out/"results.json") if (out/"results.json").exists() else {}
        if "frozen" not in results:
            results["frozen"] = bounded.kernel(problem,bounded.POLICIES["frozen"],options)
            bounded.write(out/"results.json",results)
        tested = results["frozen"]
        comparison = abs(tested["probability"]-old_frozen["probability"])
        gate = tested["numerical_pass"] and tested["seconds"]<=10 and comparison<1e-6
        results["comparison"] = dict(previous_probability=old_frozen["probability"],
                                      absolute_difference=comparison,
                                      previous_seconds=old_frozen["seconds"],
                                      speedup=old_frozen["seconds"]/tested["seconds"],
                                      gate_passed=bool(gate))
        bounded.write(out/"results.json",results)
        if "gradient_check" not in results:
            results["gradient_check"] = bounded.gradient_check(problem,fixed) if gate else dict(status="skipped_baseline_gate")
            bounded.write(out/"results.json",results)
        if "gradient_refinement" not in results:
            results["gradient_refinement"] = bounded.kernel(problem,[.5,.5],options,directions=True) if gate else dict(status="skipped_baseline_gate")
            bounded.write(out/"results.json",results)
        gradient_gate = (gate and results["gradient_refinement"].get("numerical_pass",False)
                         and results["gradient_check"].get("max_absolute_difference",float("inf"))<.001)
        if "optimization" not in results:
            results["optimization"] = bounded.optimize(problem,fixed,out) if gradient_gate else dict(status="skipped_gradient_gate")
            bounded.write(out/"results.json",results)
        if "refined_best" not in results:
            best = results["optimization"].get("best")
            results["refined_best"] = bounded.kernel(problem,best["theta"],options,directions=True) if best else None
            bounded.write(out/"results.json",results)
        results["reused_validation"] = dict(conditional=previous["conditional_mc"],
                                              full_primitive=previous["full_mc"])
        bounded.write(out/"results.json",results)
        bounded.write(out/"complete.json",dict(status="complete",results_sha256=bounded.sha(out/"results.json"),
                                                no_new_MC=True,no_AC_run=True,
                                                optimizer_scope="conditional_voltage_noise_only"))
        print(json.dumps(bounded.plain(results),ensure_ascii=True),flush=True)
    except Exception as exc:
        bounded.write(out/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("archive","public","previous","out"):
        parser.add_argument("--"+key,type=Path,required=True)
    main(parser.parse_args())
