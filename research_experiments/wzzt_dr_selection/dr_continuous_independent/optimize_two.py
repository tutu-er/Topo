"""Finish the authorized two-scale check after fine-step gradient validation.

The completed coarse-step gate failure remains in run_v2. This stage adds one
smaller finite difference and uses the same continuous probability and devices.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import traceback
import zipfile

import numpy as np

import run as bounded
from repair import GaussResiduals
from continuous import IndependentResiduals


def fine_check(problem, fixed, prior):
    point = np.array([.5, .5])
    exact = bounded.kernel(problem,point,fixed,directions=True)
    rows = []
    device = problem["device"]
    center_x = bounded.nominal(problem,point)["x"]
    _, center_cache = device.nominal(center_x,cache=True)
    for step in (1e-3,1e-4,1e-5):
        for coordinate in range(2):
            left, right = point.copy(),point.copy()
            left[coordinate] -= step
            right[coordinate] += step
            a, b = bounded.nominal(problem,left),bounded.nominal(problem,right)
            _, left_cache = device.nominal(a["x"],cache=True)
            _, right_cache = device.nominal(b["x"],cache=True)
            changes = {}
            for side, cache in (("left",left_cache),("right",right_cache)):
                changed = (cache[0]!=center_cache[0]) | (cache[1]!=center_cache[1])
                changes[side] = int(np.count_nonzero(changed))
            if step == 1e-5:
                lo = bounded.kernel(problem,left,fixed)
                hi = bounded.kernel(problem,right,fixed)
                estimate = (hi["probability"]-lo["probability"])/(2*step)
                error = abs(estimate-exact["gradient"][coordinate])
                probability = [lo["probability"],hi["probability"]]
            else:
                original = next(row for row in prior["gradient_check"]["rows"]
                                if row["step"]==step and row["coordinate"]==coordinate)
                estimate,error = original["finite_difference"],original["absolute_difference"]
                probability = [original["lower_probability"],original["upper_probability"]]
            rows.append(dict(step=step,coordinate=coordinate,finite_difference=estimate,
                             analytic=exact["gradient"][coordinate],absolute_difference=error,
                             changed_device_faces=changes,probabilities=probability))
    fine = [row for row in rows if row["step"]<=1e-4]
    return dict(point=point.tolist(),analytic=exact,rows=rows,
                maximum_fine_error=max(row["absolute_difference"] for row in fine),
                passed=max(row["absolute_difference"] for row in fine)<1e-5,
                scope="conditional probability, fixed Gauss32 nodes, same quote and physical domain")


def main(args):
    out=args.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    if (out/"complete.json").exists():
        print("Already complete",flush=True)
        return
    lock=out/"running.lock"
    with lock.open("x",encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    try:
        if not (args.previous/"complete.json").exists():
            raise ValueError("Previous Gauss check is incomplete")
        prior=bounded.read(args.previous/"results.json")
        if not prior["comparison"]["gate_passed"] or not prior["gradient_refinement"]["numerical_pass"]:
            raise ValueError("Prior probability/refined gradient checks did not pass")
        problem=bounded.build_problem(args.archive.resolve(),args.public.resolve())
        baseline=bounded.nominal(problem,bounded.POLICIES["frozen"])
        model=IndependentResiduals(baseline["a_C"],baseline["a_F"],problem["sigma"].ravel())
        problem["normalization"]=(model.sum_mean,model.sum_sd)
        if not np.array_equal(problem["normalization"],prior["frozen"]["normalization"]):
            raise ValueError("Changed frozen normalization")
        for path in (Path(__file__),bounded.HERE/"repair.py",bounded.HERE/"gauss_frequency.py",bounded.HERE/"PLAN.md"):
            problem["sources"][str(path.relative_to(bounded.TASK))]=bounded.sha(path)
        request=dict(previous=bounded.reference(args.previous/"results.json"),sources=problem["sources"],
                     inputs=problem["inputs"],normalization=problem["normalization"],
                     fixed_order=32,cutoff=512.,refinements=0,fine_steps=[1e-4,1e-5],
                     gradient_acceptance=1e-5,maxiter=15,seconds=120,
                     no_new_MC=True,no_new_AC=True,no_resource_or_noise_change=True)
        if (out/"request.json").exists() and bounded.read(out/"request.json")!=bounded.plain(request):
            raise ValueError("Changed inputs/source in existing stage")
        bounded.write(out/"request.json",request)
        if not (out/"source_snapshot.zip").exists():
            with zipfile.ZipFile(out/"source_snapshot.zip","x",zipfile.ZIP_DEFLATED) as archive:
                for name in problem["sources"]:
                    archive.write(bounded.TASK/name,name)
        bounded.IndependentResiduals=GaussResiduals
        fixed=dict(cutoff=512.,order=32,refinements=0,tolerance=1e-7,gradient_tolerance=1e-5)
        refined=dict(cutoff=512.,order=16,refinements=2,tolerance=1e-7,gradient_tolerance=1e-5)
        result=bounded.read(out/"results.json") if (out/"results.json").exists() else {}
        if "fine_check" not in result:
            result["fine_check"]=fine_check(problem,fixed,prior)
            bounded.write(out/"results.json",result)
        if "optimization" not in result:
            result["optimization"]=(bounded.optimize(problem,fixed,out)
                if result["fine_check"]["passed"] else dict(status="unresolved_fine_gradient"))
            bounded.write(out/"results.json",result)
        if "refined_best" not in result:
            best=result["optimization"].get("best")
            result["refined_best"]=(bounded.kernel(problem,best["theta"],refined,directions=True)
                                    if best else None)
            bounded.write(out/"results.json",result)
        bounded.write(out/"complete.json",dict(status="complete",results_sha256=bounded.sha(out/"results.json"),
                                                scope="conditional probability and two quote scales only",
                                                no_new_MC=True,no_new_AC=True))
        print(json.dumps(bounded.plain(result),ensure_ascii=True),flush=True)
    except Exception as exc:
        bounded.write(out/"failure.json",dict(error=repr(exc),traceback=traceback.format_exc()))
        raise
    finally:
        lock.unlink()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("archive","public","previous","out"):
        parser.add_argument("--"+name,type=Path,required=True)
    main(parser.parse_args())
