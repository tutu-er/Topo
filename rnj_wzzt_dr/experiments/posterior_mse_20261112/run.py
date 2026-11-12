"""Re-score 96 historical libraries plus four completed strict-96 libraries.

No estimator, optimizer, AC simulation, fresh observations, or random draws.
Protocol -> weights -> selection seal -> truth evaluation -> public aggregates.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile

import numpy as np

from ... import io
from ...observations import Observation, load
from .scoring import active_topology, independent_check, score, topology_id


ROOT = Path(__file__).resolve().parents[3]
PACKAGE = Path(__file__).resolve().parents[2]
METHODS = ("101", "111")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(reference):
    path = Path(reference["path"])
    if sha(path) != reference["sha256"]:
        raise ValueError("Frozen input changed: "+str(path))
    return path


def protocol_rules():
    return dict(formula="sigma2=min validation MSE; s=2*sigma2/N; w=softmax(-MSE/s)",
        noise="independent zero-mean Gaussian scalar residuals, common unknown variance plugged in",
        N="all validation scalar cells: terminal count times validation slots",
        prior="uniform across distinct active topologies",
        topology="original zero_tol active proper clades; no new impedance threshold or pruning",
        same_topology_points="retain all; use minimum-training-MSE frozen point, tie by original index",
        zero_variance="report failure; no arbitrary positive floor",
        scope="retrospective completed records, already studied; not independent new data",
        primary="48 historical inputs times 101/111; train96, validation288",
        supplement="four completed strict96 inputs, fixed relative_gain=0; train64, validation32",
        exclusions="active large96 batch, raw observations from test files, new fits and new DR",
        gates=[.9, .95, .99], reliability_edges=[0., .5, .7, .9, .95, .99, 1.],
        near_pair="existing .003 relative alternative-MSE band with original numerical tie",
        outputs="MAP correctness, coverage, confidence gates, wrong high confidence, Brier, log score",
        dependency="same history across methods and four shared seed groups; no independent-trial CI",
        budget=dict(seconds=600, workers=1, optimizer_calls=0, random_draws=0))


def prepare(out):
    if out.exists():
        raise FileExistsError("Existing run preserved; choose a new output directory")
    full = ROOT/"topology_identification_standalone/outputs/full48_mse_20261111"
    strict = ROOT/"topology_identification_standalone/outputs/strict96_20261112_002"
    archive = ROOT/"topology_identification_standalone/experiments/full48_mse/records.zip"
    if not io.read(full/"completion.json")["completion"]:
        raise ValueError("Full historical source is not complete")
    if not io.read(strict/"completion.json")["all_planned_terminal"]:
        raise ValueError("Strict96 source is not complete")
    catalog = io.read(full/"catalog.json")
    requests = []
    with zipfile.ZipFile(archive) as z:
        for job, row in sorted(catalog.items()):
            for method in METHODS:
                result = full/"runs"/job/method/"result.json"
                native = io.read(result)
                if native["status"] != "fitted" or native.get("budget_exceeded"):
                    raise ValueError("Incomplete historical method: "+str(result))
                member = f"full48_mse_20261111/runs/{job}/{method}/result.json"
                if hashlib.sha256(z.read(member)).hexdigest() != sha(result):
                    raise ValueError("Live record differs from sealed archive: "+str(result))
                train, validation = full/"runtime"/row["train"], full/"runtime"/row["validation"]
                for role, path in (("train", train), ("validation", validation)):
                    if sha(path) != native["train_val_input_hashes"][role]:
                        raise ValueError("Observation differs from original fitting input")
                requests.append(dict(id=f"full48__{job}__{method}", cohort="full48", job=job,
                    method=method, case=row["case"], seed_index=row["seed_index"],
                    voltage_sigma=row["voltage_sigma"], result=io.ref(result),
                    train=io.ref(train), validation=io.ref(validation)))
    if len(requests) != 96:
        raise ValueError("Expected 48 inputs times two methods")
    original_protocol = io.read(strict/"protocol.json")
    for input_id, row in sorted(original_protocol["inputs"].items()):
        result = strict/"runs"/input_id/"0.0.json"
        native = io.read(result)
        observed = strict/"runtime"/row["path"]
        if (native["status"] != "fitted" or native["kappa"] != 0
                or native["input_sha256"] != sha(observed) or row["sha256"] != sha(observed)):
            raise ValueError("Strict96 completed source mismatch")
        requests.append(dict(id=f"strict96__{input_id}__101", cohort="strict96", job=input_id,
            method="101", result=io.ref(result), observed=io.ref(observed),
            split=dict(train=[0, 64], validation=[64, 96])))
    if len(requests) != 100:
        raise ValueError("Expected exactly four strict96 supplement inputs")
    out.mkdir(parents=True)
    sources = {}
    for path in sorted(PACKAGE.rglob("*.py")):
        if "outputs" in path.relative_to(PACKAGE).parts or "__pycache__" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        target = out/"source_snapshot"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        sources[str(relative).replace("\\", "/")] = io.ref(path)
    protocol = dict(schema="posterior_mse_validation_v1", status="frozen_before_scoring",
        created_utc=datetime.now(timezone.utc).isoformat(), rules=protocol_rules(), expected=100,
        code_commit_at_lock=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        environment=io.environment(), source_references=sources,
        source_catalog=io.ref(full/"catalog.json"), source_archive=io.ref(archive),
        source_completions=[io.ref(full/"completion.json"), io.ref(strict/"completion.json")],
        strict_evaluation_map_path=str(strict/"evaluation_map.json"),
        truth_loader_path=str(full/"runtime/data.py"), requests=requests)
    io.write(out/"protocol.json", protocol)
    io.write(out/"prepared.json", dict(protocol_sha256=sha(out/"protocol.json")))
    return protocol


def load_request(request):
    wrapper = io.read(checked(request["result"]))
    native = wrapper["result"]
    if native.get("truth_used") or native.get("test_used"):
        raise ValueError("Native identifier used forbidden truth/test inputs")
    if request["cohort"] == "full48":
        training = load(checked(request["train"]))
        validation = load(checked(request["validation"]))
        if (training.points, validation.points) != (96, 288):
            raise ValueError("Historical day split changed")
    else:
        day = load(checked(request["observed"]))
        if day.points != 96:
            raise ValueError("Strict96 day length changed")
        training = Observation(day.p[:64], day.q[:64], day.y[:64], day.labels)
        validation = Observation(day.p[64:], day.q[64:], day.y[64:], day.labels)
    return native, training, validation


def calculate(out, protocol, began):
    scores = []
    for ordinal, request in enumerate(protocol["requests"], 1):
        if time.perf_counter()-began > protocol["rules"]["budget"]["seconds"]:
            raise TimeoutError("Predeclared wall budget exceeded")
        try:
            native, training, validation = load_request(request)
            scored = score(native, training, validation)
            scored["verification"] = independent_check(native, training, validation, scored)
            scored.update(id=request["id"], cohort=request["cohort"], method=request["method"])
        except Exception as exc:
            scored = dict(id=request["id"], cohort=request["cohort"], method=request["method"],
                          status="failed", error=repr(exc), traceback=traceback.format_exc())
        io.write(out/"scores"/(request["id"]+".json"), scored)
        scores.append(scored)
        if ordinal % 20 == 0:
            print(f"scored {ordinal}/{len(protocol['requests'])}", flush=True)
    seal = dict(status="weights_frozen_before_truth", protocol_sha256=sha(out/"protocol.json"),
                scores={r["id"]: sha(out/"scores"/(r["id"]+".json")) for r in scores})
    io.write(out/"selection_seal.json", seal)
    return scores


def evaluate(out, protocol, scores):
    seal = io.read(out/"selection_seal.json")
    if seal["protocol_sha256"] != sha(out/"protocol.json"):
        raise ValueError("Protocol changed before grading")
    for request in protocol["requests"]:
        if seal["scores"][request["id"]] != sha(out/"scores"/(request["id"]+".json")):
            raise ValueError("Frozen weights changed before grading")
    path = Path(protocol["truth_loader_path"])
    spec = importlib.util.spec_from_file_location("_posterior_frozen_truth", path)
    data = importlib.util.module_from_spec(spec); sys.modules[spec.name] = data; spec.loader.exec_module(data)
    evaluation_map = io.read(protocol["strict_evaluation_map_path"])
    references = [io.ref(path), io.ref(protocol["strict_evaluation_map_path"])]
    rows = []
    for request, scored in zip(protocol["requests"], scores, strict=True):
        if scored["status"] != "complete":
            rows.append(dict(id=request["id"], cohort=request["cohort"], method=request["method"], status="failed"))
            continue
        case = request.get("case", evaluation_map.get(request["job"], {}).get("case"))
        truth = data.case_truth(case, tuple(scored["labels"]))
        truth_clades = tuple(sorted(tuple(sorted(scored["labels"][i] for i in C))
                                  for C in truth if 1 < len(C) < len(scored["labels"])))
        truth_id = topology_id(truth_clades)
        for name in ("metadata.json", "buses.csv", "branches.csv"):
            reference = io.ref(path.parent/"cases"/case/name)
            if reference not in references: references.append(reference)
        probability = {m["topology_id"]: m["probability"] for m in scored["models"]}
        logs = {m["topology_id"]: m["log_probability"] for m in scored["models"]}
        covered = truth_id in probability
        p_true = probability.get(truth_id, 0.)
        row = dict(id=request["id"], cohort=request["cohort"], method=request["method"], case=case,
            seed_index=request.get("seed_index", 0), voltage_sigma=request.get("voltage_sigma", .0005),
            status="complete", cells=scored["cells"], raw_points=scored["raw_points"],
            distinct_topologies=scored["distinct_topologies"], truth_covered=covered,
            map_correct=scored["map_topology_id"] == truth_id,
            native_correct=scored["native_topology_id"] == truth_id,
            map_matches_native=scored["map_matches_native_topology"], p_map=scored["p_map"],
            p_true=p_true, log_p_true=logs.get(truth_id),
            brier=float(sum(p*p for p in probability.values())-2*p_true+1),
            near_pair_003=scored["near_pair_003"], pair_probability=scored["pair_probability"],
            max_probability_error=scored["verification"]["maximum_probability_error"])
        rows.append(row)
    io.write(out/"evaluation.json", rows)
    io.write(out/"evaluation_sources.json", references)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve(); began = time.perf_counter()
    protocol = prepare(out)
    scores = calculate(out, protocol, began)
    rows = evaluate(out, protocol, scores)
    from .deliver import publish
    publish(out, protocol, rows)
    io.write(out/"complete.json", dict(status="complete", expected=len(rows),
        completed=sum(r["status"] == "complete" for r in rows),
        failures=sum(r["status"] != "complete" for r in rows), seconds=time.perf_counter()-began,
        protocol_sha256=sha(out/"protocol.json"), selection_seal_sha256=sha(out/"selection_seal.json"),
        fits=0, optimizer_calls=0, AC_calls=0, random_draws=0))
    print(json.dumps(io.read(out/"public/summary.json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
