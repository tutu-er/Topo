"""One saved action: same nominal target through the actual-evaluation fitter.

No actual future measurements or generating line/load/source values are read.
This diagnostic deliberately reruns four candidate fits on the reference's
nominal mean, to expose warm-start/seed/local-attainment differences between
the predictive engine and the actual-evaluation entry point.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from . import ev_mobile_v4_audit as audit
from . import ev_mobile_v4_study as study
from . import ev_mobile_v2_review as io
from . import ev_systematic_refit as refit
from . import gradient_price_study as serial_io
from .information_replay import _obs, _meter
from . import likelihood as lk


def nominal_future(candidate, fitted, forecast, future_meter, power, base_kw, *, predictor=lk.ac_predict):
    """Build the same noiseless future mean as JointPredictiveEngine."""
    meter = _meter(future_meter)
    power = np.asarray(power, float)
    p, q = np.asarray(forecast["p"]) + power / base_kw, np.asarray(forecast["q"])
    source = forecast["source_by_candidate"][candidate.name]
    physical = predictor(candidate, fitted.r, fitted.x, p, q, source["source_p"],
                         source["source_q"], forecast["root_v"])
    T, n = p.shape
    return lk.Observations(p=p + np.broadcast_to(np.asarray(meter.bias_p), (T, n)),
        q=q + np.broadcast_to(np.asarray(meter.bias_q), (T, n)),
        v=np.asarray(physical["v"]) + np.broadcast_to(np.asarray(meter.bias_v), (T, n)),
        master_p=np.asarray(physical["master_p"]), master_q=np.asarray(physical["master_q"]),
        root_v=np.asarray(forecast["root_v"]))


def _seed(base, name):
    return (int(base) + int.from_bytes(sha256(name.encode()).digest()[:4], "little")) % 2 ** 32


def check_record(record, frozen_config, *, method="guarded__fixed_charge", reference=None,
                 evaluator=study.evaluate_future, predictor=lk.ac_predict,
                 replayer=io.replay_state):
    began = perf_counter()
    bank = audit._bank(record)
    reference = record["true_name"] if reference is None else reference
    io.require(reference in bank, "requested reference absent from candidate bank")
    method_record = record["methods"][method]
    key = method_record["action_key"]
    profile = record["reporting_cache"][key]
    io.require(profile.get("success"), "complete nominal profile required")
    reference_row = next(row for row in profile["references"] if row["a"] == reference)
    trained = {row.get("candidate", row.get("name")): row for row in record["training_result"]["fits"]}
    fits = {name: SimpleNamespace(**{field:np.asarray(row[field], float)
        for field in ("r", "x", "source_p", "source_q")}) for name, row in trained.items()}
    future = nominal_future(bank[reference], fits[reference], record["forecast"],
        record["future_meter"], method_record["power_kw"], float(record["config"]["base_kw"]), predictor=predictor)
    D, dm, fm = _obs(record["observed"]), _meter(record["observed_meter"]), _meter(record["future_meter"])
    cfg = record["config"]
    options = refit._options({**cfg["training"], **cfg["source"], "ratio_bounds":cfg["ratio_bounds"],
        "multistart":frozen_config["multistart_predictive"], "max_nfev":frozen_config["max_nfev"]})
    seed = int(record["spec"]["seed"])
    observed, future_record = record["observed"], serial_io.observations_record(future)
    blocks = (("training", observed, record["observed_meter"]),
              ("future", future_record, record["future_meter"]))
    planned_replay = {}
    for name, candidate in bank.items():
        saved = reference_row["fits"][name]
        replay = replayer(candidate, saved["state"], blocks, cfg)
        residual_error = max(float(np.max(abs(np.asarray(saved["blocks"][label]["residual"])
            - np.asarray(replay["blocks"][label]["residual"])))) for label in ("training", "future"))
        planned_replay[name] = dict(saved_loss=saved["loss"], replay_loss=replay["loss"],
            difference=float(replay["loss"] - saved["loss"]),
            max_whitened_residual_difference=residual_error,
            same_nominal_target_verified=bool(np.isclose(replay["loss"], saved["loss"], atol=2e-7, rtol=2e-8)
                                                and residual_error <= 2e-7))
    rerun = evaluator(tuple(bank.values()), D, dm, future, fm, fits, options, seed + 4000)
    complete = bool(rerun["complete"])
    old_losses = {name:float(reference_row["fits"][name]["loss"]) for name in bank}
    new_losses = {name:float(row["joint"]["loss"]) for name,row in rerun["comparisons"].items()}
    differences, gaps = {}, {}
    for index, name in enumerate(bank):
        if name in new_losses:
            differences[name] = dict(planning_loss=old_losses[name], actual_entry_loss=new_losses[name],
                difference=new_losses[name] - old_losses[name],
                planning_source_seed=_seed(seed + 3000, reference + "__" + name),
                planning_joint_seed=_seed(seed + 3000, reference + "__" + name + "joint"),
                actual_source_seed=seed + 4000 + index, actual_joint_seed=seed + 4100 + index)
        if name != reference and name in new_losses and reference in new_losses:
            before = .5 * (old_losses[name] - old_losses[reference])
            after = .5 * (new_losses[name] - new_losses[reference])
            gaps[name] = dict(planning_half_gap=before, actual_entry_half_gap=after, difference=after - before)
    max_loss = max((abs(row["difference"]) for row in differences.values()), default=None)
    max_gap = max((abs(row["difference"]) for row in gaps.values()), default=None)
    numerically_agree = complete and all(abs(row["difference"]) <= 1e-5 * (1. + abs(row["planning_loss"]))
                                        for row in differences.values())
    old_nearest = min((name for name in bank if name != reference), key=lambda name:old_losses[name])
    new_nearest = min((name for name in new_losses if name != reference), key=lambda name:new_losses[name]) if complete else None
    return dict(success=complete and all(row["same_nominal_target_verified"] for row in planned_replay.values()),
        spec=record["spec"], method=method, action_key=key, reference=reference,
        reference_choice="saved generating label identifies an already trained candidate; no generating parameters used" if reference == record["true_name"] else "explicit saved candidate",
        read_actual_future=False, read_generating_parameters=False, action_selection_calls=0,
        information_scope="saved observed D, theta_D, ordinary/source forecast, future meter, retained power",
        same_nominal_target_replays=planned_replay, options=options,
        candidate_loss_comparison=differences, signed_half_gap_comparison=gaps,
        max_absolute_candidate_loss_difference=max_loss, max_absolute_half_gap_difference=max_gap,
        numerical_agreement=numerically_agree, relative_tolerance=1e-5,
        planning_nearest_competitor=old_nearest, actual_entry_nearest_competitor=new_nearest,
        nearest_competitor_changed=new_nearest is not None and old_nearest != new_nearest,
        planning_source_initialization="candidate forecast source, at theta_D",
        actual_source_initialization="initial prefix of candidate observed-D source, at theta_D",
        joint_initialization="observed-D state plus corresponding future fixed-line source fit",
        nominal_future=future_record, rerun_evaluation=rerun,
        seconds=perf_counter() - began, fit_scope="four future source fits and four shared-line joint fits; local attained results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", required=True)
    parser.add_argument("--freeze", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", default="guarded__fixed_charge")
    parser.add_argument("--reference")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    path = Path(args.record)
    frozen = io.read(args.freeze)
    record = io.read(path)
    io.require(record["policy_id"] == frozen["policy_id"], "record/freeze policy differs")
    result = check_record(record, frozen["config"], method=args.method, reference=args.reference)
    result.update(input_file=str(path.resolve()), input_sha256=sha256(path.read_bytes()).hexdigest(),
                  checker_source_sha256=sha256(Path(__file__).read_bytes()).hexdigest())
    io.write(output, serial_io.serial(result))
    print(json.dumps({key:result[key] for key in
        ("success", "numerical_agreement", "max_absolute_candidate_loss_difference", "max_absolute_half_gap_difference", "nearest_competitor_changed")}))


if __name__ == "__main__":
    main()
