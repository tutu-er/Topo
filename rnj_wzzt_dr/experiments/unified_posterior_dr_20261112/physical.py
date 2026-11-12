"""Adapt the exact frozen large96 AC generator to original device delivery.

No legacy simulator is substituted. The original function is called unchanged;
an observer captures its already-generated background at the AC boundary.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np

from ... import io
from ...observations import Observation, load
from .history_scan import ROOT, require, sha, utc


def open_runtime(run, cohort):
    requests = io.read(run/"requests.json")
    batch = Path(next(r["batch"] for r in requests if r["cohort"] == cohort))
    protocol = io.read(batch/"protocol.json")
    directory = run/"physical/source_runtime"/cohort
    source = batch/"runtime/data.py"
    require(sha(source) == protocol["source_sha256"]["data.py"], "Physical generator source differs")
    if not directory.exists():
        directory.mkdir(parents=True)
        shutil.copyfile(source, directory/"data.py")
        for relative, expected in protocol["evaluation_case_sha256"].items():
            asset = batch/"evaluation_cases"/relative
            require(sha(asset) == expected, "Original physical case snapshot changed")
            target = directory/"cases"/relative
            target.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(asset, target)
    require(sha(directory/"data.py") == protocol["source_sha256"]["data.py"], "Adapted generator bytes differ")
    for relative, expected in protocol["evaluation_case_sha256"].items():
        require(sha(directory/"cases"/relative) == expected, "Adapted physical case differs")
    name = "_posterior_ac_"+cohort+"_"+str(abs(hash(str(directory))))
    spec = importlib.util.spec_from_file_location(name, directory/"data.py")
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module, batch, protocol


def generate_background(module, parameters):
    captured = {}; original = module._ac_voltages
    def observe(network, labels, p, q, root, **kwargs):
        require(not captured, "Generator invoked more than one physical day")
        captured.update(network=deepcopy(network), labels=tuple(labels),
                        p=np.array(p, copy=True), q=np.array(q, copy=True), root=np.array(root, copy=True))
        voltage, audit = original(network, labels, p, q, root, **kwargs)
        captured.update(voltage=voltage, ac=audit)
        return voltage, audit
    module._ac_voltages = observe
    try:
        observed, metadata = module.generate_day96(**parameters)
    finally:
        module._ac_voltages = original
    require(captured and observed.points == 96, "Original one-day generator did not produce 96 rows")
    entropy = metadata["seed_entropy"]
    shapes = (captured["p"].shape, captured["q"].shape, captured["voltage"].shape, captured["root"].shape)
    noise = [np.random.default_rng(np.random.SeedSequence([*entropy, component])).normal(size=shape)
             for component, shape in zip((2, 3, 4, 5), shapes, strict=True)]
    captured.update(observed=observed, metadata=metadata, noise=noise, parameters=dict(parameters))
    return captured


def measured_channels(background, p, q, voltage):
    metadata = background["metadata"]; noise = background["noise"]
    power_sigma, voltage_sigma = metadata["pq_sigma"], metadata["voltage_sigma"]
    pm = p*(1+power_sigma*noise[0]); qm = q*(1+power_sigma*noise[1])
    vm = voltage*(1+voltage_sigma*noise[2]); rm = background["root"]*(1+voltage_sigma*noise[3])
    return Observation(pm, qm, rm[:, None]**2-vm**2, background["labels"], vm, rm)


def execute(module, background, original, nominal_mean, multiplier):
    """Byte-identical original Devices.project/draw/fees contract.

    Mirrors old dr_heterogeneous_pilot.driver.execute: project(mean*draw).
    fees(delivered) is an execution wear proxy, NOT an additional payment.
    """
    nominal_mean, multiplier = np.asarray(nominal_mean), np.asarray(multiplier)
    delivered = original.project(nominal_mean*multiplier)
    device = original.audit(delivered)
    require(device["passed"], "Original causal delivery violates its device domain")
    dp, dq = original.aggregate(delivered)
    base = background["network"]["base_kw"]
    p, q = background["p"]+dp/base, background["q"]+dq/base
    if np.all(delivered == 0):
        voltage, ac = background["voltage"], background["ac"]
        calls = 0  # Baseline already solved by the unchanged one-day generator.
    else:
        voltage, ac = module._ac_voltages(background["network"], background["labels"], p, q, background["root"])
        calls = 1
    observed = measured_channels(background, p, q, voltage)
    saturation = np.abs(delivered-nominal_mean*multiplier) > 1e-8
    nominal_fee = original.fees(nominal_mean)
    summary = dict(status="complete", original_device_audit=device,
        planned_contract_fee=nominal_fee, contract_payment=nominal_fee,
        realized_resource_wear_proxy=original.fees(delivered), wear_proxy_is_additional_payment=False,
        clipping_cells=int(saturation.sum()), clipping_fraction=float(saturation.mean()),
        recourse_abs_kwh=float(original.dt*np.abs(delivered-nominal_mean*multiplier).sum()),
        mean_abs_delivery_error_kw=float(np.mean(np.abs(delivered-nominal_mean))),
        ac=ac, ac_day_calls=calls, voltage_violation_cells=int(((voltage < .9)|(voltage > 1.1)).sum()),
        current_thermal_limits_checked=False,
        increment_error_kw=float(max(np.max(abs((p-background["p"])*base-dp)), np.max(abs((q-background["q"])*base-dq)))))
    return observed, summary, dict(delivered_kw=delivered, multiplier=multiplier, physical_p=p, physical_q=q, voltage=voltage)


def verify_historical_bridge(run):
    out = run/"physical"; out.mkdir(exist_ok=True)
    target = out/"historical_bridge_verification.json"
    require(not target.exists(), "Existing physical bridge verification preserved")
    selection = io.read(run/"planning/protocol.json")["selected"]
    requests = {r["id"]: r for r in io.read(run/"requests.json")}
    modules = {}; receipts = []
    for item in selection:
        request = requests[item["id"]]
        if request["cohort"] not in modules:
            modules[request["cohort"]] = open_runtime(run, request["cohort"])
        module, batch, protocol = modules[request["cohort"]]
        old_meta = io.read(batch/"generation.json")[request["job"]]["metadata"]
        parameters = {k: old_meta[k] for k in ("case", "replicate", "voltage_sigma", "pq_sigma", "root_sigma", "study_seed", "profile", "level")}
        generated = generate_background(module, parameters)
        expected, actual = load(request["observed"]["path"]), generated["observed"]
        differences = {name: float(np.max(np.abs(getattr(expected, name)-getattr(actual, name))))
                       for name in ("p", "q", "y", "voltage", "root")}
        require(max(differences.values()) <= 1e-12, "Historical AC bridge did not reproduce original observed day")
        remetered = measured_channels(generated, generated["p"], generated["q"], generated["voltage"])
        require(np.array_equal(remetered.y, actual.y) and np.array_equal(remetered.p, actual.p), "Metering stream reconstruction differs")
        require(generated["metadata"]["physical_branch_sha256"] == old_meta["physical_branch_sha256"], "Pressure branch version differs")
        receipts.append(dict(id=item["id"], observed=io.ref(request["observed"]["path"]),
            maximum_channel_difference=differences, physical_branch_sha256=old_meta["physical_branch_sha256"],
            generator_sha256=protocol["source_sha256"]["data.py"], profile=request["profile"], level=request["level"]))
    io.write(target, dict(status="passed", created_utc=utc(), historical_days_replayed=len(receipts),
        actual_future_days_opened=0, identifier_refits=0, cases=receipts, source=io.ref(__file__),
        execution_law="original Devices.draw then Devices.project(nominal_mean*draw)",
        price_law="original quoted contract fee; delivered wear proxy separately recorded"))
    print("Physical bridge verified on "+str(len(receipts))+" known historical days; new independent days unopened", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args(); verify_historical_bridge(args.run.resolve())


if __name__ == "__main__":
    main()
