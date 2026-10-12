"""Auxiliary optimizer recovery on one frozen failed observed-data fit.

The original formal JSON/NPZ and its statistical thresholds are never changed.
No evaluation truth enters any fitted model or initialization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import scipy

from .likelihood import Observations, RadialCandidate, concat_observations, fit_candidate
from .simulation import model

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: finite(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):
        return [finite(item) for item in value]
    return value


def summary(fit, label, seconds):
    d = fit.diagnostics
    return dict(label=label, success=fit.success, loss=fit.loss, loglik=fit.loglik,
        status=fit.status, message=fit.message, nfev=fit.nfev,
        optimality=fit.optimality, elapsed_seconds=seconds,
        r_pu=fit.r, x_pu=fit.x, restarts=d.get("restarts"),
        ac_equation_residual=d.get("ac_equation_residual"),
        power_balance_residual=d.get("power_balance_residual"),
        min_voltage=d.get("min_voltage"), max_voltage=d.get("max_voltage"),
        physical_voltage_feasible=d.get("physical_voltage_feasible"),
        local_shared_rank=d.get("local_shared_rank"),
        shared_parameter_count=d.get("shared_parameter_count"),
        failure_kind=d.get("failure_kind"), global_optimality_certified=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", default=str(HERE/"outputs"/"formal"))
    parser.add_argument("--output", default=str(HERE/"evidence"/"optimizer_recovery.json"))
    parser.add_argument("--max-nfev", type=int, default=360)
    args = parser.parse_args()
    formal = Path(args.formal)
    json_path = formal/"strong_51007.json"
    npz_path = formal/"strong_51007_observations.npz"
    config_path = formal/"config.json"
    frozen_paths = [json_path,npz_path,config_path]
    if (formal/"thresholds.json").exists():
        frozen_paths.append(formal/"thresholds.json")
    original_hashes = {str(p.relative_to(HERE)):digest(p) for p in frozen_paths}
    original = json.loads(json_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8-sig"))
    # Relevant frozen components only; independent document/SOC repairs do
    # not affect the fitter or reconstruction of recorded P/Q/V/master data.
    for name in ("likelihood.py","simulation.py"):
        if digest(HERE/name) != config["code_sha256"][name]:
            raise RuntimeError("Frozen fitter/measurement-model source changed: "+name)
    rows = original["historical_fits"]
    index = next(i for i,row in enumerate(rows) if row["name"]=="clade_01")
    historical = rows[index]
    failed_row = next(row for row in original["records"] if row["method"]=="fixed_pulse")
    failed_fit = next(fit for fit in failed_row["fits"] if fit["name"]=="clade_01")
    if failed_fit["success"]:
        raise RuntimeError("Requested formal case is not an optimizer failure")
    n = original["n"]
    candidate = RadialCandidate.from_supports(historical["name"],n,historical["supports"])
    with np.load(npz_path,allow_pickle=False) as arrays:
        def block(prefix):
            return Observations(p=arrays[prefix+"_p"],q=arrays[prefix+"_q"],
                v=arrays[prefix+"_v"],root_v=1.,
                master_p=arrays[prefix+"_master_p"],master_q=arrays[prefix+"_master_q"])
        history = block("history")
        future = block("fixed_pulse")
    joined = concat_observations(history,future)
    meter = model(n)
    seed_history = original["seed"]+300+index
    seed_joint = original["seed"]+1000+index
    common = dict(multistart=2,max_nfev=args.max_nfev)
    started = perf_counter()
    t = perf_counter()
    cold = fit_candidate(candidate,joined,meter,initial=None,seed=seed_joint,**common)
    cold_record = summary(cold,"cold_interior_joint",perf_counter()-t)
    print(f"cold joint: success={cold.success} loss={cold.loss:.8f} nfev={cold.nfev}",flush=True)
    t = perf_counter()
    recovered_history = fit_candidate(candidate,history,meter,
        initial={"r":historical["r_pu"],"x":historical["x_pu"]},
        seed=seed_history,**common)
    history_record = summary(recovered_history,"historical_parameter_warm_refit",perf_counter()-t)
    print(f"history: success={recovered_history.success} loss={recovered_history.loss:.8f}",flush=True)
    t = perf_counter()
    warm = fit_candidate(candidate,joined,meter,initial=recovered_history,seed=seed_joint,**common)
    warm_record = summary(warm,"recovered_history_warm_joint",perf_counter()-t)
    print(f"warm joint: success={warm.success} loss={warm.loss:.8f} nfev={warm.nfev}",flush=True)
    current_hashes = {str(p.relative_to(HERE)):digest(p) for p in frozen_paths}
    if current_hashes != original_hashes:
        raise RuntimeError("Original formal evidence changed during auxiliary probe")
    failed_diagnostics = failed_fit["diagnostics"]
    original_restart = min(
        (item for item in failed_diagnostics["restarts"] if item.get("loss") is not None),
        key=lambda item:item["loss"])
    originals = dict(label="original_formal_warm_two_starts_max120",
        success=failed_fit["success"],loss=failed_fit["loss"],loglik=failed_fit["loglik"],
        status=original_restart["status"],message=original_restart["message"],
        nfev=original_restart["nfev"],optimality=original_restart["optimality"],
        elapsed_seconds=failed_diagnostics["seconds"],
        r_pu=failed_fit["r_pu"],x_pu=failed_fit["x_pu"],
        restarts=failed_diagnostics["restarts"],
        ac_equation_residual=failed_diagnostics["ac_equation_residual"],
        power_balance_residual=failed_diagnostics["power_balance_residual"],
        min_voltage=failed_diagnostics["min_voltage"],
        max_voltage=failed_diagnostics["max_voltage"],
        physical_voltage_feasible=failed_diagnostics["physical_voltage_feasible"],
        local_shared_rank=failed_diagnostics["local_shared_rank"],
        failure_kind=failed_diagnostics["failure_kind"],
        global_optimality_certified=False,diagnostics=failed_diagnostics)
    result = dict(schema=1, auxiliary_diagnostic=True, case="strong",seed=51007,
        method="fixed_pulse",candidate="clade_01",
        purpose="separate finite optimizer budget/start effects from AC physics/information",
        original_formal_evidence_unchanged=True,statistical_thresholds_unchanged=True,
        no_truth_used_for_fitting_or_initialization=True,
        source_model="one common node1; independent per-time P/Q anchored by master meters",
        likelihood_model="known diagonal Gaussian; calibrated fixed terminal biases; known root1",
        cold_initial_shared_r_pu=.012,cold_initial_shared_x_pu=.006,
        observation_shapes=dict(history=list(history.p.shape),future=list(future.p.shape),
                                joint=list(joined.p.shape)),
        comparison=[originals,cold_record,history_record,warm_record],
        objective_improvement_vs_original=dict(cold=failed_fit["loss"]-cold.loss,
                                              warm=failed_fit["loss"]-warm.loss),
        configuration=dict(multistart=2,recovery_max_nfev=args.max_nfev,
            original_max_nfev=config["max_nfev"],seed_history=seed_history,seed_joint=seed_joint,
            r_bounds_pu=[1e-6,.1],x_bounds_pu=[1e-6,.1],source_bounds_pu=[-.5,.5],
            terminal_pq_bounds_pu=[-1.,1.],accepted_voltage_bounds=[.85,1.15]),
        provenance=dict(frozen_input_sha256=original_hashes,
            script_sha256=digest(Path(__file__)),
            fitter_sha256=digest(HERE/"likelihood.py"),
            simulation_sha256=digest(HERE/"simulation.py"),
            python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
            platform=platform.platform(),
            threads={key:os.environ.get(key) for key in
                     ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS")}),
        total_elapsed_seconds=perf_counter()-started,
        global_optimality_certified=False,
        interpretation="attained local objectives; one observed-data recovery probe, no new statistical calibration")
    output = Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(finite(result),ensure_ascii=False,indent=2,allow_nan=False)+"\n",
                      encoding="utf-8")
    print(str(output),flush=True)


if __name__=="__main__":
    main()
