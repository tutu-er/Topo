"""Prepare the predeclared Soumalas Source stress design, without estimation.

Three replicates, three reference 96-point scenarios, paired no-source/rank-2
source conditions. Each replicate/condition supplies 288 fitting rows and no
validation rows. Generation requires a verified frozen runtime; importing the
input loader never activates or replaces a runtime.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CASE = "soumalas11"
VARIANTS = ("no_source", "rank2_source")
REPLICATES = (0, 1, 2)
SCENARIOS = (0, 1, 2)
SOURCE_BUS = 5  # Generation truth only; never attached to algorithm inputs.
DEFAULT_OUTPUT = ROOT / "outputs/source_advantage_soumalas11_20261018"
DEFAULT_RUNTIME = ROOT / "outputs/source_ac288_runtime_20261018"
DEFAULT_ORIGINAL = ROOT / "outputs/source_ac96_20261018/soumalas11/network"
NOISE = {"pq_noise_rel": .005, "v_noise_rel": .001, "root_meter_noise_rel": .001,
         "master_p_noise_rel": .002, "master_q_noise_rel": .002}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, pd.Index)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default)
        handle.write("\n")


def inventory(directory):
    return {p.relative_to(directory).as_posix(): sha256(p) for p in sorted(Path(directory).rglob("*")) if p.is_file()}


def source_profile(mean_kw):
    """Independent deterministic harmonics, identical for every physical day."""
    if not np.isfinite(mean_kw) or mean_kw <= 0:
        raise ValueError("source mean must be positive and finite")
    index = pd.RangeIndex(96, name="t")
    t = np.arange(96)
    p = mean_kw*(1+.65*np.sin(2*np.pi*7*t/96)+.2*np.cos(2*np.pi*13*t/96))
    q = .35*mean_kw*(1+.7*np.cos(2*np.pi*11*t/96+.4))
    if np.linalg.matrix_rank(np.column_stack((p, q))) != 2 or min(p.min(), q.min()) <= 0:
        raise ValueError("declared source must be positive with rank two")
    np.testing.assert_allclose([p.mean(), q.mean()], [mean_kw, .35*mean_kw], rtol=1e-14)
    return pd.Series(p, index=index, name="P_kw"), pd.Series(q, index=index, name="Q_kvar")


def _save_tables(directory, values):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for key, value in values.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            with (directory/f"{key}.csv").open("x", encoding="utf-8", newline="") as handle:
                value.to_csv(handle, index_label="t", float_format="%.17g")
        elif key == "metadata" and isinstance(value, dict):
            metadata.update(value)
        else:
            metadata[key] = value
    write_json(directory/"metadata.json", metadata)


def activate_generation_runtime(directory):
    """Validate before imports, reject mixed packages rather than replacing one."""
    from research_experiments.rnj.prepare_source_ac288_runtime import verify_runtime, verify_loaded_modules
    directory = Path(directory).resolve()
    manifest = verify_runtime(directory)
    loaded = any(name == "rnj_wzzt" or name.startswith("rnj_wzzt.") for name in sys.modules)
    if loaded:
        verify_loaded_modules(directory)
    else:
        sys.path.insert(0, str(directory))
    import rnj_wzzt
    if not Path(rnj_wzzt.__file__).resolve().is_relative_to(directory/"rnj_wzzt"):
        raise RuntimeError("generation imported a package outside the verified frozen runtime")
    return manifest


def load_algorithm_inputs(variant_dir, *, source_inputs_class=None):
    """Load only three observed/source folders; return scenarios, sources, meta.

    ``.../soumalas11/replicate_r/{no_source,rank2_source}`` is the variant
    directory. An exact SourceInputs class from an independently selected
    runtime can be supplied explicitly. This function does not activate one.
    """
    directory = Path(variant_dir)
    if directory.name not in VARIANTS or directory.parent.name not in {f"replicate_{r}" for r in REPLICATES}:
        raise ValueError("expected replicate_0/1/2 and a declared source variant")
    if source_inputs_class is None:
        from rnj_wzzt.estimation.unmetered_inputs import SourceInputs
        source_inputs_class = SourceInputs
    replicate = int(directory.parent.name.rsplit("_", 1)[1])
    scenarios, sources, hashes, descriptions = [], [], {}, []
    labels = None
    for i in SCENARIOS:
        folder = directory/f"scenario_{i}"
        meta = read_json(folder/"observed/metadata.json")
        scenario = {"name": f"{CASE}_{directory.parent.name}_{directory.name}_scenario_{i}",
                    "root_observation": meta["root_observation"], "scenario_settings": meta["scenario_settings"]}
        for key in ("P_terminal", "Q_terminal", "V_terminal", "drop_target"):
            frame = pd.read_csv(folder/f"observed/{key}.csv", index_col="t", float_precision="round_trip")
            frame.columns = frame.columns.astype(int)
            scenario[key] = frame
        p = scenario["P_terminal"]
        if len(p) != 96 or not p.index.is_unique or not p.columns.is_unique:
            raise ValueError("expected exactly 96 uniquely indexed observations")
        if labels is not None and not labels.equals(p.columns):
            raise ValueError("terminal order differs between scenarios")
        labels = p.columns
        for key in ("root_voltage", "P0_measured", "Q0_measured"):
            frame = pd.read_csv(folder/f"observed/{key}.csv", index_col="t", float_precision="round_trip")
            if frame.shape[1] != 1:
                raise ValueError("root/master channels must contain one series")
            scenario[key] = frame.iloc[:, 0]
        for key in ("P_terminal", "Q_terminal", "V_terminal", "drop_target", "root_voltage", "P0_measured", "Q0_measured"):
            value = scenario[key]
            if not value.index.equals(p.index) or not np.isfinite(value.to_numpy()).all():
                raise ValueError("observed channels must be finite and aligned")
            if isinstance(value, pd.DataFrame) and not value.columns.equals(labels):
                raise ValueError("observed terminal orders must agree")
        amplitudes = pd.read_csv(folder/"source/amplitudes.csv", index_col="t", usecols=["t", "P_pu", "Q_pu"], float_precision="round_trip")
        if not amplitudes.index.equals(p.index) or not np.isfinite(amplitudes.to_numpy()).all() or (amplitudes["P_pu"] < 0).any():
            raise ValueError("known source amplitudes must be finite, nonnegative-P and aligned")
        rank = np.linalg.matrix_rank(amplitudes.to_numpy())
        if rank != (0 if directory.name == "no_source" else 2):
            raise ValueError("source rank disagrees with declared source condition")
        source = source_inputs_class(amplitudes["P_pu"].copy(), amplitudes["Q_pu"].copy(), "externally_given_exact_source_time_series")
        if sources:
            np.testing.assert_array_equal(amplitudes.to_numpy(), np.column_stack((sources[0].p_pu, sources[0].q_pu)))
        scenarios.append(scenario)
        sources.append(source)
        descriptions.append({"scenario_index": i, "name": scenario["name"], "rows": 96,
                             "row_range": [96*i, 96*(i+1)], "time_labels": p.index.tolist()})
        for relative in ("observed", "source"):
            for path in sorted((folder/relative).iterdir()):
                if path.is_file():
                    hashes[path.relative_to(directory).as_posix()] = sha256(path)
    return scenarios, sources, {"case": CASE, "replicate": replicate, "variant": directory.name,
        "scenario_count": 3, "samples_per_scenario": 96, "total_training_samples": 288, "validation_samples": 0,
        "terminal_labels": labels.tolist(), "scenarios": descriptions,
        "source_information": "exact P/Q schedules known; shared non-root location unknown", "input_file_sha256": hashes}


def _paired_checks(base, current):
    for key in ("P_terminal", "Q_terminal", "root_voltage"):
        np.testing.assert_array_equal(base.observed[key], current.observed[key])
    for key in ("P_terminal_base", "Q_terminal_base", "root_voltage_true"):
        np.testing.assert_array_equal(base.truth[key], current.truth[key])
    result = {"background_and_pq_root_observations_identical": True}
    for observed, truth, relative in (("V_terminal", "V_terminal_true", .001),
                                      ("P0_measured", "P0_true", .002), ("Q0_measured", "Q0_true", .002)):
        def standardized(bundle):
            true = np.asarray(bundle.truth[truth])
            return (np.asarray(bundle.observed[observed])-true)/(relative*np.maximum(abs(true), 1e-12))
        left, right = standardized(base), standardized(current)
        np.testing.assert_allclose(left, right, atol=2e-10, rtol=2e-10)
        result[f"{observed}_standardized_noise_max_difference"] = float(np.max(abs(left-right)))
    return result


def _condition_checks(net, bundle, p_kw, q_kvar):
    obs, truth, ac = bundle.observed, bundle.truth, bundle.truth["ac"]
    if len(ac["converged"]) != 96 or not bool(ac["converged"].all()):
        raise ValueError("all 96 physical AC snapshots must converge")
    for key, expected in (("P_unmetered", p_kw/(1000*net.base_mva)), ("Q_unmetered", q_kvar/(1000*net.base_mva))):
        np.testing.assert_array_equal(truth[key], expected)
    target = obs["root_voltage"].to_numpy()[:, None]**2-obs["V_terminal"].to_numpy()**2
    np.testing.assert_array_equal(obs["drop_target"], target)
    p0, q0 = truth["P0_true"].to_numpy(), truth["Q0_true"].to_numpy()
    rp = p0-truth["P_physical"].sum(axis=1).to_numpy()-ac["branch_loss_p_pu"].sum(axis=1).to_numpy()
    rq = q0-truth["Q_physical"].sum(axis=1).to_numpy()-ac["branch_loss_q_pu"].sum(axis=1).to_numpy()
    if max(np.max(abs(rp)), np.max(abs(rq))) > 1e-7:
        raise ValueError("AC feeder conservation failed")
    source = truth["source_bus_id"]
    for channel in ("P", "Q"):
        frame = truth[f"{channel}_physical"]
        expected = truth[f"{channel}_terminal_base"].reindex(columns=frame.columns, fill_value=0.).copy()
        if source is not None:
            expected.loc[:, source] += truth[f"{channel}_unmetered"].to_numpy()
        np.testing.assert_array_equal(frame, expected)
    return {"AC_converged_samples": 96, "AC_max_iterations": int(ac["iterations"].max()),
        "p_balance_max_abs_pu": float(np.max(abs(rp))), "q_balance_max_abs_pu": float(np.max(abs(rq))),
        "minimum_voltage_pu": float(ac["V_bus_mag"].min().min()),
        "maximum_voltage_pu": float(ac["V_bus_mag"].max().max()),
        "source_pq_rank": int(np.linalg.matrix_rank(np.column_stack((p_kw, q_kvar))))}


def prepare_dataset(output_dir=DEFAULT_OUTPUT, *, runtime_dir=DEFAULT_RUNTIME, original_network_dir=DEFAULT_ORIGINAL):
    """Generate eighteen conditions in a fresh directory; retain failed rows."""
    output, runtime, original = Path(output_dir).resolve(), Path(runtime_dir).resolve(), Path(original_network_dir).resolve()
    if any(output == p or output.is_relative_to(p) or p.is_relative_to(output) for p in (runtime, original)):
        raise ValueError("new dataset must be separate from runtime and original network")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh output directory; existing results are never overwritten")
    activate_generation_runtime(runtime)
    from research_experiments.rnj.prepare_source_ac288_runtime import verify_loaded_modules
    from rnj_wzzt.scenario.simulation import _prepare_case_context
    from rnj_wzzt.scenario.unmetered_load import UnmeteredLoadSpec, simulate_unmetered_scenario
    original_hashes = inventory(original)
    net = _prepare_case_context(CASE, 96, scenario_suite="reference", root_observation="noisy",
                               root_sigma=.003, impedance_scale=1., root_meter_noise_rel=.001).net
    for name, frame in (("buses", net.buses), ("branches", net.branches)):
        pd.testing.assert_frame_equal(frame.reset_index(drop=True), pd.read_csv(original/f"{name}.csv", float_precision="round_trip"),
                                      check_dtype=False, check_exact=False, rtol=1e-14, atol=1e-14)
    old_meta = read_json(original/"metadata.json")
    if any(float(old_meta[key]) != float(getattr(net, key)) for key in ("root_bus", "base_kv", "base_mva")):
        raise ValueError("physical network base/root differs from the original")
    total_pd = float(net.buses.loc[net.buses["bus_type"] == "observed_terminal", "pd_kw"].sum())
    mean_kw = .25*total_pd
    profile_p, profile_q = source_profile(mean_kw)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    config = {"case": CASE, "variants": VARIANTS, "replicates": REPLICATES, "scenario_indices": SCENARIOS,
        "samples_per_scenario": 96, "training_samples_per_replicate_variant": 288, "validation_samples": 0,
        "scenario_suite": "reference", "impedance_scale": 1., "root_observation": "noisy", "root_sigma": .003,
        "noise": NOISE, "source_mean_p_kw": mean_kw, "source_mean_q_kvar": .35*mean_kw,
        "source_profile_P": "meanP*(1+.65*sin(2*pi*7*t/96)+.2*cos(2*pi*13*t/96))",
        "source_profile_Q": ".35*meanP*(1+.7*cos(2*pi*11*t/96+.4))", "source_curve_repeated_each_scenario_and_replicate": True,
        "source_information": "exact P/Q schedules known; physical location unknown to algorithms",
        "conditions_expected": 18, "AC_snapshots_expected": 1728, "AC_max_iter": 100, "AC_voltage_update_tolerance": 1e-10,
        "design_is_hypothesis": "no estimator or observed F1 selects this design"}
    write_json(output/"run_config.json", config)
    network = output/CASE/"network"
    network.mkdir(parents=True)
    for name, frame in (("buses", net.buses), ("branches", net.branches)):
        frame.to_csv(network/f"{name}.csv", index=False, float_format="%.17g")
    write_json(network/"metadata.json", {**old_meta, "usage": "physical simulation/evaluation truth only"})
    write_json(output/CASE/"truth/design.json", {"source_bus_id": SOURCE_BUS, "original_network_directory": original,
                                               "original_network_file_sha256": original_hashes})
    rows = []
    for replicate in REPLICATES:
        for scene in SCENARIOS:
            baseline = None
            for variant in VARIANTS:
                directory = output/CASE/f"replicate_{replicate}"/variant/f"scenario_{scene}"
                directory.mkdir(parents=True)
                row = {"case": CASE, "replicate": replicate, "variant": variant, "scenario_index": scene,
                       "status": "failed", "observed_points": 96}
                attempt = perf_counter()
                try:
                    p, q = (profile_p, profile_q) if variant == "rank2_source" else (profile_p*0., profile_q*0.)
                    source = UnmeteredLoadSpec(SOURCE_BUS, p.copy(), q.copy()) if variant == "rank2_source" else None
                    current_net, bundle = simulate_unmetered_scenario(CASE, scenario_index=scene, replicate=replicate,
                        source=source, t_count=96, scenario_suite="reference", root_observation="noisy",
                        root_sigma=.003, impedance_scale=1., **NOISE)
                    # Save physical evidence before checks so validation failures
                    # retain their raw outputs instead of only an exception.
                    _save_tables(directory/"observed", bundle.observed)
                    _save_tables(directory/"truth", {k: v for k, v in bundle.truth.items() if k != "ac"})
                    _save_tables(directory/"truth/ac", bundle.truth["ac"])
                    _save_tables(directory/"source", {"amplitudes": pd.DataFrame({"P_pu": p/(1000*net.base_mva),
                        "Q_pu": q/(1000*net.base_mva), "P_kw": p, "Q_kvar": q}),
                        "information": "externally known exact P/Q; physical location unknown", "units": "per unit; kW/kvar"})
                    pd.testing.assert_frame_equal(current_net.buses, net.buses, check_exact=True)
                    pd.testing.assert_frame_equal(current_net.branches, net.branches, check_exact=True)
                    checked = _condition_checks(net, bundle, p, q)
                    if variant == "no_source":
                        baseline = bundle
                    if baseline is None:
                        raise ValueError("paired no-source condition failed; raw source evidence retained")
                    pairing = _paired_checks(baseline, bundle)
                    write_json(directory/"diagnostics.json", {**bundle.diagnostics, **checked, "pairing": pairing})
                    row.update(status="completed", **checked)
                except Exception as exc:
                    row.update(error_type=type(exc).__name__, error=str(exc))
                    write_json(directory/"failure.json", row)
                row["elapsed_seconds"] = perf_counter()-attempt
                rows.append(row)
                pd.DataFrame(rows).to_csv(output/"summary.csv", index=False, float_format="%.17g")
                print(f"replicate_{replicate}/{variant}/scenario_{scene}: {row['status']}", flush=True)
    if original_hashes != inventory(original):
        raise RuntimeError("original network files changed during generation")
    budget = [{"replicate": r, "variant": v, "source_mean_p_kw": mean_kw if v == "rank2_source" else 0.,
        "source_peak_p_kw": float(profile_p.max()) if v == "rank2_source" else 0.,
        "source_mean_q_kvar": .35*mean_kw if v == "rank2_source" else 0.,
        "source_peak_q_kvar": float(profile_q.max()) if v == "rank2_source" else 0.,
        "source_energy_3scenarios_kwh": 72*mean_kw if v == "rank2_source" else 0.,
        "source_q_energy_3scenarios_kvarh": 72*.35*mean_kw if v == "rank2_source" else 0.,
        "source_minimum_power_factor": float(np.min(profile_p/np.sqrt(profile_p**2+profile_q**2))) if v == "rank2_source" else None,
        "source_peak_apparent_kva": float(np.max(np.sqrt(profile_p**2+profile_q**2))) if v == "rank2_source" else 0.,
        "rank": 2 if v == "rank2_source" else 0} for r in REPLICATES for v in VARIANTS]
    pd.DataFrame(budget).to_csv(output/"source_budget.csv", index=False, float_format="%.17g")
    write_json(output/"runtime_import.json", {"runtime_directory": runtime, "runtime_manifest_sha256": sha256(runtime/"runtime_manifest.json"),
        "module_origins": verify_loaded_modules(runtime), "generator_sha256": sha256(__file__),
        "dependency_versions": {name: importlib.metadata.version(name) for name in ("numpy", "pandas", "networkx", "scipy")},
        "python": sys.version, "executable": sys.executable, "platform": platform.platform()})
    completed = sum(row["status"] == "completed" for row in rows)
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "settings": config,
        "attempted": 18, "completed": completed, "failed": 18-completed,
        "all_original_network_files_unchanged": True, "estimation_performed": False,
        "elapsed_seconds": perf_counter()-started, "file_sha256": inventory(output)}
    write_json(output/"manifest.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--runtime-dir", type=Path, default=DEFAULT_RUNTIME)
    parser.add_argument("--original-network-dir", type=Path, default=DEFAULT_ORIGINAL)
    args = parser.parse_args()
    result = prepare_dataset(args.output_dir, runtime_dir=args.runtime_dir, original_network_dir=args.original_network_dir)
    print(json.dumps({k: result[k] for k in ("attempted", "completed", "failed")}))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
