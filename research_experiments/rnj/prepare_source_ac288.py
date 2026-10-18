"""Extend only pengwah18 to three paired AC days, 96 points per scenario.

Scenario 0 is copied byte-for-byte from the existing AC96 dataset.  Scenarios
1 and 2 reuse the same physical network, source site, exact source schedules,
replicate 0, meter settings and AC simulator with the existing profile seeds.
No estimator, topology search, validation split or new network is introduced.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CASE = "pengwah18"
VARIANTS = ("no_source", "constant", "intermittent", "smooth")
SCENARIOS = (0, 1, 2)
DEFAULT_BASELINE = ROOT / "outputs/source_ac96_20261018" / CASE
DEFAULT_OUTPUT = ROOT / "outputs/source_ac288_20261018"
FRAME_KEYS = ("P_terminal", "Q_terminal", "V_terminal", "drop_target")
SERIES_KEYS = ("root_voltage", "P0_measured", "Q0_measured")


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, pd.Index)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
        handle.write("\n")


def _file_inventory(directory):
    directory = Path(directory)
    return {path.relative_to(directory).as_posix(): _sha256(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


def _load_one(directory, scenario_index, source_inputs_class):
    """Read only the two estimator-visible observed/source directories."""
    observed = directory / "observed"
    description = _read_json(observed / "metadata.json")
    scenario = {"name": f"{CASE}_{directory.parent.name}_scenario_{scenario_index}",
                "root_observation": description["root_observation"],
                "scenario_settings": description["scenario_settings"]}
    for key in FRAME_KEYS:
        frame = pd.read_csv(observed / f"{key}.csv", index_col="t", float_precision="round_trip")
        frame.columns = pd.Index([int(value) for value in frame.columns])
        scenario[key] = frame
    for key in SERIES_KEYS:
        frame = pd.read_csv(observed / f"{key}.csv", index_col="t", float_precision="round_trip")
        if frame.shape[1] != 1:
            raise ValueError(f"{key} must contain exactly one observed series")
        scenario[key] = frame.iloc[:, 0]
    p = scenario["P_terminal"]
    if len(p) != 96 or not p.index.is_unique or not p.columns.is_unique:
        raise ValueError("each scenario must contain 96 uniquely indexed terminal observations")
    for key in (*FRAME_KEYS, *SERIES_KEYS):
        frame = scenario[key]
        if not frame.index.equals(p.index) or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"{key} must be finite and exactly aligned with terminal observations")
        if key in FRAME_KEYS and not frame.columns.equals(p.columns):
            raise ValueError(f"{key} terminal order must equal P_terminal")
    amplitudes = pd.read_csv(directory / "source/amplitudes.csv", index_col="t",
                             usecols=["t", "P_pu", "Q_pu"], float_precision="round_trip")
    if not amplitudes.index.equals(p.index) or not np.isfinite(amplitudes.to_numpy()).all():
        raise ValueError("source amplitudes must be finite and aligned with the 96 observations")
    if (amplitudes["P_pu"] < 0).any():
        raise ValueError("source active consumption must be nonnegative")
    source = source_inputs_class(p_pu=amplitudes["P_pu"].copy(), q_pu=amplitudes["Q_pu"].copy(),
                                 provenance="externally_given_exact_source_time_series")
    return scenario, source


def load_algorithm_inputs(variant_dir, *, source_inputs_class=None):
    """Return ``(scenarios, sources, metadata)`` with three 96-row blocks.

    ``variant_dir`` is ``.../pengwah18/{no_source,constant,intermittent,smooth}``.
    Only observed channels and externally known source P/Q are opened.  The
    optional source class is supplied by a caller using an independently
    imported/frozen runtime; exact runtime ``isinstance`` checks then work.
    The default class is imported lazily from the currently selected canonical
    runtime, never bound while importing this data-loader module.
    """
    variant_dir = Path(variant_dir)
    if variant_dir.name not in VARIANTS:
        raise ValueError("variant_dir must name one of the four declared source variants")
    if source_inputs_class is None:
        from rnj_wzzt.estimation.unmetered_inputs import SourceInputs
        source_inputs_class = SourceInputs
    scenarios, sources, inventory, descriptions = [], [], {}, []
    for index in SCENARIOS:
        directory = variant_dir / f"scenario_{index}"
        scenario, source = _load_one(directory, index, source_inputs_class)
        scenarios.append(scenario)
        sources.append(source)
        for folder in ("observed", "source"):
            for path in sorted((directory / folder).iterdir()):
                if path.is_file():
                    inventory[path.relative_to(variant_dir).as_posix()] = _sha256(path)
        descriptions.append({"scenario_index": index, "name": scenario["name"], "rows": 96,
                             "local_time_labels": scenario["P_terminal"].index.tolist(),
                             "concatenated_row_range": [96 * index, 96 * (index + 1)],
                             "profile_scenario": scenario["scenario_settings"].get("profile_scenario"),
                             "source_provenance": source.provenance})
    reference = scenarios[0]["P_terminal"].columns
    if any(not scenario["P_terminal"].columns.equals(reference) for scenario in scenarios):
        raise ValueError("all three scenarios must share the same terminal order")
    for source in sources[1:]:
        if not (np.array_equal(source.p_pu.to_numpy(), sources[0].p_pu.to_numpy())
                and np.array_equal(source.q_pu.to_numpy(), sources[0].q_pu.to_numpy())):
            raise ValueError("the same known source power curve must be supplied in all three scenarios")
    metadata = {"case": CASE, "variant": variant_dir.name, "scenario_count": 3,
                "samples_per_scenario": 96, "total_training_samples": 288, "validation_samples": 0,
                "terminal_labels": reference.tolist(), "scenario_order": list(SCENARIOS),
                "concatenation_order": "scenario_0,scenario_1,scenario_2; preserve each local observation order",
                "source_information": "exact P/Q schedules known; non-root location unknown",
                "scenarios": descriptions, "input_file_sha256": inventory}
    return scenarios, sources, metadata


def _settings_from_baseline(baseline):
    if _read_json(baseline / "network/metadata.json").get("metadata", {}).get("case_name") != "pengwah_mixed_lv_18":
        raise ValueError("baseline must be the existing pengwah18 physical case")
    settings = _read_json(baseline / "no_source/observed/metadata.json")["scenario_settings"]
    required = {"scenario_suite": "reference", "root_observation": "noisy", "replicate": 0,
                "scenario_index": 0, "physical_sample_count": 96, "observed_sample_count": 96,
                "impedance_scale": 1.0, "pq_noise_rel": .005, "v_noise_rel": .0002,
                "root_meter_noise_rel": .0002, "master_p_noise_rel": .002, "master_q_noise_rel": .002}
    for variant in VARIANTS:
        current = _read_json(baseline / variant / "observed/metadata.json")["scenario_settings"]
        for key, value in required.items():
            if current.get(key) != value:
                raise ValueError(f"existing {variant} baseline has incompatible {key}")
    return settings


def _assert_same_network(net, original):
    metadata = _read_json(original / "metadata.json")
    if int(net.root_bus) != metadata["root_bus"] or net.base_kv != metadata["base_kv"] or net.base_mva != metadata["base_mva"]:
        raise ValueError("simulator changed the physical network base or root")
    for name, current in (("buses", net.buses), ("branches", net.branches)):
        saved = pd.read_csv(original / f"{name}.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(current.reset_index(drop=True), saved, check_dtype=False,
                                      check_exact=False, rtol=1e-14, atol=1e-14)


def _saved_ac_summary(directory):
    diagnostics = _read_json(directory / "diagnostics.json")
    def series(name):
        return pd.read_csv(directory / f"truth/ac/{name}.csv", index_col="t", float_precision="round_trip").iloc[:, 0]
    converged = series("converged")
    all_converged = bool(converged.map(lambda v: str(v).lower() in {"true", "1"}).all())
    if len(converged) != 96 or not all_converged:
        raise ValueError("all 96 saved physical AC snapshots must converge")
    if max(diagnostics["p_balance_max_abs_pu"], diagnostics["q_balance_max_abs_pu"]) > 1e-7:
        raise ValueError("saved AC feeder power balance exceeds tolerance")
    return {"all_ac_samples_converged": all_converged,
            "ac_max_iterations": int(series("iterations").max()),
            "ac_max_voltage_update": float(series("max_voltage_error").max()),
            "p_balance_max_abs_pu": diagnostics["p_balance_max_abs_pu"],
            "q_balance_max_abs_pu": diagnostics["q_balance_max_abs_pu"]}


def prepare_dataset(output_dir=DEFAULT_OUTPUT, *, baseline_case_dir=DEFAULT_BASELINE):
    """Create the single-network 12-scenario dataset in a fresh directory."""
    # Generation-only helpers are lazy: importing the loader cannot change a
    # caller's selected RNJ/WZZT runtime or bind the wrong SourceInputs class.
    from research_experiments.rnj import prepare_source_ac96 as original
    from rnj_wzzt.scenario.unmetered_load import UnmeteredLoadSpec, simulate_unmetered_scenario

    output, baseline = Path(output_dir).resolve(), Path(baseline_case_dir).resolve()
    if output == baseline or output.is_relative_to(baseline) or baseline.is_relative_to(output):
        raise ValueError("new output must be separate from the existing AC96 case directory")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh AC288 output directory; existing data are never overwritten")
    settings = _settings_from_baseline(baseline)
    before = _file_inventory(baseline)
    output.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    config = {"case": CASE, "variants": list(VARIANTS), "scenario_indices": list(SCENARIOS),
              "replicate": 0, "scenario_suite": "reference", "samples_per_scenario": 96,
              "training_samples_per_variant": 288, "validation_samples": 0,
              "scenario_0_policy": "copy every original variant file byte-for-byte",
              "scenario_1_2_policy": "same physical network and source; existing scenario profiles/seeds 1 and 2",
              "noise": {key: settings[key] for key in ("pq_noise_rel", "v_noise_rel", "root_meter_noise_rel",
                                                         "master_p_noise_rel", "master_q_noise_rel")},
              "ac_max_iterations": 100, "ac_voltage_update_tolerance_pu": 1e-10,
              "source_information": "externally known exact P/Q; location unknown to estimator",
              "sample_budget": {"variants": 4, "scenarios_per_variant": 3, "points_per_scenario": 96,
                                "all_training_points": 1152, "new_AC_points": 768, "reused_AC_points": 384},
              "baseline_case_directory": str(baseline)}
    _write_json(output / "run_config.json", config)
    environment = original.code_manifest()
    environment["source_sha256"][Path(__file__).relative_to(ROOT).as_posix()] = _sha256(__file__)
    environment["data_generation_only"] = True
    _write_json(output / "code_environment.json", environment)
    case_dir = output / CASE
    shutil.copytree(baseline / "network", case_dir / "network")
    amplitudes, sites = {}, {}
    for variant in VARIANTS:
        amplitudes[variant] = pd.read_csv(baseline / variant / "source/amplitudes.csv", index_col="t",
                                         float_precision="round_trip")
        sites[variant] = _read_json(baseline / variant / "truth/metadata.json").get("source_bus_id")
    active_sites = {sites[v] for v in VARIANTS if v != "no_source"}
    if len(active_sites) != 1 or None in active_sites or sites["no_source"] is not None:
        raise ValueError("existing four variants must use one common non-root source location")
    rows, reuse_hashes = [], {}
    for scenario_index in SCENARIOS:
        baseline_bundle = None
        for variant in VARIANTS:
            directory = case_dir / variant / f"scenario_{scenario_index}"
            row = {"case": CASE, "variant": variant, "scenario_index": scenario_index,
                   "replicate": 0, "observed_points": 96, "status": "failed",
                   "origin": "copied_AC96" if scenario_index == 0 else "new_AC_simulation"}
            attempt_start = perf_counter()
            try:
                if scenario_index == 0:
                    shutil.copytree(baseline / variant, directory)
                    copied = _file_inventory(directory)
                    expected = _file_inventory(baseline / variant)
                    if copied != expected:
                        raise RuntimeError("copied scenario_0 differs from the original bytes")
                    reuse_hashes[variant] = expected
                else:
                    directory.mkdir(parents=True)
                    source_frame = amplitudes[variant]
                    source = None if variant == "no_source" else UnmeteredLoadSpec(
                        sites[variant], source_frame["P_kw"].copy(), source_frame["Q_kvar"].copy(),
                    )
                    net, bundle = simulate_unmetered_scenario(
                        CASE, scenario_index=scenario_index, replicate=0, source=source, t_count=96,
                        scenario_suite="reference", root_observation="noisy",
                        pq_noise_rel=settings["pq_noise_rel"], v_noise_rel=settings["v_noise_rel"],
                        root_meter_noise_rel=settings["root_meter_noise_rel"], root_sigma=settings["root_sigma"],
                        impedance_scale=settings["impedance_scale"], master_p_noise_rel=settings["master_p_noise_rel"],
                        master_q_noise_rel=settings["master_q_noise_rel"],
                    )
                    _assert_same_network(net, baseline / "network")
                    np.testing.assert_array_equal(bundle.truth["P_unmetered"].to_numpy(),
                                                  source_frame["P_pu"].to_numpy())
                    np.testing.assert_array_equal(bundle.truth["Q_unmetered"].to_numpy(),
                                                  source_frame["Q_pu"].to_numpy())
                    if baseline_bundle is None:
                        baseline_bundle = bundle
                    pairing = original._check_pairing(baseline_bundle, bundle, settings["v_noise_rel"])
                    original._save_tables(directory / "observed", bundle.observed)
                    shutil.copytree(baseline / variant / "source", directory / "source")
                    truth = {key: value for key, value in bundle.truth.items() if key != "ac"}
                    truth["source_location_reused_from_AC96"] = True
                    original._save_tables(directory / "truth", truth)
                    original._save_tables(directory / "truth/ac", bundle.truth["ac"])
                    source_design = original.matrix_diagnostics(source_frame[["P_pu", "Q_pu"]].to_numpy())
                    observed = bundle.observed
                    design = np.column_stack((observed["P_terminal"].to_numpy(), observed["Q_terminal"].to_numpy(),
                                              source_frame[["P_pu", "Q_pu"]].to_numpy()))
                    _write_json(directory / "diagnostics.json", {**bundle.diagnostics, "pairing": pairing,
                        "source_pq_design": source_design, "observed_pq_source_design": original.matrix_diagnostics(design)})
                    _write_json(directory / "config.json", {**config, "variant": variant,
                        "scenario_index": scenario_index, "scenario_settings": observed["scenario_settings"]})
                row.update(_saved_ac_summary(directory))
                row.update(status="completed", source_p_energy_kwh=float(amplitudes[variant]["P_kw"].sum() * .25),
                           source_q_energy_kvarh=float(amplitudes[variant]["Q_kvar"].sum() * .25),
                           source_peak_kw=float(amplitudes[variant]["P_kw"].max()),
                           source_peak_kvar=float(amplitudes[variant]["Q_kvar"].max()),
                           source_pq_rank=int(np.linalg.matrix_rank(amplitudes[variant][["P_pu", "Q_pu"]].to_numpy())))
            except Exception as exc:
                row.update(error_type=type(exc).__name__, error=str(exc))
                _write_json(directory / "failure_ac288.json", row)
            row["elapsed_seconds"] = perf_counter() - attempt_start
            rows.append(row)
            pd.DataFrame(rows).to_csv(output / "summary.csv", index=False, float_format="%.17g")
            print(f"{CASE}/{variant}/scenario_{scenario_index}: {row['status']}", flush=True)
    source_budget = []
    for variant, frame in amplitudes.items():
        source_budget.append({"variant": variant, "scenarios": 3, "points_per_scenario": 96,
            "source_mean_p_kw": float(frame["P_kw"].mean()), "source_peak_p_kw": float(frame["P_kw"].max()),
            "source_energy_per_scenario_kwh": float(frame["P_kw"].sum() * .25),
            "source_energy_total_3scenarios_kwh": float(frame["P_kw"].sum() * .25 * 3),
            "source_q_energy_per_scenario_kvarh": float(frame["Q_kvar"].sum() * .25),
            "source_q_energy_total_3scenarios_kvarh": float(frame["Q_kvar"].sum() * .25 * 3),
            "source_pq_rank": int(np.linalg.matrix_rank(frame[["P_pu", "Q_pu"]].to_numpy()))})
    pd.DataFrame(source_budget).to_csv(output / "source_budget.csv", index=False, float_format="%.17g")
    _write_json(output / "scenario_0_reuse.json", {"all_files_byte_identical": len(reuse_hashes) == 4,
        "baseline_case_directory": str(baseline), "file_sha256_by_variant": reuse_hashes})
    original_unchanged = before == _file_inventory(baseline)
    _write_json(output / "original_AC96_integrity.json", {"unchanged": original_unchanged,
        "baseline_case_directory": str(baseline), "original_file_sha256": before})
    if not original_unchanged:
        raise RuntimeError("original AC96 data changed during preparation")
    completed = sum(row["status"] == "completed" for row in rows)
    manifest = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "settings": config, "attempted": 12, "completed": completed, "failed": 12 - completed,
        "scenario_0_reused": sum(row["scenario_index"] == 0 and row["status"] == "completed" for row in rows),
        "newly_simulated": sum(row["scenario_index"] > 0 and row["status"] == "completed" for row in rows),
        "all_original_AC96_files_unchanged": original_unchanged,
        "algorithm_fitting_performed": False, "elapsed_seconds": perf_counter() - started,
        "file_sha256": _file_inventory(output)}
    _write_json(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-case-dir", type=Path, default=DEFAULT_BASELINE)
    args = parser.parse_args()
    result = prepare_dataset(args.output_dir, baseline_case_dir=args.baseline_case_dir)
    print(json.dumps({k: result[k] for k in ("attempted", "completed", "failed", "scenario_0_reused", "newly_simulated")}))
    return 0 if result["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
