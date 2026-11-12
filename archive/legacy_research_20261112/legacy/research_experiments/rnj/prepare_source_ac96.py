"""Prepare paired 96-point AC days with known extra-source P/Q, unknown site.

This script generates data only. It does not fit NJ, RNJ, or WZZT. Physical
networks, source locations, and AC truth are saved separately from observations
and externally supplied source amplitudes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
from time import perf_counter
from typing import Any

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
CORE = ROOT / "rnj_wzzt_core"
sys.path.insert(0, str(CORE))

import networkx as nx
import numpy as np
import pandas as pd

from rnj_wzzt.scenario.simulation import _prepare_case_context
from rnj_wzzt.scenario.unmetered_load import (
    ScenarioBundle,
    UnmeteredLoadSpec,
    simulate_unmetered_scenario,
)
from rnj_wzzt.estimation.unmetered_inputs import SourceInputs

CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
VARIANTS = ("no_source", "constant", "intermittent", "smooth")
SITE_RULE = (
    "Among hidden_internal branching buses with at least two downstream children "
    "and root depth >= 2, choose maximum downstream terminal nameplate Pd; "
    "break ties by ascending bus ID. If empty, use hidden root children."
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (np.ndarray, pd.Index)):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2,
                  allow_nan=False, default=_json_default)
        handle.write("\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(ROOT), *args], capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def code_manifest() -> dict:
    paths = sorted((CORE / "rnj_wzzt").rglob("*.py")) + [Path(__file__).resolve()]
    versions = {}
    for package in ("numpy", "pandas", "networkx", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status_porcelain": _git("status", "--porcelain"),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "dependency_versions": versions,
        "source_sha256": {path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths},
    }


def choose_source_bus(net) -> tuple[int, dict]:
    """Choose a simulation site before any voltage or inference is evaluated."""
    graph = net.to_networkx_graph()
    root = int(net.root_bus)
    tree = nx.bfs_tree(graph, root)
    depth = nx.single_source_shortest_path_length(tree, root)
    hidden = sorted(int(bus) for bus in net.hidden_buses())
    terminal_pd = {
        int(row.bus_id): float(row.pd_kw)
        for row in net.buses.loc[net.buses["bus_type"].eq("observed_terminal")].itertuples()
    }
    candidates = [bus for bus in hidden if depth[bus] >= 2 and tree.out_degree(bus) >= 2]
    fallback = not candidates
    if fallback:
        candidates = [bus for bus in hidden if depth[bus] == 1]
    if not candidates:
        raise ValueError("No hidden branching bus or hidden root-child source site")
    downstream_pd = {
        bus: sum(terminal_pd.get(node, 0.0) for node in nx.descendants(tree, bus))
        for bus in candidates
    }
    chosen = min(candidates, key=lambda bus: (-downstream_pd[bus], bus))
    return chosen, {
        "source_site_rule": SITE_RULE,
        "source_site_fallback_used": fallback,
        "source_bus_depth": int(depth[chosen]),
        "source_bus_downstream_terminal_pd_kw": downstream_pd[chosen],
        "source_site_candidates": [
            {"bus_id": bus, "depth": int(depth[bus]),
             "downstream_terminal_pd_kw": downstream_pd[bus]}
            for bus in candidates
        ],
    }


def source_profile(variant: str, mean_kw: float) -> tuple[pd.Series, pd.Series]:
    """Three consumption profiles have identical daily P and Q energies."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown source variant: {variant}")
    index = pd.RangeIndex(96, name="t")
    theta = 2.0 * np.pi * np.arange(96) / 96.0
    p = np.zeros(96)
    if variant == "constant":
        p[:] = mean_kw
    elif variant == "intermittent":
        p[32:48] = 3.0 * mean_kw
        p[64:80] = 3.0 * mean_kw
    elif variant == "smooth":
        p = mean_kw * (1.0 - 0.6 * np.cos(theta))
    q = p * (0.4 + 0.15 * np.sin(theta)) if variant == "smooth" else 0.4 * p
    if variant != "no_source":
        np.testing.assert_allclose(p.sum() * 0.25, mean_kw * 24.0, rtol=1e-13)
        np.testing.assert_allclose(q.sum() * 0.25, 0.4 * mean_kw * 24.0, rtol=1e-13)
    return pd.Series(p, index=index, name="P_kw"), pd.Series(q, index=index, name="Q_kvar")


def matrix_diagnostics(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    singular_values = np.linalg.svd(values, compute_uv=False)
    rank = int(np.linalg.matrix_rank(values))
    full_column_rank = rank == values.shape[1]
    condition = float(singular_values[0] / singular_values[-1]) if full_column_rank else None
    return {
        "shape": list(values.shape), "rank": rank,
        "full_column_rank": full_column_rank,
        "condition_number": condition,
        "condition_number_note": "2-norm, raw per-unit columns; null means rank deficient",
        "singular_values": singular_values.tolist(),
    }


def _save_tables(directory: Path, values: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for key, value in values.items():
        if isinstance(value, (pd.DataFrame, pd.Series)):
            path = directory / f"{key}.csv"
            with path.open("x", encoding="utf-8", newline="") as handle:
                value.to_csv(handle, index_label="t", float_format="%.17g")
        elif key == "metadata" and isinstance(value, dict):
            metadata.update(value)
        else:
            metadata[key] = value
    write_json(directory / "metadata.json", metadata)


def load_algorithm_inputs(variant_dir: str | Path) -> tuple[list[dict], list[SourceInputs]]:
    """Load only observed channels and externally known source amplitudes.

    This loader has no network/truth/diagnostics/manifest dependency. Passing a
    variant directory is required, e.g. ``.../paper15/smooth``. All-zero source
    amplitudes remain explicit SourceInputs; the existing fitter's preparation
    layer can identify that case with PreparedSourceInputs.all_zero.
    """
    variant_dir = Path(variant_dir)
    observed = variant_dir / "observed"
    scenario = {"name": f"{variant_dir.parent.name}_{variant_dir.name}"}
    metadata = json.loads((observed / "metadata.json").read_text(encoding="utf-8"))
    scenario["root_observation"] = metadata["root_observation"]
    scenario["scenario_settings"] = metadata["scenario_settings"]
    frame_keys = ("P_terminal", "Q_terminal", "V_terminal", "drop_target")
    series_keys = ("root_voltage", "P0_measured", "Q0_measured")
    for key in frame_keys:
        table = pd.read_csv(observed / f"{key}.csv", index_col="t", float_precision="round_trip")
        table.columns = pd.Index([int(column) for column in table.columns])
        scenario[key] = table
    for key in series_keys:
        table = pd.read_csv(observed / f"{key}.csv", index_col="t", float_precision="round_trip")
        if table.shape[1] != 1:
            raise ValueError(f"Expected one observed series in {key}.csv")
        scenario[key] = table.iloc[:, 0]
    amplitudes = pd.read_csv(variant_dir / "source" / "amplitudes.csv", index_col="t",
                             usecols=["t", "P_pu", "Q_pu"], float_precision="round_trip")
    source = SourceInputs(
        p_pu=amplitudes["P_pu"], q_pu=amplitudes["Q_pu"],
        provenance="externally_given_exact_source_time_series",
    )
    return [scenario], [source]


def _save_network(directory: Path, net) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in (("buses", net.buses), ("branches", net.branches)):
        with (directory / f"{name}.csv").open("x", encoding="utf-8", newline="") as handle:
            table.to_csv(handle, index=False, float_format="%.17g")
    write_json(directory / "metadata.json", {
        "root_bus": int(net.root_bus), "base_kv": float(net.base_kv),
        "base_mva": float(net.base_mva), "metadata": net.metadata,
        "usage": "simulation truth and evaluation only; never pass network to topology fitter",
    })


def _check_pairing(baseline: ScenarioBundle, current: ScenarioBundle, v_noise_rel: float) -> dict:
    """Source interventions keep the same background and measurement draws."""
    for key in ("P_terminal", "Q_terminal", "root_voltage"):
        if isinstance(current.observed[key], pd.DataFrame):
            pd.testing.assert_frame_equal(current.observed[key], baseline.observed[key], check_exact=True)
        else:
            pd.testing.assert_series_equal(current.observed[key], baseline.observed[key], check_exact=True)
    for key in ("P_terminal_base", "Q_terminal_base", "root_voltage_true"):
        np.testing.assert_array_equal(current.truth[key], baseline.truth[key])
    if v_noise_rel > 0.0:
        def standardized(bundle):
            voltage = bundle.truth["V_terminal_true"].to_numpy()
            observed = bundle.observed["V_terminal"].to_numpy()
            return (observed - voltage) / (v_noise_rel * np.maximum(np.abs(voltage), 1e-12))
        max_difference = float(np.max(np.abs(standardized(current) - standardized(baseline))))
        np.testing.assert_allclose(standardized(current), standardized(baseline), atol=2e-11, rtol=2e-11)
    else:
        max_difference = 0.0
    return {
        "terminal_pq_and_root_observations_identical": True,
        "background_pq_and_root_truth_identical": True,
        "terminal_voltage_standardized_noise_max_pair_difference": max_difference,
    }


def _dataset_readme() -> str:
    return """# Paired 96-point AC source datasets

Data generation only: no NJ/RNJ/WZZT fitting or topology-recovery results.
The four networks are project synthetic low-voltage benchmark networks.
Each variant contains one physical day, 96 samples at 15-minute intervals.
No separate training/validation/test days are generated.

Estimator-visible files are `<case>/<variant>/observed/` and
`<case>/<variant>/source/amplitudes.csv` (P_pu/Q_pu columns).
`network/`, `truth/`, diagnostics and manifests are simulation/evaluation data.
Source amplitude time series are given externally and exactly; the source bus
is unknown to the estimator and stored only below `truth/`.
No-source amplitudes are explicitly zero. Source powers are extra consumption,
not generation, using the project's nonnegative-P Source convention.

Source mean P is 5% of the network's total terminal nameplate Pd.
Constant: P=mean, Q=.4P.
Intermittent: P=3mean on zero-based t=[32,48),[64,80), zero otherwise; Q=.4P.
Smooth: theta=2pi*t/96; P=mean*(1-.6cos(theta)); Q=P*(.4+.15sin(theta)).
Three source variants share daily P and Q energies. Constant/intermittent
source P/Q design has rank 1; smooth has rank 2. This variation alone does not
establish identifiability of source location or network topology.

All power CSVs except source P_kw/Q_kvar use per unit on network base_mva.
V_bus_mag is per unit; V_bus_angle_deg uses degrees; I_branch_mag is per unit.
Branch from/to P and Q are both signed positive away from the root; losses
equal sending power minus receiving power. Network r_ohm/x_ohm use ohms,
base_kv is line-to-line kV, base_mva is three-phase MVA.
observed P/Q exclude the extra physical load. Root/master meters and AC branch
flows reflect the extra load and losses. All days share background P/Q, root
voltage, P/Q meter draws, root-meter draws, and standardized V meter draws.
Relative V noise magnitudes scale with each variant's true voltage.

`summary.csv` contains all attempted case variants, including failures.
`manifest.json` records settings, hashes, environment, and actual completion.
Tables retain 17 significant digits. AC tolerance is 1e-10 on voltage update,
at most 100 iterations; convergence and conservation must also be audited
from saved physical data. Data/code hashes describe the actual run rather
than only its Git commit; preexisting dirty files are recorded.

Load algorithm inputs without opening network/truth files:

```python
from research_experiments.rnj.prepare_source_ac96 import load_algorithm_inputs
scenarios, source_inputs = load_algorithm_inputs("outputs/source_ac96_20261018/paper15/smooth")
```
"""


def prepare_dataset(
    output_dir: str | Path,
    *,
    cases: tuple[str, ...] = CASES,
    replicate: int = 0,
    pq_noise_rel: float = 0.005,
    v_noise_rel: float = 0.0002,
    root_meter_noise_rel: float = 0.0002,
    master_noise_rel: float = 0.002,
) -> dict:
    """Save paired AC data, numerical diagnostics, and reproducibility records."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "manifest.json").exists():
        raise FileExistsError(f"Dataset manifest already exists: {output_dir}")
    if not cases or len(set(cases)) != len(cases) or any(case not in CASES for case in cases):
        raise ValueError("cases must be unique case-bank keys")
    if any((output_dir / case).exists() for case in cases):
        raise FileExistsError("A requested case directory already exists; use a fresh output directory")
    start = perf_counter()
    settings = {
        "cases": list(cases), "variants": list(VARIANTS),
        "scenario_index": 0, "replicate": replicate, "t_count": 96,
        "scenario_suite": "reference", "root_observation": "noisy",
        "pq_noise_rel": pq_noise_rel, "v_noise_rel": v_noise_rel,
        "root_meter_noise_rel": root_meter_noise_rel,
        "master_p_noise_rel": master_noise_rel, "master_q_noise_rel": master_noise_rel,
        "ac_solver": "radial_backward_forward_sweep",
        "ac_voltage_update_tolerance_pu": 1e-10, "ac_max_iterations": 100,
        "source_mean_nameplate_pd_fraction": 0.05,
        "source_site_rule": SITE_RULE,
        "source_amplitudes": "externally_given_exact; nonnegative consumption; location unknown",
        "sample_budget": "one 96-point physical day per variant; no fit or selection",
    }
    write_json(output_dir / "run_config.json", settings)
    write_json(output_dir / "code_environment.json", code_manifest())
    with (output_dir / "README.md").open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(_dataset_readme())
    rows = []
    all_files = {}
    for case in cases:
        context = _prepare_case_context(case, 96, scenario_suite="reference",
                                        root_observation="noisy",
                                        root_meter_noise_rel=root_meter_noise_rel)
        net = context.net
        case_dir = output_dir / case
        _save_network(case_dir / "network", net)
        bus_id, site_truth = choose_source_bus(net)
        total_pd = float(net.buses.loc[net.buses["bus_type"].eq("observed_terminal"), "pd_kw"].sum())
        mean_kw = 0.05 * total_pd
        baseline = None
        for variant in VARIANTS:
            variant_start = perf_counter()
            destination = case_dir / variant
            destination.mkdir(parents=True)
            row = {"case_key": case, "variant": variant, "status": "failed", "observed_points": 96,
                   "terminal_count": len(context.terminals), "nameplate_terminal_pd_kw": total_pd,
                   "source_design_mean_kw": 0.0 if variant == "no_source" else mean_kw}
            config = {**settings, "case_key": case, "variant": variant,
                      "nameplate_terminal_pd_kw": total_pd,
                      "source_design_mean_kw": row["source_design_mean_kw"]}
            write_json(destination / "config.json", config)
            try:
                p_kw, q_kvar = source_profile(variant, mean_kw)
                source = None if variant == "no_source" else UnmeteredLoadSpec(bus_id, p_kw, q_kvar)
                _, bundle = simulate_unmetered_scenario(
                    case, scenario_index=0, replicate=replicate, source=source,
                    t_count=96, scenario_suite="reference", pq_noise_rel=pq_noise_rel,
                    v_noise_rel=v_noise_rel, root_observation="noisy",
                    root_meter_noise_rel=root_meter_noise_rel,
                    master_p_noise_rel=master_noise_rel, master_q_noise_rel=master_noise_rel,
                )
                if baseline is None:
                    baseline = bundle
                pairing = _check_pairing(baseline, bundle, v_noise_rel)
                _save_tables(destination / "observed", bundle.observed)
                amplitudes = pd.DataFrame({
                    "P_pu": p_kw / (1000.0 * net.base_mva),
                    "Q_pu": q_kvar / (1000.0 * net.base_mva),
                    "P_kw": p_kw, "Q_kvar": q_kvar,
                })
                source_dir = destination / "source"
                source_dir.mkdir()
                with (source_dir / "amplitudes.csv").open("x", encoding="utf-8", newline="") as handle:
                    amplitudes.to_csv(handle, index_label="t", float_format="%.17g")
                write_json(source_dir / "metadata.json", {
                    "provenance": "externally_given_exact_source_time_series",
                    "location_available_to_estimator": False,
                    "power_convention": "positive means extra consumption",
                    "base_mva": float(net.base_mva),
                })
                truth = {key: value for key, value in bundle.truth.items() if key != "ac"}
                truth.update(site_truth if variant != "no_source" else {})
                _save_tables(destination / "truth", truth)
                _save_tables(destination / "truth" / "ac", bundle.truth["ac"])
                source_design = matrix_diagnostics(amplitudes[["P_pu", "Q_pu"]].to_numpy())
                pq = np.hstack((bundle.observed["P_terminal"].to_numpy(),
                                bundle.observed["Q_terminal"].to_numpy()))
                design = matrix_diagnostics(np.hstack((pq, amplitudes[["P_pu", "Q_pu"]].to_numpy())))
                diagnostics = {**bundle.diagnostics, "pairing": pairing,
                               "source_pq_design": source_design,
                               "observed_pq_source_design": design}
                write_json(destination / "diagnostics.json", diagnostics)
                row.update(
                    status="completed", source_p_energy_kwh=float(p_kw.sum() * 0.25),
                    source_q_energy_kvarh=float(q_kvar.sum() * 0.25),
                    source_peak_kw=float(p_kw.max()), source_peak_kvar=float(q_kvar.max()),
                    source_pq_rank=source_design["rank"],
                    design_rank=design["rank"], design_column_count=design["shape"][1],
                    design_condition_number=design["condition_number"],
                    all_ac_samples_converged=bool(bundle.truth["ac"]["converged"].all()),
                    ac_max_iterations=int(bundle.truth["ac"]["iterations"].max()),
                    ac_max_voltage_update=float(bundle.truth["ac"]["max_voltage_error"].max()),
                    voltage_true_min_pu=float(bundle.truth["ac"]["V_bus_mag"].min().min()),
                    voltage_true_max_pu=float(bundle.truth["ac"]["V_bus_mag"].max().max()),
                    p_balance_max_abs_pu=bundle.diagnostics["p_balance_max_abs_pu"],
                    q_balance_max_abs_pu=bundle.diagnostics["q_balance_max_abs_pu"],
                    paired_pq_root=True,
                    standardized_v_noise_pair_max=pairing["terminal_voltage_standardized_noise_max_pair_difference"],
                )
            except Exception as error:
                row.update(error_type=type(error).__name__, error=str(error))
                write_json(destination / "failure.json", row)
            row["elapsed_seconds"] = perf_counter() - variant_start
            rows.append(row)
            print(f"{case}/{variant}: {row['status']} ({row['elapsed_seconds']:.3f}s)", flush=True)
            pd.DataFrame(rows).to_csv(output_dir / "summary.csv", index=False, float_format="%.17g")
            for path in sorted(destination.rglob("*")):
                if path.is_file():
                    all_files[path.relative_to(output_dir).as_posix()] = _sha256(path)
        for path in sorted((case_dir / "network").rglob("*")):
            if path.is_file():
                all_files[path.relative_to(output_dir).as_posix()] = _sha256(path)
    for name in ("run_config.json", "code_environment.json", "README.md", "summary.csv"):
        all_files[name] = _sha256(output_dir / name)
    completed = sum(row["status"] == "completed" for row in rows)
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(output_dir), "settings": settings,
        "attempted": len(rows), "completed": completed, "failed": len(rows) - completed,
        "elapsed_seconds": perf_counter() - start,
        "file_sha256": all_files,
        "algorithm_fitting_performed": False,
    }
    write_json(output_dir / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "source_ac96_20261018")
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--replicate", type=int, default=0)
    parser.add_argument("--pq-noise-rel", type=float, default=0.005)
    parser.add_argument("--v-noise-rel", type=float, default=0.0002)
    parser.add_argument("--root-meter-noise-rel", type=float, default=0.0002)
    parser.add_argument("--master-noise-rel", type=float, default=0.002)
    args = parser.parse_args()
    manifest = prepare_dataset(
        args.output_dir, cases=tuple(args.cases), replicate=args.replicate,
        pq_noise_rel=args.pq_noise_rel, v_noise_rel=args.v_noise_rel,
        root_meter_noise_rel=args.root_meter_noise_rel,
        master_noise_rel=args.master_noise_rel,
    )
    print(json.dumps({key: manifest[key] for key in ("output_dir", "attempted", "completed", "failed")},
                     ensure_ascii=False), flush=True)
    return 1 if manifest["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
