"""Reviewed command-line configuration for the RNJ plus wzzT pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from rnj_wzzt.pipeline import DEFAULT_CASES, DEFAULT_OUTPUT as LEGACY_OUTPUT, run as run_pipeline
from rnj_wzzt.scenario.settings import SCENARIO_SUITES


DEFAULT_OUTPUT = Path("outputs/reference")


def _argument_parser(output: Path, default_suite: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the standalone RX75-RNJ plus L1-wzzT pipeline."
    )
    parser.add_argument("--output", type=Path, default=output)
    parser.add_argument("--cases", nargs="+", choices=DEFAULT_CASES, default=list(DEFAULT_CASES))
    parser.add_argument("--time-limit", type=float, default=1800.0,
                        help="Solver wall-time limit for each LP/MILP solve")
    parser.add_argument("--milp-solver", choices=("highs", "gurobi"), default="highs")
    parser.add_argument("--scenario-suite", choices=tuple(SCENARIO_SUITES), default=default_suite)
    parser.add_argument("--root-observation", choices=("exact", "noisy"))
    parser.add_argument("--root-sigma", type=float, help="Physical root background standard-deviation scale, p.u.")
    parser.add_argument("--root-meter-noise-rel", type=float, help="Root meter random error standard deviation, relative")
    parser.add_argument("--impedance-scale", type=float, help="Absolute multiplier of the synthetic line library")
    return parser


def _scenario_options(args) -> dict:
    return {key: getattr(args, key) for key in (
        "scenario_suite", "root_observation", "root_sigma",
        "root_meter_noise_rel", "impedance_scale",
    )}


def run(
    output_dir: str | Path = DEFAULT_OUTPUT,
    *,
    cases: tuple[str, ...] = DEFAULT_CASES,
    time_limit: float = 1800.0,
    scenario_suite: str = "reference",
    root_observation: str | None = None,
    root_meter_noise_rel: float | None = None,
    root_sigma: float | None = None,
    impedance_scale: float | None = None,
    milp_solver: str = "highs",
) -> dict:
    scenario_options = dict(
        scenario_suite=scenario_suite, root_observation=root_observation,
        root_meter_noise_rel=root_meter_noise_rel, root_sigma=root_sigma,
        impedance_scale=impedance_scale,
    )
    return run_pipeline(
        output_dir,
        cases=cases,
        scenario_count=3,
        samples_per_scenario=96,
        training_replicate=0,
        validation_replicate=1,
        pq_noise_rel=0.005,
        voltage_noise_rel=0.0002,
        tolerance_factor=0.16,
        bootstrap_replicates=100,
        block_length=4,
        confidence_threshold=0.75,
        maximum_candidate_count=2,
        selection_only=False,
        run_baseline=False,
        contract_blocks=True,
        deembedding_weight=0.5,
        time_limit=time_limit,
        milp_solver=milp_solver,
        coefficient_bound=2.0,
        **scenario_options,
    )


def main() -> None:
    args = _argument_parser(DEFAULT_OUTPUT, "reference").parse_args()
    run(args.output, cases=tuple(args.cases), time_limit=args.time_limit,
        milp_solver=args.milp_solver, **_scenario_options(args))


def advanced_main(*, runner=None) -> None:
    """Parse research options, optionally using the historical caller's runner."""
    if runner is None:
        runner = run_pipeline
    parser = _argument_parser(LEGACY_OUTPUT, "legacy")
    parser.add_argument("--scenario-count", type=int, default=3)
    parser.add_argument("--samples-per-scenario", type=int, default=96)
    parser.add_argument("--bootstrap-replicates", type=int, default=100)
    parser.add_argument("--block-length", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float, default=0.75)
    parser.add_argument("--maximum-candidate-count", type=int, default=2)
    parser.add_argument("--run-milp", action="store_true")
    parser.add_argument("--hybrid-only", action="store_true")
    parser.add_argument("--contract-blocks", action="store_true")
    parser.add_argument("--deembedding-weight", type=float, default=0.5)
    args = parser.parse_args()
    runner(
        args.output,
        cases=tuple(args.cases),
        scenario_count=args.scenario_count,
        samples_per_scenario=args.samples_per_scenario,
        bootstrap_replicates=args.bootstrap_replicates,
        block_length=args.block_length,
        confidence_threshold=args.confidence_threshold,
        maximum_candidate_count=args.maximum_candidate_count,
        selection_only=not args.run_milp,
        run_baseline=not args.hybrid_only,
        contract_blocks=args.contract_blocks,
        deembedding_weight=args.deembedding_weight,
        time_limit=args.time_limit,
        milp_solver=args.milp_solver,
        **_scenario_options(args),
    )
