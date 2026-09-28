"""Parse command-line options and delegate computation to :mod:`pipeline`.

Omitted options are not forwarded: ``pipeline.run`` owns every computation
default. The historical Python and advanced CLI entrypoints remain delegates.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from rnj_wzzt.pipeline import DEFAULT_CASES, DEFAULT_OUTPUT, run as run_pipeline
from rnj_wzzt.scenario.settings import ROOT_OBSERVATIONS, SCENARIO_SUITES


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the standalone RX75-RNJ plus L1-wzzT pipeline.",
        argument_default=argparse.SUPPRESS,
        epilog="Omitted options use pipeline.run defaults: reference scenarios and RNJ plus MILP.",
    )
    parser.add_argument("--output", dest="output_dir", type=Path)
    parser.add_argument("--cases", nargs="+", choices=DEFAULT_CASES)

    scenarios = parser.add_argument_group("scenario generation")
    scenarios.add_argument("--scenario-suite", choices=tuple(SCENARIO_SUITES))
    scenarios.add_argument("--scenario-count", type=int)
    scenarios.add_argument("--samples-per-scenario", type=int)
    scenarios.add_argument("--training-replicate", type=int)
    scenarios.add_argument("--validation-replicate", type=int)
    scenarios.add_argument("--pq-noise-rel", type=float)
    scenarios.add_argument("--voltage-noise-rel", type=float)
    scenarios.add_argument("--root-observation", choices=ROOT_OBSERVATIONS)
    scenarios.add_argument("--root-sigma", type=float,
                           help="Physical root background standard-deviation scale, p.u.")
    scenarios.add_argument("--root-meter-noise-rel", type=float,
                           help="Root meter random error standard deviation, relative")
    scenarios.add_argument("--impedance-scale", type=float,
                           help="Absolute multiplier of the synthetic line library")

    selection = parser.add_argument_group("RNJ boundary selection")
    selection.add_argument("--tolerance-factor", type=float)
    selection.add_argument("--bootstrap-replicates", type=int)
    selection.add_argument("--block-length", type=int)
    selection.add_argument("--confidence-threshold", type=float)
    selection.add_argument("--maximum-candidate-count", type=int)

    solver = parser.add_argument_group("MILP")
    solver.add_argument("--time-limit", type=float,
                        help="Solver wall-time limit for each LP/MILP solve")
    solver.add_argument("--milp-solver", choices=("highs", "gurobi"))
    solver.add_argument("--coefficient-bound", type=float)

    diagnostics = parser.add_argument_group("optional diagnostics and comparison")
    selection_mode = diagnostics.add_mutually_exclusive_group()
    selection_mode.add_argument("--selection-only", action="store_true",
                                help="Stop after RNJ boundary selection; skip MILP")
    selection_mode.add_argument("--run-milp", dest="selection_only", action="store_false",
                                help=argparse.SUPPRESS)
    baseline_mode = diagnostics.add_mutually_exclusive_group()
    baseline_mode.add_argument("--run-baseline", action="store_true",
                               help="Also solve the MILP-only comparison")
    baseline_mode.add_argument("--hybrid-only", dest="run_baseline", action="store_false",
                               help=argparse.SUPPRESS)
    return parser


def run(*args, **kwargs) -> dict:
    """Compatibility delegate; use ``pipeline.run`` for the Python API."""
    return run_pipeline(*args, **kwargs)


def main(argv: Sequence[str] | None = None, *, runner=None) -> None:
    """Parse explicit CLI overrides and invoke the shared computation entrypoint."""
    options = vars(_argument_parser().parse_args(argv))
    if "cases" in options:
        options["cases"] = tuple(options["cases"])
    (run if runner is None else runner)(**options)


def advanced_main(*, runner=None) -> None:
    """Compatibility alias for the unified CLI and its current defaults."""
    main(runner=runner)
