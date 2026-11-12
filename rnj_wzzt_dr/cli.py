"""Explicit inputs and one small CLI, without old batch controllers."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import io
from .history import fit_observed, freeze_history
from .identification.algorithm import METHODS, ALIASES, SearchConfig
from .observations import load
from .dr.pipeline import load_resources, plan, prepare_dr, zero_screen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    doctor = commands.add_parser("doctor", help="Check imports, versions and extracted source hashes; no solves")
    doctor.add_argument("--sources", action="store_true", help="Also check original source locations when available")
    check = commands.add_parser("check", help="Run deterministic interface checks, without old datasets")
    check.add_argument("--solver", action="store_true", help="Also run bounded tiny MIQP and nonzero pair tests")
    fit = commands.add_parser("fit", help="Observed training/validation -> native frozen candidate bank")
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--validation", type=Path, required=True)
    fit.add_argument("--method", choices=(*METHODS, *ALIASES), default="111")
    fit.add_argument("--seconds", type=float, default=14400.)
    fit.add_argument("--output", type=Path, required=True)
    for name in ("score", "zero", "plan"):
        command = commands.add_parser(name)
        command.add_argument("--result", type=Path, required=True)
        command.add_argument("--history", type=Path, required=True,
                             help="Original full selection/validation observations, not training")
        command.add_argument("--output", type=Path, required=True)
        if name == "score":
            command.add_argument("--new", type=Path, required=True)
        else:
            command.add_argument("--forecast", type=Path, required=True,
                                 help="NPZ with power[slots,2*n] and labels, derived from history")
            command.add_argument("--target", type=int, required=True,
                                 help="Conditionally correct target index in the complete frozen bank")
            command.add_argument("--voltage-sigma", type=float, required=True)
            command.add_argument("--power-relative-sd", type=float, default=.05)
            command.add_argument("--time-ar", type=float, default=.7)
            command.add_argument("--case", help="Match the preserved original resource allocation")
            command.add_argument("--resources", type=Path)
            if name == "plan":
                command.add_argument("--seconds", type=float, default=300.)
                command.add_argument("--mip-gap", type=float, default=.01)
    args = parser.parse_args(argv)
    if args.action == "check":
        import importlib
        import unittest
        module = importlib.import_module(__package__+".tests.test_pipeline")
        module.RUN_SOLVER = args.solver
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        return 0 if result.wasSuccessful() else 1
    if args.action == "doctor":
        from .verify import verify
        result = dict(environment=io.environment(), sources=verify(originals=args.sources))
        print(__import__("json").dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["sources"]["passed"] else 1
    if args.output.exists():
        raise FileExistsError("Use a new private output path; completed or failed attempts are preserved")
    if args.action == "fit":
        # Keep all statistical defaults, including relative_gain=0. Short runs
        # reserve a proportional output allowance, without changing search rules.
        cfg = replace(SearchConfig(), total_seconds=args.seconds,
                      output_reserve_seconds=min(600., args.seconds*.1))
        receipt = dict(status="frozen_before_fit", method=args.method, config=cfg,
                       train=io.ref(args.train), validation=io.ref(args.validation),
                       sources=io.source_receipt(), environment=io.environment())
        io.write(args.output/"request.json", receipt)
        try:
            result = fit_observed(load(args.train), load(args.validation), method=args.method, config=cfg)
            io.write(args.output/"result.json", result)
            io.write(args.output/"complete.json", dict(status="budget_exceeded" if result["total_cap_exceeded"]
                                                      else "complete", selected=result["selected"]["index"]))
        except Exception as exc:
            io.write(args.output/"failed.json", dict(status="failed", type=type(exc).__name__, error=str(exc)))
            raise
        return 0
    result = io.read(args.result)
    history = freeze_history(result, load(args.history))
    if args.action == "score":
        io.write(args.output, history.select(load(args.new)))
        return 0
    with np.load(args.forecast, allow_pickle=False) as z:
        if tuple(map(int, z["labels"])) != history.labels:
            raise ValueError("Forecast terminal order changed")
        forecast = z["power"].copy()
    resources = load_resources(args.case, labels=history.labels, directory=args.resources) if args.case else None
    data = prepare_dr(history, forecast, voltage_sigma=args.voltage_sigma,
                      resources=resources, power_relative_sd=args.power_relative_sd, time_ar=args.time_ar)
    if args.action == "zero":
        io.write(args.output, zero_screen(data, target=args.target))
    else:
        plan(data, target=args.target, output=args.output, seconds=args.seconds, mip_gap=args.mip_gap)
    return 0
