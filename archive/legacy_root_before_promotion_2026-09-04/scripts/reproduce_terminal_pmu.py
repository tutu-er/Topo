"""Run terminal-meter topology detection and micro-PMU placement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from topoident.terminal_pmu import run_terminal_pmu_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--design-samples", type=int, default=18)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--output", type=Path, default=Path("results/terminal_pmu.json"))
    args = parser.parse_args()
    result = run_terminal_pmu_experiment(
        seed=args.seed, design_samples=args.design_samples, trials=args.trials
    )
    payload = {
        "candidate_topologies": len(result.configurations),
        "terminal_smart_meter_buses_1_based": [x + 1 for x in result.terminal_nodes],
        "optimal_pmu_buses_1_based": {
            str(k): [x + 1 for x in buses] for k, buses in result.optimal_by_budget.items()
        },
        "worst_pair_separation": result.separation_by_budget,
        "identification_accuracy": result.accuracy_by_scheme,
        "assumptions": [
            "radial balanced feeder",
            "hidden transit buses are zero-injection",
            "candidate switch/line library and impedances are known",
            "terminal p/q loads are observed by smart meters",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
