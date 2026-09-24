"""Run the Zhang et al. 2020 numerical reproduction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from topoident.ieee33 import simulate_meter_data
from topoident.pseudo_refinement import fine_identification_pseudo
from topoident.zhang2020 import basic_identification, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--fine-samples", type=int, default=40)
    parser.add_argument("--iterations", type=int, default=16)
    parser.add_argument("--power-noise", type=float, default=0.02)
    parser.add_argument("--voltage-noise", type=float, default=0.0)
    parser.add_argument("--output", type=Path, default=Path("results/zhang2020.json"))
    args = parser.parse_args()

    data = simulate_meter_data(
        samples=args.samples,
        seed=args.seed,
        p_noise=args.power_noise,
        q_noise=args.power_noise,
        v_noise=args.voltage_noise,
    )
    basic = basic_identification(data.p, data.q, data.v, max_edges=60)
    fine = fine_identification_pseudo(
        data.p,
        data.q,
        data.v,
        basic,
        samples=args.fine_samples,
        iterations=args.iterations,
    )
    metrics = evaluate(fine, data.branches)
    true_edges = {tuple(sorted((branch.u, branch.v))) for branch in data.branches}
    metrics.update(
        {
            "paper_case": "IEEE33 weakly meshed, standard tie 21-8 closed",
            "seed": args.seed,
            "samples": args.samples,
            "fine_samples": args.fine_samples,
            "power_noise_relative_std": args.power_noise,
            "voltage_noise_relative_std": args.voltage_noise,
            "basic_true_edges_in_candidates": len(set(basic.edges) & true_edges),
            "angle_mae_degree": float(
                np.mean(np.abs(fine.theta - data.theta_true[-args.fine_samples :])) * 180.0 / np.pi
            ),
            "exact_reproduction": False,
            "exact_reproduction_blockers": [
                "paper author repository is no longer publicly available",
                "Irish customer traces and exact switch state are not bundled with the paper",
            ],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
