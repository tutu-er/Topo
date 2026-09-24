"""Generate terminal loads and solve nonlinear radial AC line flows."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from terminal_case33.data.case33bw_raw import load_raw_case33bw
from terminal_case33.models.timeseries import simulate_terminal_load_timeseries
from terminal_case33.scenario.terminalize import terminalize_case33
from terminal_case33.scenario.validate_scenario import validate_terminal_load_only_scenario
from terminal_case33.utils.io import ensure_dir, write_json
from terminal_case33.visualization.plots import plot_network


def _save_frame_dict(result: dict, out_dir: Path) -> None:
    """Write tabular AC power-flow outputs."""

    frame_names = [
        "V_bus_mag",
        "V_bus_angle_deg",
        "I_branch_mag",
        "branch_p_from_pu",
        "branch_q_from_pu",
        "branch_p_to_pu",
        "branch_q_to_pu",
        "branch_loss_p_pu",
        "branch_loss_q_pu",
    ]
    for name in frame_names:
        result[name].to_csv(out_dir / f"{name}.csv")
    pd.concat([result["converged"], result["iterations"], result["max_voltage_error"]], axis=1).to_csv(
        out_dir / "solver_iterations.csv"
    )


def _branch_flow_summary(result: dict, base_mva: float) -> pd.DataFrame:
    """Summarize branch flows and losses in engineering units."""

    p_from = result["branch_p_from_pu"]
    q_from = result["branch_q_from_pu"]
    p_loss = result["branch_loss_p_pu"]
    q_loss = result["branch_loss_q_pu"]
    scale = base_mva * 1000.0
    rows = []
    for col in p_from.columns:
        rows.append(
            {
                "branch": col,
                "mean_p_from_kw": float(p_from[col].mean() * scale),
                "max_abs_p_from_kw": float(p_from[col].abs().max() * scale),
                "mean_q_from_kvar": float(q_from[col].mean() * scale),
                "max_abs_q_from_kvar": float(q_from[col].abs().max() * scale),
                "mean_p_loss_kw": float(p_loss[col].mean() * scale),
                "max_p_loss_kw": float(p_loss[col].max() * scale),
                "mean_q_loss_kvar": float(q_loss[col].mean() * scale),
            }
        )
    return pd.DataFrame(rows)


def _plot_voltage_envelope(result: dict, path: Path) -> None:
    """Plot min/mean/max bus-voltage envelope over time."""

    v_mag = result["V_bus_mag"]
    plt.figure(figsize=(9, 4))
    plt.plot(v_mag.index, v_mag.min(axis=1), label="min")
    plt.plot(v_mag.index, v_mag.mean(axis=1), label="mean")
    plt.plot(v_mag.index, v_mag.max(axis=1), label="max")
    plt.xlabel("time step")
    plt.ylabel("voltage magnitude [pu]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def _plot_branch_heatmap(frame: pd.DataFrame, path: Path, title: str) -> None:
    """Plot branch-flow heatmap."""

    plt.figure(figsize=(11, 6))
    plt.imshow(frame.T.to_numpy(), aspect="auto", cmap="coolwarm")
    plt.colorbar(label="pu")
    plt.xlabel("time step")
    plt.ylabel("oriented branch")
    plt.yticks(range(len(frame.columns)), frame.columns, fontsize=5)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def run(
    output: str | Path = "outputs/ac_line_flow_case33",
    T: int = 96,
    seed: int = 7,
    profile_mode: str = "pv_ev_mixed",
    noise_rel: float = 0.005,
) -> dict:
    """Build the terminalized case33 feeder and solve AC flows per time step."""

    out_dir = ensure_dir(output)
    raw = load_raw_case33bw(include_tie_lines=False)
    net = terminalize_case33(raw, mode="hybrid_leaf", service_impedance_mode="scaled_original", seed=seed)
    scenario = validate_terminal_load_only_scenario(net, strict=True, output_path=out_dir / "scenario_summary.json")

    sim = simulate_terminal_load_timeseries(
        net,
        T=T,
        dt_seconds=900,
        profile_mode=profile_mode,
        pf_mode="time_varying_pf",
        noise_config={"p_rel_std": noise_rel, "q_rel_std": noise_rel, "v_rel_std": noise_rel},
        v0_config={"sigma": 0.001},
        seed=seed,
    )
    ac = {
        "V_bus_mag": sim["V_bus_true"],
        "V_bus_angle_deg": sim["V_bus_angle_deg_true"],
        "I_branch_mag": sim["I_branch_mag_true"],
        "branch_p_from_pu": sim["branch_p_from_pu_true"],
        "branch_q_from_pu": sim["branch_q_from_pu_true"],
        "branch_p_to_pu": sim["branch_p_to_pu_true"],
        "branch_q_to_pu": sim["branch_q_to_pu_true"],
        "branch_loss_p_pu": sim["branch_loss_p_pu_true"],
        "branch_loss_q_pu": sim["branch_loss_q_pu_true"],
        "converged": sim["ac_converged"],
        "iterations": sim["ac_iterations"],
        "max_voltage_error": sim["ac_max_voltage_error"],
        "metadata": sim["topology_metadata"]["power_flow_solver"],
    }

    _save_frame_dict(ac, out_dir)
    sim["P_terminal_true"].to_csv(out_dir / "P_terminal_true_pu.csv")
    sim["Q_terminal_true"].to_csv(out_dir / "Q_terminal_true_pu.csv")
    sim["P_terminal_meas"].to_csv(out_dir / "P_terminal_meas_pu.csv")
    sim["Q_terminal_meas"].to_csv(out_dir / "Q_terminal_meas_pu.csv")
    net.buses.to_csv(out_dir / "buses.csv", index=False)
    net.branches.to_csv(out_dir / "branches.csv", index=False)
    plot_network(net, out_dir / "topology_terminalized.png", "terminalized case33 AC-flow model")
    _plot_voltage_envelope(ac, out_dir / "voltage_envelope.png")
    _plot_branch_heatmap(ac["branch_p_from_pu"], out_dir / "branch_p_from_heatmap.png", "sending-end active branch flow")

    branch_summary = _branch_flow_summary(ac, net.base_mva)
    branch_summary.to_csv(out_dir / "branch_flow_summary.csv", index=False)

    root_edges = [item["label"] for item in ac["metadata"]["branch_edges"] if item["from_bus"] == net.root_bus]
    root_p = ac["branch_p_from_pu"][root_edges].sum(axis=1)
    root_q = ac["branch_q_from_pu"][root_edges].sum(axis=1)
    total_load_p = sim["P_terminal_true"].sum(axis=1)
    total_load_q = sim["Q_terminal_true"].sum(axis=1)
    total_loss_p = ac["branch_loss_p_pu"].sum(axis=1)
    total_loss_q = ac["branch_loss_q_pu"].sum(axis=1)
    balance = pd.DataFrame(
        {
            "root_p_from_pu": root_p,
            "terminal_load_p_pu": total_load_p,
            "total_loss_p_pu": total_loss_p,
            "p_balance_error_pu": root_p - total_load_p - total_loss_p,
            "root_q_from_pu": root_q,
            "terminal_load_q_pu": total_load_q,
            "total_loss_q_pu": total_loss_q,
            "q_balance_error_pu": root_q - total_load_q - total_loss_q,
        }
    )
    balance.to_csv(out_dir / "power_balance.csv")

    solver_summary = {
        "case": "terminalized_case33_hybrid_leaf",
        "power_flow_model": "nonlinear radial AC backward-forward sweep",
        "root_reference": "root bus is slack/reference voltage phasor; angle fixed at 0",
        "T": int(T),
        "profile_mode": profile_mode,
        "measurement_noise_rel_std": float(noise_rel),
        "all_converged": bool(ac["converged"].all()),
        "max_iterations": int(ac["iterations"].max()),
        "max_voltage_error": float(ac["max_voltage_error"].max()),
        "min_voltage_pu": float(ac["V_bus_mag"].min().min()),
        "max_voltage_pu": float(ac["V_bus_mag"].max().max()),
        "max_abs_p_balance_error_pu": float(balance["p_balance_error_pu"].abs().max()),
        "max_abs_q_balance_error_pu": float(balance["q_balance_error_pu"].abs().max()),
        "scenario": scenario,
        "ac_metadata": ac["metadata"],
    }
    write_json(out_dir / "solver_summary.json", solver_summary)
    (out_dir / "report.md").write_text(
        "# AC line-flow time-series\n\n"
        "This run uses the terminalized case33 feeder and solves a nonlinear radial AC "
        "backward-forward sweep at every time step. The root bus is the slack/reference "
        "phasor. Terminal P/Q profiles are generated first, then branch currents, "
        "sending-end powers, receiving-end powers, and line losses are computed from "
        "the AC equations.\n\n"
        f"- time steps: {T}\n"
        f"- all converged: {solver_summary['all_converged']}\n"
        f"- voltage range: {solver_summary['min_voltage_pu']:.6f} to {solver_summary['max_voltage_pu']:.6f} pu\n"
        f"- max active-power balance error: {solver_summary['max_abs_p_balance_error_pu']:.3e} pu\n"
        f"- relative measurement noise saved on measured P/Q channels: {noise_rel:.4%}\n",
        encoding="utf-8",
    )
    return solver_summary


def main() -> None:
    """CLI entry point."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/ac_line_flow_case33")
    parser.add_argument("--T", type=int, default=96)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile-mode", default="pv_ev_mixed")
    parser.add_argument("--noise-rel", type=float, default=0.005)
    args = parser.parse_args()
    run(args.output, T=args.T, seed=args.seed, profile_mode=args.profile_mode, noise_rel=args.noise_rel)


if __name__ == "__main__":
    main()
