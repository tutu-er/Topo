"""Plot frozen spatial DR plans and completed AC evaluation aggregates only.

This script performs no simulation, fitting, search, or repricing.  It requires
price_freeze.json, summary.json, and the completion receipt to exist already.
The comprehensive figure includes planned controls, never household load data.
An aggregate-only figure is also produced for sharing without control schedules.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


STAGE = Path(__file__).resolve().parents[1]
ORDER = ["zero", "legacy_iid_pair", "fixed_correlated", "open_correlated",
         "rotation_uniform", "rotation_selected_capacity"]
LABELS = ["No DR", "Legacy pair", "Fixed pair", "Open nodes",
          "Uniform\nrotation", "Selected-cap.\nrotation"]
RECOVERY_COLOR = "#24806E"
ERROR_COLOR = "#D76A46"
OPEN_COLOR = "#336A9A"
GRAY = "#BCC5CE"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_frozen(run):
    required = ["problem_frozen.json", "price_freeze.json", "protocol.json", "summary.json", "ended.json"]
    missing = [name for name in required if not (run / name).exists()]
    if missing:
        raise FileNotFoundError("Completed run required; missing " + ", ".join(missing))
    problem, prices, protocol, summary, ended = [read(run / name) for name in required]
    if summary["status"] != "complete" or ended["status"] != "complete" or prices["status"] != "frozen":
        raise ValueError("Expected frozen prices and completed evaluation")
    for ref in [problem["input"], prices["problem"], prices["protocol"], ended["summary"]]:
        if sha(ref["path"]) != ref["sha256"]:
            raise ValueError("Hash mismatch: " + ref["path"])
    if problem["input"] != prices["problem"]:
        raise ValueError("Price and displayed problem do not match")
    policies = prices["policies"]
    if set(policies) != set(ORDER):
        raise ValueError("Expected the six predeclared comparison policies")
    table = {(row["law"], row["policy"]): row for row in summary["tables"]}
    if set(table) != {(law, policy) for law in ("main", "misspec") for policy in ORDER}:
        raise ValueError("Incomplete six-policy, two-law summary")
    for law in ("main", "misspec"):
        for policy in ORDER:
            row = table[(law, policy)]
            if row["complete"] + row["unresolved"] != row["planned"]:
                raise ValueError("Denominator mismatch")
            if not np.isclose(row["fee_per_day"], policies[policy]["cost"], rtol=1e-12, atol=1e-12):
                raise ValueError("Displayed fee differs from frozen policy")
    u = np.asarray(policies["open_correlated"]["mean_u"], float)
    capacity = np.asarray(policies["open_correlated"]["capacity_kw"], float)
    caps = np.asarray(problem["caps"], float)
    if u.shape != (len(problem["labels"]), protocol["T"]) or capacity.shape != caps.shape:
        raise ValueError("Planned controls/capacities have unexpected shape")
    if not np.isfinite(u).all() or not np.isfinite(capacity).all():
        raise ValueError("Nonfinite planned values")
    if np.max(np.abs(u) - capacity[:, None]) > 1e-6 or np.max(capacity - caps) > 1e-6 or capacity.min() < -1e-6:
        raise ValueError("Planned allocation is outside the declared domain")
    if capacity.sum() > protocol["resource_total_kw"] + 1e-6:
        raise ValueError("Total allocation exceeds the frozen domain")
    return problem, prices, protocol, summary, table, u, capacity, caps


def outcome_panel(ax, law, table):
    rows = [table[(law, policy)] for policy in ORDER]
    x = np.arange(len(ORDER))
    width = .34
    recoveries = [row["recoveries"] for row in rows]
    errors = [row["total_errors"] for row in rows]
    first = ax.bar(x-width/2, recoveries, width, color=RECOVERY_COLOR, label="Full recovery (days)")
    second = ax.bar(x+width/2, errors, width, color=ERROR_COLOR, label="Missing + extra blocks")
    ax.bar_label(first, padding=3, fontsize=8)
    ax.bar_label(second, padding=3, fontsize=8)
    maximum = max(max(recoveries), max(errors), max(row["planned"] for row in rows))
    ax.set_ylim(0, max(1, maximum)*1.23)
    ax.set_xticks(x, LABELS, rotation=27, ha="right", fontsize=8.5)
    ax.set_ylabel("Count")
    ax.set_title("Main response law" if law == "main" else "Response-law stress test", loc="left", weight="bold", pad=24)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E9ED", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    for tick, policy in zip(ax.get_xticklabels(), ORDER):
        if policy == "open_correlated":
            tick.set_color(OPEN_COLOR)
            tick.set_weight("bold")
    unfinished = [row["unresolved"] for row in rows]
    if any(unfinished):
        text = "Unresolved by policy: " + "/".join(str(v) for v in unfinished)
    else:
        planned = sorted({row["planned"] for row in rows})
        text = f"All {planned[0]} daily paths retained per strategy" if len(planned) == 1 else "All predeclared paths retained"
    ax.text(0, 1.01, text, transform=ax.transAxes, fontsize=8, color="#58616B")
    return first, second


def cost_panel(ax, prices):
    x = np.arange(len(ORDER))
    values = [prices["policies"][name]["cost"] for name in ORDER]
    bars = ax.bar(x, values, color=[OPEN_COLOR if name == "open_correlated" else GRAY for name in ORDER], width=.66)
    ax.bar_label(bars, labels=[f"{v:.1f}" for v in values], padding=3, fontsize=8)
    ax.set_ylim(0, max(1, max(values))*1.2)
    ax.set_xticks(x, LABELS, rotation=27, ha="right", fontsize=8.5)
    ax.set_title("Frozen daily fee", loc="left", weight="bold", pad=24)
    ax.set_ylabel("Contract units / day")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E5E9ED", linewidth=.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0, 1.01, "Same fee under both response laws", transform=ax.transAxes, fontsize=8, color="#58616B")


def save_formats(figure, output, stem):
    paths = []
    for extension in ("png", "pdf", "svg"):
        destination = output / (stem + "." + extension)
        figure.savefig(destination, dpi=200, bbox_inches="tight", facecolor="white")
        paths.append({"path": str(destination), "sha256": sha(destination)})
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=STAGE / "run_v1")
    parser.add_argument("--output", type=Path, default=STAGE / "public_figures")
    args = parser.parse_args()
    problem, prices, protocol, summary, table, u, capacity, caps = load_frozen(args.run)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                               "axes.titlesize": 11, "axes.labelsize": 9,
                               "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"})
    args.output.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(17.4, 10.3), layout="constrained")
    outer = figure.add_gridspec(2, 1, height_ratios=[1.15, 1.0], hspace=.17)
    top = outer[0].subgridspec(1, 2, width_ratios=[4.5, 1.3], wspace=.08)
    heat = figure.add_subplot(top[0])
    extent = max(.05, float(np.max(np.abs(u))))
    image = heat.imshow(u, aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-extent, vmax=extent,
                        extent=[0, 24, len(capacity)-.5, -.5])
    heat.set_yticks(np.arange(len(capacity)), [str(label) for label in problem["labels"]])
    heat.set_xticks(np.arange(0, 25, 3))
    heat.set_xlabel("Hour of day")
    heat.set_ylabel("Node")
    heat.set_title("A  Open-node planned mean response", loc="left", weight="bold")
    colorbar = figure.colorbar(image, ax=heat, pad=.015, fraction=.025)
    colorbar.set_label("Mean response u (kW)")
    heat.text(.995, 1.015, "+ increases load; − decreases load", transform=heat.transAxes, ha="right", fontsize=9, color="#58616B")
    allocation = figure.add_subplot(top[1], sharey=heat)
    y = np.arange(len(capacity))
    allocation.barh(y, caps, height=.64, color="white", edgecolor="#AAB3BD", linewidth=1., label="Historical cap")
    allocation.barh(y, capacity, height=.42, color=OPEN_COLOR, label="Allocated a")
    allocation.set_xlim(0, max(1, caps.max())*1.2)
    allocation.set_xlabel("Capacity (kW)")
    allocation.set_title("B  Daily allocation", loc="left", weight="bold", pad=25)
    allocation.tick_params(axis="y", labelleft=False)
    allocation.spines[["top", "right"]].set_visible(False)
    for yi, value in zip(y, capacity):
        allocation.text(value+.015, yi, f"{value:.3f}", va="center", fontsize=8)
    allocation.legend(loc="lower right", fontsize=7, frameon=False)
    allocation.text(0, 1.02, f"Total {capacity.sum():.3f} / {protocol['resource_total_kw']:g} kW", transform=allocation.transAxes, fontsize=9)
    bottom = outer[1].subgridspec(1, 3, width_ratios=[1.35, 1.35, 1.05], wspace=.16)
    main_ax, stress_ax, fee_ax = [figure.add_subplot(bottom[i]) for i in range(3)]
    outcome_panel(main_ax, "main", table)
    outcome_panel(stress_ax, "misspec", table)
    cost_panel(fee_ax, prices)
    main_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    stress_ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    main_ax.legend(loc="upper right", fontsize=8, frameon=False)
    figure.suptitle("Spatial DR allocation and independent AC evaluation", fontsize=16, weight="bold")
    delivery_note = "Uncapped Gaussian delivery: open-plan device violations 12/12 main, 9/12 stress; both rotations 0/12."
    footer = (f"Frozen historical candidate pool and R/X; 288 training + 288 validation rows; 96 new slots; original 3% rule.  "
              f"{summary['independent_backgrounds']} paired background days; two laws do not double the independent denominator.\n"
              "Correlated response assumptions are synthetic. Fees are contract units, not calibrated currency. Planned u contains no household observations.\n"
              + delivery_note)
    figure.supxlabel(footer, fontsize=8, color="#58616B")
    outputs = save_formats(figure, args.output, "spatial_correlated_comprehensive")
    plt.close(figure)

    aggregate, axes = plt.subplots(1, 3, figsize=(16.2, 5.0), layout="constrained", gridspec_kw={"width_ratios": [1.35,1.35,1.05]})
    outcome_panel(axes[0], "main", table)
    outcome_panel(axes[1], "misspec", table)
    cost_panel(axes[2], prices)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    aggregate.suptitle("Final selection and fee: independent AC evaluation", fontsize=15, weight="bold")
    aggregate.supxlabel(footer.split("\n")[0] + "\n" + delivery_note + "\nAggregate results only; no load or control trajectories.", fontsize=8, color="#58616B")
    outputs.extend(save_formats(aggregate, args.output, "spatial_correlated_aggregate"))
    plt.close(aggregate)
    manifest = {"status": "generated", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source": {"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
                "inputs": {str(args.run/name): sha(args.run/name) for name in ("protocol.json", "problem_frozen.json", "price_freeze.json", "summary.json", "ended.json")},
                "outputs": outputs, "privacy": "Only planned controls/capacities and aggregate outcomes. No household P/Q/V observations read.",
                "scope": "Visualization of frozen results, no simulation or optimization."}
    (args.output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "generated", "files": len(outputs), "output": str(args.output)}))


if __name__ == "__main__":
    main()
