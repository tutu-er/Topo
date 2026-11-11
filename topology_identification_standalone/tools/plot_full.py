"""Plot independently analyzed full-panel results, without loading an optimizer.

Usage: python tools/plot_full.py PATH_TO_RUN [--output PATH_TO_FIGURES]
Figures describe the supplied finite panel. Missing/invalid denominators stay
visible; confidence intervals treating the 48 inputs as independent are not
invented. Matplotlib is needed only for this optional postprocessing script.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import combinations, permutations
import json
from pathlib import Path

import numpy as np


ALL = ("rnj", "nj", "000", "100", "010", "001", "110", "101", "011", "111")
MAIN = ("rnj", "nj", "000", "100", "111")
CELLS = ("000", "100", "010", "001", "110", "101", "011", "111")
NAMES = {"rnj": "RNJ", "nj": "NJ", "000": "WZZT", "100": "R + WZZT", "111": "R + WZZT + T + E"}
CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
COLORS = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    if not path.exists() or path.stat().st_size <= 1:
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, *keys, default=np.nan):
    for key in keys:
        if row.get(key) not in (None, "", "None", "null"):
            try:
                value = float(row[key])
                if np.isfinite(value):
                    return value
            except (ValueError, TypeError):
                continue
    return default


def boolean(value):
    return str(value).lower() in {"true", "1", "yes"}


def usable(row):
    if any(boolean(row.get(key, False)) for key in ("cap_exceeded", "total_cap_exceeded", "budget_exceeded")):
        return False
    for key in ("valid", "verified", "numerically_verified"):
        if row.get(key) not in (None, ""):
            return boolean(row[key])
    return row.get("status", "").lower() in {"complete", "verified", "valid", "numerically_verified"}


def metric(row, key):
    if key == "F1":
        return number(row, "F1", "clade_f1", "f1")
    if key == "exact":
        return float(boolean(row.get("exact", row.get("exact_recovery", False))))
    return number(row, "total_seconds", "reported_seconds", "charged_seconds")


def plot(run, output=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run).resolve()
    analysis = run/"analysis" if (run/"analysis").is_dir() else run
    output = Path(output).resolve() if output else run/"figures"
    output.mkdir(parents=True, exist_ok=True)
    inputs = [p for p in analysis.glob("*.csv") if p.is_file()]
    rows = table(analysis/"per_model.csv")
    if not rows:
        raise ValueError("Run independent analysis first: a nonempty per_model.csv is required")
    by_method = {method: [row for row in rows if row.get("method") == method] for method in ALL}
    by_job = {}
    for row in rows:
        job = row.get("job", row.get("input", ""))
        by_job.setdefault(job, {})[row.get("method")] = row
    # Full protocol requires 48 inputs per method, including unresolved rows.
    planned = 48
    protocol_path = run/"protocol.json"
    protocol = {}
    if protocol_path.exists():
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        planned = int(protocol.get("planned_inputs", len(protocol.get("jobs", [])) or planned))
    verification_path = analysis/"verification.json"
    verification = json.loads(verification_path.read_text(encoding="utf-8")) if verification_path.exists() else {}
    selection = str(protocol.get("primary_selection", protocol.get("selection_metric", verification.get("selection_metric", "")))).lower()
    if "mse" not in selection:
        raise ValueError("This plotter requires an explicitly declared MSE-primary protocol or verification record")
    style = {"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
             "axes.labelsize": 9, "legend.fontsize": 8, "svg.fonttype": "none"}
    plt.rcParams.update(style)
    written = []
    valid_counts = {m: sum(usable(row) for row in group) for m, group in by_method.items()}

    def save(fig, name, caption):
        if protocol.get("synthetic_fixture"):
            caption = "SYNTHETIC SCHEMA FIXTURE - NOT BENCHMARK RESULTS. " + caption
        fig.text(.01, .012, caption, fontsize=7, va="bottom", wrap=True)
        fig.tight_layout(rect=(0, .075, 1, .94))
        for suffix in ("png", "svg", "pdf"):
            path = output/(name+"."+suffix)
            fig.savefig(path, dpi=220, bbox_inches="tight")
            written.append(path)
        plt.close(fig)

    def aggregate(method):
        group = by_method[method]
        valid = [row for row in group if usable(row)]
        f1 = [metric(row, "F1") for row in valid]
        seconds = [metric(row, "time") for row in group]
        return (float(np.mean(f1)) if f1 else np.nan,
                sum(metric(row, "exact") for row in valid)/planned,
                float(np.nansum(seconds))/3600 if any(np.isfinite(seconds)) else np.nan,
                len(valid))

    def summary_figure(methods, name, title):
        values = np.array([aggregate(m) for m in methods], float)
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        labels = [NAMES.get(m, m) for m in methods]
        palette = [COLORS[i % len(COLORS)] for i in range(len(methods))]
        for i, (ylabel, col) in enumerate((("F1 (valid inputs)", 0), ("Exact / planned", 1), ("Recorded charged method-hours", 2))):
            axes[i].bar(range(len(methods)), values[:, col], color=palette)
            axes[i].set_xticks(range(len(methods)), labels, rotation=28, ha="right")
            axes[i].set_ylabel(ylabel); axes[i].grid(axis="y", alpha=.2)
            if col < 2:
                axes[i].set_ylim(0, 1.09)
            else:
                finite = values[np.isfinite(values[:, col]), col]
                if len(finite):
                    axes[i].set_ylim(0, max(1e-6, float(finite.max())*1.16))
            for k, method in enumerate(methods):
                if col == 0:
                    if np.isfinite(values[k, col]):
                        axes[i].annotate(f"{values[k, col]:.3f}\nn={int(values[k, 3])}/{planned}", (k, values[k, col]),
                                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
                    else:
                        axes[i].text(k, .03, f"n=0/{planned}\nNA", ha="center", fontsize=7)
                if col == 1:
                    axes[i].annotate(f"{round(values[k, col]*planned)}/{planned}", (k, values[k, col]),
                                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
                if col == 2:
                    if np.isfinite(values[k, col]):
                        axes[i].annotate(f"{values[k, col]:.3g}", (k, values[k, col]),
                                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
                    unknown = max(0, planned-len(by_method[method]))+sum(not np.isfinite(metric(row, "time")) for row in by_method[method])
                    if unknown:
                        axes[i].text(k, .03, f"unknown={unknown}", transform=axes[i].get_xaxis_transform(),
                                     rotation=90, ha="center", va="bottom", fontsize=6)
        fig.suptitle(title)
        save(fig, name, "F1 uses explicitly counted valid inputs; exact recovery keeps the planned denominator. "
             "Time sums reported attempts including failures; unknown costs are disclosed and omitted from recorded totals.")

    summary_figure(MAIN, "main_comparison", "Main algorithms: validation-MSE selection")
    summary_figure(CELLS, "factorial_ablation", "All eight R/T/E cells (bit order R, T, E)")

    fig, axes = plt.subplots(len(CASES), 2, figsize=(10, 10), squeeze=False)
    for i, case in enumerate(CASES):
        case_rows = [row for row in rows if row.get("case") == case]
        sigmas = sorted({number(row, "voltage_sigma", "sigma") for row in case_rows if np.isfinite(number(row, "voltage_sigma", "sigma"))})
        for j, method in enumerate(MAIN):
            means, exact, counts = [], [], []
            for sigma in sigmas:
                group = [row for row in case_rows if row.get("method") == method and number(row, "voltage_sigma", "sigma") == sigma]
                valid = [row for row in group if usable(row)]
                means.append(np.mean([metric(row, "F1") for row in valid]) if valid else np.nan)
                exact.append(sum(metric(row, "exact") for row in valid)/4.)
                counts.append(len(valid))
            axes[i, 0].plot(sigmas, means, "o-", color=COLORS[j], label=NAMES[method], ms=4)
            axes[i, 1].plot(sigmas, exact, "o-", color=COLORS[j], label=NAMES[method], ms=4)
            for sigma, mean, count in zip(sigmas, means, counts):
                if count < 4:
                    axes[i, 0].annotate(f"n={count}/4", (sigma, mean if np.isfinite(mean) else .02),
                                        xytext=(2, 4), textcoords="offset points", fontsize=6, color=COLORS[j])
        for col in range(2):
            axes[i, col].set_title(case); axes[i, col].set_ylim(-.03, 1.05)
            axes[i, col].set_xlabel("Voltage / root-meter relative noise sigma")
            axes[i, col].ticklabel_format(axis="x", style="sci", scilimits=(-3, -3))
            axes[i, col].grid(alpha=.2)
        axes[i, 0].set_ylabel("F1 (valid inputs)"); axes[i, 1].set_ylabel("Exact / 4 planned")
    axes[0, 0].legend(loc="lower left", fontsize=7)
    fig.suptitle("Case and measurement-noise strata")
    save(fig, "case_noise", "Four shared seed groups per case/noise cell; lines are descriptive means, not independent-sample confidence intervals. "
         "Missing/failed runs reduce exact/planned recovery; F1 is restricted to valid runs.")

    def contrast(coefficients, key):
        values = []
        for group in by_job.values():
            if all(method in group and usable(group[method]) for method in coefficients):
                terms = [weight*metric(group[method], key) for method, weight in coefficients.items()]
                if all(np.isfinite(terms)):
                    values.append(sum(terms))
        return (float(np.mean(values)) if values else np.nan), len(values)

    def effect_figure(effects, name, title):
        fig, axes = plt.subplots(1, 3, figsize=(13, 5))
        labels = [label for label, _ in effects]
        for ax, key, ylabel, scale in zip(axes, ("F1", "exact", "time"),
                                         ("Mean delta F1", "Mean delta exact (percentage points)", "Mean delta charged seconds"), (1., 100., 1.)):
            data_rows = [contrast(coef, key) for _, coef in effects]
            values = np.array([value*scale for value, _ in data_rows])
            ax.barh(range(len(labels)), values, color=["#0072B2" if v >= 0 else "#D55E00" for v in values])
            ax.axvline(0, color="black", linewidth=.7)
            ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis()
            ax.set_xlabel(ylabel); ax.grid(axis="x", alpha=.2)
            for k, (value, count) in enumerate(data_rows):
                if np.isfinite(value):
                    ax.annotate(f" n={count}", (value*scale, k), fontsize=6, va="center")
        fig.suptitle(title)
        save(fig, name, "Contrasts pair the same input across all required cells; incomplete contrasts are omitted with n disclosed. "
             "Negative effects are retained. Exact deltas are differences of 0/1 recovery indicators.")

    marginal = []
    for module in range(3):
        others = [i for i in range(3) if i != module]
        for mask in range(4):
            low = ["0"]*3
            for k, other in enumerate(others):
                low[other] = str((mask >> k) & 1)
            high = low.copy(); high[module] = "1"
            context = ",".join("RTE"[i]+low[i] for i in others)
            marginal.append(("RTE"[module]+" | "+context, {"".join(high): 1., "".join(low): -1.}))
    effect_figure(marginal, "conditional_marginals", "Conditional module contributions")

    interactions = []
    for pair in combinations(range(3), 2):
        remaining = next(i for i in range(3) if i not in pair)
        for fixed in (0, 1):
            coefficients = {}
            for a, b in ((0, 0), (1, 0), (0, 1), (1, 1)):
                bits = ["0"]*3; bits[remaining] = str(fixed)
                bits[pair[0]], bits[pair[1]] = str(a), str(b)
                coefficients["".join(bits)] = 1. if a == b else -1.
            interactions.append(("RTE"[pair[0]]+" x "+"RTE"[pair[1]]+" | "+"RTE"[remaining]+str(fixed), coefficients))
    interactions.append(("R x T x E", {cell: (-1.)**(3-cell.count("1")) for cell in CELLS}))
    effect_figure(interactions, "factorial_interactions", "Second- and third-order factorial interactions")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    complete = [group for group in by_job.values() if all(cell in group and usable(group[cell]) for cell in CELLS)]
    for order in permutations(range(3)):
        bits = ["0"]*3; route = ["000"]
        for module in order:
            bits[module] = "1"; route.append("".join(bits))
        for ax, key in zip(axes, ("F1", "exact")):
            values = [np.mean([metric(group[cell], key) for group in complete]) if complete else np.nan for cell in route]
            ax.plot(range(4), values, "o-", label=" -> ".join("RTE"[i] for i in order), ms=4)
            ax.set_xticks(range(4), ("WZZT", "1 module", "2 modules", "R+T+E"))
            ax.set_ylabel("Mean F1" if key == "F1" else "Exact fraction")
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=7)
    fig.suptitle(f"Six module-addition views; complete paired inputs n={len(complete)}/{planned}")
    save(fig, "module_orders", "These paths read the same eight factorial cells; they are not six different temporal search pipelines.")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    times = [[metric(row, "time") for row in by_method[m] if np.isfinite(metric(row, "time")) and metric(row, "time") > 0] for m in ALL]
    tick_kw = {"tick_labels": ALL} if tuple(map(int, matplotlib.__version__.split(".")[:2])) >= (3, 9) else {"labels": ALL}
    axes[0, 0].boxplot([values or [np.nan] for values in times], showfliers=True, **tick_kw)
    axes[0, 0].set_yscale("log"); axes[0, 0].set_ylabel("Reported charged seconds (log)")
    axes[0, 0].set_title("Per-input method cost"); axes[0, 0].tick_params(axis="x", rotation=45)
    categories = ("valid", "failed/invalid", "budget exceeded", "pending")
    counts = {category: [] for category in categories}
    for method in ALL:
        group = by_method[method]
        valid = sum(usable(row) for row in group)
        budget = sum(row.get("status") == "budget_exceeded" or boolean(row.get("cap_exceeded", False)) for row in group)
        pending = sum(row.get("status") in {"pending", "missing", "queued", "running"} for row in group)+max(0, planned-len(group))
        counts["valid"].append(valid); counts["budget exceeded"].append(budget)
        counts["pending"].append(pending); counts["failed/invalid"].append(max(0, planned-valid-budget-pending))
    bottom = np.zeros(len(ALL))
    for category, color in zip(categories, ("#009E73", "#D55E00", "#E69F00", "#BBBBBB")):
        axes[0, 1].bar(range(len(ALL)), counts[category], bottom=bottom, label=category, color=color)
        bottom += counts[category]
    axes[0, 1].set_xticks(range(len(ALL)), ALL, rotation=45); axes[0, 1].set_ylabel("Planned inputs")
    axes[0, 1].set_title("All denominators retained"); axes[0, 1].legend(fontsize=7)
    solvers = table(analysis/"solver.csv")
    for stage, color in zip(("W", "T", "E"), COLORS):
        group = [row for row in solvers if row.get("stage") == stage and np.isfinite(number(row, "seconds")) and np.isfinite(number(row, "gap", "relative_gap"))]
        axes[1, 0].scatter([number(row, "seconds") for row in group],
                           [max(number(row, "gap", "relative_gap"), 1e-12) for row in group], s=10, alpha=.6, label=stage, color=color)
    axes[1, 0].set_yscale("log"); axes[1, 0].set_xlabel("Native MIQP seconds")
    axes[1, 0].set_ylabel("Relative gap (zero shown at 1e-12)"); axes[1, 0].legend()
    statuses = sorted({str(row.get("status", "missing")) for row in solvers})
    axes[1, 1].bar(statuses, [sum(str(row.get("status", "missing")) == status for row in solvers) for status in statuses])
    axes[1, 1].set_xlabel("Native solver status code"); axes[1, 1].set_ylabel("MIQP calls")
    fig.suptitle("Runtime, completion status and solver evidence")
    save(fig, "runtime_solver", "Costs include the declared shared-preprocessing charge. A native gap only covers its conditional W/T/E domain; "
         "zero gaps use a plotting floor. No topology-correctness probability is inferred.")

    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    comparisons = (("100", "000"), ("111", "100"), ("111", "nj"), ("111", "rnj"))
    for ax, (new, old) in zip(axes.ravel(), comparisons):
        paired = [(metric(group[new], "F1")-metric(group[old], "F1"), group[new].get("case", ""))
                  for group in by_job.values() if all(m in group and usable(group[m]) for m in (new, old))]
        paired.sort(key=lambda value: value[0])
        for case, color in zip(CASES, COLORS):
            selected = [(i+1, value) for i, (value, c) in enumerate(paired) if c == case]
            ax.scatter([i for i, _ in selected], [v for _, v in selected], s=22, color=color, label=case)
        ax.axhline(0, color="black", linewidth=.7); ax.grid(axis="y", alpha=.2)
        win = sum(v > 1e-12 for v, _ in paired); loss = sum(v < -1e-12 for v, _ in paired)
        ax.set_title(f"{NAMES[new]} minus {NAMES[old]}: {win} win / {len(paired)-win-loss} tie / {loss} loss")
        ax.set_xlabel(f"Paired input sorted by delta ({len(paired)}/{planned})"); ax.set_ylabel("Delta F1")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Paired topology changes")
    save(fig, "paired_changes", "Each point compares the same observations; colours identify networks. "
         "Shared seed groups are correlated across networks and noise levels; these are descriptive paired differences.")

    payload = dict(plot_source_sha256=sha(__file__), run=str(run), planned_inputs_per_method=planned,
                   valid_counts=valid_counts, methods=list(ALL),
                   analysis_input_sha256={p.name: sha(p) for p in [*inputs, *([verification_path] if verification_path.exists() else [])]},
                   protocol_sha256=sha(protocol_path) if protocol_path.exists() else None,
                   outputs={p.name: sha(p) for p in written},
                   selection="validation MSE; MAE is diagnostic only",
                   interpretation="Descriptive finite-panel plots; unresolved denominators disclosed; no independent-input confidence intervals")
    (output/"plot_manifest.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(dict(figures=len(written)//3, files=len(written), output=str(output), valid_counts=valid_counts), indent=2))
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="Full run directory containing analysis/per_model.csv")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plot(args.run, args.output)
