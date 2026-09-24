"""Summarize and plot the serial one-extension Benders/MILP benchmark.

Read-only with respect to benchmark reports and solver sources. Timed-out and
uncertified solves remain in summaries and runtime plots; their elapsed times
are not presented as times to certification. No optimization is performed.
"""

from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/wzzt_benders_scaling_20260922"
METHODS = ("original_milp", "benders")
LABELS = {"original_milp": "Monolithic MILP", "benders": "Benders prototype"}
COLORS = {"original_milp": "#426887", "benders": "#BD663E"}


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_events(record, report_path):
    if record.get("events"):
        return record["events"], "report"
    candidates = []
    if record.get("events_file"):
        candidates.append(Path(record["events_file"]))
    result_path = record.get("specification", {}).get("result_path")
    if result_path:
        candidates.append(Path(result_path).with_suffix(".events.jsonl"))
    for path in candidates:
        if not path.is_absolute():
            path = report_path.parent / path
        if path.exists():
            events = []
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if line.strip():
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"Invalid event JSON: {path}:{number}") from exc
            return events, str(path)
    return [], "unavailable"


def final_or_last(record, events, key):
    if finite(record.get(key)):
        return float(record[key])
    for event in reversed(events):
        if finite(event.get(key)):
            return float(event[key])
    return None


def first_event_time(events, predicate):
    for event in events:
        seconds = event.get("elapsed_seconds")
        if finite(seconds) and predicate(event):
            return float(seconds)
    return None


def summarize_record(record, report_path):
    spec, inputs = record["specification"], record.get("input", {})
    events, event_source = load_events(record, report_path)
    lb = final_or_last(record, events, "lower_bound")
    ub = final_or_last(record, events, "upper_bound")
    wall = record.get("wall_time_seconds")
    solver_time = record.get("solver_time_seconds")
    truth = inputs.get("generation_only_true_new_support")
    target = tuple(truth) if truth is not None else None
    method = spec["method"]
    subproblem_events = [event for event in events if event.get("event") == "subproblem"]
    master_events = [event for event in events if event.get("event") == "master"]
    # A final event alone does not reveal when the incumbent was first found.
    first_final_ub = None if ub is None or method != "benders" else first_event_time(
        subproblem_events,
        lambda event: finite(event.get("upper_bound"))
        and abs(event["upper_bound"] - ub) <= 1e-10 + 1e-8 * abs(ub),
    )
    first_truth_master = None if target is None else first_event_time(
        master_events, lambda event: tuple(event.get("support") or ()) == target,
    )
    first_truth_lp = None if target is None else first_event_time(
        subproblem_events, lambda event: tuple(event.get("support") or ()) == target,
    )
    initial_ub = next((float(event["upper_bound"]) for event in events
                       if finite(event.get("upper_bound"))), None)
    master_time = record.get("master_time_seconds")
    if master_time is None and method == "benders":
        master_time = sum(event.get("call_seconds", 0) for event in master_events)
    subproblem_time = record.get("subproblem_time_seconds")
    if subproblem_time is None and method == "benders":
        subproblem_time = sum(event.get("call_seconds", 0) for event in events
                              if str(event.get("event", "")).startswith("subproblem"))
    row = {
        "source_report": str(report_path), "case_id": spec["case_id"],
        "case_family": spec["case_family"], "method": method,
        "n": inputs.get("n", spec.get("n")), "repeat": spec.get("repeat"),
        "seed": spec.get("case_seed"),
        "samples_per_scenario": inputs.get("samples_per_scenario", spec.get("samples_per_scenario")),
        "scenario_count": inputs.get("scenario_count"),
        "observation_count": inputs.get("observation_count"),
        "time_budget_seconds": spec["time_limit"], "status": record.get("status"),
        "stop_reason": record.get("stop_reason", record.get("error")),
        "proven_optimal": bool(record.get("proven_optimal", False)),
        "wall_time_seconds": wall,
        "watchdog_limit_seconds": record.get("watchdog_seconds"),
        "solver_time_seconds": solver_time, "master_time_seconds": master_time,
        "subproblem_time_seconds": subproblem_time,
        "non_solver_wall_seconds": max(0, wall - solver_time)
        if finite(wall) and finite(solver_time) else None,
        "lower_bound": lb, "upper_bound": ub,
        "absolute_gap": record.get("absolute_gap"),
        "signed_upper_minus_lower": None if lb is None or ub is None else ub - lb,
        "relative_gap": record.get("relative_gap"),
        "initial_upper_bound": initial_ub,
        "first_final_upper_bound_seconds": first_final_ub,
        "final_upper_bound_match_tolerance": "1e-10 + 1e-8 * abs(final UB)",
        "first_generation_support_selected_seconds": first_truth_master,
        "first_generation_support_evaluated_seconds": first_truth_lp,
        "generation_support": truth,
        "generation_support_is_optimality_evidence": False,
        "incumbent_support": (record.get("incumbent") or {}).get("support"),
        "matches_generation_support": (record.get("incumbent") or {}).get("matches_generation_support"),
        "independent_mae": (record.get("incumbent") or {}).get("independent_mae"),
        "master_count": record.get("master_count", len(master_events) if method == "benders" else None),
        "lp_attempt_count": record.get("lp_count"),
        "lp_completed_count": len(subproblem_events) if method == "benders" else None,
        "cut_count": record.get("cut_count"), "node_count": record.get("node_count"),
        "event_count": len(events), "event_source": event_source,
        "source_integrity_unchanged": record.get("source_integrity", {}).get("unchanged"),
    }
    return row, events


def common_sample_count(row):
    samples = row["samples_per_scenario"]
    if isinstance(samples, (list, tuple)):
        return samples[0] if samples and all(value == samples[0] for value in samples) else None
    return samples


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["case_family"], row["n"], common_sample_count(row), row["method"])].append(row)
    result = []
    for (family, n, samples, method), items in sorted(groups.items(), key=lambda item: str(item[0])):
        times = [row["wall_time_seconds"] for row in items]
        all_timed = all(finite(value) for value in times)
        all_certified = all(row["proven_optimal"] for row in items)
        median = statistics.median(times) if all_timed else None
        result.append({
            "case_family": family, "n": n, "samples_per_scenario": samples, "method": method,
            "run_count": len(items), "certified_count": sum(row["proven_optimal"] for row in items),
            "unfinished_or_failed_count": sum(not row["proven_optimal"] for row in items),
            "all_wall_times_available": all_timed,
            "median_stop_seconds_all_runs": median,
            "median_time_to_certification_seconds": median if all_certified else None,
            "median_stop_time_is_certification_time": all_certified and all_timed,
            "wall_times_seconds": times,
            "budgets_seconds": sorted({row["time_budget_seconds"] for row in items}),
        })
    return result


def paired_comparisons(rows):
    pairs = defaultdict(dict)
    for row in rows:
        key = (row["source_report"], row["case_id"])
        if row["method"] in pairs[key]:
            raise ValueError(f"Duplicate method record: {key} {row['method']}")
        pairs[key][row["method"]] = row
    results = []
    for (report, case), methods in pairs.items():
        native, benders = methods.get("original_milp"), methods.get("benders")
        if native is None or benders is None:
            results.append({"source_report": report, "case_id": case, "pair_complete": False})
            continue
        a, b = native["upper_bound"], benders["upper_bound"]
        signed_diff = None if a is None or b is None else b - a
        both = native["proven_optimal"] and benders["proven_optimal"]
        at, bt = native["wall_time_seconds"], benders["wall_time_seconds"]
        ratio = bt / at if finite(at) and finite(bt) and at > 0 else None
        results.append({
            "source_report": report, "case_id": case, "pair_complete": True,
            "both_certified": both,
            "benders_minus_native_incumbent_objective": signed_diff,
            "incumbent_objective_absolute_difference": None if signed_diff is None else abs(signed_diff),
            "benders_minus_native_relative_to_native_incumbent":
                None if signed_diff is None else signed_diff / max(abs(a), 1e-15),
            "objective_difference_is_between_proven_optima": both,
            "equal_optimum_within_1e_minus_7": None if not both or signed_diff is None else abs(signed_diff) <= 1e-7,
            "benders_over_native_stop_time_ratio": ratio,
            "benders_over_native_certification_time_ratio": ratio if both else None,
        })
    return results


def plot_runtime(rows, output_dir, plt, Line2D):
    chosen = [row for row in rows if row["case_family"] == "synthetic_single_clade"
              and common_sample_count(row) == 24 and row["n"] is not None]
    sizes = sorted({row["n"] for row in chosen})
    fig, ax = plt.subplots(figsize=(8.4, 4.8), constrained_layout=True)
    if not sizes:
        ax.text(0.5, 0.5, "No synthetic cases with 24 samples per scenario", ha="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        groups = defaultdict(list)
        for row in chosen:
            groups[(row["n"], row["method"])].append(row)
        for index, n in enumerate(sizes):
            budgets = sorted({row["time_budget_seconds"] for row in chosen if row["n"] == n})
            for budget in budgets:
                ax.plot([index - 0.38, index + 0.38], [budget, budget], color="#929292", linestyle=":", linewidth=1.1)
                ax.annotate(f"{budget:g}s budget", (index + 0.38, budget), xytext=(0, 4), textcoords="offset points", ha="right", fontsize=7, color="#666666")
            for method_index, method in enumerate(METHODS):
                items = groups.get((n, method), [])
                center = index + (-0.13 if method_index == 0 else 0.13)
                times = []
                for repeat_index, row in enumerate(items):
                    shift = (repeat_index - (len(items) - 1) / 2) * 0.055
                    seconds = row["wall_time_seconds"]
                    known_time = finite(seconds) and seconds > 0
                    if known_time:
                        times.append(seconds)
                    else:
                        seconds = row["time_budget_seconds"]
                    marker = "o" if row["proven_optimal"] else "x"
                    ax.scatter(center + shift, seconds, c=COLORS[method], marker=marker,
                               s=38, linewidths=1.3, alpha=0.85, zorder=3)
                    if not known_time:
                        ax.annotate("no wall time", (center + shift, seconds), xytext=(0, -12),
                                    textcoords="offset points", ha="center", fontsize=6)
                if items and len(times) == len(items):
                    median = statistics.median(times)
                    ax.plot([center - 0.085, center + 0.085], [median, median],
                            color=COLORS[method], linewidth=2.2, zorder=4)
        ax.set_xticks(range(len(sizes)), sizes)
        ax.set_yscale("log")
        ax.set_xlabel("Terminal count n")
        ax.set_ylabel("Observed stop time (s; logarithmic scale)")
        ax.set_title("One extension: synthetic observations, 2 scenarios × 24 samples\n"
                     "Stopping time includes certification effort; it is not first-good-incumbent time", fontsize=10)
        ax.grid(axis="y", alpha=0.2, which="major")
        ax.set_xlim(-0.6, len(sizes) - 0.4)
        handles = [Line2D([], [], color=COLORS[method], linewidth=2, label=LABELS[method]) for method in METHODS]
        handles += [Line2D([], [], color="#444444", marker="o", linestyle="none", label="Certified run"),
                    Line2D([], [], color="#444444", marker="x", linestyle="none", label="Uncertified / failed run"),
                    Line2D([], [], color="#444444", linewidth=2, label="Median stop time, all runs")]
        fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False, fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"runtime_scaling.{suffix}", dpi=220)
    plt.close(fig)


def plot_convergence(rows, events_by_record, output_dir, plt, Line2D):
    requests = [("synthetic_single_clade", 8), ("synthetic_single_clade", 12),
                ("synthetic_single_clade", 18), ("synthetic_single_clade", 32),
                ("archived_ac", 11), ("archived_ac", 18)]
    fig, axes = plt.subplots(3, 2, figsize=(11.0, 10.0), constrained_layout=True)
    selected = []
    for ax, (family, n) in zip(axes.flat, requests):
        candidates = [row for row in rows if row["method"] == "benders" and row["case_family"] == family
                      and row["n"] == n and (family != "synthetic_single_clade" or common_sample_count(row) == 24)]
        candidates.sort(key=lambda row: (row["repeat"] if row["repeat"] is not None else 10**9, row["source_report"]))
        if not candidates:
            ax.text(0.5, 0.5, f"No report: {family}, n={n}", ha="center", transform=ax.transAxes, fontsize=9)
            ax.set_axis_off()
            selected.append({"family": family, "n": n, "available": False})
            continue
        row = candidates[0]
        native = next((item for item in rows if item["method"] == "original_milp"
                       and item["source_report"] == row["source_report"]
                       and item["case_id"] == row["case_id"]), None)
        events = events_by_record[(row["source_report"], row["case_id"], "benders")]
        scale = row["initial_upper_bound"]
        scale = scale if finite(scale) and scale > 0 else 1.0
        selected.append({"family": family, "n": n, "available": True, "case_id": row["case_id"],
                         "source_report": row["source_report"], "normalization": scale,
                         "normalization_rule": "first finite positive UB; fallback 1",
                         "native_final_upper_bound": None if native is None else native["upper_bound"],
                         "native_final_lower_bound": None if native is None else native["lower_bound"]})
        if native is not None:
            for key, color, style in (("lower_bound", "#5E8660", ":"), ("upper_bound", "#7F6F9F", "-.")):
                if finite(native.get(key)):
                    ax.axhline(native[key] / scale, color=color, linestyle=style, linewidth=1.3)
        for key, color, style in (("lower_bound", "#426887", "--"), ("upper_bound", "#BD663E", "-")):
            points = [(event["elapsed_seconds"], event[key] / scale) for event in events
                      if finite(event.get("elapsed_seconds")) and finite(event.get(key))]
            if points:
                times, values = zip(*points)
                ax.step(times, values, where="post", color=color, linestyle=style, linewidth=1.6)
        for event in events:
            event_type = event.get("event")
            if event_type not in ("master", "subproblem", "final"):
                continue
            key = "lower_bound" if event_type == "master" else "upper_bound"
            if finite(event.get(key)) and finite(event.get("elapsed_seconds")):
                marker = {"master": "s", "subproblem": "o", "final": "D"}[event_type]
                color = "#426887" if event_type == "master" else "#BD663E"
                ax.plot(event["elapsed_seconds"], event[key] / scale, marker=marker,
                        markersize=3.5 if event_type != "final" else 5,
                        markerfacecolor="white", markeredgecolor=color, linestyle="none", alpha=0.85)
        wall = row["wall_time_seconds"]
        budget = row["time_budget_seconds"]
        xmax = max(wall if finite(wall) else budget, 0.01)
        ax.set_xlim(0, xmax * 1.025)
        ax.set_ylim(-0.025, 1.05)
        label = "Archived AC" if family == "archived_ac" else "Synthetic"
        state = "certified" if row["proven_optimal"] else "uncertified"
        ax.set_title(f"{label}, n={n}; {budget:g}s budget; {state}\n"
                     f"{row['master_count']} masters / {row['lp_completed_count']} completed LPs", fontsize=9)
        ax.set_xlabel("Elapsed time (s)", fontsize=9)
        ax.set_ylabel("Bound / first finite UB", fontsize=9)
        ax.grid(alpha=0.2)
        native_ub = None if native is None else native["upper_bound"]
        benders_ub = row["upper_bound"]
        native_label = "no incumbent" if native_ub is None else f"{native_ub:.4g}"
        benders_label = "no incumbent" if benders_ub is None else f"{benders_ub:.4g}"
        ax.text(0.97, 0.92, f"Initial UB = {scale:.3g}\nFinal UB: Benders {benders_label}; MILP {native_label}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 2})
    handles = [Line2D([], [], color="#426887", linestyle="--", label="Global lower bound"),
               Line2D([], [], color="#BD663E", label="Best feasible upper bound"),
               Line2D([], [], color="#555555", marker="s", markerfacecolor="white", linestyle="none", label="Master solve"),
               Line2D([], [], color="#555555", marker="o", markerfacecolor="white", linestyle="none", label="LP solve"),
               Line2D([], [], color="#555555", marker="D", markerfacecolor="white", linestyle="none", label="Final event"),
               Line2D([], [], color="#5E8660", linestyle=":", label="MILP final LB (same instance)"),
               Line2D([], [], color="#7F6F9F", linestyle="-.", label="MILP final UB (same instance)")]
    fig.legend(handles=handles, loc="outside lower center", ncols=3, frameon=False, fontsize=8)
    fig.suptitle("Benders bound trajectories for one full-domain extension", fontsize=12)
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"bound_trajectories.{suffix}", dpi=220)
    plt.close(fig)
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, nargs="+",
                        default=[BASE / part / "benchmark_report.json" for part in ("main", "larger", "archive", "samples")],
                        help="One or more completed or partial benchmark reports; missing files are errors.")
    parser.add_argument("--output-dir", type=Path, default=BASE / "figures")
    args = parser.parse_args()
    paths = [path.resolve() for path in args.reports]
    for path in paths:
        if not path.is_file():
            parser.error(f"Report does not exist: {path}")
    if len(set(paths)) != len(paths):
        parser.error("Duplicate report paths are not allowed")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows, events_by_record, provenance = [], {}, []
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(report.get("cases"), list):
            parser.error(f"Report lacks cases list: {path}")
        provenance.append({"path": str(path), "sha256": sha256(path), "case_method_count": len(report["cases"]),
                           "scope": report.get("scope"), "source_integrity": report.get("source_integrity"),
                           "certification_tolerance": report.get("certification_tolerance")})
        for record in report["cases"]:
            row, events = summarize_record(record, path)
            rows.append(row)
            key = (row["source_report"], row["case_id"], row["method"])
            if key in events_by_record:
                parser.error(f"Duplicate record: {key}")
            events_by_record[key] = events
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    plot_runtime(rows, output, plt, Line2D)
    trajectory_cases = plot_convergence(rows, events_by_record, output, plt, Line2D)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Full-domain one extension from singleton supports; not complete topology recovery.",
        "notes": [
            "All failed, interrupted, and uncertified runs are retained.",
            "Median stop time is called certification time only when every run in the group certified.",
            "Native solve event exposes final bounds only; its first incumbent time is unavailable.",
            "First attainment of the final UB is an observed event time, not a proof time.",
            "Generation support is used only for retrospective synthetic-case diagnostics.",
            "Paired objective differences refer to incumbents unless both methods certified.",
            "Trajectories use all finite master, subproblem and final events, including final master-only certification.",
            "Plots use the first repeat for each requested convergence panel; all repeats remain in summary.",
        ],
        "reports": provenance, "records": rows, "groups": aggregate(rows),
        "comparisons": paired_comparisons(rows), "trajectory_panels": trajectory_cases,
        "certification_counts": {
            method: {"certified": sum(row["proven_optimal"] for row in rows if row["method"] == method),
                     "total": sum(row["method"] == method for row in rows)} for method in METHODS
        },
        "script_sha256": sha256(__file__), "matplotlib_version": matplotlib.__version__,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "records": len(rows),
                      "certification_counts": summary["certification_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
