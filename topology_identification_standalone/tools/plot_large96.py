"""Plot sealed strict-96 analysis, without reading observations or optimizers.

Usage: python tools/plot_large96.py RUN [--output FIGURES]
Only independent analysis tables are read. Figure denominators retain failed,
interrupted, invalid and over-budget runs; unknown quantities are shown as NA.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np


ORDER = ("rnj", "nj", "000", "100", "010", "001", "110", "101", "011", "111")
DISPLAY = ("rnj", "nj", "000", "100", "101", "111")
CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
NAMES = {"rnj": "RNJ", "nj": "NJ", "000": "WZZT", "100": "R + W",
         "010": "W + T", "001": "W + E", "110": "R + W + T",
         "101": "R + W + E", "011": "W + T + E", "111": "R + W + T + E"}
PALETTE = ("#0072B2", "#E69F00", "#009E73", "#56B4E9", "#D55E00", "#CC79A7")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def table(path):
    path = Path(path)
    if not path.exists() or path.stat().st_size <= 1:
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(row, key, default=np.nan):
    value = row.get(key)
    if value in (None, "", "None", "null"):
        return default
    try:
        answer = float(value)
    except (ValueError, TypeError):
        return default
    return answer if np.isfinite(answer) else default


def boolean(value):
    return str(value).lower() in {"true", "1", "yes"}


def verified(row):
    return row.get("status") == "verified"


def rate(numerator, denominator):
    return numerator/denominator if denominator else np.nan


def plot(run, output=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run = Path(run).resolve()
    analysis = run/"analysis" if (run/"analysis").is_dir() else run
    output = Path(output).resolve() if output else run/"figures"
    rows = table(analysis/"results.csv")
    summary = read_json(analysis/"summary.json")
    verification_path = analysis/"verification.json"
    verification = read_json(verification_path) if verification_path.exists() else summary
    if not verification.get("verified", verification.get("passed", False)):
        raise ValueError("Independent analysis must pass before plotting")
    if not rows:
        raise ValueError("A nonempty analysis/results.csv is required")
    method_rows = {row["method"]: row for row in summary["methods"]}
    methods = [m for m in ORDER if m in method_rows]
    if not methods:
        raise ValueError("No recognized methods are present")
    for method in methods:
        group = [r for r in rows if r["method"] == method]
        counted = sum(verified(r) for r in group)
        if len(group) != int(method_rows[method]["planned"]) or counted != int(method_rows[method]["verified"]):
            raise ValueError("Summary and result denominators disagree: "+method)
        if any(not 0 <= number(r, "F1") <= 1 or r.get("exact") in (None, "") for r in group if verified(r)):
            raise ValueError("A verified result has no finite F1/exact score: "+method)
    shown = [m for m in DISPLAY if m in methods]
    protocol_path = run/"protocol.json"
    protocol = read_json(protocol_path) if protocol_path.exists() else {}
    fixture = bool(protocol.get("synthetic_fixture", summary.get("synthetic_fixture", False)))
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "svg.fonttype": "none"})
    written, captions, skipped = [], {}, []

    def save(fig, name, caption, *, bottom=.14):
        if fixture:
            fig.text(.5, 1.025, "SYNTHETIC SCHEMA FIXTURE - NOT BENCHMARK RESULTS",
                     ha="center", va="bottom", color="#A00000", weight="bold", fontsize=9)
        fig.text(.012, .018, caption, fontsize=7, va="bottom", wrap=True)
        fig.tight_layout(rect=(0, bottom, 1, .96))
        for suffix in ("png", "pdf", "svg"):
            target = output/(name+"."+suffix)
            fig.savefig(target, dpi=220, bbox_inches="tight")
            written.append(target)
        captions[name] = caption
        plt.close(fig)

    def method_summary(selected, filename, title):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
        labels = [NAMES[m] for m in selected]
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(selected))]
        keys = ("F1_verified_mean", "exact_rate_planned", "total_seconds_recorded")
        for column, (ax, key) in enumerate(zip(axes, keys, strict=True)):
            vals = np.array([number(method_rows[m], key) for m in selected])
            if column == 2:
                vals /= 3600
            ax.bar(range(len(selected)), vals, color=colors)
            ax.set_xticks(range(len(selected)), labels, rotation=32, ha="right")
            ax.grid(axis="y", alpha=.2)
            if column < 2:
                ax.set_ylim(0, 1.16)
            elif np.any(vals > 0):
                ax.set_yscale("log")
                positive = vals[vals > 0]
                ax.set_ylim(float(positive.min())*.45, float(positive.max())*3)
            ax.set_ylabel(("F1: verified inputs", "Exact recovery / planned", "Recorded charged method-hours (log)")[column])
            for index, method in enumerate(selected):
                row = method_rows[method]
                valid, planned = int(row["verified"]), int(row["planned"])
                value = vals[index]
                if np.isfinite(value) and (column < 2 or value > 0):
                    text = f"{value:.3f}\nn={valid}/{planned}" if column == 0 else (
                        f"{int(row['exact'])}/{planned}" if column == 1 else f"{value:.3g}")
                    ax.annotate(text, (index, value), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7)
                else:
                    ax.text(index, .03, "NA" if not np.isfinite(value) else "0", transform=ax.get_xaxis_transform(), ha="center", fontsize=7)
                if column == 0:
                    zero_mean = number(row, "F1_failure_zero_mean")
                    if np.isfinite(zero_mean):
                        ax.plot(index, zero_mean, marker="D", ms=3, color="black")
                if column == 2 and int(row.get("timing_unknown", 0)):
                    ax.text(index, .03, f"unknown={row['timing_unknown']}", transform=ax.get_xaxis_transform(), rotation=90, ha="center", fontsize=6)
        fig.suptitle(title)
        save(fig, filename,
             "F1 bars use verified runs; black diamonds count failures as F1=0. Exact recovery retains every planned run.\n"
             "Recorded time includes returned failures; unknown times are omitted and labelled. Confidence intervals use clustered report tables.")

    method_summary(shown, "main_comparison", "Strict 96-point comparison: frozen selection-MSE primary")

    case_order = [c for c in CASES if any(r["case"] == c for r in rows)]
    case_order += sorted({r["case"] for r in rows}-set(case_order))
    sigmas = sorted({number(r, "voltage_sigma") for r in rows if np.isfinite(number(r, "voltage_sigma"))})
    use_noise = len(sigmas) > 1
    conditions = sigmas if use_noise else sorted({(r.get("profile", "baseline"), r.get("level", "1")) for r in rows})
    labels = [f"{s:g}" for s in sigmas] if use_noise else [f"{p}\n{l}" for p, l in conditions]
    fig, axes = plt.subplots(len(case_order), 2, figsize=(12, max(4., 2.55*len(case_order))), squeeze=False)
    for i, case in enumerate(case_order):
        for j, method in enumerate(shown):
            means, exact_rates, denoms = [], [], []
            for condition in conditions:
                group = [r for r in rows if r["case"] == case and r["method"] == method and (
                    number(r, "voltage_sigma") == condition if use_noise else
                    (r.get("profile", "baseline"), r.get("level", "1")) == condition)]
                valid = [r for r in group if verified(r)]
                means.append(float(np.mean([number(r, "F1") for r in valid])) if valid else np.nan)
                exact_rates.append(rate(sum(boolean(r.get("exact")) for r in valid), len(group)))
                denoms.append((len(valid), len(group)))
            color = PALETTE[j % len(PALETTE)]
            for column, values in enumerate((means, exact_rates)):
                axes[i, column].plot(range(len(conditions)), values, "o-", color=color, label=NAMES[method], ms=3)
            for index, ((valid, planned), value) in enumerate(zip(denoms, means, strict=True)):
                if valid < planned:
                    axes[i, 0].annotate(f"{valid}/{planned}", (index, value if np.isfinite(value) else .02), xytext=(2, 3), textcoords="offset points", color=color, fontsize=6)
        for column, ax in enumerate(axes[i]):
            ax.set_title(case)
            ax.set_ylim(-.025, 1.06)
            ax.set_xticks(range(len(conditions)), labels, rotation=0 if use_noise else 32, ha="center" if use_noise else "right")
            ax.set_xlabel("Voltage / root-meter noise sigma" if use_noise else "Predeclared stress profile / level")
            ax.set_ylabel("F1: verified" if column == 0 else "Exact / planned in stratum")
            ax.grid(alpha=.2)
    axes[0, 0].legend(loc="lower left", fontsize=7)
    save(fig, "case_noise" if use_noise else "case_profile",
         "Methods and noise siblings share each day's trajectory. Daily replicate groups, not condition rows, are independent repeats.\n"
         "Lines are descriptive stratum means. Failed/invalid/over-budget runs remain in exact-recovery denominators.", bottom=.09)

    dr_rows = table(analysis/"dr_curve.csv")
    if dr_rows:
        kappas = sorted({number(r, "kappa") for r in dr_rows if np.isfinite(number(r, "kappa"))})
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        wrong_pair = "wrong_truth_pair_rate" in dr_rows[0]
        triggered_wrong_pair = "triggered_wrong_truth_pair_rate" in dr_rows[0]
        dr_keys = (("trigger_rate_eligible", "Pair emitted / eligible"),
                   ("wrong_capture_rate", "Wrong primaries flagged / wrong primaries"),
                   (("wrong_truth_pair_rate", "Wrong primaries with truth in pair / wrong primaries") if wrong_pair else
                    ("returned_truth_rate", "Truth in returned set / eligible (legacy schema)")),
                   (("triggered_wrong_truth_pair_rate", "Truth in pair / triggered wrong primaries") if triggered_wrong_pair else
                    ("triggered_truth_coverage_rate", "Truth in pair / emitted pairs (legacy schema)")))
        for ax, (key, ylabel) in zip(axes.flat, dr_keys, strict=True):
            has_rates = False
            for j, method in enumerate(shown):
                group = [r for r in dr_rows if r.get("method") == method]
                values = [next((number(r, key) for r in group if number(r, "kappa") == k), np.nan) for k in kappas]
                has_rates = has_rates or any(np.isfinite(values))
                ax.plot(range(len(kappas)), values, "o-", ms=4, color=PALETTE[j % len(PALETTE)], label=NAMES[method])
            if not has_rates:
                ax.text(.5, .5, "NA: no defined rates", transform=ax.transAxes, ha="center", va="center")
            ax.set_xticks(range(len(kappas)), [f"{k:g}" for k in kappas])
            ax.set_xlabel("Declared output kappa grid")
            ax.set_ylabel(ylabel)
            ax.set_ylim(-.03, 1.05)
            ax.grid(alpha=.2)
        axes[0, 0].legend(loc="best", fontsize=7)
        fig.suptitle("Conditional competing topology output: review load and saved-library coverage")
        save(fig, "dr_tradeoff", "Kappa changes the returned set, never the primary topology or its F1. No DR intervention was run.\n"
             "Wrong-primary pair coverage measures an available correct alternative, not observed DR recovery. Undefined rates stay NA; no confidence levels.", bottom=.105)
    else:
        skipped.append("dr_tradeoff: no dr_curve.csv rows")

    fig, ax = plt.subplots(figsize=(11, 4.8))
    categories = (("two_near_tied", "Wrong primary: pair emitted", "#E69F00"),
                  ("single_separated", "Wrong primary: single separated", "#D55E00"),
                  ("single_no_saved_alternative", "Wrong primary: no saved alternative", "#CC79A7"),
                  ("unknown", "Wrong primary: unclassified output", "#777777"))
    bottom = np.zeros(len(shown), int)
    for decision, label, color in categories:
        values = []
        for method in shown:
            wrong = [r for r in rows if r["method"] == method and verified(r) and not boolean(r.get("exact"))]
            values.append(sum(r.get("decision") == decision if decision != "unknown" else
                              r.get("decision") not in {c[0] for c in categories[:-1]} for r in wrong))
        ax.bar(range(len(shown)), values, bottom=bottom, color=color, label=label)
        bottom += np.asarray(values, int)
    ax.set_xticks(range(len(shown)), [NAMES[m] for m in shown], rotation=25, ha="right")
    ax.set_ylabel("Wrong-primary count among verified runs")
    ax.set_ylim(0, max(1., float(bottom.max())*1.15))
    ax.grid(axis="y", alpha=.2)
    for index, total in enumerate(bottom):
        ax.annotate(str(total), (index, total), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)
    save(fig, "single_error_breakdown", "A separated saved alternative and no saved alternative are different outcomes. Neither proves global identifiability.\n"
         "Failed, interrupted, invalid and over-budget runs have no verified primary and are excluded from this descriptive error breakdown.")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    columns = ((0, 0), (1, 0), (0, 1), (1, 1))
    for ax, (key, title) in zip(axes, (("F1_verified_mean", "F1: verified inputs"),
                                      ("exact_rate_planned", "Exact recovery / planned")), strict=True):
        matrix = np.full((2, 4), np.nan)
        annotations = {}
        for e in (0, 1):
            for col, (r, t) in enumerate(columns):
                method = f"{r}{t}{e}"
                item = method_rows.get(method)
                if item:
                    matrix[e, col] = number(item, key)
                    annotations[e, col] = f"{method}\n{matrix[e, col]:.3f}" if np.isfinite(matrix[e, col]) else f"{method}\nNA"
                    annotations[e, col] += f"\nn={item['verified']}/{item['planned']}"
                else:
                    annotations[e, col] = f"{method}\nnot planned"
        cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("#DDDDDD")
        image = ax.imshow(np.ma.masked_invalid(matrix), vmin=0, vmax=1, cmap=cmap, aspect="auto")
        for (e, col), label in annotations.items():
            value = matrix[e, col]
            ax.text(col, e, label, ha="center", va="center", fontsize=8,
                    color="white" if np.isfinite(value) and value < .6 else "black")
        ax.set_xticks(range(4), [f"R{r} T{t}" for r, t in columns])
        ax.set_yticks([0, 1], ["E0", "E1"])
        ax.set_title(title)
        fig.colorbar(image, ax=ax, shrink=.8)
    fig.suptitle("R/T/E factorial cells (bit order R, T, E)")
    save(fig, "rte_heatmap", "Gray cells are not planned or have no verified F1; they are never filled with another method's result.\n"
         "Module effects include the declared algorithm's stopping and budget allocation policy.")

    effect_rows = table(analysis/"ablation_effects.csv")
    if effect_rows:
        fig, axes = plt.subplots(1, 3, figsize=(14, max(4.5, .34*len(effect_rows)+1.4)))
        labels = [f"{r['module']}: {r['off']} -> {r['on']}" for r in effect_rows]
        for ax, key, scale, xlabel in zip(axes,
                ("delta_F1_mean", "delta_exact_mean", "delta_seconds_mean"), (1., 100., 1.),
                ("Paired delta F1", "Paired delta exact (percentage points)", "Paired delta charged seconds"), strict=True):
            values = np.array([number(r, key)*scale for r in effect_rows])
            ax.barh(range(len(labels)), values, color=["#0072B2" if v >= 0 else "#D55E00" for v in values])
            ax.axvline(0, color="black", lw=.7)
            ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis()
            ax.set_xlabel(xlabel); ax.grid(axis="x", alpha=.2)
            for i, (row, value) in enumerate(zip(effect_rows, values, strict=True)):
                text = f" n={row['paired']}/{row['planned']}" if np.isfinite(value) else f" NA n={row['paired']}/{row['planned']}"
                ax.annotate(text, (value if np.isfinite(value) else 0., i), va="center", fontsize=6)
        fig.suptitle("Conditional module effects: verified paired inputs")
        save(fig, "ablation_effects", "Only pairs with both endpoints independently verified contribute to deltas; planned and paired denominators are displayed.\n"
             "These bars are descriptive paired means. Clustered uncertainty is reported separately; module order is not re-executed here.", bottom=.11)
    else:
        skipped.append("ablation_effects: no ablation_effects.csv rows")

    used = [analysis/name for name in ("results.csv", "summary.json", "dr_curve.csv", "ablation_effects.csv", "verification.json")
            if (analysis/name).exists()]
    if protocol_path.exists():
        used.append(protocol_path)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
                    input_sha256={str(p.relative_to(run)): sha(p) for p in used},
                    figure_sha256={p.name: sha(p) for p in written}, captions=captions,
                    figures=len(captions), exports=len(written), methods=methods,
                    planned_by_method={m: int(method_rows[m]["planned"]) for m in methods},
                    verified_by_method={m: int(method_rows[m]["verified"]) for m in methods},
                    synthetic_fixture=fixture, skipped=skipped, optimizer_imported=False,
                    observations_or_truth_opened=False,
                    scope="Descriptive plots of independently verified finite panel; no new inference, refitting or DR intervention")
    (output/"plot_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path)
    result = plot(**vars(parser.parse_args()))
    print(json.dumps({key: result[key] for key in ("figures", "exports", "methods", "synthetic_fixture", "skipped")}, indent=2))
