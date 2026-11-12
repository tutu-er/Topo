"""Cross-check old audit metrics and publish only non-private summaries."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil

import numpy as np

from ... import io
from ..posterior_mse_20261112.deliver import write_csv
from .history_scan import ARCHIVES, BATCHES, BETAS, ROOT, require, sha, utc


def yes(value):
    return str(value).lower() in ("true", "1")


def audit(run):
    protocol = io.read(run/"protocol.json"); seal = io.read(run/"selection_seal.json")
    source_audit = io.read(run/"source_audit.json")
    require(sha(run/"protocol.json") == seal["protocol_sha256"], "Protocol changed")
    require(sha(run/"source_audit.json") == seal["source_audit_sha256"], "Source receipt changed")
    for path, expected in source_audit["sources_checked"].items():
        require(sha(path) == expected, "Consumed source/input/result changed")
    for ref in protocol["source_references"].values():
        require(sha(ref["path"]) == ref["sha256"], "Historical scoring source changed")
    evaluation = {r["id"]: r for r in io.read(run/"evaluation.json")}
    manifest = io.read(ARCHIVES/"manifest.json")
    baselines = []; checked = []; maximum_error = 0.
    for cohort, (name, _, _) in BATCHES.items():
        batch = Path(source_audit["batches"][cohort]["path"])
        path = batch/"analysis/results.csv"
        require(sha(path) == manifest["members"][name+"/analysis/results.csv"]["sha256"], "Original audited CSV differs")
        with path.open(encoding="utf-8-sig", newline="") as f:
            old = list(csv.DictReader(f))
        grouped = defaultdict(list)
        for row in old:
            grouped[row["method"]].append(row)
            identity = f"{cohort}__{row['job']}__{row['method']}"
            if identity not in evaluation:
                continue
            new = evaluation[identity]
            require(new["native_correct"] == yes(row["exact"]), "Original selected topology grading differs")
            require(new["truth_covered"] == yes(row["library_truth_present"]), "Original frozen library coverage differs")
            checked.append(identity)
        for method, values in grouped.items():
            baselines.append(dict(cohort=cohort, method=method, planned=len(values),
                exact=sum(yes(r["exact"]) for r in values), covered=sum(yes(r["library_truth_present"]) for r in values),
                total_fresh_seconds=sum(float(r.get("fresh_seconds") or 0) for r in values),
                invalid=sum(not yes(r["model_audit_passed"]) for r in values)))
    require(len(checked) == 800, "Not all 800 original metrics were cross-checked")
    for identity, expected in seal["scores"].items():
        path = run/"scores"/(identity+".json")
        require(sha(path) == expected, "Sealed weights differ")
        scored = io.read(path)
        require(scored["status"] == "complete" and scored["independent_verification"]["status"] == "passed", "Incomplete independent score")
        maximum_error = max(maximum_error, scored["independent_verification"]["maximum_probability_error"])
    receipt = dict(status="passed", created_utc=utc(), libraries_checked=800,
        owning_coverage_and_native_map_matches=800, maximum_independent_probability_error=maximum_error,
        source_inputs_rechecked=len(source_audit["sources_checked"]), source=io.ref(__file__),
        protocol_sha256=sha(run/"protocol.json"), selection_seal_sha256=sha(run/"selection_seal.json"))
    io.write(run/"independent_audit.json", receipt)
    write_csv(run/"public/algorithm_baselines.csv", baselines)
    io.write(run/"public/INDEPENDENT_AUDIT.json", {k: v for k, v in receipt.items() if k != "source"})
    return receipt


def publish(run, destination):
    audit_receipt = audit(run)
    public = run/"public"
    rows = io.read(run/"evaluation.json")
    development = run/"development"
    if (development/"complete.json").exists():
        values = io.read(development/"results.json")
        summary = io.read(development/"summary.json")
        io.write(public/"development_summary.json", summary)
        flat = []
        for row in values:
            for beta, gate in row["gates"].items():
                flat.append(dict(id=row["id"], case=row["case"], target=row["target"],
                    target_correct=row["target_correct"], historical_weight=row["historical_weight"], beta=float(beta),
                    truth_retained_at_beta_star=row["truth_retained_at_beta_star"],
                    candidate_count=row["candidate_count"], active_competitors=len(gate["allocation"]["active"]),
                    log_uniform_tail_bound=gate["allocation"]["log_tail"],
                    success_lower=gate["bound"]["success_lower"], zero_successes=gate["successes"],
                    paths=gate["paths"], probability=gate["probability"],
                    interval_lower=gate["interval95"][0], interval_upper=gate["interval95"][1],
                    zero_certified=gate["zero_certified"], gap_evidence=gate["gap_evidence"],
                    active_resource_count=gate["active_resource_count"], status=row["status"]))
        write_csv(public/"development_qualification.csv", flat)
        with (public/"REPORT.md").open("a", encoding="utf-8") as f:
            f.write("\n## 零DR开发检查\n\n")
            f.write(f"按冻结结构强度排序及离线资格检查选中{summary['inputs']}输入、{summary['targets']}条件目标。每目标2048条工作模型路径；未打开实际未来AC。\n\n")
            f.write("| 后验门槛 | 零动作充分界通过 | 剩余缺口证据 | 其中目标正确 |\n|---|---:|---:|---:|\n")
            for beta, data in summary["by_beta"].items():
                f.write(f"| {float(beta):.0%} | {data['zero_certified']} | {data['gap_evidence']} | {data['correct_target_gaps']} |\n")
            f.write(f"\n展示门槛按事前登记的模型结果规则锁为{summary['beta_star']:.0%}；门槛选择未读取目标正确标签。独立模型确认、报价和实际AC尚未完成。\n")
    figures(public, rows)
    if destination.exists():
        raise FileExistsError("Existing public delivery must not be overwritten")
    destination.mkdir(parents=True)
    for path in public.iterdir():
        if path.is_file():
            shutil.copyfile(path, destination/path.name)
    io.write(destination/"DELIVERY.json", dict(stage="history_and_zero_development", created_utc=utc(),
        independent_audit_sha256=sha(run/"independent_audit.json"),
        files={p.name: sha(p) for p in destination.iterdir() if p.is_file()},
        raw_data_and_parameters_published=False, optimization_and_physical_stage_complete=False))
    print(json.dumps(audit_receipt, ensure_ascii=False), flush=True)


def figures(public, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    groups = sorted({r["cohort"]+"/"+r["method"] for r in rows})
    for group in groups:
        valid = [r for r in rows if r["status"] == "complete" and r["cohort"]+"/"+r["method"] == group]
        for axis, key in zip(axes.flat, ("correct_singleton", "wrong_singleton", "true_excluded", "size")):
            y = [np.mean([r["sets"][str(b)][key] for r in valid]) for b in BETAS]
            axis.plot(BETAS, y, "o-", label=group)
    for axis, title in zip(axes.flat, ("Correct unique output / all inputs", "Wrong unique output / all inputs",
                                     "True topology excluded / all inputs", "Mean retained candidate count")):
        axis.set(title=title, xlabel="Posterior decision threshold", xticks=BETAS)
        axis.grid(alpha=.25)
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Sealed single-day histories: fixed scale, no new fitting")
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(public/("history_thresholds."+suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True); parser.add_argument("--public", type=Path, required=True)
    args = parser.parse_args(); publish(args.run.resolve(), args.public.resolve())


if __name__ == "__main__":
    main()
