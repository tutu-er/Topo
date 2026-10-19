"""Build concise JSON and LaTeX tables from completed frozen v9 episodes.

No optimization/simulation is run. Action evidence is compared on each case's
fixed INITIAL candidate bank. Different updated bank counts are kept separate.
Partial output explicitly records completed/expected cases.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


POLICIES = ("no_EV", "selected_single", "matched_constant", "fixed_single",
            "selected_mobile", "fixed_mobile")
MODELS = ("independent_box", "shared_gain")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False)+"\n", encoding="utf-8")


def case_name(config):
    n, weak, history, meter, seed = config
    return f'n{n}_{"weak" if weak else "ordinary"}_{history}_{meter}_{seed}'


def family(supports, n):
    return {tuple(sorted(s)) for s in supports if 1 < len(s) < n}


def summarize_case(result, code):
    n, weak, history, meter, seed = result["config"]
    initial = {r["id"] for r in result["proposals"]["candidates"]}
    rows = []
    for row in result["proposals"]["candidates"]:
        rows.append(dict(id=row["id"], rank=row["rank"],
                         origins=row["proposal_origins"],
                         metrics=result["candidate_truth_metrics"][row["id"]]))
    policies = {}
    for label in POLICIES:
        if label not in result["policies"]:
            policies[label] = None
            continue
        source = result["policies"][label]
        key = source["action"]
        counts = {}
        for model in MODELS:
            recorded = result["records"][key][model]
            if not initial <= set(recorded["records"]):
                raise ValueError("Updated bank is missing an initial comparison candidate")
            retained = initial & set(recorded["retained"])
            counts[model] = dict(initial_bank_retained_count=len(retained),
                initial_bank_excluded_count=len(initial)-len(retained),
                initial_bank_retained_ids=sorted(retained),
                updated_bank_count=len(recorded["records"]),
                updated_bank_retained_count=len(recorded["retained"]))
            if "initial_bank_retained_count" in source[model]:
                if len(retained) != source[model]["initial_bank_retained_count"]:
                    raise ValueError("Stored initial-bank comparison count disagrees")
        kind, *sites = key.split("_")
        location = list(map(int, sites)) if kind != "none" else []
        policies[label] = dict(action=key, kind=kind, sites_code_zero_based=location,
            terminal_numbers_one_based=[j+1 for j in location],
            grid_energy_kwh=0 if kind == "none" else 27,
            travel_energy_kwh=2 if kind == "mobile" else 0,
            models=counts,
            post_best_metrics=source.get("post_passive_best_metrics"),
            truth_in_updated_bank=source.get("truth_in_updated_bank"))
    wzzt = result["proposals"]["diagnostics"].get("wzzt", {})
    wfamily = family(wzzt.get("support_indices", []), n)
    matched = None if "support_indices" not in wzzt else next(
        (r for r in result["proposals"]["candidates"] if family(r["supports"], n) == wfamily), None)
    wmetrics = None if matched is None else result["candidate_truth_metrics"][matched["id"]]
    best = rows[0]
    return dict(code=code, case=result["case"], config=result["config"],
        n_terminal=n, weak=weak, history=history, meter=meter, seed=seed,
        initial_bank_count=len(initial), initial_best_metrics=result["passive_best_metrics"],
        initial_best_origins=best["origins"], truth_in_initial_bank=result["truth_in_bank"],
        initial_candidates=rows,
        historical_shared_gain_retained_count=len(result["historical_evidence"]["retained"]),
        historical_status=result["historical_evidence"]["status"],
        design_status=result["design"]["status"],
        used_nonwitness_references=result["design"].get("used_nonwitness_references"),
        references_verified_in_shared_gain_domain=result["design"].get("references_verified_in_shared_gain_domain", False),
        wzzt=dict(called=wzzt.get("called", False), stop_reason=wzzt.get("stop_reason"),
            path_length=wzzt.get("path_length"), bound_active=wzzt.get("bound_active"),
            selected_internal_supports=sorted(wfamily), selected_metrics_if_in_initial_bank=wmetrics,
            accepted_solver_stages=sum(bool(s.get("accepted")) for s in wzzt.get("solver_attempts", [])),
            timed_out_stages=sum(s.get("status") == 1 for s in wzzt.get("solver_attempts", [])),
            global_topology_optimality=False),
        policies=policies, seconds=result["seconds"])


def _cell(value):
    return "--" if value is None else str(value)


def _yes(value):
    return "--" if value is None else ("是" if value else "否")


def _post_cell(policy):
    if policy is None or policy["post_best_metrics"] is None:
        return "--"
    return f'{policy["post_best_metrics"]["f1"]:.2f}/{_yes(policy["truth_in_updated_bank"])}'


def _site_cell(policy):
    if policy is None:
        return "--"
    return ",".join(map(str, policy["sites_code_zero_based"]))


def latex(summary):
    lines = ["% Generated by build_summary_v9.py; requires booktabs and Chinese support.",
        "% Evidence counts below always use the same initial bank within a case.",
        f'% Completed {summary["completed_cases"]}/{summary["expected_cases"]}; complete={summary["complete"]}.',
        rf'\noindent 本表覆盖已完成的 {summary["completed_cases"]}/{summary["expected_cases"]} 个冻结机制算例。',
        r'表中“无 EV”也加入相同长度的未来自然观测；它不等于仅历史筛选。',
        r'每例各策略比较同一初始候选库；新生成候选库大小不用于横向比较。']
    for model, title in (("independent_box", "独立计量误差盒"),
                         ("shared_gain", "共享增益及有界扰动")):
        lines += [r'\begin{table}[htbp]\centering\small',
            r'\begin{tabular}{lrrrrrr}\toprule',
            r'案例（初始数） & 无 EV & 选单 & 恒充 & 固定单 & 选双 & 固定双\\\midrule']
        for row in summary["cases"]:
            counts = [_cell(None if row["policies"][p] is None else
                row["policies"][p]["models"][model]["initial_bank_retained_count"]) for p in POLICIES]
            lines.append(f'{row["code"]}（{row["initial_bank_count"]}） & '+" & ".join(counts)+r'\\')
        lines += [r'\bottomrule\end{tabular}',
            rf'\caption{{{title}：联合重拟合后初始候选库的剩余数；越小表示排除证据越多，但不自动证明正确。}}',
            r'\end{table}']
    lines += [r'\begin{table}[htbp]\centering\small',
        r'\begin{tabular}{llrrllll}\toprule',
        r'案例 & 条件 & 初始 F1 & 历史余数 & 初始在库 & 无 EV 后 & 选单后 & 选双后\\\midrule']
    for row in summary["cases"]:
        history = "R" if row["history"] == "rich" else "P"
        meter = {"gain_dominated": "G", "jitter_dominated": "J", "precision_control": "C"}[row["meter"]]
        condition = f'{row["n_terminal"]}/{history}/{meter}'+("/W" if row["weak"] else "")
        lines.append(" & ".join((row["code"], condition,
            f'{row["initial_best_metrics"]["f1"]:.2f}', str(row["historical_shared_gain_retained_count"]),
            _yes(row["truth_in_initial_bank"]), _post_cell(row["policies"]["no_EV"]),
            _post_cell(row["policies"]["selected_single"]),
            _post_cell(row["policies"]["selected_mobile"])))+r'\\')
    lines += [r'\bottomrule\end{tabular}',
        r'\caption{被动候选质量与新数据后重建。R：丰富历史；P：部分相关历史；G：固定增益主导；J：时变扰动主导；C：精度控制；W：弱支路。无 EV/选单/选双后单元格为首选 F1/真结构是否在更新库中。历史余数采用共享增益验证。}',
        r'\end{table}',
        r'\begin{table}[htbp]\centering\small',
        r'\begin{tabular}{lll}\toprule 案例 & 所选单地点 & 所选双地点\\\midrule']
    for row in summary["cases"]:
        lines.append(" & ".join((row["code"], _site_cell(row["policies"]["selected_single"]),
                                     _site_cell(row["policies"]["selected_mobile"])))+r'\\')
    lines += [r'\bottomrule\end{tabular}',
        r'\caption{EV 地点采用从 0 开始的终端标签；地点 0 不是物理根。所有充电策略为 27 kWh；双地点另有 2 kWh 行驶耗电。}',
        r'\end{table}']
    return "\n".join(lines)+"\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    protocol_file = args.directory/"protocol.json"
    protocol = read(protocol_file)
    rows, pending = [], []
    for index, config in enumerate(protocol["cases"], 1):
        file = args.directory/case_name(config)/"result.json"
        if not file.exists():
            pending.append(case_name(config))
            continue
        result = read(file)
        if result["config"] != config or result["case"] != case_name(config):
            raise ValueError("Result config/name differs from frozen protocol")
        item = summarize_case(result, f"C{index}")
        item["result_sha256"] = sha(file)
        rows.append(item)
    if args.require_complete and pending:
        raise RuntimeError(f"Only {len(rows)}/{len(protocol['cases'])} cases complete")
    diagnostics_file = args.directory/"diagnostics.json"
    diagnosis = read(diagnostics_file) if diagnostics_file.exists() else None
    diagnosis_cases = {} if diagnosis is None else {r["case"]: r for r in diagnosis["results"]}
    for row in rows:
        diagnostic = diagnosis_cases.get(row["case"])
        if diagnostic is None:
            row["diagnosis_status"] = "not_available"
        elif diagnostic["result_sha256"] != row["result_sha256"]:
            row["diagnosis_status"] = "stale_result_hash"
        else:
            row["diagnosis_status"] = "matching_result_hash"
            row["same_bank_policy_comparisons"] = diagnostic["comparison"]["comparisons"]
            row["deleted_pair_witnesses"] = {key: dict(
                compatible=value.get("compatible", False),
                fixed_gain_compatible=value.get("fixed_gain_witness", {}).get("compatible", False),
                refitted_gain_compatible=value.get("refitted_gain_witness", {}).get("compatible", False),
                fixed_joint_jitter_margin=value.get("fixed_gain_witness", {}).get("min_joint_jitter_margin"),
                refitted_joint_jitter_margin=value.get("refitted_gain_witness", {}).get("min_joint_jitter_margin"),
                status=value["status"])
                for key, value in diagnostic["deletion_witnesses"].items()}
    summary = dict(schema=9, complete=not pending, completed_cases=len(rows),
        expected_cases=len(protocol["cases"]), pending_cases=pending, cases=rows,
        protocol_sha256=sha(protocol_file), summary_script_sha256=sha(__file__),
        diagnosis_sha256=sha(diagnostics_file) if diagnosis is not None else None,
        aggregate=dict(initial_best_exact_count=sum(r["initial_best_metrics"]["exact"] for r in rows),
            truth_in_initial_bank_count=sum(r["truth_in_initial_bank"] for r in rows),
            wzzt_stop_reasons=dict(Counter(r["wzzt"]["stop_reason"] for r in rows)),
            policies={p: dict(available_cases=sum(r["policies"][p] is not None for r in rows),
                post_best_exact_count=sum(r["policies"][p] is not None and
                    bool(r["policies"][p]["post_best_metrics"] and r["policies"][p]["post_best_metrics"]["exact"]) for r in rows),
                truth_in_updated_bank_count=sum(r["policies"][p] is not None and
                    bool(r["policies"][p]["truth_in_updated_bank"]) for r in rows)) for p in POLICIES}),
        semantics=["Descriptive frozen mechanism cases, not statistical/field success rates.",
            "All cross-policy evidence counts restrict to each case's initial candidate bank.",
            "No-EV adds future passive data; historical screening is a distinct quantity.",
            "The two LP outer sets are not claimed nested.",
            "Source/candidate/reference ranking quality is not AC feasibility or topology probability.",
            "Shared gain and jitter decomposition is a declared additional assumption."])
    save(args.directory/"summary.json", summary)
    (args.directory/"summary_tables.tex").write_text(latex(summary), encoding="utf-8")
    print(json.dumps(dict(complete=summary["complete"], completed_cases=len(rows),
                         expected_cases=summary["expected_cases"], aggregate=summary["aggregate"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
