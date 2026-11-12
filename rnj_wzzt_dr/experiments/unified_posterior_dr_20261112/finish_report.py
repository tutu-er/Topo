"""Report bounded pricing failures and actual observations without filling gaps."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from ... import io
from ...observations import load
from ..posterior_mse_20261112.deliver import write_csv
from .history_scan import BETAS, require, sha, utc
from .posterior import retained
from .prospective import binomial_interval


def actual_audit(run, confirmation, protocol, rows):
    seen = set(); probability_error = 0.; checked = 0
    seal = io.read(confirmation/"policy_seal.json")
    history_seal = io.read(run/"selection_seal.json")
    truth = {r["id"]: r for r in io.read(run/"evaluation.json")}
    checked_scores = set()
    require(sha(confirmation/"protocol.json") == seal["protocol_sha256"], "Policy protocol changed")
    for path, checksum in seal["policy_files"].items():
        require(sha(path) == checksum, "Executed nominal policy changed")
    for row in rows:
        key = row["id"], row["target"], row["method"], row["strategy"], row["replicate"]
        require(key not in seen, "Duplicate actual trial key"); seen.add(key)
        if row["status"] != "complete":
            continue
        ref = row["observation"]; require(sha(ref["path"]) == ref["sha256"], "Actual observations changed")
        with np.load(ref["path"], allow_pickle=False) as data:
            require(set(data.files) == {"P", "Q", "V", "root", "labels"}, "Algorithm observation contains latent data")
        observed = load(ref["path"])
        identity = row["id"].rsplit("__", 1)[0]+"__"+row["method"]
        if identity not in checked_scores:
            require(sha(run/"scores"/(identity+".json")) == history_seal["scores"][identity], "Historical score changed after seal")
            checked_scores.add(identity)
        scored = io.read(run/"scores"/(identity+".json"))
        losses = []
        for parameter in scored["parameters"]:
            point = parameter["point"]; predicted = np.zeros_like(observed.y, dtype=np.longdouble)
            for C, r, x in zip(point["supports"], point["r"], point["x"], strict=True):
                contribution = np.longdouble(r)*observed.p[:, C].astype(np.longdouble).sum(axis=1)
                contribution += np.longdouble(x)*observed.q[:, C].astype(np.longdouble).sum(axis=1)
                predicted[:, C] += contribution[:, None]
            loss = np.sum((observed.y.astype(np.longdouble)-predicted)**2)+np.longdouble(parameter["history_sse"])
            losses.append(loss)
        loss = np.array(losses, dtype=np.longdouble); ll = -(loss-loss.min())/(2*np.longdouble(scored["sigma2"]))
        expected = np.exp(ll); expected /= expected.sum()
        prefix_ref = row["prefix"]; require(sha(prefix_ref["path"]) == prefix_ref["sha256"], "Evidence path changed")
        with np.load(prefix_ref["path"], allow_pickle=False) as prefix:
            final = np.exp(prefix["final_log_probability"])
            require(np.max(abs(np.exp(prefix["prefix_log_probability"][-1])-final)) < 1e-10, "Prefix endpoint differs")
        error = float(np.max(abs(expected-final)))
        probability_error = max(probability_error, error); require(error < 1e-8, "Independent actual probability differs")
        best = int(np.argmax(expected)); ids = [m["topology_id"] for m in scored["models"]]
        require(ids[best] == row["selected_topology_id"], "Independent selected topology differs")
        real = set(map(tuple, truth[row["id"]]["truth_clades"]))
        predicted = set(map(tuple, scored["models"][best]["active_clades"]))
        require(row["missing"] == len(real-predicted) and row["extra"] == len(predicted-real), "Independent structure counts differ")
        require(row["map_correct"] == (ids[best] == truth[row["id"]]["truth_id"]), "Independent recovery differs")
        for label in ("target", "true"):
            index = row[label+"_index"]
            if index is not None:
                require(abs(float(expected[index])-row["p_"+label]) < 1e-8, "Target or true probability differs")
        for beta in BETAS:
            order = np.argsort(-expected, kind="stable")
            count = min(int(np.searchsorted(np.cumsum(expected[order]), beta))+1, len(order))
            kept = [int(j) for j in order if expected[j] >= expected[order[count-1]]]
            contains = row["true_index"] in kept
            decision = row["decisions"][str(beta)]
            require(decision["size"] == len(kept) and decision["truth_retained"] == contains, "Independent retained set differs")
            require(decision["correct_unique"] == (len(kept) == 1 and contains), "Independent correct unique differs")
            require(decision["wrong_unique"] == (len(kept) == 1 and not contains), "Independent wrong unique differs")
        checked += 1
    receipt = dict(status="passed", created_utc=utc(), actual_trial_keys=len(seen), scored_rows=checked,
        maximum_independent_probability_error=probability_error, policies_frozen_before_actual=True,
        observation_channels_only=True, parameter_refits=0, frozen_score_banks_checked=len(checked_scores),
        checks="all final weights, prefix endpoint, target/true probabilities, selected topology, missing/extra, retained sets and unique decisions",
        source=io.ref(__file__))
    io.write(confirmation/"independent_actual_audit_v2.json", receipt)
    return receipt


def aggregate(rows, protocol, truth_by_input):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["id"], row["target"], row["method"], row["strategy"]].append(row)
    entries = {(e["id"], e["target"]): e for e in protocol["policies"]}
    table = []
    for (identity, target, method, strategy), trials in grouped.items():
        require(len(trials) == 32, "Actual planned path count differs")
        available = entries[identity, target]["policies"][strategy]["available"]
        valid = [r for r in trials if r["status"] == "complete"]
        target_correct = entries[identity, target]["target_topology_id"] == truth_by_input[identity]["truth_id"]
        for beta in BETAS:
            unique = sum(r["decisions"][str(beta)]["correct_unique"] for r in valid)
            wrong = sum(r["decisions"][str(beta)]["wrong_unique"] for r in valid)
            recovered = sum(r["map_correct"] for r in valid)
            table.append(dict(id=identity, case=trials[0]["case"], target=target, method=method, strategy=strategy,
                role=trials[0]["role"], target_correct=target_correct, beta=beta, available=available,
                planned=32, attempted=32 if available else 0, valid=len(valid),
                correct_unique=unique if available else None, wrong_unique=wrong if available else None,
                full_recovery=recovered if available else None,
                correct_unique_rate=unique/32 if available else None,
                wrong_unique_rate=wrong/32 if available else None,
                recovery_rate=recovered/32 if available else None,
                interval95=binomial_interval(unique, 32) if available else None,
                true_retained=sum(r["decisions"][str(beta)]["truth_retained"] for r in valid) if available else None,
                pair_pass_full_fail=sum(r["decisions"][str(beta)]["pair_pass_but_full_fail"] for r in valid) if available else None,
                mean_target_probability=float(np.mean([r["p_target"] for r in valid if r["p_target"] is not None])) if any(r["p_target"] is not None for r in valid) else None,
                mean_missing=float(np.mean([r["missing"] for r in valid])) if valid else None,
                mean_extra=float(np.mean([r["extra"] for r in valid])) if valid else None,
                mean_absolute_DR_kwh=float(np.mean([r["execution"]["original_device_audit"]["absolute_DR_kwh"] for r in valid])) if valid else None,
                clipping_fraction=float(np.mean([r["execution"]["clipping_fraction"] for r in valid])) if valid else None))
    return table


def publish(run, destination):
    require(not destination.exists(), "Existing final delivery preserved")
    destination.mkdir(parents=True)
    confirmation = run/"confirmation"; protocol = io.read(confirmation/"protocol.json")
    require((confirmation/"actual_complete.json").exists(), "Finish actual paths before reporting")
    rows = io.read(confirmation/"actual_results.json")
    truth = {r["id"]: r for r in io.read(run/"evaluation.json")}
    audit = actual_audit(run, confirmation, protocol, rows)
    actual = aggregate(rows, protocol, truth)
    write_csv(destination/"actual_all_comparisons.csv", actual)
    primary = [r for r in actual if r["target_correct"] and r["method"] == "101"]
    write_csv(destination/"actual_primary_comparison.csv", primary)
    development = {(r["id"], r["target"]): r for r in io.read(run/"development/results.json")}
    planning = io.read(run/"planning/results.json"); pricing = []
    positive = [r["objective"] for r in planning if r.get("qualified") and (r.get("objective") or 0) > 0]
    fee_unit = min(positive) if positive else None
    for row in planning:
        request = io.read(run/"planning/targets"/(row["id"]+f"__target{row['target']}")/"request.json")
        original_pairs = {p["competitor"]: p for p in request["zero_bound"]["pairs"]}
        log_bound_floor = float(logsumexp([original_pairs[c["competitor"]]["log_c"]-c["moment_upper"] for c in request["capacity"]]))
        pricing.append(dict(id=row["id"], target=row["target"], role=row["role"],
            target_correct=development[row["id"], row["target"]]["target_correct"], status=row["status"],
            qualified=row["qualified"], native_status=row.get("native_status"),
            normalized_fee=row["objective"]/fee_unit if row.get("objective") is not None and fee_unit else 0. if row.get("objective") == 0 else None,
            native_gap=row.get("native_gap"), constraints=len(request["allocation"]["active"]),
            log_exponential_bound_capacity_floor=log_bound_floor,
            full_exponential_certificate_capacity_infeasible=log_bound_floor > np.log(.1),
            original_chance_infeasibility_proved=row["status"] == "original_chance_infeasible",
            bound_scope=row.get("bound_scope", row.get("infeasible_scope", "fixed allocation only"))))
    write_csv(destination/"pricing_summary.csv", pricing)
    model = io.read(confirmation/"model_results.json"); model_table = []
    for row in model:
        for beta, result in row["gates"].items():
            model_table.append(dict(id=row["id"], target=row["target"], strategy=row["strategy"], beta=float(beta),
                target_correct=development[row["id"], row["target"]]["target_correct"], **result))
    write_csv(destination/"model_confirmation.csv", model_table)
    status = Counter(r["status"] for r in planning)
    summary = dict(created_utc=utc(), historical_libraries=800, development_inputs=14, development_targets=28,
        pricing_statuses=status, qualified_zero=sum(r["qualified"] and r.get("objective") == 0 for r in planning),
        qualified_nonzero=len(positive), beta_star=protocol["beta_star"], physical_inputs=6,
        actual_statuses=Counter(r["status"] for r in rows), actual_audit=audit,
        original_chance_infeasibility_proved=sum(r["status"] == "original_chance_infeasible" for r in planning),
        full_run_scope="Bounded attempted pipeline; unavailable quotes not replaced by zero optimized policies.")
    io.write(run/"summary.json", summary)
    io.write(destination/"summary.json", {k: v for k, v in summary.items() if k != "actual_audit"})
    plot_trajectories(run, confirmation, protocol, rows, primary, destination)
    plot_model_actual(model_table, primary, protocol["beta_star"], destination)
    status_names = {"solver_finished": "求解超时，未找到可行报价", "fixed_allocation_certificate_infeasible": "容量检查排除本次保守条件", "zero_certified": "零动作已满足保证条件"}
    lines = ["# 统一后验与DR：本批执行结果", "", "已完成800个历史候选库评分、14个输入的28个条件目标检查、12次有界报价尝试，以及6个输入各32条独立交流潮流新日。未取得合格非零报价，因此本批尚未验证非零DR的改善。", "",
        "101表示RNJ提供候选＋WZZT逐步加线＋结束前联合重选连接；111在此基础上增加末端局部重选。三位开关依次为R/T/E，WZZT加线始终存在。T/E是训练阶段的结构搜索，两种方法最后都按验证MSE选择输出。本批沿用原拟合结果，门槛在实际新日前锁为90%，另保留80%、95%、99%的完整结果。", "",
        "## 报价结果", "", "| 结果 | 目标数 |", "|---|---:|"]
    lines += [f"| {status_names.get(name, name)} | {count} |" for name, count in status.items()]
    excluded = sum(r["full_exponential_certificate_capacity_infeasible"] for r in pricing)
    lines += ["", f"容量检查进一步表明：{excluded}个目标连完整指数概率界也无法在原容量范围内达到要求。这说明本次保证条件过严或原资源不足；它没有证明实际后验达标概率不可达。其余1个非零目标按30分钟预算终止，未找到可行点，保留未决。4个零费目标因零动作可行且费用非负，最小费用确为0（声明规划模型下）。", "",
        "## 独立新日：无需DR时究竟有多难", "", "下表只看目标正确的101库。正确唯一输出指真拓扑权重达到90%；完整恢复指最高权重拓扑的整棵结构正确。两项分别计数。模型达标率每目标2048条，交流潮流模拟每输入32条。", "",
        "| 网络／输入 | 模型内达标率 | 新日正确唯一／32 | 新日完整恢复／32 | 新日错误唯一／32 |",
        "|---|---:|---:|---:|---:|"]
    model_lookup = {(r["id"], r["target"], r["strategy"], r["beta"]): r for r in model_table}
    for row in primary:
        if row["beta"] == protocol["beta_star"] and row["strategy"] == "no_dr":
            mr = model_lookup[row["id"], row["target"], "no_dr", row["beta"]]
            lines.append(f"| {row['case']}／{row['id'].rsplit('__', 1)[0]} | {mr['successes']}/{mr['paths']}（{mr['probability']:.1%}） | {row['correct_unique']}/32 | {row['full_recovery']}/32 | {row['wrong_unique']}/32 |")
    lines += ["", "前4个是主要难例，后2个是零费用对照。两个零费对照的三策略均为零动作，结果均为32/32；主要难例的优化DR和同费轮换因缺少合格报价而未执行。详细表保留全部目标、101/111及四种门槛：", "",
        "- [报价状态](pricing_summary.csv)", "- [独立工作模型确认](model_confirmation.csv)", "- [目标正确的主比较](actual_primary_comparison.csv)", "- [全部实际评分](actual_all_comparisons.csv)", "",
        "无DR给所有锁定输入各32条原网络新日；仅对存在合格报价的目标执行同费轮换／优化策略。未执行不计作0/32，也不计作零动作优化。101/111、两个目标及零费策略复用的观测不当作独立样本。", "",
        "## 这批结果支持什么", "", "当前指标能把‘最高权重选对’与‘证据足够、可唯一输出’分开，也找到了保留真拓扑而仍难以区分的实例。后续DR求解有明确对象。但本次保守保证在7个目标上超出资源所能满足的范围，剩余非零求解未成功；尚无非零DR增益或费用—效果证据。", "",
        "历史800库中503库缺少完整真拓扑，90%门槛下有349次错误唯一输出；另有12库含真但最高权重错误。该概率仍是冻结有限库和假设误差模型内的相对权重，实际错误率要看这些统计。历史全量表见[前阶段报告](../development/REPORT.md)。", "",
        "下一项最小工作是先检查Flynn单主要竞争者的保证界与原两候选高斯公式之间的差距，并处理本次限时未决求解。其余案例应先判断保守界是否过严，再决定是否值得继续求解。保持现有观测、资源、噪声和门槛，避免靠改条件制造改善。", "",
        "模型确认采用目标冻结响应与声明误差法则；交流潮流模拟采用原物理网络、原相对P/Q/V及根量测噪声、整段相关设备响应与因果截断。结构正确不保证冻结参数正好等于实际条件均值，因此模型概率和模拟频率分别报告；这里的实际新日均为独立模拟，未做现场试验。", "",
        "合同费用沿用原报价规则；实际交付的费用函数值是资源损耗代理，不是额外支付。没有正费可行报价时，不绘制或声称已得到最小费用—效果曲线。", "",
        f"已独立逐支持复算{audit['scored_rows']}行的全部候选权重、最终拓扑、错漏和各门槛决定，最大概率差{audit['maximum_independent_probability_error']:.3g}；原始量测文件仅含P/Q/V/root/labels，真实功率和执行日志另存私人目录。没有重新拟合辨识器或参数。"]
    (destination/"REPORT.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    io.write(destination/"RECEIPT.json", dict(created_utc=utc(), protocol_sha256=sha(confirmation/"protocol.json"),
        independent_audit_sha256=sha(confirmation/"independent_actual_audit_v2.json"),
        files={p.name: sha(p) for p in destination.iterdir() if p.is_file()}, private_data_published=False))
    io.write(run/"execution_complete.json", dict(status="bounded_protocol_completed", created_utc=utc(),
        public_report=str(destination/"REPORT.md"), qualified_nonzero=len(positive),
        planning_failures_retained=True, actual_paths_complete=True, source=io.ref(__file__)))
    print({k: v for k, v in summary.items() if k != "actual_audit"}, flush=True)


def save_figure(fig, destination, name):
    for suffix in ("png", "svg"):
        path = destination/(name+"."+suffix); fig.savefig(path, dpi=180, bbox_inches="tight")
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())+"\n", encoding="utf-8")


def plot_trajectories(run, confirmation, protocol, rows, primary, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "svg.fonttype": "none"})
    identities = list(dict.fromkeys(r["id"] for r in primary)); fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, identity in zip(axes.flat, identities):
        relevant = [r for r in rows if r["id"] == identity and r.get("target_correct") and r["method"] == "101" and r["status"] == "complete"]
        for name in ("no_dr", "rotation", "optimized"):
            values = []
            for row in relevant:
                if row["strategy"] != name:
                    continue
                with np.load(row["prefix"]["path"], allow_pickle=False) as z:
                    values.append(np.exp(z["prefix_log_probability"][:, row["true_index"]]))
            if values:
                q = np.quantile(values, [.1, .5, .9], axis=0)
                line = axis.plot(np.arange(97), q[1], label=name)[0]
                axis.fill_between(np.arange(97), q[0], q[2], alpha=.1, color=line.get_color())
        axis.axhline(protocol["beta_star"], color=".4", linestyle="--", linewidth=.8)
        axis.set(title=identity, ylim=(0, 1.02), xlabel="New-day observations", ylabel="True-candidate weight")
        axis.grid(alpha=.2)
    labels = {}
    for axis in axes.flat:
        handles, names = axis.get_legend_handles_labels()
        labels.update(dict(zip(names, handles)))
    fig.legend(labels.values(), labels.keys(), loc="lower center", ncol=3, frameon=False)
    fig.suptitle("Actual AC days: median and 10-90% path range; fixed historical scale")
    fig.tight_layout(rect=(0, .04, 1, .95)); save_figure(fig, destination, "actual_evidence_paths"); plt.close(fig)


def plot_model_actual(model_table, primary, beta, destination):
    import matplotlib.pyplot as plt
    lookup = {(r["id"], r["target"], r["strategy"]): r for r in model_table if r["beta"] == beta}
    values = [r for r in primary if r["beta"] == beta and r["strategy"] == "no_dr"]
    x = np.arange(len(values)); fig, ax = plt.subplots(figsize=(11, 4))
    theoretical = [lookup[r["id"], r["target"], "no_dr"]["success_lower"] for r in values]
    model = [lookup[r["id"], r["target"], "no_dr"]["probability"] for r in values]
    actual = [r["correct_unique_rate"] for r in values]
    ax.bar(x-.25, theoretical, .25, label="Declared-model analytical lower")
    ax.bar(x, model, .25, label="Independent declared-model frequency")
    ax.bar(x+.25, actual, .25, label="Actual AC correct unique output")
    ax.set(xticks=x, xticklabels=[r["id"].replace("__101", "") for r in values], ylim=(0, 1.05), ylabel="Probability / frequency")
    ax.tick_params(axis="x", rotation=20); ax.legend(frameon=False); ax.grid(axis="y", alpha=.2)
    fig.tight_layout(); save_figure(fig, destination, "model_and_actual_no_dr"); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True); parser.add_argument("--public", type=Path, required=True)
    args = parser.parse_args(); publish(args.run.resolve(), args.public.resolve())


if __name__ == "__main__":
    main()
