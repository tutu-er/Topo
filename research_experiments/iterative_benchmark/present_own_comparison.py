"""Reproduce the four-method comparison without final RNJ-tree fallback.

This presenter reads frozen results. It runs no simulator, LP, MILP, tuning or
new validation selection. Historical stage diagnostics use saved coefficients.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/rnj_union_long_20261031"
CURRENT = ROOT / "docs/iterative_exit_matched120_20261101"
DEST = ROOT / "docs/own_mainline_four_methods_20261101"
METHODS = ("NJ_matched", "pure_wzzt", "ordinary", "current_own")
LABELS = {
    "NJ_matched": "NJ：同尺度收缩与重拟合",
    "pure_wzzt": "纯WZZT",
    "ordinary": "普通RNJ＋WZZT",
    "current_own": "当前主线：无整树备用择优",
}
CASES = ("paper15", "soumalas11", "flynn16", "pengwah18")
FLOW = '''flowchart LR
    A["① RNJ小块预选<br/>至多两个不相交的2～3终端块"]
    B["② 原始WZZT逐步扩展<br/>联合拟合，及时释放零块"]
    C["③ ≤5直属块联合末端<br/>枚举候选连接，联合MILP重构"]
    D["④ 一次最终退出<br/>收缩、重拟合、自身路径验证择优"]
    A --> B --> C --> D
'''


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8", newline="\n")


def markdown(frame: pd.DataFrame) -> str:
    def cell(value):
        return f"{value:.6f}" if isinstance(value, (float, np.floating)) else str(value)
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    lines.extend("| " + " | ".join(cell(v) for v in row) + " |"
                 for row in frame.itertuples(index=False, name=None))
    return "\n".join(lines)


def checked_sources():
    sources = {}
    for folder, names in ((BASE, ("per_model.csv", "cost_per_input.csv", "protocol.json", "verification.json")),
                          (CURRENT, ("per_model.csv", "protocol.json", "verification.json", "domain_diagnostics.json"))):
        manifest = read(folder / "manifest.json")["files"]
        assert read(folder / "verification.json")["passed"]
        for name in names:
            path = folder / name
            assert digest(path) == manifest[name]["sha256"], path
            sources[str(path.relative_to(ROOT)).replace("\\", "/")] = digest(path)
    a, b = read(BASE / "protocol.json"), read(CURRENT / "protocol.json")
    assert a["jobs"] == b["jobs"] and len(a["jobs"]) == 120
    assert a["input_hashes"] == b["input_hashes"]
    assert read(CURRENT / "domain_diagnostics.json")["both_base_subproblems_gap_met"] == 120
    return sources, b


def observations():
    old = pd.read_csv(BASE / "per_model.csv")
    now = pd.read_csv(CURRENT / "per_model.csv")
    own = now[(now.method == "after") & (now.view == "zero_only_own")].set_index("job")
    assert len(own) == 120 and own.index.is_unique
    assert own.selected_source.eq("baseline").all()
    assert abs(own.clade_f1.mean() - .8217551430051431) < 1e-12 and own.exact.sum() == 60
    metadata = ["panel", "case", "bundle", "N", "voltage_sigma", "condition"]
    metrics = ["clade_f1", "precision", "recall", "exact", "train_mae", "validation_mae", "test_mae",
               "R_relative", "X_relative", "internal", "selected_source"]
    parts = []
    for method in METHODS:
        if method == "current_own":
            part = own.copy()
        else:
            part = old[(old.method == method) & (old.resolution == .24) & (old.view == "selected")].set_index("job")
        assert len(part) == 120 and part.index.is_unique and set(part.index) == set(own.index)
        part = part.reindex(own.index)
        # The current catalogue supplies identical condition labels for all methods.
        result = own[metadata].join(part[metrics])
        result["output_origin"] = part["origin"] if "origin" in part else part["selected_source"]
        result["algorithm"] = method
        parts.append(result.reset_index())
    frame = pd.concat(parts, ignore_index=True)
    assert len(frame) == 480 and not any("seconds" in c for c in frame.columns)
    return frame, now


def summarize(frame: pd.DataFrame, keys):
    rows = []
    for key, group in frame.groupby(keys, sort=False, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(keys, key))
        row.update(inputs=len(group), F1=float(group.clade_f1.mean()), exact=int(group.exact.sum()),
                   recovery_rate=float(group.exact.mean()), precision=float(group.precision.mean()),
                   recall=float(group.recall.mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def paired(frame: pd.DataFrame):
    own = frame[frame.algorithm == "current_own"].set_index("job")
    rows, pairs = [], []
    for method in METHODS[:-1]:
        base = frame[frame.algorithm == method].set_index("job").reindex(own.index)
        delta = own.clade_f1 - base.clade_f1
        rows.append(dict(reference=method, inputs=120, delta_F1=float(delta.mean()),
                         wins=int((delta > 1e-12).sum()), ties=int((abs(delta) <= 1e-12).sum()),
                         losses=int((delta < -1e-12).sum()),
                         new_exact=int((own.exact & ~base.exact).sum()),
                         lost_exact=int((~own.exact & base.exact).sum())))
        pairs.extend(dict(job=job, case=own.loc[job, "case"], reference=method,
                          baseline_F1=float(base.loc[job, "clade_f1"]), own_F1=float(own.loc[job, "clade_f1"]),
                          delta_F1=float(delta.loc[job]), baseline_exact=bool(base.loc[job, "exact"]),
                          own_exact=bool(own.loc[job, "exact"])) for job in own.index)
    return pd.DataFrame(rows), pd.DataFrame(pairs)


def timing(protocol):
    cost = pd.read_csv(BASE / "cost_per_input.csv")
    rows = []
    for method in METHODS[:-1]:
        group = cost[cost.method == method]
        assert len(group) == 120
        rows.append(dict(algorithm=method, inputs=120, recorded_total_seconds=float(group.total_seconds.sum()),
                         stage_subtotal_seconds=None, standalone_timing_measured=method != "NJ_matched",
                         interpretation="shared frontend and NJ/selector accounting" if method == "NJ_matched" else "recorded algorithm components"))
    raw = ROOT / "outputs/iterative_exit_matched120_20261101"
    pieces, hashes = [], {}
    for job in protocol["jobs"]:
        result_path = raw / "jobs" / job / "after/result.json"
        result = read(result_path)
        event = result["branches"]["baseline"]["exit_event"]
        pieces.append(dict(job=job, frontend=result["common_seconds"], prefix=result["prefix"]["seconds"],
                           exit_construction=event["construction_seconds"], exit_solve=event["seconds"]))
        hashes[str(result_path.relative_to(ROOT)).replace("\\", "/")] = digest(result_path)
    pieces = pd.DataFrame(pieces)
    total = float(pieces.drop(columns="job").to_numpy().sum())
    assert abs(total - 30461.17236880002) < 1e-6
    rows.append(dict(algorithm="current_own", inputs=120, recorded_total_seconds=None,
                     stage_subtotal_seconds=total, standalone_timing_measured=False,
                     interpretation="disjoint recorded frontend/prefix/exit stages; excludes some LP closure/output costs"))
    return pd.DataFrame(rows), pieces, hashes


def report(summary, cases, panels, noise, stress, pairs, history):
    display = summary[["algorithm", "inputs", "F1", "exact", "recovery_rate"]].copy()
    display["algorithm"] = display.algorithm.map(LABELS)
    text = "# 四算法对照：无整树备用择优的RNJ–WZZT主线\n\n"
    text += "第四算法固定为after/zero_only_own：少量RNJ小块→原始WZZT及时清零→≤5直属块联合末端→一次最终退出、收缩和自身路径验证择优。输出只比较自身路径与单点候选，120个输出均来自自身baseline路径。完整RNJ树备用择优、额外停滞恢复和屏蔽分支均未纳入。旧版含备用图表保持原样，本目录是本次修订口径。\n\n"
    text += "```mermaid\n" + FLOW + "```\n\n"
    text += "## 四算法主结果\n\n" + markdown(display) + "\n\n"
    text += "![总体结果](four_algorithm_overall.png)\n\n"
    text += "当前主线自身F1为0.821755、完整恢复60/120（50%）。相对普通RNJ＋WZZT，F1增加0.074800，完整恢复净增5个；逐输入44胜60平16负，完整恢复新增13个、损失8个。相对NJ、纯WZZT分别净增22、32个完整恢复。四个网络的平均F1均最高；优势是本批数据上的平均拓扑恢复质量。\n\n"
    text += "## 分网络与条件\n\n" + markdown(cases) + "\n\n![分网络结果](four_algorithm_cases.png)\n\n"
    text += "pengwah18的平均F1改善，但完整恢复为15/30，低于普通方法17/30。常规108输入为F1=0.841743、55个完整恢复，高于普通方法0.753671、49个；压力12输入则为0.641865、5个，低于普通方法0.686508、6个。\n\n"
    text += markdown(panels) + "\n\n" + markdown(noise) + "\n\n![噪声对比](four_algorithm_noise.png)\n\n"
    text += "每个噪声等级36个常规输入；每个噪声×N组合12个。中高电压噪声（0.05%、0.10%）下当前主线优于普通方法；低噪声0.02%下普通方法更好。9个噪声×N格中当前主线6格F1改善、1格相同、2格下降，不能写成所有条件占优。训练N为三天合计24/48/96，P/Q噪声固定0.5%。\n\n"
    text += markdown(stress) + "\n\n当前主线弱激励仅F1=0.071429、0/4完整恢复；根表噪声0.895833、2/4；异常值0.958333、3/4。此前含备用版本的压力图不适用于本口径。\n\n"
    text += "## 历史消融与阶段诊断\n\n"
    text += "及时清零的成对历史比较使用before/own与after/zero_only_own：共同输入、4小时总预算、gap=0.0001、相同惩罚、收缩与自身验证规则，120对base子问题都达到设定精度。历史分支分配的时限不同，但未出现求解时钟截断。它衡量及时清零及随后的路径变化，不是全树最优证明。\n\n"
    zero = history["zero_summary"].set_index("scope").loc["all"]
    zero_display = pd.DataFrame([
        ["无及时清零", zero.F1_before, int(zero.exact_before), zero.exact_before / zero.inputs],
        ["有及时清零", zero.F1_after, int(zero.exact_after), zero.exact_after / zero.inputs],
    ], columns=["自身输出对照", "F1", "完整恢复", "完整恢复率"])
    text += markdown(zero_display) + "\n\n![及时清零消融](historical_zero_ablation.png)\n\n"
    text += "无及时清零F1=0.818889，加入后0.821755，净增0.002866；1胜116平3负，完整恢复均60/120。当前方法相对普通方法的总体优势不能主要归因于及时清零。\n\n"
    text += "另外独立读取保存的最终退出前、退出后权重，并从物理网络重建真实后代集合：\n\n"
    stage_display = history["stage_summary"].query("scope == 'all'").drop(columns="scope")
    text += markdown(stage_display) + "\n\n![历史阶段诊断](historical_stage_diagnostic.png)\n\n"
    text += "原始退出前F1=0.655460、28个完整恢复；原始退出后0.799636、48个；最终自身输出0.821755、60个。退出前后是保存轨迹的阶段诊断，未应用统一输出收缩与选模，不能代替“关闭最终退出后按同规则重新选模”的独立消融；最终输出的增益也同时包含收缩、重拟合和路径验证选择。末端枚举、线路数惩罚及最终退出的独立去除消融目前缺失，本次未用其他模型的历史参数实验替代。\n\n"
    text += "## 时间与比较边界\n\n"
    time_table = pd.DataFrame([
        [LABELS["NJ_matched"], "163.23秒（2.72分钟）", "共享前端＋NJ/选择开销"],
        [LABELS["pure_wzzt"], "10459.37秒（2.91小时）", "已记录组件累计"],
        [LABELS["ordinary"], "1714.16秒（28.57分钟）", "已记录组件累计"],
        [LABELS["current_own"], "独立总耗时缺测；已归属阶段30461.17秒（8.46小时）", "阶段累计缺少部分LP关闭、清零及输出成本"],
    ], columns=["算法", "120输入累计用时", "口径"])
    text += markdown(time_table) + "\n\n"
    text += "累计时间是逐输入算法阶段墙钟之和，不是并行整批等待时间。NJ记账包含公共RNJ候选库前端，非最小独立NJ计时。当前主线的16.15小时旧字段属于含恢复分支的完整运行，禁止用于本表；8.46小时也不作为独立总耗时或计算加速的证据。最终退出约占已归属阶段85%，质量优势伴随较重计算。\n\n"
    text += "普通RNJ＋WZZT为历史300秒结构预算、单MIP/LP30秒、gap=0.001；纯WZZT与当前主线为总预算14400秒、LP300秒、gap=0.0001（单步WZZT≤3600秒）。普通方法280个MIP无超时但20个输入保留no_gain_unresolved；纯WZZT1130个MIP达到gap，1个输入保留no_gain_unresolved。这是同输入的历史算法对照，预算、精度及正则项并非完全一致。\n\n"
    text += "普通RNJ＋WZZT的历史最终输出仍保留原完整RNJ备用政策，107个来源为complete_rnj、13个来源为path；本次按用户指定保留这个原基线。第四算法不参加完整树备用竞争，但仍是RNJ指导的混合方法：RNJ用于小块预选、参考损失、几何校准、退出候选池和热启动。公共候选池还包含NJ/RNJ几何产生的连接；因此准确表述为“无完整树备用择优”，而不是“完全不使用其他算法信息”。\n\n"
    text += "## 复现与核验\n\n"
    text += "本次未新增仿真、LP、MILP或参数调节。读取已有结果和保存权重，核对两个协议全部120个job与各split哈希、归档manifest及480条质量记录；未改变运行默认或原结果。此前多轮设计已使用本批数据，本图是历史比较，不是新盲测。\n\n"
    text += "运行 `python -B -m research_experiments.iterative_benchmark.present_own_comparison`。逐输入与分组CSV、配对变化、阶段诊断、来源哈希和核验回执在本目录；PNG用于预览，PDF为同图矢量版。\n"
    return text


def main():
    from .history_diagnostics import analyze_history
    from .own_figures import render_figures

    DEST.mkdir(parents=True, exist_ok=True)
    source_hashes, protocol = checked_sources()
    frame, original = observations()
    summary = summarize(frame, ["algorithm"])
    cases = summarize(frame, ["case", "algorithm"])
    panels = summarize(frame, ["panel", "algorithm"])
    noise = summarize(frame[frame.panel == "main"], ["voltage_sigma", "algorithm"])
    cells = summarize(frame[frame.panel == "main"], ["voltage_sigma", "N", "algorithm"])
    stress = summarize(frame[frame.panel == "stress"], ["condition", "algorithm"])
    paired_summary, pair_rows = paired(frame)
    times, time_pieces, time_hashes = timing(protocol)
    history = analyze_history(ROOT)
    products = {"per_input_quality": frame, "overall_summary": summary, "case_summary": cases,
                "panel_summary": panels, "noise_summary": noise, "noise_sample_summary": cells,
                "stress_summary": stress, "paired_summary": paired_summary, "paired_inputs": pair_rows,
                "timing": times, "recorded_current_stages": time_pieces}
    products.update({name: value for name, value in history.items() if isinstance(value, pd.DataFrame)})
    for name, product in products.items():
        product.to_csv(DEST / f"{name}.csv", index=False, lineterminator="\n")
    figures = render_figures(DEST, frame, history)
    (DEST / "four_core_flow.mmd").write_text(FLOW, encoding="utf-8", newline="\n")
    (DEST / "RESULTS.md").write_text(report(summary, cases, panels, noise, stress, paired_summary, history),
                                      encoding="utf-8", newline="\n")
    profile = dict(main_case="current_own", source_method="after", source_view="zero_only_own",
                   includes_complete_tree_output_fallback=False, includes_singleton_candidate=True,
                   includes_stall_recovery=False, includes_mask_branch=False,
                   RNJ_guidance_retained=["small seeds", "reference loss", "calibration", "exit candidate pool", "warm starts"],
                   original_ordinary_fallback_preserved=True, standalone_current_timing_measured=False,
                   runtime_defaults_modified=False, new_optimization_solves=0, retuned=False,
                   parameters=dict(max_seeds=2, seed_size=[2, 3], terminal_blocks=5, lambda_over_f0=.003,
                                   contraction=.24, response_bounds=[0, 2], zero_tolerance=1e-9),
                   source_hashes=source_hashes, current_raw_record_hashes=time_hashes,
                   history_metadata=history.get("metadata", {}),
                   producer_hashes={name: digest(Path(__file__).parent / name) for name in
                                    ("present_own_comparison.py", "history_diagnostics.py", "own_figures.py")})
    write_json(DEST / "main_case_profile.json", profile)
    assert summary.set_index("algorithm").loc["current_own", "exact"] == 60
    assert frame.query("algorithm == 'current_own'").selected_source.eq("baseline").all()
    assert times.loc[times.algorithm == "current_own", "recorded_total_seconds"].isna().all()
    assert original.query("method == 'after' and view == 'zero_only_own'").exact.sum() == 60
    write_json(DEST / "verification.json", dict(passed=True, paired_inputs=120, methods=4, quality_rows=480,
               same_split_hashes=True, source_manifests_verified=True, no_complete_RNJ_output_selection=True,
               current_F1=float(summary.set_index("algorithm").loc["current_own", "F1"]), current_exact=60,
               historical_stage_labels_distinguished=True, new_simulations=0, new_LPs=0, new_MILPs=0,
               standalone_current_total_time_unavailable=True, figures=figures))
    write_json(DEST / "manifest.json", {"files": {p.name: {"bytes": p.stat().st_size, "sha256": digest(p)}
               for p in sorted(DEST.iterdir()) if p.is_file() and p.name != "manifest.json"}})
    print(summary.to_string(index=False))
    print("PASS: four methods, matched 120 inputs, no final RNJ-tree fallback, no new optimization.")


if __name__ == "__main__":
    main()
