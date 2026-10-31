"""Descriptive post-freeze tables, including old-24 versus new-84 separation."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from .run import ROOT, LABELS, read, write, sha
from research_experiments.joint_exit.analyze import markdown


def report(output):
    output = Path(output)
    p = read(output / 'protocol.json')
    audit = read(output / 'analysis/verification.json')
    assert audit['passed'] and audit['returned'] == 480
    dest = ROOT / 'docs/rnj_four_methods_20261031'
    frame = pd.read_csv(dest / 'per_model.csv')
    seed_differences = []
    for job in p['jobs']:
        coarse = read(output / 'evaluation' / job / 'small3_coarse_exit/result.json')['trajectory']['seeding']['seeds']
        original = read(output / 'evaluation' / job / 'small3_original_exit/result.json')['trajectory']['seeding']['seeds']
        if coarse != original:
            seed_differences.append(job)
    events = pd.read_csv(dest / 'events.csv')
    ordinary = events[events.method == 'ordinary']
    write(dest / 'mechanism_checks.json', dict(seed_pairs_checked=120, seed_differences=seed_differences,
        ordinary_stop_reasons=ordinary.groupby('stop_reason').size().to_dict(),
        ordinary_basic_timeouts=int(ordinary.basic_timeouts.sum()), no_parameter_changes=True))
    catalog = read(output / 'catalog.json')
    truth_directories = sorted({catalog[job]['truth_network'] for job in p['jobs']})
    truth_files = sorted({str(file) for directory in truth_directories for file in Path(directory).iterdir() if file.is_file()})
    write(dest / 'scoring_data_manifest.json', dict(protocol_sha256=sha(output / 'protocol.json'),
        catalog_sha256=sha(output / 'catalog.json'), observed_hashes_verified_in='verification.json', truth_directories=truth_directories,
        truth_files={name: dict(bytes=Path(name).stat().st_size, sha256=sha(name)) for name in truth_files}))
    base = frame[(frame.resolution == .24) & (frame['view'] == 'bank')].copy()
    previous = set(read(output / 'previous_base_protocol.json')['jobs'])
    base['scope'] = ['prior_main24' if j in previous else 'new_main84' if panel == 'main' else 'new_stress12'
                     for j, panel in zip(base.job, base.panel)]
    aggregate = dict(inputs=('job', 'size'), F1=('clade_f1', 'mean'), exact=('exact', 'sum'),
                     train=('train_mae', 'mean'), validation=('validation_mae', 'mean'), test=('test_mae', 'mean'))
    scopes = base.groupby(['scope', 'method']).agg(**aggregate).reset_index()
    scopes.to_csv(dest / 'prior_new_scopes.csv', index=False)
    all120 = base.groupby('method').agg(**aggregate).reset_index()
    all120.to_csv(dest / 'all120.csv', index=False)
    summary = pd.read_csv(dest / 'summary.csv')
    main = summary[(summary.panel == 'main') & (summary.resolution == .24) & (summary['view'] == 'bank')]
    stress = summary[(summary.panel == 'stress') & (summary.resolution == .24) & (summary['view'] == 'bank')]
    cases = pd.read_csv(dest / 'cases.csv')
    cases = cases[(cases.panel == 'main') & (cases.resolution == .24) & (cases['view'] == 'bank')]
    case_table = cases.pivot(index='case', columns='method', values='F1').reset_index()
    exact_table = cases.pivot(index='case', columns='method', values='exact').reset_index()
    timing = pd.read_csv(dest / 'timing_summary.csv')
    paired = pd.read_csv(dest / 'paired.csv')
    paired = paired[(paired.panel == 'main') & (paired.resolution == .24) & (paired['view'] == 'bank')]
    origins = pd.read_csv(dest / 'selection_origins.csv')
    origins = origins[(origins.panel == 'main') & (origins.resolution == .24)]
    errors = pd.read_csv(dest / 'size_errors.csv')
    errors = errors[(errors.panel == 'main') & (errors.resolution == .24) & (errors['view'] == 'bank')]
    conditions = pd.read_csv(dest / 'conditions.csv')
    sensitivity = summary[(summary.panel == 'main') & (summary.resolution == .16) & (summary['view'] == 'bank')]
    paths = summary[(summary.panel == 'main') & (summary.resolution == .24) & summary['view'].isin(['path', 'prefix_bank', 'prefix_path'])]
    columns = ['method', 'inputs', 'F1', 'exact', 'train', 'validation', 'test']
    winner = main.sort_values(['F1', 'exact'], ascending=False).iloc[0]
    lines = ['# 四种RNJ／WZZT方法：完整测算与分析', '',
        f'480/480条方法—输入记录已齐并核验。主108输入平均F1最高为{LABELS[winner.method]}，F1={winner.F1:.6f}、完整恢复{int(winner.exact)}/108。',
        '264条沿用已完成且逐字节核验的300秒记录，216条为本轮新增求解；没有把复用写成重跑。四网络、观测、数据划分和最终评价一致，不重新调参。', '',
        '## 1. 方法和预算', '',
        '普通版采用此前确认的近期完整RNJ初值，关闭预联合、末端联合和退出，使用原无计数惩罚的逐步扩展；不是生产bootstrap最多两个边界块的配置。其余三项定义见METHODS.md。',
        '总结构预算300秒，单次基础MILP30秒、固定家族LP30秒；扩展版前缀180秒，剩余预算用于退出，完整RNJ预联合最多60秒。普通版全部300秒供单候选搜索使用。共同前端和最终选模LP另计；worker硬上限900秒。',
        '扩展版训练目标为MAE＋0.003 f0×内部连接数，普通版仅为MAE；最终均经0.24有效深度收缩、重新训练拟合和严格验证选择，另报0.16。故本轮评估整个方法包，不是只改变一个模块的消融。', '',
        '## 2. 主108输入与压力12输入', '', markdown(main[columns]), '', markdown(stress[columns]), '',
        '完整恢复数的分母分别为108和12；每个输入权重相同，失败不得移出分母。F1按终端内部集合族计算，完整恢复要求集合族完全一致。', '',
        '合并全120输入的描述性均值：', '', markdown(all120), '',
        '## 3. 四个算例分别比较', '', '主面板每个网络27输入。平均F1：', '', markdown(case_table), '',
        '对应完整恢复数（每项分母27）：', '', markdown(exact_table), '',
        '逐噪声／样本数条件见conditions.csv；本批包含N24/48/96和三档电压噪声，不能只取较容易的一档代表全部。', '',
        '## 4. 配对胜负与原24输入的推广', '', markdown(paired[['reference', 'method', 'inputs', 'delta', 'wins', 'ties', 'losses', 'new_exact', 'lost_exact']]), '',
        'delta为method减reference；new_exact和lost_exact分别为新增和损失的完整恢复输入数。', '',
        markdown(scopes), '',
        'prior_main24是前轮机制对照；new_main84和new_stress12是原始WZZT组合本轮补齐的输入。三个范围难度不同，不将范围间均值变化解释成同一输入上的算法退化；在每个范围内比较相同输入上的方法差异。参数始终冻结，没有根据这些F1再选择small3、惩罚或收缩阈值。', '',
        '## 5. 收益来自哪里', '', markdown(paths[['view', *columns]]), '', markdown(origins), '',
        '最终库包含公共完整RNJ候选与各自路径；complete_rnj表示选择公共RNJ，并非该方法新发现了这个结构。自身路径仍可能包含退出热启动带来的RNJ结构。prefix视角排除了退出阶段，但保留其前缀算法；差值是同一流程的阶段诊断，不自动等于单个模块的独立因果贡献。', '',
        '## 6. 大块和小块错误', '', markdown(errors[['method', 'size', 'TP', 'FP', 'FN']]), '',
        'small按终端集合大小≤5定义，与触发联合求解的直属块数≤5不同。假集合若与真集合交叉，会阻挡前向补全；相容的多余细分则可能由收缩删除。上表只统计最终结构，过程中的种子和阻塞需要结合events.csv及原始记录。', '',
        '## 7. 时间、参与率和求解状态', '', markdown(timing), '',
        'basic_inputs只统计实际启动基础MILP的输入，排除了“调用扩展函数后发现已无相容候选”的情况。退出达到设定gap只适用于其候选域；超时记录与可行解全部保留。普通方法没有退出，相关列不构成其失败率。不同退出池和初值同时影响速度，不能将所有耗时变化归为求解器性能。',
        'per_model.csv中普通版的candidate_pool_true_recall只指公共前端池的覆盖，不代表其自由单步MILP的全部可选集合；普通版没有显式退出池，events.csv对应列为空。', '',
        '## 8. 收缩敏感性与证据边界', '', markdown(sensitivity[columns]), '',
        f'数值审查{audit["checks"]}项通过；固定家族核验共{audit["independent_families"]}个，新增独立HiGHS LP {audit["newly_solved_LP"]}次，复用同输入已核验LP {audit["reused_verified_LP"]}个。旧交付{audit["preserved_previous_files"]}文件哈希保持不变。相关行为测试15项通过。',
        '这些结果覆盖固定的论文系列合成网络、有限噪声和样本设置。训练目标降低、计数惩罚和求解gap分别反映拟合、复杂度和数值搜索，均不能替代拓扑F1与完整恢复。更高平均F1与更多完整恢复也应分别判断。', '',
        '源代码、冻结协议、逐输入记录、原始分析器、独立LP回执及旧源码恢复映射见records.zip。每条复用记录保留原协议哈希，当前目录的manifest.json用于校验最终交付。', '']
    (dest / 'ANALYSIS.md').write_text('\n'.join(lines), encoding='utf-8')
    write(dest / 'report_provenance.json', dict(source_sha256=sha(Path(__file__)), truth_used_only_after_freeze=True,
          no_new_optimization=True, primary_resolution=.24, previous_subset_separated=True))
    print(main[columns].to_string(index=False))


if __name__ == '__main__':
    import sys
    report(Path(sys.argv[1]).resolve())
