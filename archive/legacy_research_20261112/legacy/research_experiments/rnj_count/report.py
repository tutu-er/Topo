"""Report F1/exact recovery first, with regularization and error-source evidence."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd

from .run import ROOT,PARENT,read,write,sha
from .diagnose import diagnose

NAMES=dict(original_rnj_wzzt='原RNJ+WZZT',joint_unregularized='五块联合，无惩罚',
    count_search='仅搜索惩罚',count_selection='仅最终选模惩罚',count_both='搜索与选模都惩罚',count_both_strict='两处惩罚，严格最小规则')


def audit_inputs(output):
    parent=ROOT/'outputs/paper_cases_lp_rebuild_20261031';old=read(parent/'input_audit.json')
    protocol=read(output/'protocol.json');catalog=read(output/'catalog.json');frozen=read(output/'frozen.json');hashes=dict(old['input_hashes'])
    for job in protocol['development']:
        common=read(output/'common'/job/'result.json');meta=catalog[job]
        hashes[meta['train']]=common['source_input_hashes']['train'];hashes[meta['validation']]=common['source_input_hashes']['validation']
        hashes[meta['test']]=frozen['development_outer_hashes'][job]
    for name,digest in hashes.items():
        assert sha(name)==digest
    networks={q['case']:q for q in old['networks']};seen=set()
    for meta in catalog.values():
        network=Path(meta['truth_network'])
        if network in seen:
            continue
        seen.add(network);q=networks[meta['case']]
        assert sha(network/'buses.csv')==q['buses_sha256'] and sha(network/'branches.csv')==q['branches_sha256']
    receipt=dict(passed=True,observation_archives=len(hashes),network_directories=len(seen),input_hashes=hashes)
    write(output/'input_audit.json',receipt)
    return receipt


def generate(output,destination):
    output=Path(output);destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    verification=read(output/'analysis/verification.json');frozen=read(output/'frozen.json');assert verification['passed']
    diagnose(output);input_audit=audit_inputs(output)
    frame=pd.read_csv(output/'analysis/per_model.csv');summary=pd.read_csv(output/'analysis/summary.csv')
    pairs=pd.read_csv(output/'analysis/paired.csv');sizes=pd.read_csv(output/'analysis/size_summary.csv')
    origins=pd.read_csv(output/'diagnosis/error_summary.csv');source_receipt=read(output/'diagnosis/verification.json')
    methods=list(NAMES)
    if frozen['methods']['count_both']==frozen['methods']['count_both_strict']:
        methods.remove('count_both_strict')
    main=summary[(summary.panel=='main') & (summary.view=='bank')].set_index('method')
    chosen=main.loc['count_search'];zero=main.loc['joint_unregularized'];original=main.loc['original_rnj_wzzt']
    lines=['# RNJ小块误选、过度细分与线路数正则化', '',
        f"开发112次搜索全部返回；冻结后正式搜索返回{verification['returned']}/{verification['planned']}。"
        f"{verification['scored_bank_outputs']}个方法输出通过核查，独立重解{verification['unique_independent_LP']}个不同的输入/最终家族组合。",
        '同一批paper15、soumalas11、flynn16、pengwah18，108主输入+12压力输入；沿用完整RNJ初值，'
        '生产默认bootstrap两边界块口径未覆盖、默认代码未修改。', '',
        f"本批最有效的已冻结配置是仅搜索惩罚lambda={frozen['selected']['search_only']['coefficient']:g}："
        f"平均F1为{chosen.F1:.4f}，完整恢复{int(chosen.exact)}/108。"
        f"无惩罚联合模型为{zero.F1:.4f}、{int(zero.exact)}/108；原RNJ+WZZT为{original.F1:.4f}、{int(original.exact)}/108。"
        '改善主要来自减少小块误检，仍有交叉错误和压力条件限制。', '', '## 正则项和调参', '',
        '采用 J=原始量测MAE+lambda*f0*K。K是非平凡内部连接数；单点和全集不计，固定粗骨架的数量代价作为常数记入。'
        'f0是未加惩罚时RNJ初值的训练MAE，用于统一尺度，并非已知噪声标准差。'
        '可选零响应线路规范化为不选，没有加入新的正阻抗下界。', '',
        '开发集为bundle0/1、四网络、两档噪声、N48，共16输入。训练拟合、内层验证选候选、外层开发观测选lambda；'
        '历史上名为test_observed的开发归档明确改作调参观测。最终bundle2/3/4的测试观测和真实拓扑均未用于选参。', '',
        '| 惩罚位置 | 冻结lambda | 开发平均内部连接数 | 开发预测误差对数比 |', '|---|---:|---:|---:|']
    for key,label in [('search_only','仅搜索'),('selection_only','仅最终选模'),('both','两处都加')]:
        q=frozen['selected'][key]
        lines.append(f"| {label} | {q['coefficient']:g} | {q['mean_internal']:.3f} | {q['mean_log_ratio']:+.6f} |")
    lines += ['', '先最小化外层开发预测MAE，再在配对一标准误差或0.1%带内优先较少线路。'
              '标准误差按case×bundle八组计算；这是一种简约选模规则，不作置信保证。'
              '两处惩罚的严格最小误差控制取lambda='+f"{frozen['strict_min']['both']['coefficient']:g}。",
              '参数冻结后才计算开发F1；没有因为开发或最终F1重新选择强度。完整设置见[PROTOCOL.md](PROTOCOL.md)。', '',
              '## 主面板：F1和完整恢复', '',
              '| 方法 | 平均F1 | 完整恢复 | 平均预测内部连接数 |', '|---|---:|---:|---:|']
    for method in methods:
        q=main.loc[method]
        lines.append(f"| {NAMES[method]} | {q.F1:.4f} | {int(q.exact)}/108 ({q.exact/108:.1%}) | {q.internal:.3f} |")
    if 'count_both_strict' not in methods:
        lines += ['', '严格最小误差规则与简约规则选择了相同的两处惩罚配置，因此合并显示，未把别名算成独立搜索。']
    b=main.loc['count_both'];u=main.loc['joint_unregularized'];o=main.loc['original_rnj_wzzt']
    lines += ['', f"预先选定的两处惩罚相对无惩罚联合模型：平均F1变化{b.F1-u.F1:+.5f}，完整恢复变化{int(b.exact-u.exact):+d}/108。"
              f"相对原RNJ+WZZT：平均F1变化{b.F1-o.F1:+.5f}，完整恢复变化{int(b.exact-o.exact):+d}/108。"]
    historical=pd.read_csv(ROOT/'docs/paper_cases_lp_rebuild_20261031/panel_summary.csv')
    historical=historical[(historical.panel=='main') & (historical.view=='bank') & historical.method.isin(['pool','original_fullwarm','best_hybrid'])]
    lines += ['', '同输入的较早基线参照（本轮未重跑、结构搜索域不同）：', '', '| 历史方法 | F1 | 完整恢复 |', '|---|---:|---:|']
    for code,label in [('pool','池内逐步修正'),('original_fullwarm','叶初值自由支撑MILP'),('best_hybrid','先逐步后局部重构的正则混合')]:
        q=historical[historical.method==code].iloc[0]
        lines.append(f'| {label} | {q.F1:.4f} | {int(q.exact)}/108 |')
    lines += ['', '相对无惩罚联合模型的逐输入配对：', '']
    for method in ('count_search','count_selection','count_both'):
        q=pairs[(pairs.panel=='main') & (pairs.view=='bank') & (pairs.method==method) & (pairs.reference=='joint_unregularized')].iloc[0]
        lines.append(f"- {NAMES[method]}：平均F1差{q.mean_delta:+.5f}，{int(q.wins)}胜/{int(q.ties)}平/{int(q.losses)}负。")
    lines += ['', '三个位置各自调到了不同强度，以上比较是完整配置的效果，不把差异全部归因于惩罚放置位置。'
              f"仅搜索惩罚的搜索中位数为{chosen.search_median:.2f}秒，无惩罚为{zero.search_median:.2f}秒，恢复改善伴随更多求解时间。"]
    lines += ['', '| 算例，每行27输入 | 原RNJ+WZZT | 无惩罚联合 | 仅搜索 | 仅选模 | 两处都加 |', '|---|---:|---:|---:|---:|---:|']
    case=frame[(frame.panel=='main') & (frame.view=='bank')].groupby(['case','method']).clade_f1.mean().unstack()
    shown=['original_rnj_wzzt','joint_unregularized','count_search','count_selection','count_both']
    for network in ('paper15','soumalas11','flynn16','pengwah18'):
        lines.append('| '+network+' | '+' | '.join(f'{case.loc[network,m]:.4f}' for m in shown)+' |')
    lines += ['', '108行包含相关的噪声/采样条件，不作为108次独立随机拓扑重复。'
              '完整恢复指非平凡终端支撑集合完全一致。', '',
              '## 是RNJ输入已错，还是后续辨识出错', '',
              'RNJ自身输出是合法层状树，内部非法交叉数为0。这里“交叉错误”指预测块与真实拓扑块交叉。'
              '“相容的多余细分”则与所有真块包含或分离，但多出了不存在的线路。', '',
              '| 原流程主面板来源 | 相容的多余细分 | 与真树交叉的错误分组 |', '|---|---:|---:|']
    for stage,label in [('rnj_input','RNJ初始固定输入'),('wzzt_added','WZZT后续新增'),('other_rnj_bank','最终回退库中另外出现的假块')]:
        g=origins[(origins.panel=='main') & (origins.stage==stage)].groupby('error_type').occurrences.sum()
        lines.append(f"| {label} | {int(g.get('compatible_overresolution',0))} | {int(g.get('crosses_truth',0))} |")
    final_origin=pd.read_csv(output/'diagnosis/final_tree_sources.csv')
    f=final_origin[final_origin.panel=='main'].groupby('selected_origin').occurrences.sum()
    lines += ['', f"按最终实际选择的树来源，误检中{int(f.get('complete_rnj',0))}次位于公共RNJ库输出，"
              f"{int(f.get('path',0))}次位于自身搜索路径输出。候选生成历史和最终选中来源分别统计，不能相互替代。"]
    lines += ['', f"{source_receipt['initial_cross_cases']}/108个输入在RNJ阶段已经含与真树交叉的错误块；"
              f"{source_receipt['later_cross_cases']}/108在后续又新增交叉错误。"
              f"其中两阶段同时出现的为{source_receipt['both_cross_cases']}例，初始无交叉而后来新增交叉为{source_receipt['clean_input_later_cross']}例。",
              '这个分布受完整RNJ固定支撑限制影响：若初值已保留真支撑，后续相容扩展本身就受到它们约束。'
              '因此不能把“未在干净初值后出现交叉”外推为WZZT一般不会出错。', '',
              f"RNJ输入共有{source_receipt['initial_total']}次结构连接，其中{source_receipt['initial_true']}次为真；"
              f"遗漏的{source_receipt['initial_missing']}次真线路中，{source_receipt['initial_blocked']}次在进入WZZT前已被错误固定块排除。"
              f"WZZT随后新增{source_receipt['added_total']}次连接，其中{source_receipt['added_true']}次为真、{source_receipt['added_false']}次为假。"
              '总数和新增误检比例需结合看：输入先限制了可恢复范围，后续无复杂度代价的扩展又增加了较多假线。', '',
              '具体例子：paper15、bundle2、0.05%电压噪声、N24，RNJ输入含{106,110}，训练权重为零，'
              '仍排除了真块{108,109,110}；后续WZZT又加入{107,108,109}。'
              '这说明零权响应与结构解除是两件事。固定粗骨架的线路数代价只是常数，惩罚不能自动解除这些错误。', '',
              '以上是同一支撑的阶段追踪及搜索域排除关系。最终公共RNJ库回退还可能换树，不能把所有最终误检直接计为后续搜索新造。', '',
              '## 惩罚减少了哪类错误', '',
              '| 主面板最终输出 | 小块FP | 小块FN | 大块FP | 大块FN |', '|---|---:|---:|---:|---:|']
    for method in methods:
        g=sizes[(sizes.panel=='main') & (sizes.view=='bank') & (sizes.method==method)].set_index('size')
        lines.append(f"| {NAMES[method]} | {int(g.loc['small','FP'])} | {int(g.loc['small','FN'])} | {int(g.loc['large','FP'])} | {int(g.loc['large','FN'])} |")
    final_types=pd.read_csv(output/'analysis/final_error_summary.csv')
    lines += ['', '| 最终假块类型，主面板 | 相容的多余细分 | 与真树交叉 |', '|---|---:|---:|']
    for method in methods:
        g=final_types[(final_types.panel=='main') & (final_types.method==method)].groupby('error_type').occurrences.sum()
        lines.append(f"| {NAMES[method]} | {int(g.get('compatible_overresolution',0))} | {int(g.get('crosses_truth',0))} |")
    lines += ['', '小块指覆盖2..5个终端，大块指>5个终端；与联合模型的“五个直属子树块”分别统计。'
              '主面板真线路次数为小块540、大块81。', '', '## 路径与最终选模', '',
              '| 方法 | 仅路径F1 | 路径完整恢复 | 未收缩终点F1 | 最终来自公共RNJ库 |', '|---|---:|---:|---:|---:|']
    for method in methods:
        q=summary[(summary.panel=='main') & (summary.view=='path') & (summary.method==method)].iloc[0]
        end=summary[(summary.panel=='main') & (summary.view=='endpoint') & (summary.method==method)].iloc[0]
        g=frame[(frame.panel=='main') & (frame.view=='bank') & (frame.method==method)]
        lines.append(f"| {NAMES[method]} | {q.F1:.4f} | {int(q.exact)}/108 | {end.F1:.4f} | {int((g.selected_origin=='complete_rnj').sum())}/108 |")
    zero_path=summary[(summary.panel=='main') & (summary.view=='path') & (summary.method=='joint_unregularized')].iloc[0]
    reg_path=summary[(summary.panel=='main') & (summary.view=='path') & (summary.method=='count_search')].iloc[0]
    lines += ['', f"不使用公共RNJ库时，仅搜索惩罚的路径选模F1仍由{zero_path.F1:.4f}升至{reg_path.F1:.4f}，"
              f"完整恢复由{int(zero_path.exact)}升至{int(reg_path.exact)}。因此本批改善也出现在搜索路径本身，并非仅由公共RNJ回退造成。"]
    coverage=pd.read_csv(output/'analysis/coverage_summary.csv')
    lines += ['', '按实际结构状态检查搜索域是否得到修复（主面板累计；初始被阻断真线路为124次）：', '',
              '| 训练lambda | 阶段 | 保留真线路 | 相容假细分 | 交叉假分组 | 被阻断真线路 |', '|---:|---|---:|---:|---:|---:|']
    for cid,config in frozen['evaluation_settings'].items():
        for stage,label in [('after_pre','预选后'),('endpoint','终点')]:
            q=coverage[(coverage.panel=='main') & (coverage.config_id==cid) & (coverage.stage==stage)].iloc[0]
            lines.append(f"| {config['coefficient']:g} | {label} | {int(q.true_present)} | {int(q.false_compatible)} | {int(q.false_cross)} | {int(q.blocked_true)} |")
    lines += ['', '搜索惩罚比较的是J而非单独训练MAE，J沿接受路径下降。'
              '仅选模惩罚复用零惩罚路径，能揭示收益来自结构搜索还是从已有候选中选择更简约的树。'
              '只在搜索中加惩罚时，最终仍可能退回较复杂的原RNJ候选；不同lambda还会改变搜索路径和时限内的解，最终线路数不要求单调。'
              '对相同连接数的错误/正确树，数量惩罚没有直接偏好；保留的错误粗边界仍需单独解决。'
              '若修复一个错误分组需要同时补入更多真实线路，数量惩罚还会增加这次修复的代价。'
              '因此它针对过度细分有明确作用方向，对错误分组的改善依赖可释放范围和量测区分能力。', '',
              '## 压力面板', '', '| 方法 | F1 | 完整恢复 |', '|---|---:|---:|']
    for method in methods:
        q=summary[(summary.panel=='stress') & (summary.view=='bank') & (summary.method==method)].iloc[0]
        lines.append(f"| {NAMES[method]} | {q.F1:.4f} | {int(q.exact)}/12 |")
    lines += ['', '## 核验与复现', '',
        f"正则目标、原始损失、界、树结构、选择分数、输入/源码等{verification['checks']}项核查通过。"
        f"不同的输入/最终家族组合共{verification['unique_independent_LP']}个，均以HiGHS独立重解；"
        f"共享家族复用结果，核查覆盖{verification['scored_bank_outputs']}个方法输出，未把别名当作独立证据。", '',
        '相同观测、公共RNJ库、0.16收缩和0..2响应界；每条搜索20秒软预算，预选5秒、单步2秒，收尾用剩余预算。'
        '原RNJ+WZZT使用上批同输入结果作参照；其计时与本批非同一次运行。'
        'MIP达到0.1%相对gap才计为达到求解容差，触时可行解单列；[solver_summary.csv](solver_summary.csv)保留各配置状态。'
        f"输入核验覆盖{input_audit['observation_archives']}个观测归档和{input_audit['network_directories']}个网络目录。"
        '源代码、参数、逐输入结果、失败和MIP状态/上下界均保存在[records.zip](records.zip)。', '',
        '```powershell','python -m research_experiments.rnj_count.run freeze outputs/rnj_count_NEW',
        'python -m research_experiments.rnj_count.run development outputs/rnj_count_NEW',
        'python -m research_experiments.rnj_count.run select outputs/rnj_count_NEW',
        'python -m research_experiments.rnj_count.run evaluation outputs/rnj_count_NEW',
        'python -m research_experiments.rnj_count.run analyze outputs/rnj_count_NEW','```','',
        '本地完整记录：`outputs/rnj_small_count_20261031/`。原始观测归档沿用既有本地基准，未在交付包重复存储。', '',
        '![F1与完整恢复](comparison.png)', '', '![开发强度扫描，F1在冻结后计算](development.png)', '']
    (destination/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    (destination/'.gitattributes').write_text('* -text\n',encoding='utf-8',newline='\n')
    for file in (output/'analysis').iterdir():
        if file.suffix=='.csv' or file.name=='verification.json':
            shutil.copy2(file,destination/file.name)
    for name in ('tuning_grid.csv','tuning_grid.json','frozen.json','development_verification.json'):
        shutil.copy2(output/name,destination/name)
    historical.to_csv(destination/'historical_references.csv',index=False)
    event_frame=pd.DataFrame([e for e in read(output/'analysis/events.json') if e.get('status') is not None])
    solver_rows=[]
    for (cid,kind),g in event_frame.groupby(['config_id','kind']):
        solver_rows.append(dict(config_id=cid,kind=kind,calls=len(g),gap_met=int((g.status==2).sum()),
            time_limit=int((g.status==9).sum()),median_native_gap=g.relative_gap.median(),median_solver_seconds=g.seconds.median()))
    pd.DataFrame(solver_rows).to_csv(destination/'solver_summary.csv',index=False)
    for name in ('error_summary.csv','error_ledger.csv','case_origins.csv','final_tree_sources.csv','verification.json'):
        shutil.copy2(output/'diagnosis'/name,destination/('origin_'+name))
    plot(frame,pd.read_csv(output/'analysis/development_summary.csv'),destination,shown)
    with zipfile.ZipFile(destination/'records.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for file in sorted(output.rglob('*')):
            if file.is_file() and file.suffix in ('.json','.py','.csv') and '__pycache__' not in file.parts:
                archive.write(file,file.relative_to(output).as_posix())
        for name in ('report.py','analyze.py','diagnose.py'):
            archive.write(Path(__file__).with_name(name),'controllers/'+name)
        for q in read(output/'analysis/baseline_references.json'):
            assert sha(q['path'])==q['sha256']
            archive.write(q['path'],'references/original_rnj_wzzt/'+q['job']+'/result.json')
        archive.write(PARENT/'protocol.json','references/original_protocol.json')
        archive.write(Path(__file__).parents[1]/'tests/test_rnj_count.py','tests/test_rnj_count.py')
        archive.write(destination/'PROTOCOL.md','PROTOCOL.md')
    with zipfile.ZipFile(destination/'records.zip') as archive:
        assert archive.testzip() is None
    for file in destination.iterdir():
        if file.is_file() and (file.suffix in ('.md','.csv','.json') or file.name=='.gitattributes'):
            file.write_bytes(file.read_bytes().replace(b'\r\n',b'\n'))
    write(destination/'delivery_manifest.json',dict(report_sha256=sha(__file__),inference_frozen_sha256=sha(output/'frozen.json'),
        files={f.name:dict(sha256=sha(f),bytes=f.stat().st_size) for f in destination.iterdir() if f.is_file() and f.name!='delivery_manifest.json'}))
    f=destination/'delivery_manifest.json';f.write_bytes(f.read_bytes().replace(b'\r\n',b'\n'))
    print('PACKAGED',destination,(destination/'records.zip').stat().st_size,flush=True)


def plot(frame,development,destination,methods):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=dict(original_rnj_wzzt='Original RNJ + WZZT',joint_unregularized='Joint, no penalty',count_search='Search penalty',count_selection='Selection penalty',count_both='Both penalties')
    bank=frame[(frame.panel=='main') & (frame.view=='bank')];case=bank.groupby(['case','method']).clade_f1.mean().unstack()
    names=['paper15','soumalas11','flynn16','pengwah18'];fig,axes=plt.subplots(1,2,figsize=(13,4.8),gridspec_kw={'width_ratios':[1.5,1]})
    x=np.arange(4);width=.15
    for k,method in enumerate(methods):
        axes[0].bar(x+(k-2)*width,[case.loc[n,method] for n in names],width,label=labels[method])
    axes[0].set_xticks(x,names);axes[0].set_ylim(0,1.05);axes[0].set_ylabel('Mean support F1')
    axes[0].legend(fontsize=8,loc='upper center',bbox_to_anchor=(.5,1.18),ncol=3)
    exact=bank.groupby('method').exact.sum()
    axes[1].barh([labels[m] for m in methods],[exact[m] for m in methods]);axes[1].invert_yaxis();axes[1].set_xlim(0,108)
    axes[1].set_xlabel('Complete support recovery / 108')
    for i,m in enumerate(methods):
        axes[1].text(exact[m]+1,i,str(exact[m]),va='center')
    fig.tight_layout();fig.savefig(destination/'comparison.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.2))
    for mode,label in [('search_only','Search'),('selection_only','Selection'),('both','Both')]:
        g=development[development['mode']==mode].sort_values('coefficient')
        axes[0].plot(g.coefficient,g.F1,'o-',label=label);axes[1].plot(g.coefficient,g.internal,'o-',label=label)
    for ax in axes:
        ax.set_xscale('symlog',linthresh=.0003);ax.set_xlabel('Penalty coefficient');ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Development F1 (scored after freeze)');axes[1].set_ylabel('Mean internal supports')
    fig.tight_layout();fig.savefig(destination/'development.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('destination');a=parser.parse_args()
    generate(Path(a.output).resolve(),Path(a.destination).resolve())
