"""Archive the stage comparison and separate structural error mechanisms."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import zipfile
import numpy as np
import pandas as pd

from .run import PARENT, read, write, sha

NAMES = dict(baseline='原RNJ+WZZT', rnj_pre='只改RNJ预选', wzzt_tail='只改WZZT收尾', both='两处都改')


def audit_inputs(output):
    old=read(PARENT/'input_audit.json');catalog=read(output/'catalog.json');checks=0
    for name,digest in old['input_hashes'].items():
        assert sha(name)==digest;checks+=1
    networks={q['case']:q for q in old['networks']}
    seen=set()
    for meta in catalog.values():
        root=Path(meta['truth_network'])
        if root in seen:
            continue
        seen.add(root);q=networks[meta['case']]
        assert sha(root/'buses.csv')==q['buses_sha256'] and sha(root/'branches.csv')==q['branches_sha256'];checks+=2
    receipt=dict(passed=True,checks=checks,observation_archives=len(old['input_hashes']),network_directories=len(seen),parent_audit_sha256=sha(PARENT/'input_audit.json'))
    write(output/'input_audit_verification.json',receipt)
    return receipt


def generate(output, destination):
    output = Path(output); destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    analysis = output/'analysis'; verification = read(analysis/'verification.json')
    assert verification['passed']
    input_audit=audit_inputs(output)
    frame = pd.read_csv(analysis/'per_model.csv'); summary = pd.read_csv(analysis/'summary.csv')
    errors = pd.read_csv(analysis/'size_summary.csv'); blocked = pd.read_csv(analysis/'blocking_summary.csv')
    paired = pd.read_csv(analysis/'paired.csv'); bank = frame[(frame.panel=='main') & (frame.view=='bank')]
    events = read(analysis/'events.json'); first = read(analysis/'first_errors.json')
    main=summary[(summary.panel=='main') & (summary.view=='bank')].set_index('method')
    speed=1-main.loc['wzzt_tail','search_median']/main.loc['baseline','search_median']
    endpoint_counts=errors[(errors.panel=='main') & (errors.stage=='endpoint') & (~errors.structural)].groupby('method')[['TP','FP']].sum()
    extra_tp=int(endpoint_counts.loc['both','TP']-endpoint_counts.loc['baseline','TP'])
    extra_fp=int(endpoint_counts.loc['both','FP']-endpoint_counts.loc['baseline','FP'])
    lines = ['# 五块以内一次联合MILP：RNJ预选与WZZT收尾', '',
        f"120输入×4方法，返回{verification['returned']}/{verification['planned']}；"
        f"{verification['independent_LP']}次独立HiGHS最终家族重解通过。",
        '本批使用近期四算例的完整RNJ初始层级，不等同于生产默认bootstrap最多两个边界块。'
        '矩阵为108主输入和12压力输入，配置见[PROTOCOL.md](PROTOCOL.md)。生产默认未修改。', '',
        f"主要计算收益来自联合收尾：搜索中位数减少{speed:.1%}，平均F1近似持平，完整恢复为"
        f"{int(main.loc['baseline','exact'])}→{int(main.loc['wzzt_tail','exact'])}。"
        '两处都改的平均F1仅略升，配对胜负和压力面板不支持稳定的恢复优势。当前保留为研究选项，不替换默认流程。', '',
        '## 模型和两处介入', '',
        '每个参与区域最多五个直属块，列出至多25个候选线路；全部区域在同一次MILP中选择连接，'
        '共同拟合原始终端P/Q/Y。交叉冲突约束保证层状结构，原有外围权重也联合重拟合。'
        '同一次求解可删除已释放的小合并、同时加入多条线路，没有逐候选LP接续。', '',
        'RNJ预选介入只打开可在五块区域内重选的≤5终端合并，保留较大骨架边界。'
        'WZZT收尾在最大剩余直属块数≤5时调用一次联合模型，随后结束搜索。'
        '四组共用20秒搜索软预算，预选联合最多5秒，原单步MILP2秒，收尾用剩余预算；无新正下界、无连接数惩罚。', '',
        '## 主面板结果', '',
        '| 方法 | 平均F1 | 完整恢复 | 搜索中位数(s) | 验证MAE | 测试MAE |',
        '|---|---:|---:|---:|---:|---:|']
    for method in NAMES:
        q = summary[(summary.panel=='main') & (summary.view=='bank') & (summary.method==method)].iloc[0]
        lines.append(f"| {NAMES[method]} | {q.F1:.4f} | {int(q.exact)}/108 | {q.search_median:.2f} | {q.validation_MAE:.7f} | {q.test_MAE:.7f} |")
    lines += ['', '| 算例，每行27输入 | 原流程 | 只改预选 | 只改收尾 | 两处都改 |', '|---|---:|---:|---:|---:|']
    cases = bank.groupby(['case','method']).clade_f1.mean().unstack().reindex(columns=NAMES)
    for case in ('paper15','soumalas11','flynn16','pengwah18'):
        lines.append('| '+case+' | '+' | '.join(f'{cases.loc[case,m]:.4f}' for m in NAMES)+' |')
    lines += ['', '主面板为三个bundle下相关的噪声和采样条件，不把108行当作108次独立随机网络重复。'
              'F1评价非平凡终端支撑；完整恢复是该支撑集合完全一致。', '', '相对原流程的配对F1：', '']
    for method in ('rnj_pre','wzzt_tail','both'):
        q = paired[(paired.panel=='main') & (paired.view=='bank') & (paired.method==method)].iloc[0]
        lines.append(f"- {NAMES[method]}：平均差{q.mean_delta:+.5f}，{int(q.wins)}胜/{int(q.ties)}平/{int(q.losses)}负。")
    lines += ['', '仅自身路径的平均F1（不使用公共RNJ库回退）：', '', '| 方法 | F1 | 完整恢复 |', '|---|---:|---:|']
    for method in NAMES:
        q=summary[(summary.panel=='main') & (summary.view=='path') & (summary.method==method)].iloc[0]
        lines.append(f"| {NAMES[method]} | {q.F1:.4f} | {int(q.exact)}/108 |")
    lines += ['', '## 联合模型实际执行情况', '', '| 方法 | 预选调用/接受 | 收尾调用/接受 | 最终来自公共RNJ库 |', '|---|---:|---:|---:|']
    for method in NAMES:
        g=bank[bank.method==method]
        lines.append(f"| {NAMES[method]} | {int(g.pre_invoked.sum())}/{int(g.pre_accepted.sum())} | "
                     f"{int(g.tail_invoked.sum())}/{int(g.tail_accepted.sum())} | {int((g.selected_origin=='complete_rnj').sum())}/108 |")
    gaprows=[]
    for method in NAMES:
        for kind in ('rnj_pre_joint','wzzt_tail_joint','wzzt_extension'):
            group=[e for e in events if e['method']==method and e['kind']==kind and e.get('status') is not None]
            if not group:
                continue
            gaps=[e['relative_gap'] for e in group if e.get('relative_gap') is not None]
            gaprows.append(dict(method=method,kind=kind,calls=len(group),gap_met=sum(e['status']==2 for e in group),
                time_limit=sum(e['status']==9 for e in group),median_gap=float(np.median(gaps)) if gaps else None,
                median_solver_seconds=float(np.median([e['seconds'] for e in group]))))
    pd.DataFrame(gaprows).to_csv(destination/'solver_summary.csv',index=False)
    lines += ['', '状态2表示达到0.1%相对gap容差，触时incumbent和有效bound单独保留。'
              '求解器汇总见[solver_summary.csv](solver_summary.csv)，未达容差不计为已证明最优。', '',
              '## 为什么训练拟合改善而结构恢复有限', '',
              '| 未收缩搜索终点 | 平均训练MAE | 平均F1 | 正确线路TP | 假线路FP |',
              '|---|---:|---:|---:|---:|']
    for method in ('baseline','both'):
        q=summary[(summary.panel=='main') & (summary.view=='endpoint') & (summary.method==method)].iloc[0]
        g=errors[(errors.panel=='main') & (errors.method==method) & (errors.stage=='endpoint') & (~errors.structural)]
        lines.append(f"| {NAMES[method]} | {q.train_MAE:.7f} | {q.F1:.4f} | {int(g.TP.sum())} | {int(g.FP.sum())} |")
    lines += ['', f'按未收缩终点的正权重口径（阈值1e-9），两处都改在108输入上累计正确检出变化{extra_tp:+d}次、误检变化{extra_fp:+d}次。'
              '原始L1目标没有线路数代价，额外连接可用于吸收噪声和线性模型偏差；本轮计数显示这种自由度带来了更多误检。'
              '最终收缩、验证选模和公共RNJ库回退对维持输出F1起主要作用。'
              '因此本轮的小幅最终均值改善，不能改写成联合训练终点的拓扑质量提高。', '',
              '## 当前错误：大块还是小块', '',
              '错误按覆盖终端数分组：小块2..5个终端，大块>5个终端。'
              '这与切换使用的“直属块数≤5”是两个口径。下表只看原流程108个最终有效输出：', '',
              '| 线路大小 | 预测数 | 真线路数 | 误检FP | 漏检FN | 误检比例FP/预测 | 漏检比例FN/真线路 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for size in ('small','large'):
        q=errors[(errors.panel=='main') & (errors.method=='baseline') & (errors.stage=='bank') & (~errors.structural) & (errors['size']==size)].iloc[0]
        lines.append(f"| {'小块' if size=='small' else '大块'} | {int(q.predicted)} | {int(q.truth)} | {int(q.FP)} | {int(q.FN)} | "
                     f"{q.false_discovery_rate:.1%} | {q.miss_rate:.1%} |")
    lines += ['', '四个算例的真内部支撑终端数分别为：paper15为3/3/3/5/5/5；soumalas11为3/3/3/5；'
              'flynn16为3/3/4/4/8/8；pengwah18为3/3/3/4/4/5/6。'
              '主面板共有540次小块真线路、81次大块真线路，原始错误数受分母差异影响，需结合比例判断。']
    lines += ['', '在该基准中，小块的误检总数、漏检总数以及各自错误比例都更高。'
              '修正重点应放在小合并的过度细分和错误锁定；大块仍需核查，不能仅因覆盖终端多就当作可靠先验。']
    lines += ['', '再看原流程搜索终点中，错误结构排除了多少真实线路。初始固定集合即使权重为零，也计入结构限制。'
              '同一真线路可能同时被多个来源阻断，以下列之间有重叠：', '',
              '| 阻断来源 | 被排除的真线路次数 |', '|---|---:|']
    b=blocked[(blocked.panel=='main') & (blocked.method=='baseline')].iloc[0]
    for field,label in [('rnj_small','RNJ预选的小块'),('rnj_large','RNJ预选的大块'),('wzzt_small','WZZT后加的小块'),('wzzt_large','WZZT后加的大块')]:
        lines.append(f'| {label} | {int(b[field])} |')
    lines += [f"| 去重后的全部阻断 | {int(b.blocked)} |", '',
              '这项统计核查的是实际搜索域排除关系：错误集合与真集合交叉，后续相容扩展就无法加入它。'
              '它比只数误检更接近结构损害，但仍不把重叠计数解释为互斥因果贡献。', '',
              '原流程初始RNJ结构与后续新增错误：', '']
    exclusive=pd.read_csv(analysis/'blocking_exclusive.csv')
    x=exclusive[(exclusive.panel=='main') & (exclusive.method=='baseline')].set_index('category').true_line_occurrences
    lines.append(f"按大小去重：仅受小块阻断{int(x.get('small_only',0))}次，仅受大块阻断{int(x.get('large_only',0))}次，"
                 f"同时受两者阻断{int(x.get('both_sizes',0))}次；与当前集合相容但仍遗漏{int(x.get('compatible_but_missing',0))}次。")
    baseline_first=[q for q in first if q['panel']=='main' and q['method']=='baseline']
    new=[q['first_new_wrong'] for q in baseline_first if q['first_new_wrong'] is not None]
    lines.append(f"- 初始RNJ含错误固定集合：{sum(q['initial_wrong']>0 for q in baseline_first)}/108输入。")
    lines.append(f"- 后续路径首次新增错误：{len(new)}/108输入；其中首次错误批次含小块{sum(q['small']>0 for q in new)}例、含大块{sum(q['large']>0 for q in new)}例。")
    lines += ['', '逐输入、大小和阶段的TP/FP/FN见[size_errors.csv](size_errors.csv)，'
              '逐条真线路及其阻断者见[blocking_rows.csv](blocking_rows.csv)。'
              'RNJ、预选修正和后续WZZT的责任需结合这些阶段记录判断；最终验证选模的回退来源也已单列。', '',
              '## 压力面板', '', '| 方法 | 平均F1 | 完整恢复 |', '|---|---:|---:|']
    for method in NAMES:
        q=summary[(summary.panel=='stress') & (summary.view=='bank') & (summary.method==method)].iloc[0]
        lines.append(f"| {NAMES[method]} | {q.F1:.4f} | {int(q.exact)}/12 |")
    lines += ['', '## 证据与复现', '',
        f"原始损失、层状性、界、单调性、切换、状态与输入/源码等{verification['checks']}项核查通过；"
        f"独立HiGHS重解{verification['independent_LP']}次；失败{len(verification['failures'])}。"
        f"另核验{input_audit['observation_archives']}个观测归档及{input_audit['network_directories']}个网络目录与父批次一致。", '',
        '```powershell', 'python -m research_experiments.joint_five.run freeze outputs/joint_five_NEW',
        'python -m research_experiments.joint_five.run run outputs/joint_five_NEW',
        'python -m research_experiments.joint_five.run analyze outputs/joint_five_NEW', '```', '',
        '完整本地记录：`outputs/joint_five_rnj_wzzt_20261031/`。'
        '[records.zip](records.zip)保存冻结源码、输入索引、公共前端、全部逐项结果和核查；原始观测NPZ沿用既有本地基准。', '',
        '![F1与错误类型](comparison.png)', '']
    (destination/'README.md').write_text('\n'.join(lines),encoding='utf-8',newline='\n')
    (destination/'.gitattributes').write_text('* -text\n',encoding='utf-8')
    for file in analysis.iterdir():
        if file.suffix=='.csv' or file.name=='verification.json':
            shutil.copy2(file,destination/file.name)
    plot(cases,errors,destination/'comparison.png')
    with zipfile.ZipFile(destination/'records.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
        for file in sorted(output.rglob('*')):
            if file.is_file() and file.suffix in ('.json','.py') and '__pycache__' not in file.parts:
                archive.write(file,file.relative_to(output).as_posix())
        archive.write(__file__,'report_controller.py')
        archive.write(Path(__file__).with_name('analyze.py'),'analysis_controller.py')
        archive.write(Path(__file__).parents[1]/'tests/test_joint_five.py','tests/test_joint_five.py')
        archive.write(destination/'PROTOCOL.md','PROTOCOL.md')
    with zipfile.ZipFile(destination/'records.zip') as archive:
        assert archive.testzip() is None
    for file in destination.iterdir():
        if file.is_file() and (file.suffix in ('.md','.csv','.json') or file.name=='.gitattributes'):
            file.write_bytes(file.read_bytes().replace(b'\r\n',b'\n'))
    write(destination/'delivery_manifest.json',dict(planned=verification['planned'],returned=verification['returned'],
        report_sha256=sha(__file__),files={f.name:dict(sha256=sha(f),bytes=f.stat().st_size) for f in destination.iterdir()
            if f.is_file() and f.name!='delivery_manifest.json'}))
    manifest=destination/'delivery_manifest.json';manifest.write_bytes(manifest.read_bytes().replace(b'\r\n',b'\n'))
    print('PACKAGED',destination,(destination/'records.zip').stat().st_size,flush=True)


def plot(cases,errors,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=dict(baseline='Original RNJ + WZZT',rnj_pre='Joint RNJ preselection',wzzt_tail='Joint WZZT tail',both='Both stages')
    cases=cases.reindex(['paper15','soumalas11','flynn16','pengwah18'])
    fig,axes=plt.subplots(1,2,figsize=(12,4.6),gridspec_kw={'width_ratios':[1.6,1]})
    x=np.arange(4);width=.19
    for k,m in enumerate(NAMES):
        axes[0].bar(x+(k-1.5)*width,cases[m],width,label=labels[m])
    axes[0].set_xticks(x,cases.index);axes[0].set_ylim(0,1.05);axes[0].set_ylabel('Mean nontrivial-support F1')
    axes[0].set_title('Main panel: 27 inputs per network');axes[0].legend(fontsize=8)
    g=errors[(errors.panel=='main') & (errors.method=='baseline') & (errors.stage=='bank') & (~errors.structural)].set_index('size')
    for k,kind in enumerate(('FP','FN')):
        axes[1].bar(np.arange(2)+(k-.5)*.34,[g.loc[s,kind] for s in ('small','large')],.34,label=kind)
    axes[1].set_xticks([0,1],['2-5 terminals','>5 terminals']);axes[1].set_title('Original final-output errors');axes[1].legend()
    for ax in axes:
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.tight_layout();fig.savefig(path,dpi=180);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output');parser.add_argument('destination')
    args=parser.parse_args();generate(Path(args.output).resolve(),Path(args.destination).resolve())
