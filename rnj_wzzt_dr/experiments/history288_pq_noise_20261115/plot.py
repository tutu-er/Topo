"""Plot audited historical-noise sensitivity; no fitting, planning or AC calls.

Usage: python -B -m rnj_wzzt_dr.experiments.history288_pq_noise_20261115.plot
       --report <report.json> --out <new-directory> [--fee total|activation]
The self-test uses labelled synthetic summaries and never opens experiment data.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager
from matplotlib.ticker import PercentFormatter
import numpy as np


CASES = ('soumalas11', 'pengwah18')
NOISE_LEVELS = (0., .0025, .005, .01, .02)
POLICIES = ('no_dfs', 'single_normal_fixed', 'single_mgf_fixed', 'single_mgf_forecast')
NAMES = {
    'no_dfs': '无采购',
    'single_normal_fixed': '固定功率正态门槛',
    'single_mgf_fixed': '固定功率指数门槛',
    'single_mgf_forecast': '随机功率指数门槛',
}
EN_NAMES = {
    'no_dfs': 'No procurement',
    'single_normal_fixed': 'Normal, fixed power',
    'single_mgf_fixed': 'Exponential, fixed power',
    'single_mgf_forecast': 'Exponential, random power',
}
COLORS = dict(zip(POLICIES, ('#626262', '#7A5195', '#0072B2', '#009E73')))
MARKERS = dict(zip(POLICIES, ('o', 'D', 's', '^')))
STYLES = dict(zip(POLICIES, ('--', '-', '-.', ':')))


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=digest(path))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf8')


def checked_json(reference):
    path = Path(reference['path'])
    if digest(path) != reference['sha256']:
        raise ValueError('Input hash differs: ' + str(path))
    return json.loads(path.read_text(encoding='utf8'))


def count(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError('Expected nonnegative integer for ' + label)
    return int(value)


def statistic(value, label):
    n = count(value['n'], label + '.n')
    mean = value['mean']
    if n == 0:
        if mean is not None:
            raise ValueError('Empty mean must be null: ' + label)
        return None
    if mean is None or not math.isfinite(mean) or mean < -1e-9:
        raise ValueError('Invalid nonnegative fee mean: ' + label)
    return float(mean)


def prepare_data(report):
    """Validate planned denominators and retain failures rather than drop rows."""
    if report.get('status') != 'complete':
        raise ValueError('A completed report is required')
    repeats = count(report['repeats'], 'repeats')
    if not repeats:
        raise ValueError('Execution repeat count must be positive')
    groups = report['groups']
    keys = [(g['case'], float(g['pq_relative_sd'])) for g in groups]
    if len(set(keys)) != len(keys) or set(k[0] for k in keys) != set(CASES):
        raise ValueError('Exactly the two declared networks and unique groups are required')
    if not report.get('smoke'):
        expected = {(c, sd) for c in CASES for sd in NOISE_LEVELS}
        if set(keys) != expected or repeats != 16 or report['planned_inputs'] != 40:
            raise ValueError('Formal 2-network x 5-noise x 4-history x 16-path ledger differs')
    cells = {c['id']: c for c in report['cells']}
    if len(cells) != len(report['cells']):
        raise ValueError('Duplicate cell id')
    quotes = {}
    for quote in report['quote_records']:
        name = quote.get('policy', quote.get('strategy'))
        key = (quote['id'], name)
        if key in quotes or quote['id'] not in cells or name not in POLICIES[1:]:
            raise ValueError('Duplicate, unmatched or unexpected quote: ' + str(key))
        quotes[key] = quote
    result, flat = [], []
    for g in sorted(groups, key=lambda x: (CASES.index(x['case']), x['pq_relative_sd'])):
        histories = count(g['historical_repetitions'], 'historical_repetitions')
        if not report.get('smoke') and histories != 4:
            raise ValueError('Each formal noise group must retain four histories')
        ids = {tag for tag, c in cells.items()
               if c['case'] == g['case'] and float(c['pq_relative_sd']) == float(g['pq_relative_sd'])}
        if len(ids) != histories:
            raise ValueError('Cell ledger and group denominator differ')
        item = dict(case=g['case'], pq_relative_sd=float(g['pq_relative_sd']),
                    historical_repetitions=histories, history={}, strategies={})
        for view in ('original', 'enriched'):
            h = g['history'][view]
            n, complete, failed, correct, covered = [count(h[k], view + '.' + k)
                for k in ('n', 'completed', 'failed', 'correct', 'covered')]
            if n != histories or complete + failed != n or not 0 <= correct <= covered <= complete:
                raise ValueError('Historical counts are inconsistent')
            item['history'][view] = dict(n=n, completed=complete, failed=failed,
                correct=correct, covered=covered, correct_fraction=correct / n,
                coverage_fraction=covered / n,
                full_support_correct=count(h['full_support_correct'], 'full_support_correct'),
                full_support_covered=count(h['full_support_covered'], 'full_support_covered'))
            flat.append(dict(case=g['case'], pq_relative_sd=g['pq_relative_sd'],
                view='history_' + view, policy='', planned=n, completed=complete,
                confirmed_internal_correct=correct, candidate_internal_coverage=covered,
                full_support_correct=h['full_support_correct'], mean_activation_yuan=None,
                mean_total_yuan=None, zero_fallback_quotes=None, unavailable_quotes=None))
        for name in POLICIES:
            s = g['strategies'][name]
            n, complete, recovered = [count(s[k], name + '.' + k) for k in ('n', 'completed', 'recovered')]
            if n != histories * repeats or not 0 <= recovered <= complete <= n:
                raise ValueError('Execution counts are inconsistent')
            costs = {}
            for kind in ('activation', 'total'):
                stats = s['settled_' + kind + '_yuan']
                if stats['n'] != complete:
                    raise ValueError('Fee mean must cover every completed path and no missing path')
                costs[kind] = statistic(stats, name + '.' + kind)
            qq = [quotes[(tag, name)] for tag in ids if (tag, name) in quotes]
            fallback = sum(bool(q.get('zero_fallback')) for q in qq)
            planned_quotes = 0 if name == 'no_dfs' else histories
            unavailable = planned_quotes - len(qq)
            if unavailable < 0:
                raise ValueError('Too many quotes')
            item['strategies'][name] = dict(n=n, completed=complete, unavailable_rows=n-complete,
                recovered=recovered, recovery_fraction=recovered/n,
                full_support_recovered=count(s['full_support_recovered'], 'full_support_recovered'),
                wrong_high=count(s['wrong_high'], 'wrong_high'),
                mean_activation_yuan=costs['activation'], mean_total_yuan=costs['total'],
                zero_fallback_rows=count(s['zero_fallback_rows'], 'zero_fallback_rows'),
                zero_fallback_quotes=fallback, planned_quotes=planned_quotes,
                observed_quotes=len(qq), unavailable_quotes=unavailable,
                fallback_fraction=fallback/planned_quotes if planned_quotes else None)
            flat.append(dict(case=g['case'], pq_relative_sd=g['pq_relative_sd'], view='execution',
                policy=name, planned=n, completed=complete, confirmed_internal_correct=recovered,
                candidate_internal_coverage=None, full_support_correct=s['full_support_recovered'],
                mean_activation_yuan=costs['activation'], mean_total_yuan=costs['total'],
                zero_fallback_quotes=fallback, unavailable_quotes=unavailable))
        result.append(item)
    return result, flat


def select_font():
    for path in (Path('C:/Windows/Fonts/msyh.ttc'), Path('C:/Windows/Fonts/simhei.ttf')):
        if path.is_file():
            fontManager.addfont(str(path))
            name = FontProperties(fname=str(path)).get_name()
            return name, str(path), True
    return 'DejaVu Sans', None, False


def draw(groups, report, out, fee):
    font, font_path, zh = select_font()
    tr = lambda chinese, english: chinese if zh else english
    names = NAMES if zh else EN_NAMES
    style = {'font.family': font, 'font.size': 10, 'axes.unicode_minus': False,
             'axes.spines.top': False, 'axes.spines.right': False,
             'svg.fonttype': 'none', 'svg.hashsalt': 'history288-pq-noise-v1',
             'savefig.facecolor': 'white'}
    with plt.rc_context(style):
        fig = plt.figure(figsize=(17, 10.7))
        grid = fig.add_gridspec(2, 3, left=.065, right=.985, top=.83, bottom=.17,
                               wspace=.27, hspace=.43)
        axes = []
        for row, case in enumerate(CASES):
            gg = [g for g in groups if g['case'] == case]
            x = np.array([100*g['pq_relative_sd'] for g in gg])
            ticks = [f'{v:g}' for v in x]
            history = fig.add_subplot(grid[row, 0])
            execution = fig.add_subplot(grid[row, 1])
            cost_grid = grid[row, 2].subgridspec(2, 1, height_ratios=(3.1, 1), hspace=.60)
            cost = fig.add_subplot(cost_grid[0])
            table_ax = fig.add_subplot(cost_grid[1])
            table_ax.set_axis_off()
            axes.extend((history, execution, cost))
            for view, color in (('original', '#0072B2'), ('enriched', '#D55E00')):
                for key, marker, line in (('correct_fraction', 'o' if view=='original' else 'D', '-'),
                                          ('coverage_fraction', 's' if view=='original' else '^', '--')):
                    label = tr(('原辨识' if view=='original' else '追加候选后') +
                               ('：正确' if key=='correct_fraction' else '：覆盖'),
                               ('Original' if view=='original' else 'Enriched') +
                               (' correct' if key=='correct_fraction' else ' coverage'))
                    y = [g['history'][view][key] for g in gg]
                    history.plot(x, y, label=label, color=color, marker=marker,
                                 linestyle=line, markersize=6, linewidth=1.65,
                                 markerfacecolor='white' if key=='coverage_fraction' else color)
                    incomplete = [i for i,g in enumerate(gg)
                                  if g['history'][view]['completed']<g['history'][view]['n']]
                    history.scatter(x[incomplete], np.array(y)[incomplete], marker='x',
                                    color='#B00020', s=65, linewidth=1.5, zorder=7)
            for name in POLICIES:
                y = [g['strategies'][name]['recovery_fraction'] for g in gg]
                execution.plot(x, y, label=names[name], color=COLORS[name], marker=MARKERS[name],
                               linestyle=STYLES[name], markersize=6, linewidth=1.8,
                               markerfacecolor='white' if name=='no_dfs' else COLORS[name])
                incomplete = [i for i,g in enumerate(gg)
                              if g['strategies'][name]['completed']<g['strategies'][name]['n']]
                execution.scatter(x[incomplete], np.array(y)[incomplete], marker='x',
                                  color='#B00020', s=65, linewidth=1.5, zorder=7)
                if name != 'no_dfs':
                    values = [g['strategies'][name]['mean_'+fee+'_yuan'] for g in gg]
                    values = np.array([np.nan if v is None else v for v in values])
                    cost.plot(x, values, color=COLORS[name], marker=MARKERS[name],
                              linestyle=STYLES[name], linewidth=1.8, markersize=6)
                    cost.scatter(x[incomplete], values[incomplete], marker='x',
                                 color='#B00020', s=65, linewidth=1.5, zorder=7)
            fallback_rows = []
            for name in POLICIES[1:]:
                cells = []
                for g in gg:
                    s=g['strategies'][name]
                    text=f"{s['zero_fallback_quotes']}/{s['planned_quotes']}"
                    if s['unavailable_quotes']:
                        text+=f" (+{s['unavailable_quotes']}?)"
                    cells.append(text)
                fallback_rows.append(cells)
            tab = table_ax.table(cellText=fallback_rows, colLabels=ticks,
                                 rowLabels=[tr('正态', 'Normal'),tr('固定指数','Fixed exp.'),
                                            tr('随机指数','Random exp.')],
                                 loc='center', cellLoc='center', colLoc='center')
            tab.auto_set_font_size(False);tab.set_fontsize(8.3);tab.scale(1,1.12)
            for (r,c),cell in tab.get_celld().items():
                cell.set_edgecolor('#E1E1E1');cell.set_linewidth(.5)
                if r==0:cell.set_facecolor('#F1F3F5')
                if c==-1 and r>0:cell.get_text().set_color(COLORS[POLICIES[r]])
            table_ax.set_title(tr('零动作回退报价 / 计划报价（列为 P/Q 误差 %）',
                                  'Zero fallback / planned quotes (columns: P/Q SD %)'),
                               fontsize=8.3, pad=7)
            case_label = 'Soumalas · 11' if case=='soumalas11' else 'Pengwah · 18'
            history.set_title(case_label+'  |  '+tr('历史内部拓扑', 'Historical internal topology'),loc='left')
            execution.set_title(case_label+'  |  '+tr('第4天实际内部恢复', 'Day-4 actual internal recovery'),loc='left')
            cost.set_title(case_label+'  |  '+tr('服务总费' if fee=='total' else '调用费',
                                              'Total service fee' if fee=='total' else 'Activation fee'),loc='left')
            for ax in (history, execution):
                ax.set_ylim(-.04,1.07);ax.set_yticks([0,.25,.5,.75,1])
                ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
                ax.set_xlabel(tr('P/Q 附加量测标准差（%）','Additional P/Q metering SD (%)'))
                ax.set_ylabel(tr('已确认正确 / 原定数量','Confirmed correct / planned'))
            history.set_ylabel(tr('正确或覆盖 / 原定历史数','Correct or covered / planned histories'))
            cost.set_ylabel(tr('元 / 执行日','Yuan / execution day'));cost.set_ylim(bottom=0)
            for ax in (history, execution, cost):
                ax.set_xticks(x, ticks);ax.set_xlim(min(x)-.05,max(x)+.07 if len(x)>1 else max(x)+.05)
                ax.grid(axis='y',alpha=.23);ax.set_axisbelow(True)
            if row==0:
                history.legend(loc='lower left',bbox_to_anchor=(-.01,1.14),ncol=2,
                               frameon=False,fontsize=9,columnspacing=1.0,handlelength=2.1)
                execution.legend(loc='lower left',bbox_to_anchor=(-.01,1.14),ncol=2,
                                 frameon=False,fontsize=9,columnspacing=1.0,handlelength=2.1)
        tag = tr('静态示例，仅核对绘图', 'SYNTHETIC PLOTTING FIXTURE') if report.get('synthetic_fixture') else (
              tr('接线烟测，非正式结果','SMOKE TEST, NOT FORMAL RESULTS') if report.get('smoke') else '')
        title = tr('P/Q 附加量测误差：历史辨识与第4天执行',
                   'Added P/Q metering error: historical identification and day-4 execution')
        fig.suptitle(title + ('  —  '+tag if tag else ''), fontsize=17, y=.985)
        subtitle = tr('192 点拟合 + 96 点验证；V / 根电压标准差固定 0.05%；预测误差固定 5%',
                      '192 fit + 96 validation samples; V/root SD fixed at 0.05%; forecast error fixed at 5%')
        fig.text(.5,.947,subtitle,ha='center',fontsize=11,color='#444444')
        hcounts=sorted({g['historical_repetitions'] for g in groups})
        pcounts=sorted({g['strategies']['no_dfs']['n'] for g in groups})
        scope = tr(f"每网络每档：{'/'.join(map(str,hcounts))} 次历史噪声重复 × {report['repeats']} 条执行路径 = {'/'.join(map(str,pcounts))} 条条件路径。\n"
                   '各网络仅一个固定物理窗口；条件路径不等于独立物理日。覆盖指候选库含正确内部拓扑，完整响应支持另见报告。',
                   f"Per network/noise level: {'/'.join(map(str,hcounts))} history-noise draws × {report['repeats']} execution draws = {'/'.join(map(str,pcounts))} conditional paths.\n"
                   'One fixed physical window per network. Conditional paths are not independent physical days; coverage concerns internal topology.')
        fee_note = tr('费用为已完成执行记录的均值；总费 = 调用费 + 预留费，优化目标仅为调用费。',
                      'Fees average completed executions; total = activation + availability. Only activation is optimized.')
        warning = tr('红叉：存在缺失/失败，正确率分母仍保留；表中 ? 为未形成报价。零回退单列，不等同零采购已满足规划门槛。\n'
                     '图中为实际恢复频率，不是模型内概率界或“拓扑为真”的概率。0% 档仍含原始来源未校准的量测误差。',
                     'Red cross: missing/failed records, retained in the recovery denominator; ? indicates missing quotes. Fallback is not a zero-sufficient plan.\n'
                     'Curves are actual recovery frequencies, not model probability bounds. 0% adds no P/Q error; source measurement error remains unknown.')
        fig.text(.065,.112,scope,va='top',fontsize=9.6,linespacing=1.6)
        fig.text(.065,.062,fee_note+'\n'+warning,va='top',fontsize=9.0,linespacing=1.5,color='#444444')
        png=out/'noise_sensitivity.png';svg=out/'noise_sensitivity.svg'
        fig.savefig(png,dpi=180)
        fig.savefig(svg,metadata={'Date':None,'Creator':'history288_pq_noise.plot'})
        plt.close(fig)
        # Hash the final, whitespace-normalized SVG rather than a pre-cleanup file.
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf8').splitlines())+'\n',encoding='utf8')
    return dict(font_family=font,font_path=font_path,chinese_labels=zh,
                matplotlib_version=matplotlib.__version__,numpy_version=np.__version__)


def plot(report_path, out, *, fee='total'):
    started=time.perf_counter();report_path=Path(report_path).resolve();out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    if fee not in ('total','activation'):raise ValueError('Unknown fee')
    script_ref=ref(__file__);report_ref=ref(report_path);data=checked_json(report_ref)
    audit_ref=data['source_audit'];audit=checked_json(audit_ref)
    if audit.get('status')!='passed' or audit.get('task_errors') or audit['audited_rows']!=audit['expected_rows']:
        raise ValueError('A complete passed independent audit is required')
    groups,flat=prepare_data(data)
    out.mkdir(parents=True)
    rendering=draw(groups,data,out,fee)
    write(out/'plot_data.json',dict(source_report=report_ref,groups=groups))
    with (out/'plot_data.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    fixture=bool(data.get('synthetic_fixture'))
    text=['# P/Q 误差曲线','',
          '本目录仅为静态示例核对绘图，数值不是实验结果。' if fixture else
          ('本目录为短预算接线烟测，不用于研究结论。' if data.get('smoke') else '本目录直接绘制独立审计通过后的汇总数据。'),'',
          '左列：原辨识与追加候选后的内部拓扑正确率，以及候选库覆盖率。',
          '中列：第4天无采购及三种概率门槛规划的实际内部拓扑恢复频率。',
          '右列：'+('调用费与预留费之和' if fee=='total' else '调用费')+'，下表是零动作回退报价数/原定报价数。',
          '内部拓扑与包含单节点、共同响应项的完整支持不同；完整支持结果保留于 plot_data.json 和原报告。','',
          '正式设计每档4次历史噪声，每次16条条件执行路径；两网络各只有一个固定物理窗口，不能把64条条件路径当作64个独立物理日。',
          '缺失或失败在正确率原定分母中保留，红叉标识未齐记录；费用只对已经完成的执行记录取均值，缺失费用不填零。',
          '零回退是报价失败后执行零动作，与已验证零动作足够分开。表中 (+n?) 为没有报价的历史组数。',
          '图中没有把模型内概率界或归一化候选权重当作实际恢复频率；0%只表示不额外模拟P/Q误差。',
          '服务总费是调用费加预留费；求解目标只最小化调用费，系统购售电费用没有混入本图。','',
          ('字体为 Microsoft YaHei。' if rendering['chinese_labels'] else
           '机器无可用中文字体，图用英文：Original/Enriched=原辨识/追加候选，coverage=候选内部真族覆盖，Normal/Exponential=正态/指数门槛。'),
          '',f"来源：{report_path}",'']
    (out/'README.md').write_text('\n'.join(text),encoding='utf8')
    if (digest(report_path)!=report_ref['sha256'] or digest(audit_ref['path'])!=audit_ref['sha256']
            or digest(__file__)!=script_ref['sha256']):
        raise ValueError('Source changed while plotting')
    receipt=dict(status='complete',created_utc=datetime.now(timezone.utc).isoformat(),
        script=script_ref,source_report=report_ref,source_audit=audit_ref,
        source_unchanged=True,synthetic_fixture=fixture,smoke=bool(data.get('smoke')),
        fee=fee,groups=len(groups),planned_histories=sum(g['historical_repetitions'] for g in groups),
        physical_backgrounds=2,executions_per_history=data['repeats'],
        interpretation='Descriptive paired sensitivity on two fixed physical windows; actual internal recovery frequencies, not model probabilities.',
        failed_rows_retained=True,missing_costs_imputed=False,
        rendering=rendering,seconds=time.perf_counter()-started,
        files={name:ref(out/name) for name in ('noise_sensitivity.png','noise_sensitivity.svg','plot_data.json','plot_data.csv','README.md')})
    write(out/'receipt.json',receipt)
    return receipt


def synthetic_fixture():
    """Small static schema fixture, including failure and missing-quote cases."""
    groups,cells,quotes=[],[],[]
    for ci,case in enumerate(CASES):
        for ni,sd in enumerate(NOISE_LEVELS):
            failed=1 if ci==0 and ni==3 else 0;complete=4-failed
            histories={}
            for view,extra in (('original',0),('enriched',1)):
                covered=min(complete,4-ni//2+extra);correct=max(0,covered-(ni>0))
                histories[view]=dict(n=4,completed=complete,failed=failed,correct=correct,
                    covered=covered,full_support_correct=0,full_support_covered=0)
            strategies={}
            for pi,policy in enumerate(POLICIES):
                completed=complete*16;recovered=max(0,min(completed,64-ni*5+(pi-1)*3))
                cost=0. if pi==0 else (pi+1)*(.2+sd*100)*(ci+1)
                strategies[policy]=dict(n=64,completed=completed,recorded_failures=64-completed,
                    recovered=recovered,full_support_recovered=0,wrong_high=completed-recovered,
                    zero_fallback_rows=(16 if pi>0 and ni>=2 else 0),
                    settled_activation_yuan=dict(n=completed,mean=cost*.6),
                    settled_total_yuan=dict(n=completed,mean=cost))
            groups.append(dict(case=case,pq_relative_sd=sd,historical_repetitions=4,
                               history=histories,strategies=strategies))
            for repeat in range(4):
                tag=f'{case}__n{ni}__h{repeat}';cells.append(dict(id=tag,case=case,pq_relative_sd=sd,history_repeat=repeat))
                if failed and repeat==3:continue
                for policy in POLICIES[1:]:quotes.append(dict(id=tag,policy=policy,zero_fallback=ni>=2 and repeat==0))
    return dict(status='complete',synthetic_fixture=True,smoke=False,repeats=16,planned_inputs=40,
                groups=groups,cells=cells,quote_records=quotes)


def self_test(out):
    import copy
    out=Path(out).resolve()
    if out.exists():raise FileExistsError(out)
    sample=synthetic_fixture();groups,_=prepare_data(sample)
    missing=next(g for g in groups if g['case']=='soumalas11' and g['pq_relative_sd']==.01)
    assert missing['strategies']['single_normal_fixed']['unavailable_quotes']==1
    assert missing['strategies']['single_normal_fixed']['n']==64
    assert missing['history']['original']['n']==4
    assert missing['strategies']['single_normal_fixed']['zero_fallback_quotes']==1
    for damage in ('duplicate','success_over_completed','unreported_noise','fee_missing_count'):
        broken=copy.deepcopy(sample)
        if damage=='duplicate':broken['groups'].append(broken['groups'][0])
        if damage=='success_over_completed':broken['groups'][0]['strategies']['no_dfs']['recovered']=65
        if damage=='unreported_noise':broken['groups'].pop()
        if damage=='fee_missing_count':broken['groups'][0]['strategies']['no_dfs']['settled_total_yuan']['n']=63
        try:prepare_data(broken)
        except ValueError:pass
        else:raise AssertionError('Invalid fixture accepted: '+damage)
    no_rows=copy.deepcopy(sample)
    for s in no_rows['groups'][0]['strategies'].values():
        s.update(completed=0,recovered=0,wrong_high=0,zero_fallback_rows=0)
        for key in ('settled_activation_yuan','settled_total_yuan'):s[key]=dict(n=0,mean=None)
    validated,_=prepare_data(no_rows)
    assert validated[0]['strategies']['no_dfs']['mean_total_yuan'] is None
    out.mkdir(parents=True)
    audit_path=out/'synthetic_audit.json';write(audit_path,dict(status='passed',task_errors=[],audited_rows=2560,expected_rows=2560,synthetic_fixture=True))
    sample['source_audit']=ref(audit_path);source=out/'synthetic_report.json';write(source,sample)
    answer=plot(source,out/'figure')
    assert answer['synthetic_fixture'] and answer['source_unchanged']
    write(out/'checks.json',dict(status='passed',synthetic_fixture=True,checks=[
        'failure denominator retained','missing quote distinct from zero fallback',
        'duplicate groups rejected','correct count bounded by completed',
        'missing formal noise group rejected','fee count matches completed paths',
        'missing fee remains null','PNG/SVG source hashes sealed'],receipt=ref(out/'figure/receipt.json')))
    return answer


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report');parser.add_argument('--out',required=True)
    parser.add_argument('--fee',choices=('total','activation'),default='total')
    parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:
        if args.report:parser.error('--self-test does not read experiment reports')
        result=self_test(args.out)
    else:
        if not args.report:parser.error('--report is required')
        result=plot(args.report,args.out,fee=args.fee)
    print(json.dumps(dict(status=result['status'],files=result['files']),ensure_ascii=False))
