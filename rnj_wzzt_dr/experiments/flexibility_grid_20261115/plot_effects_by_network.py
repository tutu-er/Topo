"""Plot sealed grid effects and resource use without running any experiment."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import numpy as np
from ... import io

CASES = ('pengwah18', 'flynn16', 'paper15', 'soumalas11')
NAMES = ('Pengwah18', 'Flynn16', 'Paper15', 'Soumalas11')
MODES = ('normal_fixed', 'mgf_fixed', 'mgf_forecast')
TITLES = ('正态约束', '固定功率指数约束', '含预测误差指数约束')
STRIDES = (1, 2, 4)
MULTIPLIERS = (.5, 1., 2.)


def checked(ref):
    if io.ref(ref['path'])['sha256'] != ref['sha256']:
        raise ValueError('Changed input: '+ref['path'])
    return Path(ref['path'])


def plotting():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    path = Path('C:/Windows/Fonts/msyh.ttc')
    if path.exists():
        font_manager.fontManager.addfont(str(path))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(path)).get_name()
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['pdf.fonttype'] = 42
    return plt


def prepare(report_path):
    report = io.read(report_path)
    if (report.get('smoke') or not report.get('inputs_unchanged')
            or len(report['cells']) != 36 or report['repeats'] != 16
            or report['forecasts'] != 1):
        raise ValueError('Complete 36-cell audited report required')
    audit = io.read(checked(report['audit']))
    if audit['status'] != 'passed' or audit['audited_rows'] != 4608:
        raise ValueError('Full independent audit required')
    entries = io.read(checked(report['source_inputs']))
    entries = {e['id']: e for e in entries}
    capacities, contracts, rows = {}, {}, []
    lookup = {}
    for cell in report['cells']:
        d = cell['dimensions']; e = entries[d['id']]
        key = (d['case'], d['stride'], d['meter_multiplier'])
        if key in lookup:
            raise ValueError('Repeated grid identity')
        lookup[key] = cell
        frozen = json.dumps([e['p_resources'], e['q_resources']], sort_keys=True)
        if d['case'] in contracts and contracts[d['case']] != frozen:
            raise ValueError('Resource contracts changed within network')
        contracts[d['case']] = frozen
        hours = 0.
        for q in e['q_resources']:
            start, end = q['available_clock_hours']
            if (start, end) != (0, 24):
                raise ValueError('Review denominator for non-all-day Q resources')
            hours += float(q['capacity_kvar'])*(end-start)
        capacities[d['case']] = dict(contract_Q_kvar=sum(q['capacity_kvar'] for q in e['q_resources']),
            contract_Q_capacity_hours_kvarh=hours, Q_nodes=len(e['q_resources']))
        for mode in MODES:
            g = cell['groups'][mode]; zero = cell['groups']['no_dfs']
            if g['n'] != 16 or g['completed'] != 16 or zero['completed'] != 16:
                raise ValueError('Unexpected or failed execution rows')
            q_energy = g['delivered_Q_kvarh_mean_completed']
            utilization = q_energy/hours if hours else 0.
            if not 0 <= utilization <= 1+1e-9:
                raise ValueError('Contract-hour utilization outside bounds')
            rows.append(dict(id=d['id'], case=d['case'], stride=d['stride'],
                meter_multiplier=d['meter_multiplier'], strategy=mode,
                truth_in_bank=cell['coverage']['truth_in_native_bank'],
                baseline_recovered=zero['recovered'], recovered=g['recovered'], paths=g['n'],
                gained=cell['paired_vs_no_dfs'][mode]['gained'],
                lost=cell['paired_vs_no_dfs'][mode]['lost'],
                settled_yuan=g['settled_cost_mean_completed']['total_yuan'],
                requested_P_kwh=g['requested_P_kwh_mean_completed'],
                delivered_P_kwh=g['delivered_P_kwh_mean_completed'],
                requested_Q_kvarh=g['requested_Q_kvarh_mean_completed'],
                delivered_Q_kvarh=q_energy, contract_Q_capacity_hours_kvarh=hours,
                delivered_Q_contract_hour_fraction=utilization,
                anchor_fallbacks=cell['solver'][mode]['anchor_fallbacks'],
                zero_fallbacks=cell['solver'][mode]['zero_fallbacks']))
    expected = {(c, s, m) for c in CASES for s in STRIDES for m in MULTIPLIERS}
    if set(lookup) != expected:
        raise ValueError('Missing or additional network/grid condition')
    return report, rows, capacities


def run(report_path, out):
    from matplotlib.colors import TwoSlopeNorm, Normalize
    from matplotlib.patches import Rectangle
    report_path, out = Path(report_path).resolve(), Path(out).resolve()
    source_ref = io.ref(report_path)
    report, rows, capacities = prepare(report_path)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    plt = plotting()
    data = {(r['case'], r['stride'], r['meter_multiplier'], r['strategy']): r for r in rows}
    images = []
    max_use = max(100*r['delivered_Q_contract_hour_fraction'] for r in rows)
    for kind in ('recovery', 'usage'):
        fig, axes = plt.subplots(4, 3, figsize=(13.4, 14.5), layout='constrained')
        for i, (case, name) in enumerate(zip(CASES, NAMES)):
            for j, (mode, title) in enumerate(zip(MODES, TITLES)):
                ax = axes[i, j]
                table = [[data[case, s, m, mode] for m in MULTIPLIERS] for s in STRIDES]
                values = np.array([[100*(r['recovered']-r['baseline_recovered'])/16
                    if kind == 'recovery' else 100*r['delivered_Q_contract_hour_fraction']
                    for r in row] for row in table])
                norm = TwoSlopeNorm(vmin=-100., vcenter=0, vmax=100.) if kind == 'recovery' else Normalize(0, max(1., max_use))
                im = ax.imshow(values, cmap='RdBu' if kind == 'recovery' else 'YlGnBu', norm=norm)
                ax.set_title(f'{name} · {title}', fontsize=11)
                ax.set_xticks(range(3), ['0.25 / 0.025', '0.5 / 0.05', '1 / 0.1'], fontsize=8)
                ax.set_yticks(range(3), ['192/96', '96/48', '48/24'], fontsize=9)
                ax.set_ylabel('训练／验证点数', fontsize=9)
                ax.set_xlabel('P/Q ／ 电压量测标准差（%）', fontsize=9)
                for y, line in enumerate(table):
                    for x, r in enumerate(line):
                        missing = not r['truth_in_bank']
                        fallback = r['anchor_fallbacks'] or r['zero_fallbacks']
                        if kind == 'recovery':
                            if missing:
                                ax.add_patch(Rectangle((x-.5, y-.5), 1, 1, facecolor='#eeeeee',
                                    edgecolor='#c4c4c4', hatch='///', linewidth=.4))
                            label = ('缺真候选\n' if missing else '') + f"{r['baseline_recovered']}→{r['recovered']}" + ('†' if fallback else '')
                            label += f"\n{r['settled_yuan']:.2f}元"
                            white = not missing and abs(values[y, x]) > 60
                        else:
                            label = f"Q {r['delivered_Q_kvarh']:.1f}\nP {r['delivered_P_kwh']:.1f}"
                            if missing:
                                label += ' *'
                            white = values[y, x] > .65*max_use
                        ax.text(x, y, label, ha='center', va='center', fontsize=10,
                                color='white' if white else '#182534')
        cb = fig.colorbar(im, ax=axes, fraction=.018, pad=.012)
        cb.set_label('内部恢复率变化（百分点）' if kind == 'recovery' else '无功合同容量小时使用率（%）', fontsize=10)
        if kind == 'recovery':
            title = '各网络的采购效果：每格16条配对路径'
            footer = '数字为无采购→采购后的恢复数及结算均费；斜线格缺少正确内部拓扑候选。†执行了旧方案回退。'
        else:
            title = '同一资源配置下的实际调用：各网络的九个设置'
            footer = 'Q为平均绝对无功调用量(kvarh)，P为平均绝对有功调用量(kWh)；*表示缺真候选。颜色分母为全天合同幅度累计。'
        fig.suptitle(title, fontsize=16)
        fig.supxlabel(footer, fontsize=10)
        stem = 'effects_by_network' if kind == 'recovery' else 'resource_use_by_network'
        for ext in ('png', 'pdf'):
            path = out/f'{stem}.{ext}'
            fig.savefig(path, dpi=170, bbox_inches='tight')
            images.append(io.ref(path))
        plt.close(fig)
    with (out/'cell_resource_effects.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    if io.ref(report_path) != source_ref:
        raise ValueError('Report changed while plotting')
    checked(report['audit']); checked(report['source_inputs'])
    io.write(out/'receipt.json', dict(status='complete', report=source_ref,
        inputs=report['source_inputs'], audit=report['audit'], script=io.ref(__file__),
        rows=len(rows), cases=capacities, files=images, csv=io.ref(out/'cell_resource_effects.csv'),
        scope='Saved audited observations only; no new fitting, optimization, AC, or random paths.',
        configuration='Contracts and order verified identical across all nine conditions within each network.',
        utilization='Mean delivered absolute Q energy divided by sum(contract Q amplitude times available hours); not measured installed capacity or stored energy.',
        comparison='Different networks and data conditions under frozen offers; not a randomized resource placement or capacity ablation.'))
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    print(run(args.report, args.out))
