"""Plot recorded Q requests and algebraic causes of the Soumalas regressions."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from ... import io
from .plot_effects_by_network import checked, plotting

POLICIES = ('anchor', 'normal_fixed', 'mgf_fixed', 'mgf_forecast')
TITLES = ('旧方案', '正态约束', '固定功率指数约束', '含预测误差指数约束')


def save(fig, out, stem):
    files = []
    for ext in ('png', 'pdf'):
        path = out/f'{stem}.{ext}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        files.append(io.ref(path))
    return files


def run(diagnosis, waterfall, run_root, out):
    from matplotlib.colors import TwoSlopeNorm
    diagnosis, waterfall, run_root, out = map(lambda x: Path(x).resolve(), (diagnosis, waterfall, run_root, out))
    refs = [io.ref(diagnosis), io.ref(waterfall), io.ref(run_root/'inputs.json')]
    data, decomposition = io.read(diagnosis), io.read(waterfall)
    if (not data['all_sources_unchanged'] or data['selection_mismatches'] != 0
            or data['records'] != 256 or len(data['cases']) != 2):
        raise ValueError('Checked two-cell diagnosis required')
    expected_cases = ['soumalas11__2024-01-04__s2__m0p5',
                      'soumalas11__2024-01-04__s2__m1p0']
    if [case['id'] for case in data['cases']] != expected_cases:
        raise ValueError('Review panel labels for a different case order')
    checked(decomposition['source'])
    entries = {e['id']:e for e in io.read(run_root/'inputs.json')}
    for ref in data['source_refs']:
        checked(ref)
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)
    plt = plotting(); files = []
    fig, axes = plt.subplots(2, 4, figsize=(15.2, 8.8), layout='constrained')
    used = []
    for row, case in enumerate(data['cases']):
        tag = case['id']; entry = entries[tag]
        qs = entry['q_resources']
        path = run_root/'fraction0.10'/tag/'forecast00'/'planner.npz'
        known = {r['path']:r['sha256'] for r in data['source_refs']}
        if str(path) not in known:
            raise ValueError('Planner is absent from sealed diagnosis references')
        checked(dict(path=str(path), sha256=known[str(path)])); refs.append(io.ref(path))
        with np.load(path) as arrays:
            for col, (policy, title) in enumerate(zip(POLICIES, TITLES)):
                p = case['policies'][policy]
                ids = [r['id'] for r in p['resources']]
                action = arrays[policy]
                if action.shape != (24, len(ids)):
                    raise ValueError('Unexpected action dimensions')
                matrix = np.array([action[:, ids.index(q['id'])] for q in qs])
                for k,q in enumerate(qs):
                    if np.max(np.abs(matrix[k])) > q['capacity_kvar']+1e-6:
                        raise ValueError('Action exceeds stated Q capacity')
                    record = p['resources'][ids.index(q['id'])]
                    if not np.isclose(np.abs(matrix[k]).sum(), record['abs_amount'], atol=1e-7):
                        raise ValueError('Resource order or amount differs from diagnosis')
                ax = axes[row,col]
                requested_p = sum(r['abs_amount'] for r in p['resources'] if r['unit']=='kW')
                im = ax.imshow(matrix, aspect='auto', cmap='RdBu_r',
                    norm=TwoSlopeNorm(vmin=-5., vcenter=0., vmax=5.), interpolation='nearest')
                ax.set_title(f"{title}\n恢复 {p['observed_correct']}/16；{p['summary']['settled_fee_yuan']['mean']:.2f}元\n另有P请求 {requested_p:.2f} kWh", fontsize=10)
                ax.set_xticks([0,6,12,18,23], ['0','6','12','18','23'])
                ax.set_xlabel('方案时段（每段1小时）',fontsize=9)
                ax.set_yticks(range(len(qs)), [f"{q['node_label']} (±{q['capacity_kvar']:g})" for q in qs], fontsize=8)
                noise = 'P/Q 0.25%；V 0.025%' if row == 0 else 'P/Q 0.5%；V 0.05%'
                ax.set_ylabel(noise+'\n节点（合同kvar上限）',fontsize=9)
                used.append(dict(id=tag, strategy=policy, ids=[q['id'] for q in qs],
                    requested_Q=matrix.tolist(), contract_kvar=[q['capacity_kvar'] for q in qs],
                    observed_recovered=p['observed_correct'],
                    requested_P_kwh=requested_p,
                    requested_Q_kvarh=sum(r['abs_amount'] for r in p['resources'] if r['unit']=='kvar')))
    cb = fig.colorbar(im, ax=axes, fraction=.018, pad=.012)
    cb.set_label('请求无功（kvar）：正=吸收，负=注入',fontsize=10)
    fig.suptitle('Soumalas负例：同一21 kvar合同配置，调整的节点、方向和时段不同',fontsize=15)
    fig.supxlabel('训练96点、验证48点；图为冻结采购请求，恢复与费用来自16条实际执行。零采购两格均16/16。',fontsize=10)
    files += save(fig,out,'negative_Q_requests');plt.close(fig)

    panels = decomposition['panels']
    fig, axes = plt.subplots(1, len(panels), figsize=(15.4, 5.4), layout='constrained')
    titles = ('正态：相对旧竞争者6', '同一正态动作：相对新赢家3', '固定指数：相对竞争者6')
    steps = ('baseline_observed_L','delta_J','delta_response_bias','delta_AC_remainder','delta_meter','observed_L')
    labels = ('原优势','距离\n变化','响应矩阵\n偏差','AC近似\n余项','量测\n变化','最终优势')
    for ax,p,title in zip(axes,panels,titles):
        values = [float(p[k]) for k in steps]
        if abs(sum(values[:-1])-values[-1]) > 1e-8:
            raise ValueError('Waterfall does not conserve the actual score difference')
        previous = 0.; extrema=[0.]
        for k,value in enumerate(values):
            total = k in (0,5)
            start = 0. if total else previous
            finish = value if total else previous+value
            color = '#43556b' if total else '#258673' if value>=0 else '#c74b4b'
            ax.bar(k, finish-start, bottom=start, width=.7, color=color, zorder=3)
            if k<5:
                ax.plot([k+.35,k+.65],[finish,finish],color='#aab1b8',lw=.8)
            extrema += [start, finish]
            previous=finish
        span=max(extrema)-min(extrema)
        for k,value in enumerate(values):
            if k==0: tip=value
            elif k==5: tip=value
            else: tip=sum(values[:k+1])
            ax.annotate(f'{value:+.2f}',(k,tip),xytext=(0,6 if value>=0 else -6),
                textcoords='offset points',ha='center',va='bottom' if value>=0 else 'top',fontsize=10,
                bbox=dict(facecolor='white',edgecolor='none',alpha=.85,pad=.5))
        ax.set_ylim(min(extrema)-.19*span,max(extrema)+.22*span)
        ax.set_xticks(range(6),labels,fontsize=9)
        ax.axhline(0,color='#333',lw=.8)
        ax.grid(axis='y',alpha=.17,zorder=0)
        ax.set_ylabel('累计评分优势（正值有利于真候选5）',fontsize=9)
        ax.set_title(f"{title}；整库恢复 {p['observed_correct']}/16\n规划均值 {p['forecast_K_mean']:+.1f}；最终实测 {p['observed_L']:+.1f}",fontsize=11)
    fig.suptitle('负例成因：响应矩阵误差抵消了候选距离收益',fontsize=15)
    fig.supxlabel('Soumalas 96/48点，P/Q标准差0.5%、电压0.05%。各面板纵轴独立；16条配对执行均值。真线路只用于离线分解。',fontsize=10)
    files += save(fig,out,'negative_evidence_decomposition');plt.close(fig)
    for ref in refs:
        checked(ref)
    checked(decomposition['source'])
    io.write(out/'request_plot_data.json',used)
    io.write(out/'receipt.json',dict(status='complete',script=io.ref(__file__),
        helper=io.ref(Path(__file__).with_name('plot_effects_by_network.py')),
        inputs=refs, physical_split=decomposition['source'], files=files,
        data=io.ref(out/'request_plot_data.json'),
        source_diagnosis='256 saved paths with zero independent selection mismatches',
        scope='Post hoc plots of stored requests and exact score identities. No new fitting, optimization, AC solve, or causal resource-location ablation.',
        sign='Q positive increases net consumption (absorption); negative injects Q.',
        caveat='Response-matrix bias includes all frozen response-support terms. It is not a per-line impedance estimate.'))
    return out


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnosis',type=Path,required=True)
    p.add_argument('--waterfall',type=Path,required=True)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(run(a.diagnosis,a.waterfall,a.run,a.out))
