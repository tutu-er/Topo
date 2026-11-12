"""Review saved linear-DR amplitudes only; no optimization, noise draw or AC rerun."""
from pathlib import Path
from collections import defaultdict
import argparse
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import require,sha,utc
from ..posterior_mse_20261112.deliver import write_csv


def analyze(run,public):
    require((run/'confirmation/complete.json').exists(),'Finish locked physical tests first')
    require(not (public/'MODEL_REVIEW.md').exists(),'Preserve previous model review')
    inputs={e['request']['id']:e for e in io.read(run/'inputs.json')};records=[]
    for phase in ('population','confirmation'):
        for directory in sorted((run/phase/'jobs').iterdir()):
            entry=inputs[directory.name];scale=np.array(entry['scale'])
            for d in sorted(directory.glob('day*')):
                if not (d/'physical.npz').exists():continue
                seal=io.read(d/'policy_seal.json');require(sha(seal['planner']['path'])==seal['planner']['sha256'],'Frozen controls changed')
                with np.load(d/'planner.npz',allow_pickle=False) as packet,np.load(d/'physical.npz',allow_pickle=False) as actual:
                    baseline=actual['baseline_P'];daily=float(baseline.sum());peak=np.max(abs(baseline),axis=0)
                    for name in packet.files:
                        if not name.startswith('optimized_'):continue
                        x=packet[name];delta=np.repeat(x,4,axis=0)*scale
                        instantaneous=np.divide(abs(delta),abs(baseline),out=np.zeros_like(delta),where=baseline!=0)
                        actual_peak=np.divide(abs(delta),peak[None],out=np.zeros_like(delta),where=peak[None]!=0)
                        records.append(dict(phase=phase,id=directory.name,gamma_label=name.split('_',1)[1],
                            request_rms=float(np.sqrt(np.mean(x*x))),request_max=float(abs(x).max()),
                            actual_peak_relative_max=float(actual_peak.max()),instantaneous_relative_max=float(instantaneous.max()),
                            absolute_energy_ratio=float(np.abs(delta).sum()/daily),net_energy_ratio=float(delta.sum()/daily),
                            negative_power_fraction=float(np.mean(baseline+delta<0)),negative_power_day=bool(np.any(baseline+delta<0)),
                            above_current_load_day=bool(np.any(abs(delta)>abs(baseline))),
                            zero_load_control_cells=int(np.sum((baseline==0)&(delta!=0)))))
    groups=defaultdict(list)
    for r in records:groups[r['phase'],'all',r['gamma_label']].append(r)
    for r in records:
        if r['phase']=='confirmation':groups[r['phase'],r['id'],r['gamma_label']].append(r)
    output=[]
    for (phase,identity,gamma),rows in sorted(groups.items()):
        row=dict(phase=phase,id=identity,gamma_label=gamma,days=len(rows))
        for name in ('request_rms','request_max','actual_peak_relative_max','instantaneous_relative_max','absolute_energy_ratio','net_energy_ratio','negative_power_fraction'):
            values=np.array([r[name] for r in rows])
            row[name+'_mean']=float(values.mean());row[name+'_median']=float(np.median(values))
            row[name+'_p95']=float(np.quantile(values,.95));row[name+'_max']=float(values.max())
        row.update(negative_power_days=sum(r['negative_power_day'] for r in rows),above_current_load_days=sum(r['above_current_load_day'] for r in rows),
                   zero_load_control_cells=sum(r['zero_load_control_cells'] for r in rows))
        output.append(row)
    write_csv(public/'strength.csv',output)
    lookup={(r['phase'],r['id'],r['gamma_label']):r for r in output}
    global4=lookup['population','all','g4']
    ids=['main__input121__101','main__input195__101','pressure__input026__101','pressure__input025__101','main__input039__101']
    lines=['# 纯线性DR建模与强度核查','',
        '结论：当前模型适合检验理想可控净注入能否改善辨识；不宜直接当作实际负荷DR能力或经济性的主结果。强度无上限，使少数请求远超自然负荷。费用小或AC收敛都不足以判定调节幅度合理。','',
        '## 模型究竟表示什么','',
        '代码规定deltaP=m_i*x_hour、deltaQ=.35deltaP，m_i为历史节点功率峰值；请求确定性、精确交付，没有价格弹性估计或参与随机性。线性的是执行映射，电网电压仍由原AC计算。若把x解释为归一化有符号价格，响应系数就是人为给定的m_i，未由用户行为数据校准。',
        '', '费用mean(x²)是相对历史节点峰值的归一化调节代价，适合本算例内同代价比较；它不按实际kW或kWh结算，且跨所有节点和时段取平均会掩盖局部大动作。它不支持直接的货币报价、真实用户响应或跨系统经济性结论。',
        '', '没有单节点幅度上限、能量限制或净负荷非负限制，因此该模型允许原负荷节点变成净注入。若研究对象是双向可控资源，这可以作为理想假设；若称为纯负荷削减／转移，就超出了该含义。固定deltaQ/deltaP=.35也是声明假设，不能同时代表所有设备类型。','',
        '## 全体强度分布','',
        f"gamma=4主体1600新日：单节点小时请求最大为历史峰值的{global4['request_max_max']:.2f}倍；日最大请求的中位数为{global4['request_max_median']:.3%}、95%分位为{global4['request_max_p95']:.3%}。共有{global4['negative_power_days']}/1600日使至少一个负荷节点成为净注入，{global4['above_current_load_days']}/1600日出现调节绝对值超过当时自然负荷。",'',
        '报告中使用三种不同的强度，避免混淆：平均RMS请求按全部节点小时计算；单节点最大请求相对该节点历史功率峰值；瞬时相对变化则用当时真实自然负荷作分母，只用于事后检查，不提供给规划。','',
        '## 64日复验中的代表情况（gamma=4）','',
        '| 输入 | 平均RMS请求／历史峰值 | 单节点最大请求／历史峰值 | 最大瞬时变化／自然负荷 | 绝对调节能量／原日能量（均值） | 净注入日／64 |',
        '|---|---:|---:|---:|---:|---:|']
    for identity in ids:
        r=lookup['confirmation',identity,'g4']
        lines.append(f"| {identity} | {r['request_rms_mean']:.2%} | {r['request_max_max']:.2%} | {r['instantaneous_relative_max_max']:.2%} | {r['absolute_energy_ratio_mean']:.2%} | {r['negative_power_days']} |")
    lines+=['','前3个是明显改善且没有电压越限的例子，但它们仍有40%—56%的单节点峰值请求，10—13/64日出现局部净注入。全局RMS约6%—9%并不代表每个节点都只调整了这个比例。Paper121、Soumalas195、Pengwah026的平均净能量变化约+0.19%、−0.26%、+0.19%，绝对调节能量却为3.81%、8.25%、5.63%；这主要体现时空正负调节抵消，不能据此推出每个设备满足能量平衡。',
        '', 'Pengwah025虽有明显后验改善，但最大单节点请求约为历史峰值15.77倍，绝对调节能量约为自然日能量190%，64/64日电压越限，不作为可部署正例。Soumalas039既有过大局部动作，也有平均真后验下降。完整失败和总体统计仍保留在REPORT.md。','',
        '## 对下一版主实验的建议','',
        '保留线性响应及原AC评价，先补一个简单的幅度边界即可，不必恢复SOC、充放电互斥等复杂模型。若解释为负荷DR，可采用相对规划预测负荷的限幅，例如|deltaP_it|<=alpha*|Phat_it|；alpha=10%、20%、30%可作事前固定的数值敏感性档，不能称为实证可调容量。这样同时避免低负荷时按历史峰值下发过大的削减。若解释为双向资源，则应使用事前声明的装机／可用容量上限，明确允许净注入。',
        '', '这一限幅仍是线性约束，整体仍为QCQP。单对无约束解析解一般不再直接适用，需要有界QCQP求解；不因此自动需要ADMM。换求解算法本身不能使无上限响应变得合理。',
        '', '本次核查只读取已冻结的全部动作和真实新日；没有重规划、重抽噪声、重跑AC或删除极端结果。无上限批次保留为理想响应参考；有幅度边界的结果需要另外锁定并验证，不能用当前数值替代。']
    (public/'MODEL_REVIEW.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    receipt=io.read(public/'RECEIPT.json')
    receipt['strength_review_source']=io.ref(__file__);receipt['strength_review_created_utc']=utc()
    receipt['files']={p.name:sha(p) for p in sorted(public.iterdir()) if p.is_file() and p.name!='RECEIPT.json'}
    io.write(public/'RECEIPT.json',receipt)
    io.write(run/'strength_review.json',dict(status='complete',created_utc=utc(),new_days=0,new_solver_calls=0,new_AC_calls=0,
        action_sets_reviewed=len(records),source=io.ref(__file__)))
    print({k:global4[k] for k in ('days','request_max_max','request_max_median','request_max_p95','negative_power_days','above_current_load_days')},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    a=p.parse_args();analyze(a.run.resolve(),a.public.resolve())
