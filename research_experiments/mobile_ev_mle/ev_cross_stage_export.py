"""Export descriptive frozen-v2 tables; no planning or fitting is run."""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
from statistics import mean
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--directory', type=Path, default=root/'evidence/ev_cross_all/v2')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
folder = args.directory
stage = args.output
if stage.exists():
    raise ValueError('choose a new output directory to preserve prior derived exports')
stage.mkdir(parents=True)
review = json.loads((folder / 'review_v2_full.json').read_text(encoding='utf-8'))
assert review['success'] and review['complete_trial_family']
trials = [json.loads(gzip.decompress(p.read_bytes())) for p in sorted(folder.glob('trial_*.json.gz'))]
assert len(trials) == 16 and len({t['base_seed'] for t in trials}) == 8
methods = trials[0]['config'].get('methods') or [r['method'] for r in trials[0]['rows']]
rows = []
for t in trials:
    assert len(t['rows']) == 7 and t['selection']['fallback'] is False
    for r in t['rows']:
        rows.append(dict(seed=t['base_seed'], quality=t['quality']['kind'], method=r['method'],
                         status=r['status'], margin=r.get('margin'), candidate_miss=r['candidate_miss'],
                         u_kw=r['u_kw'], impedance_rmse=r.get('impedance_rmse'),
                         source_state_rmse=r.get('source_state_rmse'), truth_best=r.get('margin', -1) >= -1e-9))
method_summary = {}
contrasts = {}
for method in methods:
    subset = [r for r in rows if r['method'] == method]
    good = [r for r in subset if r['status'] == 'complete']
    method_summary[method] = dict(available=len(subset), complete=len(good),
        truth_is_min_attained_cost=sum(r['truth_best'] for r in good),
        mean_margin=mean(r['margin'] for r in good),
        mean_impedance_rmse_pu=mean(r['impedance_rmse'] for r in good),
        mean_source_state_rmse_pu=mean(r['source_state_rmse'] for r in good))
    by_history = []
    for seed in sorted({r['seed'] for r in subset}):
        deltas = []
        for quality in ('uniform', 'temporal'):
            a = next(r for r in rows if (r['seed'], r['quality'], r['method']) == (seed, quality, method))
            b = next(r for r in rows if (r['seed'], r['quality'], r['method']) == (seed, quality, 'constant'))
            if a['status'] == b['status'] == 'complete':
                deltas.append(a['margin'] - b['margin'])
        assert len(deltas) == 2
        by_history.append(dict(seed=seed, delta=mean(deltas),
            development_group='seen_for_numerical_repair' if seed < 987103 else 'post_freeze_unseen_development'))
    contrasts[method] = dict(independent_histories=8, per_history=by_history,
        mean_delta=mean(r['delta'] for r in by_history),
        positive_histories=sum(r['delta'] > 1e-8 for r in by_history),
        negative_histories=sum(r['delta'] < -1e-8 for r in by_history),
        seen_H_mean=mean(r['delta'] for r in by_history[:2]),
        post_freeze_H_mean=mean(r['delta'] for r in by_history[2:]))
fits = []
for t in trials:
    for f in t['historical_fits']:
        fits.append(dict(seed=t['base_seed'], quality=t['quality']['kind'], scope='H', method='history',
            candidate=f['candidate'], success=f['success'], loss=f['loss'], loglik=f['loglik'],
            global_MLE_certified=f['global_MLE_certified'], cache_hit='',
            r_pu=json.dumps(f['r']), x_pu=json.dumps(f['x']),
            r_ohm=json.dumps(f['r_ohm']), x_ohm=json.dumps(f['x_ohm'])))
    for r in t['rows']:
        for f in r.get('fits', []):
            fits.append(dict(seed=t['base_seed'], quality=t['quality']['kind'], scope='H+N', method=r['method'],
                candidate=f['candidate'], success=f['success'], loss=f['loss'], loglik=f['loglik'],
                global_MLE_certified=f['global_MLE_certified'], cache_hit=r['cache_hit'],
                r_pu=json.dumps(f['r']), x_pu=json.dumps(f['x']),
                r_ohm=json.dumps(f['r_ohm']), x_ohm=json.dumps(f['x_ohm'])))
assert len(fits) == 512 and not any(f['global_MLE_certified'] for f in fits)
with (stage / 'ev_cross_fit_table.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(fits[0])); writer.writeheader(); writer.writerows(fits)
test_ids = set()
unit_paths = ['stage_a_unit.xml', 'stage_bcd_unit.xml', 'restore_price_unit.xml', 'correlated_unit.xml', 'reviewer_final_unit.xml']
for name in unit_paths:
    xmlpath = root/'evidence/ev_ac_moving_cross'/name if name == 'stage_a_unit.xml' else root/'evidence/ev_cross_all/diagnostics'/name
    xml = ET.parse(xmlpath)
    for c in xml.iter('testcase'):
        assert c.find('failure') is None and c.find('error') is None
        test_ids.add((c.attrib['classname'], c.attrib['name']))
analysis = dict(schema='cross_stage_descriptive_analysis_v1',
    independent_histories=8, quality_conditions=16, methods=method_summary,
    paired_margin_vs_constant=contrasts, seconds=sum(t['seconds'] for t in trials),
    selection_seconds=sum(t['selection']['seconds'] for t in trials),
    actual_counts=review['numbers']['executed_calls'],
    read_only_review_numbers=review['numbers'],
    fit_csv_saved_rows=len(fits), fit_csv_rows_are_not_independent_fit_calls=True,
    targeted_test_unique_cases=len(test_ids), targeted_test_xml=unit_paths,
    formal_calibration=0, formal_evaluation=0,
    research_complete=False,
    actual_margin_scope='local attained AC costs; no rejection thresholds or topology probabilities',
    source_sha256=json.loads((folder/'freeze.json').read_text(encoding='utf-8'))['source_sha256'],
    reviewed_file_sha256=hashlib.sha256((folder/'review_v2_full.json').read_bytes()).hexdigest())
(stage / 'stage_analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding='utf-8')
table = '\n'.join(f"| {m} | {method_summary[m]['complete']}/16 | {method_summary[m]['truth_is_min_attained_cost']}/16 | {method_summary[m]['mean_margin']:.6f} | {contrasts[m]['mean_delta']:.6f} |" for m in methods)
md = f'''# 响应修正、全候选曲线与EV位置：本轮阶段结果

2026-10-14。执行协议为 `ev_cross_execution_protocol.md`；此批全部为开发数据。完整研究未完成，正式校准0、正式评价0。阻抗下界仍为1e−6 pu数值正性条件，未根据结果抬高。

## 1. 响应修正已经实现，区域误差门仍未全过

移动未来总P/Q中心时同时加入参考和竞争控制导数，再保留功率×线路及功率×背景状态交叉项。54份旧合法见证中逐通道白化RMS≤.25由18/54增至48/54，未来残差误差范数中位数13.7953降至.1762。剩余6行保留为远参数和线路×背景项失效；没有据此开展正式排错校准。P/Q直接通道误差仅浮点量级，源无独立遥测通道。

## 2. 全候选连续优化与计算修复

四候选全部进入联合AC拟合，12有向竞争对均参加代理；单点0、连续0–6kW、四段15分钟、同3kWh电量、效率.95、终端26.85kWh。竞争R/X共享历史与未来，同点源逐时重新拟合。规划只读取历史、公开计量质量和设备条件；未来真实负荷/噪声仅用于评价。

v1四条件/两已见star历史保留：2次无效proxy上下界、2个最终方法拟合失败。v2用可行恒充QP解释初始化，收紧残差LP范围、缩放实际得分、撤销冲突上界；同配置新版本完成16/16条件。首2H已见用于数值修复，其余6H在冻结后运行，二者均属于开发。

16/16初始见证界达到声明.01规划间隙，开放竞争主MILP实际调用0；恒充已处于该代理容差内，raw/guarded均与恒充相同。固定MLE分段MILP实际调用16。严格1e−7矩阵回归仍有有效正间隙8.56e−7，保留unfinished。最小AC接受增益.01为规划分辨率，未用作统计显著性门槛。

## 3. 实际AC曲线评价

定义真实候选a的有符号平方成本裕量 M=min_wrong J_b−J_a；实际对数似然领先量为M/2。表中真候选取得最小成本只表示本批局部排名，尚无可靠排错判定。

| 方法 | 完整条件 | 真候选最小取得成本 | 平均M | 对恒充平均M增量 |
|---|---:|---:|---:|---:|
{table}

同一历史的两种质量是配对重复，平均增量先在每个历史内取两质量平均，再按8历史汇总。简单质量LP相对恒充在4/8历史增加、4/8降低；主要增加发生于接入点0位于cherry01/02内部的四份历史。star与cherry12的接入信息较弱，失败及负裕量保留。随机价格LP为独立声明价格优化曲线，不是从曲线库均匀抽样。

此批显示合成相关负荷条件下高低曲线可能比恒充更有区分信号；当前全对最坏指标未选出额外曲线，也未优于简单质量对照。高低曲线的收益与新指标的有效性分别检验。

完整逐候选R/X及最佳已取得loglik表有512保存H/联合行，位于 `evidence/ev_cross_all/v2/ev_cross_fit_table.csv`；含缓存和历史复制，行数不等于实际拟合次数。全部原始状态及独立重放在同目录。R/X误差和源状态误差的逐条件量及方法均值在 `stage_analysis.json`；共同源是推断状态，没有虚构独立源表。

## 4. 指标与价格的独立数学检查

活动边界价格为 rᵀ(B_t+A_tz)+.5(μ_U−μ_L)ᵀS_t，明确下约束 l(u)−z≤0、上约束 z−h(u)≤0。固定外盒去边界项。价格表示得分/kW的一阶变化；有限步回代检查，MILP全域界承担代理搜索。

弱对可限制全部最坏目标。三模型解析例给最坏全局曲线u1=1/(1+2ε)，平均任务曲线(0,1)提高两对距离但弱对继续未决。proper toy EIG对三套显式先验发生排序翻转，40/80/160节点只作为积分收敛诊断。平均profile的有效MILP上下界已推导，尚未接AC主策略。

相关负荷三末端线性例开放公共干线与所有末端斜率重拟合，严格成本 q=(rc²/(2σ²))(Σu²−E²/L)。同3kWh恒充2.7，高低7.2，曲线由共享平方MILP产生。已知P、固定Q/源、相同精度、线性近似和投影解域覆盖是明确前提；两末端开放公共干线时任何曲线均可精确补偿的失败也保留。

## 5. 标准物理尺度与EV空间价值

case33原五条Ω线路、12.66kV/1MVA、120/90/90kW负荷参考，配合合成时序与噪声。负荷节点代表等效功率扰动，实际EV充电/低压及变压器接口尚未建模。6kW为负荷5–6.67%，不同位置电压变化只有.08–.324个σ。两独立历史/40候选拟合全部有效；同曲线三个位置诊断中bus23相对无EV增加.161和.504 SSE，其他位置可能降低，R/X相对误差仍约1到6。位置会改变响应，净空间优势尚待设计选址和成本公平比较。

EV/ESS同位置同注入AC差0。驾驶.2kWh后网侧任务3.210526kWh，行驶和接线各占一段时只剩两段、最多3kWh，任务不可行；明确提前到达才可行，工作人员延迟又可能使其不可行。10操作情景5可行/5不可行均保留。详细结果见 `ev_physical_spatial_fit_results.md`。

## 6. 物理资料和类似论文

当前没有现场线路表。已核查5篇原始论文的算例章节：常用IEEE13/34/37标准网络，部分引入真实消费者负荷后再AC仿真；它们通常使用当地全负荷容量逆变器、按秒多次探测，或600个时刻估计参数。Cavraro图算法rmin=.0014pu来自其IEEE37模型，不能直接当现场或case33拟合下界。详见 `ev_related_case_review.md` 及出处JSON。

## 7. 验证、失败分母和续接

v2独立重放1024保存拟合状态、112物理响应、1728QP状态通过。实际604次候选拟合，计数AC调用22450，仿真真实响应另120次；8独立H才是统计单位。所有112方法条件完整、候选漏真0。针对新路线{len(test_ids)}个唯一测试用例通过；未重复跑与改动无关的旧全套。

源码字节快照、版本、种子和reviewer字节已保存。`ev_cross_source_restore`默认只读检查33份v2（可指定v1的31份）外部冻结源码；仅换行差时显式 `--restore-line-endings` 才恢复。实际Git对象导出21份原字节相同、12份仅换行不同，恢复后33/33相符，0新增科学拟合。

下一协议见 `ev_cross_next_protocol.md`。严格响应门、正式风险校准、平均目标AC部署、扣成本的位置收益、滚动与扩网仍未完成。结果分别定位信息/接入不足、代理或指标不足、数值求解与操作失败，不用人为参数下界制造改善。
'''
(stage / 'ev_cross_stage_results.md').write_text(md, encoding='utf-8')
tex_rows = []
for m in methods:
    label = m.replace('_', '\\_')
    tex_rows.append(f"{label} & {method_summary[m]['truth_is_min_attained_cost']}/16 & {method_summary[m]['mean_margin']:.4f} & {contrasts[m]['mean_delta']:.4f}\\\\")
tex = '\n'.join(tex_rows)
tex = '''8份独立历史覆盖star和三种cherry、各高/低相关；每份重复两种公开计量质量，共16条件、112方法条件。首2历史已见用于数值修复，后6历史在v2冻结后运行，均为开发。四段15 min、固定点0、同3 kWh，比较无EV、恒充、质量LP、独立随机价格LP、固定MLE平方MILP、开放竞争原始及AC检查策略。所有112方法条件完整，候选漏真0。

\\begin{center}\\begin{tabular}{lrrr}
\\toprule 方法 & 真候选最小取得成本 & 平均$M$ & 对恒充平均增量\\\\\\midrule
''' + tex + '''
\\bottomrule\\end{tabular}\\end{center}
平均先在每份历史内汇总两质量，再按8历史计算。质量LP对恒充4/8历史提高、4/8降低，主要改善出现在点0位于cherry01/02模糊块内的四历史；star和不含点0的cherry12保留负例。最小取得成本是局部排名；本批正式校准和正式评价均为0，尚未使用排错阈值。

开放竞争12对在16条件均由初始恒充见证界达到声明.01代理容差，主MILP实际调用0；原始和AC检查策略均返回恒充，AC检查接受0/16。固定MLE代理实际调用16次MILP。此结果定位全对最坏目标的瓶颈，未把基线返回当作算出了更好的曲线。

独立重放1024份保存拟合状态、112物理响应和1728QP通过；复制状态区别于实际604次拟合。计数AC调用22,450，仿真真实响应另120次。全部候选逐边R/X与最佳已取得loglik的512行H/联合表随证据保存；源为同点推断状态。对应新路线的唯一测试用例总数见结果稿。
'''
(stage / 'report_cross_results.tex').write_text(tex, encoding='utf-8')
print(json.dumps(dict(tests=len(test_ids), seconds=analysis['seconds'], methods=method_summary,
                     contrasts={m:contrasts[m]['mean_delta'] for m in methods}),ensure_ascii=False))
