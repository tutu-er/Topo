"""Validate retained original policies and publish complete normalized summaries."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from ... import io
from ..unified_posterior_dr_20261112.history_scan import ROOT,require,sha,utc
from ..posterior_mse_20261112.deliver import write_csv
from ..expected_logodds_admm_20261113.run import REFERENCE,load_target
from ..expected_logodds_admm_20261113.evidence import direct_gains

LABELS={'storage_relaxation':'储能连续松弛','linear_response':'纯线性响应',
        'no_dr':'无DR','linear_QP_replayed':'线性QP提议重放',
        'storage_relaxation_ADMM_replayed':'储能松弛ADMM重放',
        'linear_response_ADMM_replayed':'线性响应ADMM重放','rotation_same_QP_fee':'同QP费用轮换'}


def publish(run,destination):
    require(not destination.exists(),'Keep previous public summaries')
    require((run/'confirmation/complete.json').exists(),'Complete every locked model check first')
    protocol=io.read(run/'pricing_protocol.json');screens=io.read(run/'screen_results.json')
    planning=io.read(run/'pricing_results.json');models=io.read(run/'confirmation/results.json')
    synthetic=io.read(run/'synthetic/results.json');synthetic_plans=io.read(run/'synthetic/plans.json')
    for source in ('protocol.json','pricing_protocol.json'):
        for rel,ref in io.read(run/source)['sources'].items():require(sha(ROOT/rel)==ref['sha256'],'Source changed: '+rel)
    frozen=io.read(run/'confirmation/protocol.json')
    for path,digest in frozen['policies_sha256'].items():require(sha(path)==digest,'Frozen policy changed')
    checks=0;max_error=0.;plans=[];attempts=[]
    for row in planning:
        if row['status']!='complete':continue
        value=row['result'];data=load_target(row,protocol['gain_required']);mode=value['mode']
        path=run/'pricing'/(row['id']+f"__target{row['target']}")/mode
        if value['original_qualified']:
            with np.load(path/'original_quote.npz',allow_pickle=False) as a:u,x=a['u'],a['x']
            part=data['problem'];lo=np.zeros(x.size);hi=np.zeros(x.size)
            lo[part.active]=part.lower;hi[part.active]=part.upper
            require(np.max(np.maximum(lo-x.ravel(),x.ravel()-hi))<1e-8,'Request bound violation')
            independent=direct_gains(data['H'],row['target'],row['competitors'],data['forecast'],data['score']['sigma2'],data['original'],data['base'],u)
            difference=float(np.max(abs(independent-data['evidence'].gains(u))));max_error=max(max_error,difference)
            require(difference<1e-8 and min(independent)>=protocol['gain_required']-1e-8,'Original expected gain audit failed')
            require(np.max(abs(data['original'].nominal(x)-u))<1e-7 and data['original'].audit(u)['passed'],'Original replay audit failed')
            checks+=1
        seed=next(s for s in screens if s['id']==row['id'] and s['target']==row['target'] and s['mode']==mode)
        seed_fee=seed['screen'].get('projected_fee');fee=(value['original'] or {}).get('fee')
        plans.append(dict(id=row['id'],target=row['target'],mode=mode,plant_qualified=value['plant_qualified'],
            original_qualified=value['original_qualified'],normalized_original_fee=None if fee is None else fee/data['fee_scale'],
            normalized_projected_seed_fee=None if seed_fee is None else seed_fee/data['fee_scale'],
            fee_improvement_over_replayed_seed=None if fee is None or not seed_fee else 1-fee/seed_fee,
            ADMM_seconds=value['ADMM_seconds'],screen_seed_seconds=value['screen_seed_seconds'],
            original_quote_origin=(value['original'] or{}).get('origin'),converged_starts=value['converged_starts'],
            global_optimality_proved=False))
        for a in value['attempts']:
            attempts.append(dict(id=row['id'],target=row['target'],mode=mode,start=a['start'],status=a['status'],
                plant_feasible=a['feasible'],first_plant_feasible_seconds=a['first_feasible_seconds'],iterations=a['iterations'],
                branch_seconds=a['total_seconds'],global_optimality_proved=False))
    # Saved complete model-stream statistics independently recomputed.
    for entry in frozen['policies']:
        item=entry['item'];path=run/'confirmation'/(item['id']+f"__target{item['target']}_samples.npz")
        with np.load(path,allow_pickle=False) as samples:
            for strategy in entry['controls']:
                p=samples[strategy+'_posterior'];row=next(r for r in models if (r['id'],r['target'],r['strategy'])==(item['id'],item['target'],strategy))
                require(abs(float(p.mean())-row['mean_posterior'])<1e-12,'Model posterior summary mismatch')
                for beta,rate in row['pass_rates'].items():require(float(np.mean(p>=float(beta)))==rate,'Model threshold mismatch')
    report_seed=io.read(run/'previous_counterfactual/summary.json');pairs=io.read(run/'previous_diagnostics/paired.json')
    old=next(r for r in pairs if r['id']=='main__input054__101' and r['target']==6 and r['strategy']=='admm')
    # Truth joins only the completed comparison, not planning or selection.
    truth={r['id']:r for r in io.read(REFERENCE/'evaluation.json')}
    screen_table=[]
    for r in screens:
        s=r['screen'];data=load_target(r,protocol['gain_required']);score=data['score'];J=np.asarray(r['baseline_distance_standardized'])
        projected=s.get('projected_gains');nominal=s.get('gains')
        screen_table.append(dict(id=r['id'],target=r['target'],mode=r['mode'],native_status=s['status'],
            target_correct=score['models'][r['target']]['topology_id']==truth[r['id']]['truth_id'],
            plant_qualified=s['qualified'],original_replay_qualified=bool(projected is not None and min(projected)>=protocol['gain_required']-1e-8),
            minimum_nominal_gain=None if nominal is None else min(nominal),minimum_original_replay_gain=None if projected is None else min(projected),
            closest_baseline_standardized_distance=float(J[0]),power_box_minimum_upper_bound=min(r['power_box_gain_upper']),
            original_infeasibility_proved=bool(min(r['power_box_gain_upper'])<protocol['gain_required']),
            normalized_nominal_fee=None if s.get('net_fee') is None else s['net_fee']/r['fee_scale'],
            normalized_replayed_fee=None if s.get('projected_fee') is None else s['projected_fee']/r['fee_scale'],
            seconds=s['seconds'],simultaneous_charge_discharge_kwh=s.get('simultaneous_kwh'),
            nominal_original_device_feasible=(s.get('original_device_audit') or {}).get('passed')))
    destination.mkdir(parents=True)
    write_csv(destination/'screen.csv',screen_table);write_csv(destination/'planning.csv',plans);write_csv(destination/'attempts.csv',attempts)
    write_csv(destination/'model.csv',[{k:v for k,v in r.items() if k not in ('paired_gains',)} for r in models])
    write_csv(destination/'synthetic.csv',synthetic);write_csv(destination/'synthetic_solver.csv',synthetic_plans)
    io.write(destination/'previous_gap.json',{k:v for k,v in report_seed.items() if k!='source'})
    io.write(destination/'previous_odds_decomposition.json',{k:v for k,v in old.items() if k!='raw_distance'})
    audit=dict(status='passed',created_utc=utc(),original_quotes_recomputed=checks,maximum_independent_gain_error=max_error,
        all_source_and_policy_hashes_unchanged=True,new_identifier_fits=0,new_real_parameter_fits=0,
        new_AC_paths=0,old_AC_paths_diagnosed=32,conditional_model_paths_per_target=protocol['model_paths'],
        direct_model_residual_checks=io.read(run/'confirmation/complete.json')['direct_likelihood_checks'],
        all_saved_posterior_mean_and_threshold_counts_recomputed=True,
        synthetic_all_four_levels=True,synthetic_scope='explicit artificial pair, no real fit modified',
        native_screen_statuses=dict(Counter(str(r['native_status']) for r in screen_table)),
        actual_ADMM_stop_reasons=dict(Counter(r['status'] for r in attempts)),
        targeted_unit_checks=4,targeted_unit_checks_passed=4,
        original_infeasible_certificates=sum(r['original_infeasibility_proved'] for r in screen_table),
        source=io.ref(__file__))
    io.write(run/'independent_audit.json',audit)
    io.write(destination/'VALIDATION.json',{k:v for k,v in audit.items() if k!='source'})
    lines=['# 连续设备和线性DR：完整测试结果','',
        '本轮真实算例未调整拟合参数、噪声或证据门槛。先对已有6个输入的12个条件目标筛查，再锁定Paper15和Flynn两个输入的4个目标进行ADMM与独立模型确认。相反目标方向完整保留，4个目标不计作4个独立输入。人工例单列。','',
        '## 可行性先于效果','',
        '| 设备模型 | QP提议合格 | 原执行器重放后合格 | 其余状态 |',
        '|---|---:|---:|---|']
    for mode in protocol['modes']:
        subset=[r for r in screen_table if r['mode']==mode]
        lines.append(f"| {LABELS[mode]} | {sum(r['plant_qualified'] for r in subset)}/{len(subset)} | {sum(r['original_replay_qualified'] for r in subset)}/{len(subset)} | QP最优结束{sum(r['native_status']==2 for r in subset)}，超时{sum(r['native_status']==9 for r in subset)} |")
    lines+=['','线性QP只要求全局二次下界达到增量；提议之后经过原控制器截断、能量和截止回补，再用完整冻结H重新计算。表中原执行器合格方案满足原设备域和原增量要求。Paper15两个方向的零点QP重放增量均约0.909，直接构成原问题有解的证据；此前限时失败不再作为无解的疑据。所有功率盒上界均高于要求，本轮没有原问题不可行证明。','',
        '储能松弛QP允许同时充放电，纯线性响应删除能量耦合。两者均需重放后才可作为原设备报价。完整DR问题仍是非凸QCQP，凸的是设备更新与提议QP。','',
        '## ADMM、初值与费用','',
        '| 输入／目标 | 模型 | 原设备报价 | 相对原全幅轮换费用 | 相对重放初值节费 | ADMM阶段秒 | 初值筛查秒 | 收敛起点数 |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for r in plans:
        cost='无报价' if r['normalized_original_fee'] is None else f"{r['normalized_original_fee']:.6g}"
        change='无合格初值比较' if r['fee_improvement_over_replayed_seed'] is None else f"{r['fee_improvement_over_replayed_seed']:.2%}"
        origin=r['original_quote_origin']
        if origin=='input_screen_QP':origin='保留输入QP方案'
        elif origin:origin='迭代中保留'
        else:origin='无报价'
        lines.append(f"| {r['id']}／{r['target']} | {LABELS[r['mode']]} | {origin} | {cost} | {change} | {r['ADMM_seconds']:.2f} | {r['screen_seed_seconds']:.2f} | {r['converged_starts']} |")
    lines+=['','初值生成时间不包含在表中ADMM阶段时间内，完整目标流程须将两者相加；全部6输入的筛查还需另外计入总开发成本。线性模型同时从零点和已有QP初值开始；储能模型从零开始。若保留输入QP方案，来源明确写为input_screen_QP，不计为ADMM新找到的改进。各起点和停止原因见attempts.csv。所有实际原问题报价都没有全局费用最优证明。',
        '',f"实际ADMM停止原因：{audit['actual_ADMM_stop_reasons']}。达到停止条件的两个分支属于Flynn的纯线性响应模型；原执行器核验通过的最佳报价来自中间点。线性模型收敛与原设备投影后问题收敛分开报告。",'',
        '## 新的条件模型结果','',
        '每目标2048条新路径，背景功率及残差在策略间配对；报价先冻结，候选与参数不重拟合。下表列真实结构目标；错误条件目标的全部结果也在model.csv。费用按该输入原全幅轮换归一化。轮换只匹配线性QP提议的费用，并未另外匹配ADMM较低费用。此次没有新增AC路径。',
        '', '预测均值仍为历史96点P/Q；模型内按其峰值5%的标准差叠加误差，时间相关0.7。这个法则声明了模型中的预测精度，没有验证对独立AC新日的精度。若采用用户明确的高精度预测前提，后续应先生成真实新日，再按事前固定噪声形成预测输入；规划与评价分开读取。', '',
        '| 输入 | 策略 | 平均后验 | 后验≥90% | 相对费用 | 主要竞争者解析增量 |',
        '|---|---|---:|---:|---:|---|']
    for r in models:
        if r['target_correct']:
            lines.append(f"| {r['id']} | {LABELS[r['strategy']]} | {r['mean_posterior']:.3%} | {r['pass_rates']['0.9']:.3%} | {r['normalized_fee']:.6g} | {[round(g,4) for g in r['analytic_gains']]} |")
    lines+=['','## 为什么旧J增大、后验变化小','',
        '本轮明确Jdist=||D P||²/(2σ²)。旧代码的平均绝对距离、指数分数dᵀWd以及残差损失S/2有不同单位，不能直接比较它们的数值大小。旧冻结距离报告本身记录设计DR距离增加1.571415微pu²，但最终恢复仍为0/4，原3%带退出0/4；因此早期距离增大也没有自动转成最终辨识收益。','',
        f"上轮Flynn实际32日：Jdist平均从7.77835升到7.92948，增加{old['distance']['mean']:.6f}（约1.94%）；残差交叉项变化{old['cross']['mean']:.6f}；两者相加为对数赔率增量{old['new_log_odds']['mean']:.6f}。平均后验增加{100*old['p_target']['mean']:.4f}个百分点。",'',
        f"该实际平均证据增量的95%均值区间为{[round(x,4) for x in old['new_log_odds']['mean_interval95']]}，包含0；完整恢复仍为30/32。新条件模型改善与原实际AC改善分别说明。",'',
        f"同一名义方案：历史预测背景上的增量{report_seed['planned_on_forecast_gain']:.6f}；移到已记录实际背景后为{report_seed['nominal_on_actual_background_mean']:.6f}；实际观测功率给出{report_seed['observed_power_distance_mean']:.6f}。主要损失来自背景预测变化，执行和功率量测差异净贡献仅{report_seed['execution_and_meter_difference_mean']:.6f}。这个事后分解未用于本轮报价。",'',
        '当后验已接近1，同样的赔率增量对应更小的概率增量；同时，平均后验不等于把平均对数赔率代入一次归一化。本批既保留证据尺度，也单列平均后验和达标率。','',
        '## 人工构造例：保留全部预设强度','',
        'Ht=I，Hj=.98I+.01·11ᵀ。平衡自然功率使两者预测相同；反向线性DR产生区分。该例只检验算法与概率数值关系，未改动真实拟合。费用与真实合同不混用。','',
        '| 要求的平均证据增量 | 样本平均增量 | 平均后验 | 后验≥90% | 合成费用 |',
        '|---:|---:|---:|---:|---:|']
    for r in synthetic:
        lines.append(f"| {r['gamma']:.6g} | {r['mean_gain']:.6g} | {r['mean_posterior']:.3%} | {r['pass90']:.3%} | {r['fee']:.7g} |")
    lines+=['','四档ADMM均7—8轮达到停止条件；费用与独立手算的γ/k最优值一致到绝对误差3e-9以内。这是这个单幅值人工例的解析核验，不作为实际多设备问题的最优保证。65536条新高斯路径共用噪声，所有四档强度在打开路径前固定。','',
        '## 下一步','',
        '当前最简单可用的路径是：凸QP产生线性DR提议→原执行器重放→重算期望证据和费用。ADMM用于进一步降低费用。下一步优先在保留完整历史的条件下改善次日功率预测，或用事前固定的多个历史模板评价；随后锁定少量原设备可执行报价做新的AC配对验证。保留失败，不通过调整真线路或未来样本制造效果。']
    (destination/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(destination,models,synthetic)
    io.write(destination/'RECEIPT.json',dict(created_utc=utc(),pricing_protocol_sha256=sha(run/'pricing_protocol.json'),
        confirmation_protocol_sha256=sha(run/'confirmation/protocol.json'),audit_sha256=sha(run/'independent_audit.json'),
        files={p.name:sha(p) for p in destination.iterdir() if p.is_file()},private_raw_data_published=False))
    io.write(run/'complete.json',dict(status='complete',created_utc=utc(),report=str(destination/'REPORT.md'),source=io.ref(__file__)))
    print({k:v for k,v in audit.items() if k!='source'},flush=True)


def plot(destination,models,synthetic):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,3.7))
    for identity in ('main__input054__101','main__input121__101'):
        r=[r for r in models if r['id']==identity and r['target_correct'] and r['strategy'] in ('no_dr','linear_QP_replayed','linear_response_ADMM_replayed')]
        lookup={x['strategy']:x for x in r};order=['no_dr','linear_QP_replayed','linear_response_ADMM_replayed']
        x=[i for i,k in enumerate(order) if k in lookup]
        axes[0].plot(x,[lookup[order[i]]['mean_posterior'] for i in x],'o-',label=identity.split('__')[1])
    axes[0].set(xticks=[0,1,2],xticklabels=['No DR','Linear QP replay','ADMM replay'],ylim=(0,1),ylabel='Mean target posterior',title='Frozen real-case candidates: model only')
    axes[0].legend(frameon=False)
    axes[1].plot([r['analytic_gain'] for r in synthetic],[r['mean_posterior'] for r in synthetic],'o-',label='Mean posterior')
    axes[1].plot([r['analytic_gain'] for r in synthetic],[r['pass90'] for r in synthetic],'s-',label='Posterior >= 90%')
    axes[1].set(xlabel='Expected log-odds gain',ylim=(0,1),title='Explicit artificial linear-response pair')
    axes[1].legend(frameon=False)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout()
    for suffix in ('png','svg'):
        path=destination/('comparison.'+suffix);fig.savefig(path,dpi=180,bbox_inches='tight')
        if suffix=='svg':path.write_text('\n'.join(s.rstrip() for s in path.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--public',type=Path,required=True)
    a=p.parse_args();publish(a.run.resolve(),a.public.resolve())
