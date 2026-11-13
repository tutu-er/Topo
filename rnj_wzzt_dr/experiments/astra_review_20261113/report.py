"""Verify sealed fresh-day results and report all predeclared groups separately."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from scipy.stats import binomtest, binom, norm
from ... import io
from ..customer_flexibility_20261113.model import CustomerCase

LABELS = dict(no_dfs='无服务',old_gamma2='旧增量2',old_rotation='旧方案同费轮换',
    nominal='名义恢复约束',nominal_rotation='名义方案同费轮换',
    calibrated='均值修正恢复约束',calibrated_rotation='均值修正同费轮换')
ROOT=Path(__file__).resolve().parents[3]

def numeric_mean(values):
    values=[float(x) for x in values if x is not None and np.isfinite(x)]
    return None if not values else float(np.mean(values))

def bootstrap_interval(x, seed, repetitions=2000):
    x=np.asarray(x,float)
    if not len(x): return None
    rng=np.random.default_rng(seed)
    means=x[rng.integers(0,len(x),(repetitions,len(x)))].mean(1)
    return np.quantile(means,[.025,.975]).tolist()

def paired(rows, a, b, days, seed):
    first={r['day']:r for r in rows if r['strategy']==a}
    second={r['day']:r for r in rows if r['strategy']==b}
    correct=lambda row: bool(row.get('status')=='complete' and row.get('native_map_correct',False))
    recovery=np.array([int(correct(first.get(d,{})))-int(correct(second.get(d,{}))) for d in range(days)])
    probability=[]; both=[]
    for day in range(days):
        x,y=first.get(day,{}),second.get(day,{})
        if x.get('status')==y.get('status')=='complete' and x.get('p_true') is not None and y.get('p_true') is not None:
            probability.append(x['p_true']-y['p_true']);both.append(day)
    p=np.asarray(probability)
    rescued=int(np.sum(recovery>0));lost=int(np.sum(recovery<0))
    gain_ci=binomtest(rescued,days).proportion_ci(confidence_level=.975,method='exact')
    loss_ci=binomtest(lost,days).proportion_ci(confidence_level=.975,method='exact')
    return dict(strategy=a,reference=b,planned_pairs=days,complete_probability_pairs=len(p),
        new_recoveries=rescued,lost_recoveries=lost,recovery_gain=float(recovery.mean()),
        recovery_gain_bootstrap95=bootstrap_interval(recovery,seed),
        recovery_gain_conservative_exact95=[float(gain_ci.low-loss_ci.high),float(gain_ci.high-loss_ci.low)],
        recovery_exact_interval_method='Gain and loss multinomial proportions each receive a97.5% Clopper-Pearson interval; subtract opposite endpoints. Bonferroni joint coverage at least95%, conservative.',
        exact_mcnemar_two_sided_p=1. if rescued+lost==0 else float(binomtest(rescued,rescued+lost,.5).pvalue),
        recovery_bootstrap_degenerate=bool(np.all(recovery==recovery[0])),
        zero_discordance_95_upper=None if rescued+lost else float(1-.05**(1/days)),
        mean_true_weight_gain=None if not len(p) else float(p.mean()),
        true_weight_gain_bootstrap95=bootstrap_interval(p,seed+1),
        improved_weight_days=int(np.sum(p>1e-12)),declined_weight_days=int(np.sum(p< -1e-12)),
        worst_weight_change=None if not len(p) else float(p.min()),
        worst_day=None if not len(p) else both[int(np.argmin(p))],
        failure_handling='Missing or failed AC rows count as nonrecovery; mean weights use explicitly counted complete pairs.')

def verify(run,protocol,rows):
    checks=dict(day_seals=0,observation_files=0,native_selections=0,resource_actions=0,
        probability_max_error=0.,sse_max_error=0.,fee_max_error=0.,delivered_energy_max_error=0.,
        rejection_probability_max_error=0.,source_hash_checks=0,errors=[])
    def check(condition,message):
        if not condition: checks['errors'].append(message)
    lock=io.read(run/'lock.json')
    check(io.ref(run/'protocol.json')['sha256']==lock['protocol']['sha256'],'Protocol hash drift')
    for relative,expected in protocol['sources'].items():
        p=run/'source_snapshot'/relative
        check(p.exists() and io.ref(p)['sha256']==expected,'Snapshot hash drift: '+relative)
        checks['source_hash_checks']+=1
    truth=io.read(run/'offline_truth.json')['truth_id']
    rowmap={(r['id'],r['day'],r['strategy']):r for r in rows}
    for entry in protocol['entries']:
        score=io.read(entry['score']['path'])
        check(io.ref(entry['score']['path'])['sha256']==entry['score']['sha256'],'Score hash drift: '+entry['id'])
        case=CustomerCase(io.read(run/'resource_configs'/f"{entry['adoption']}.json"))
        H=np.asarray([p['H'] for p in score['parameters']]);history=np.array([p['history_sse'] for p in score['parameters']])
        bank=score['native_bank'];nativeH=np.asarray([p['H'] for p in bank]);native_history=np.array([p['history_sse'] for p in bank])
        ids=[m['topology_id'] for m in score['models']]
        target=score['map_index'];truth_i=next((i for i,x in enumerate(ids) if x==truth),None)
        others=[i for i in range(len(H)) if i!=target]
        runner=max(others,key=lambda i:score['models'][i]['log_probability']) if others else None
        with np.load(entry['observed_refs']['validation']['path']) as vobs:
            Xv=np.c_[vobs['P'],vobs['Q']];Yv=vobs['root'][:,None]**2-vobs['V']**2
        target_prediction=Xv@H[target].T;residual=Yv-target_prediction
        centered=target_prediction-target_prediction.mean(0)
        slope=np.sum(centered*(residual-residual.mean(0)),axis=0)/(np.sum(centered**2,axis=0)+len(Xv)*score['sigma2'])
        intercept=residual.mean(0)-slope*target_prediction.mean(0)
        for day in range(entry['new_days']):
            d=run/'confirmation'/entry['id']/f'day{day:03d}'
            if not (d/'policy_seal.json').exists(): continue
            seal=io.read(d/'policy_seal.json')
            for key in ('planner','plans','score','protocol'):
                check(io.ref(seal[key]['path'])['sha256']==seal[key]['sha256'],f'Seal mismatch {entry["id"]}/{day}/{key}')
            checks['day_seals']+=1
            with np.load(d/'physical_background.npz') as physical:
                eta=np.array(physical['response_multiplier'])
                true_root=np.array(physical['root'])
            base_kw=io.read(d/'generation.json')['base_kw']
            with np.load(d/'planner.npz') as planner:
                controls={s:np.array(planner[s]) for s in protocol['strategies']}
                forecast=np.array(planner['forecast'])
            for strategy,u in controls.items():
                key=(entry['id'],day,strategy);r=rowmap.get(key)
                if r is None: continue
                check(case.audit(u)['passed'] and case.audit(u*eta[None])['passed'],f'Resource infeasible {key}')
                checks['resource_actions']+=1
                checks['fee_max_error']=max(checks['fee_max_error'],abs(case.fee(u)-r['planned_fee_yuan']),abs(case.fee(u*eta[None])-r['settled_fee_yuan']))
                checks['delivered_energy_max_error']=max(checks['delivered_energy_max_error'],abs(case.dt*np.abs(u*eta[None]).sum()-r['delivered_abs_kwh']))
                delivered=u*eta[None]
                resource_energy=case.dt*np.abs(delivered).sum(0)
                resource_fee=case.dt*(case.fee_plus*np.maximum(delivered,0)+case.fee_minus*np.maximum(-delivered,0)).sum(0)
                kinds={};nodes={}
                for k,resource in enumerate(case.resources):
                    kind=resource['kind'];node=str(resource['node_label'])
                    kinds.setdefault(kind,dict(abs_kwh=0.,fee_yuan=0.));nodes.setdefault(node,dict(abs_kwh=0.,fee_yuan=0.))
                    for item in (kinds[kind],nodes[node]):
                        item['abs_kwh']+=float(resource_energy[k]);item['fee_yuan']+=float(resource_fee[k])
                r['postprocessed_resource_delivery']=dict(by_kind=kinds,by_node=nodes)
                if r['status']!='complete': continue
                with np.load(d/(strategy+'.npz')) as obs:
                    power=np.c_[obs['P'],obs['Q']];y=obs['root'][:,None]**2-obs['V']**2
                    losses=history+np.sum((y[None]-np.einsum('tc,knc->ktn',power,H))**2,axis=(1,2))
                    ll=-losses/(2*score['sigma2']);logp=ll-logsumexp(ll)
                    checks['sse_max_error']=max(checks['sse_max_error'],float(np.max(abs(losses-obs['cumulative_sse']))))
                    checks['probability_max_error']=max(checks['probability_max_error'],float(np.max(abs(np.exp(logp)-np.exp(obs['log_probability'])))),abs(float(np.exp(logp[target]))-r['p_target']))
                    if truth_i is not None:
                        checks['probability_max_error']=max(checks['probability_max_error'],abs(float(np.exp(logp[truth_i]))-r['p_true']))
                    nl=native_history+np.sum((y[None]-np.einsum('tc,knc->ktn',power,nativeH))**2,axis=(1,2))
                    scaled=nl/(score['cells']+y.size)/score['reference_mse']
                    eligible=np.flatnonzero(scaled<=float(scaled.min())+score['validation_tie'])
                    native=min(eligible,key=lambda i:(bank[i]['internal'],scaled[i],i))
                    check((bank[native]['topology_id']==truth)==r['native_map_correct'],f'Native selection disagreement {key}')
                    check(int(native)==r['native_selected_index'],f'Native index disagreement {key}')
                    checks['native_selections']+=1;checks['observation_files']+=1
                    if runner is not None:
                        actual=np.c_[obs['true_P'],obs['true_Q']]
                        actual_y=true_root[:,None]**2-obs['true_V']**2
                        D=H[target]-H[runner];dactual=actual@D.T
                        actual_bias=actual_y-actual@H[target].T
                        nominal=forecast+case.shift_pq_pu(u,base_kw)
                        dnominal=nominal@D.T
                        Jactual=float(np.sum(dactual*dactual)/(2*score['sigma2']))
                        Jforecast=float(np.sum(dnominal*dnominal)/(2*score['sigma2']))
                        bias=float(np.sum(actual_bias*dactual)/score['sigma2'])
                        Lh=(history[runner]-history[target])/(2*score['sigma2'])
                        affine_bias=intercept+(nominal@H[target].T)*slope
                        affine=float(np.sum(affine_bias*dnominal)/score['sigma2'])
                        observed=float(logp[target]-logp[runner])
                        r['postprocessed_mechanism']=dict(forecast_nominal_J=Jforecast,actual_power_J=Jactual,
                            frozen_parameter_bias=bias,clean_total_logodds=Lh+Jactual+bias,
                            observed_total_logodds=observed,meter_logodds_error=observed-Lh-Jactual-bias,
                            affine_predicted_bias=affine,affine_predicted_total_logodds=Lh+Jforecast+affine)
                    r['postprocessed_native_rejection']={}
                    for factor,value in r['rejection_rules'].items():
                        B=(score['cells']+y.size)*float(factor)*8*entry['voltage_sigma']**2
                        lognull=-B/(2*score['sigma2']);normal=logsumexp(np.r_[ll,lognull])
                        rejected=bool(lognull>=max(ll));selected=int(np.argmax(ll))
                        check(rejected==value['rejected'],f'Rejection decision mismatch {key}/{factor}')
                        check(bool(not rejected and selected==truth_i)==value['correct'],f'Rejection correctness mismatch {key}/{factor}')
                        checks['rejection_probability_max_error']=max(checks['rejection_probability_max_error'],abs(float(np.exp(ll[target]-normal))-value['p_target']))
                        native_rejected=bool(np.min(nl)>=B)
                        native_correct=bool(not native_rejected and bank[native]['topology_id']==truth)
                        r['postprocessed_native_rejection'][factor]=dict(rejected=native_rejected,
                            correct=native_correct,wrong=bool(not native_rejected and not native_correct))
    for key in ('probability_max_error','sse_max_error','fee_max_error','delivered_energy_max_error','rejection_probability_max_error'):
        check(checks[key]<1e-8,'Numerical reconstruction failed: '+key)
    checks['passed']=not checks['errors']
    return checks

def solver_summary(plans):
    feasible=[p for p in plans if p.get('feasible')]
    nonzero=[p for p in feasible if p.get('cost_yuan',0)>1e-9]
    certified=[p for p in feasible if p.get('final',p.get('baseline',{})).get('model_recovery_lower',0)>=.9-1e-6]
    return dict(quotes=len(plans),feasible=len(feasible),nonzero_feasible=len(nonzero),
        zero_sufficient=sum(p.get('source')=='zero' for p in feasible),
        statuses=dict(Counter(p.get('status','unknown') for p in plans)),sources=dict(Counter(p.get('source','unavailable') for p in plans)),
        certified_in_declared_model=len(certified),
        infeasibility_proved=sum(bool(p.get('infeasibility_proved')) for p in plans),
        infeasibility_scope='The locked QCQP: Gaussian sufficient constraints for new methods, fixed increment and historical guards for the old baseline. Not proof of physical or original chance-problem infeasibility.',
        zero_fallbacks=sum(bool(p.get('explicit_zero_fallback')) for p in plans),
        native_time_limits=sum(p.get('native_status')==9 for p in plans),
        global_bound_closed=sum(bool(p.get('global_optimality_verified')) for p in feasible),
        nonzero_global_bound_closed=sum(bool(p.get('global_optimality_verified')) for p in nonzero),
        initial_retained=sum(p.get('source') in ('verified_initial_retained','supporting_LP_initial_point_retained') for p in plans),
        native_found_without_verified_initial=sum(p.get('source') in ('native_found','native_QCQP_found_feasible') for p in plans),
        native_improved_initial=sum(p.get('source') in ('native_improved_initial','native_QCQP_improved_initial_point') for p in plans),
        mean_improvement_from_verified_initial=numeric_mean([p.get('improvement_from_verified_initial_yuan') for p in plans]),
        mean_relative_gap_nonzero=numeric_mean([p.get('relative_gap') for p in nonzero]),
        mean_valid_lower_bound_nonzero=numeric_mean([p.get('lower_bound_yuan') for p in nonzero]),
        mean_feasible_upper_bound_nonzero=numeric_mean([p.get('cost_yuan') for p in nonzero]),
        withdrawn_bounds=sum(bool(p.get('lower_bound_withdrawn')) for p in plans),
        mean_driver_seconds=numeric_mean([p.get('driver_planning_seconds',p.get('seconds')) for p in plans]),
        maximum_driver_seconds=max([p.get('driver_planning_seconds',p.get('seconds',0)) for p in plans],default=None),
        time_scope='Driver wall time includes initial construction, scalar rejection threshold, model construction and native solving.',
        same_planned_fee_failures=sum(not p.get('rotation_same_planned_fee',False) for p in plans))

def report(run,destination):
    if not (run/'complete.json').exists(): raise ValueError('Run is not complete; do not publish a partial success subset')
    protocol=io.read(run/'protocol.json')
    rows=[r for p in sorted((run/'confirmation').glob('*/day*/results.json')) for r in io.read(p)]
    plans={(p.parent.parent.name,int(p.parent.name[3:]),k):v for p in sorted((run/'confirmation').glob('*/day*/plans.json')) for k,v in io.read(p).items()}
    outcome=dict(protocol=io.ref(run/'protocol.json'),completion=io.read(run/'complete.json'),
        report_source=io.ref(__file__),bootstrap=dict(repetitions=2000,seed=1001,unit='paired independent new day conditional on fixed historical input'),
        interpretation='Old32 paths are development only and excluded. Three selected historical conditions reported separately. Native-bank complete recovery was the locked primary outcome; all pairwise statistical comparisons are exploratory. Recovery intervals use conservative exact paired-proportion bounds; weight intervals use paired percentile bootstrap. p values are unadjusted for multiple policies.',
        verification=verify(run,protocol,rows),groups={})
    seed=1001
    for entry in protocol['entries']:
        own=[r for r in rows if r['id']==entry['id']]
        group=dict(role=entry['role'],planned_days=entry['new_days'],strategies={},paired={},solver={})
        for strategy in protocol['strategies']:
            attempted=[r for r in own if r['strategy']==strategy]
            complete=[r for r in attempted if r['status']=='complete']
            rr=dict(planned_days=entry['new_days'],recorded_rows=len(attempted),complete_rows=len(complete),
                failed_or_missing_rows=entry['new_days']-len(complete),
                native_correct=sum(r['native_map_correct'] for r in complete),representative_correct=sum(r['map_correct'] for r in complete),
                mean_true_weight=numeric_mean([r['p_true'] for r in complete]),mean_target_weight=numeric_mean([r['p_target'] for r in complete]),
                true_weight_above90=sum(bool(r['true_pass90']) for r in complete),
                mean_planned_fee=numeric_mean([r['planned_fee_yuan'] for r in attempted]),
                mean_settled_fee=numeric_mean([r['settled_fee_yuan'] for r in attempted]),
                mean_delivered_abs_kwh=numeric_mean([r['delivered_abs_kwh'] for r in attempted]),
                voltage_violation_cells=sum(r['voltage_cells_outside_screen'] for r in complete),
                voltage_violation_days=sum(r['voltage_cells_outside_screen']>0 for r in complete),
                minimum_voltage=min([r['minimum_voltage_pu'] for r in complete],default=None),
                maximum_voltage=max([r['maximum_voltage_pu'] for r in complete],default=None),
                device_failed_days=sum(not r['device']['passed'] for r in complete),
                zero_fallbacks=sum(r['zero_fallback'] for r in attempted),
                rejection={},native_with_rejection={})
            for factor in ('1.25','2.0','4.0'):
                vals=[r['rejection_rules'][factor] for r in complete]
                rr['rejection'][factor]=dict(correct=sum(v['correct'] for v in vals),wrong=sum(v['wrong'] for v in vals),
                    rejected=sum(v['rejected'] for v in vals),failed_or_missing=entry['new_days']-len(vals),
                    mean_true_weight=numeric_mean([v['p_true'] for v in vals]),mean_target_weight=numeric_mean([v['p_target'] for v in vals]))
                nativevals=[r['postprocessed_native_rejection'][factor] for r in complete]
                rr['native_with_rejection'][factor]=dict(correct=sum(v['correct'] for v in nativevals),wrong=sum(v['wrong'] for v in nativevals),
                    rejected=sum(v['rejected'] for v in nativevals),failed_or_missing=entry['new_days']-len(nativevals))
            group['strategies'][strategy]=rr
            mechanism=[r['postprocessed_mechanism'] for r in complete if 'postprocessed_mechanism' in r]
            rr['mechanism_means']=None if not mechanism else {key:numeric_mean([r[key] for r in mechanism]) for key in mechanism[0]}
            delivery=[r['postprocessed_resource_delivery'] for r in attempted if 'postprocessed_resource_delivery' in r]
            rr['resource_delivery_means']={}
            for level in ('by_kind','by_node'):
                keys=sorted({key for r in delivery for key in r[level]})
                rr['resource_delivery_means'][level]={key:{measure:numeric_mean([r[level].get(key,{}).get(measure,0.) for r in delivery]) for measure in ('abs_kwh','fee_yuan')} for key in keys}
            if strategy!='no_dfs':
                group['paired'][strategy+'_vs_no_dfs']=paired(own,strategy,'no_dfs',entry['new_days'],seed);seed+=2
        for mode,rotation in [('old_gamma2','old_rotation'),('nominal','nominal_rotation'),('calibrated','calibrated_rotation')]:
            ps=[plans[(entry['id'],day,mode)] for day in range(entry['new_days']) if (entry['id'],day,mode) in plans]
            group['solver'][mode]=solver_summary(ps)
            group['paired'][mode+'_vs_rotation']=paired(own,mode,rotation,entry['new_days'],seed);seed+=2
            if mode!='old_gamma2':
                certified=[day for day in range(entry['new_days']) if (p:=plans.get((entry['id'],day,mode))) is not None and p.get('feasible') and p.get('final',p.get('baseline',{})).get('model_recovery_lower',0)>=.9-1e-6]
                chosen={r['day']:r for r in own if r['strategy']==mode and r['status']=='complete'}
                group['solver'][mode].update(model_certified_days=len(certified),
                    actual_correct_among_certified=sum(chosen.get(d,{}).get('native_map_correct',False) for d in certified),
                    actual_nonrecovery_among_certified=sum(not chosen.get(d,{}).get('native_map_correct',False) for d in certified),
                    actual_native_with_rejection_correct_among_certified=sum(chosen.get(d,{}).get('postprocessed_native_rejection',{}).get('2.0',{}).get('correct',False) for d in certified))
                ss=group['solver'][mode]
                failures=len(certified)-ss['actual_native_with_rejection_correct_among_certified']
                ss.update(actual_native_with_rejection_failures_among_certified=failures,
                    model90_binomial_tail=None if entry['role']=='missing_truth' or not certified else float(binom.sf(failures-1,len(certified),.1)),
                    model90_binomial_interpretation='One-sided tail under independent declared-model failure probabilities at most0.10, conditional on the fixed history and certified forecasts. Diagnoses model mismatch; not proof of calibration. Missing-truth controls excluded.')
                score=io.read(entry['score']['path'])
                sums=[];largest=[];ratios=[]
                for day in certified:
                    p=plans[(entry['id'],day,mode)];point=p.get('final',p.get('baseline',{}))
                    mu=np.asarray(point['mu']);J=np.asarray(point['J'])
                    ratio=p['calibration']['variance']/score['sigma2']
                    sd=np.sqrt(np.maximum(0,2*ratio*J))
                    tie=max(1e-8,(score['cells']+1440)*score['reference_mse']*score['validation_tie']/(2*score['sigma2']))
                    tails=norm.cdf(np.divide(tie-mu,sd,out=np.full_like(mu,-np.inf),where=sd>0))
                    tails[(sd==0)&(mu<=tie)]=1.
                    total=float(tails.sum());big=float(tails.max(initial=0))
                    sums.append(total);largest.append(big)
                    if big>0:ratios.append(total/big)
                ss['competitor_conservatism']=dict(mean_pair_failure_sum=numeric_mean(sums),
                    mean_largest_pair_failure=numeric_mean(largest),mean_sum_to_largest_ratio=numeric_mean(ratios),
                    max_sum_to_largest_ratio=max(ratios,default=None),
                    interpretation='Sum versus largest pair bound quantifies risk allocation across comparisons; exact union probability also depends on correlation.')
        outcome['groups'][entry['id']]=group
    destination.mkdir(parents=True,exist_ok=True)
    io.write(destination/'REPORT.json',outcome)
    lines=['# 新48日完整实验分析','',
        '旧32日用于方法选择，不与本轮合并。弱证据32日、容易8日、候选库缺真拓扑8日分别报告。每条新日的预测、响应和表计随机路径在各方案间配对；未决报价执行零动作，AC失败或缺失计为未恢复。',
        '',f"重算核验：{'通过' if outcome['verification']['passed'] else '失败，详见REPORT.json'}；{outcome['verification']['observation_files']}份观测，{outcome['verification']['resource_actions']}份请求及实际交付，{outcome['verification']['day_seals']}个策略封存记录。",'']
    fmt=lambda x,digits=2:'—' if x is None else f'{x:.{digits}f}'
    for identity,g in outcome['groups'].items():
        lines += [f'## {g["role"]}：{identity}，{g["planned_days"]}日','',
            '|方案|完整恢复|代表模型恢复|平均真拓扑权重|平均目标权重|实际激活费/元|调节电量/kWh|未决等零回退|',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
        for strategy,r in g['strategies'].items():
            lines.append(f"|{LABELS[strategy]}|{r['native_correct']}/{g['planned_days']}|{r['representative_correct']}/{g['planned_days']}|{fmt(None if r['mean_true_weight'] is None else 100*r['mean_true_weight'])}%|{fmt(None if r['mean_target_weight'] is None else 100*r['mean_target_weight'])}%|{fmt(r['mean_settled_fee'])}|{fmt(r['mean_delivered_abs_kwh'])}|{r['zero_fallbacks']}|")
        lines += ['','|方案相对无服务|新增／丢失恢复|恢复率变化95%区间/百分点|平均权重变化95%区间/百分点|权重提高／下降日|最坏下降/百分点|精确p值|','|---|---:|---:|---:|---:|---:|---:|']
        for mode in ('old_gamma2','nominal','calibrated'):
            p=g['paired'][mode+'_vs_no_dfs']
            ri=p['recovery_gain_conservative_exact95'];wi=p['true_weight_gain_bootstrap95']
            lines.append(f"|{LABELS[mode]}|{p['new_recoveries']}/{p['lost_recoveries']}|[{100*ri[0]:.2f},{100*ri[1]:.2f}]|{'—' if wi is None else f'[{100*wi[0]:.2f},{100*wi[1]:.2f}]'}|{p['improved_weight_days']}/{p['declined_weight_days']}|{fmt(None if p['worst_weight_change'] is None else 100*p['worst_weight_change'])}|{p['exact_mcnemar_two_sided_p']:.4f}|")
        if g['role']=='weak':
            lines += ['','|方案|规划预测J|实际功率J|冻结参数偏差项|真实无噪累计赔率|实测累计赔率|仿射修正预测累计赔率|','|---|---:|---:|---:|---:|---:|---:|']
            for mode in ('no_dfs','old_gamma2','nominal','calibrated'):
                m=g['strategies'][mode]['mechanism_means']
                lines.append('|'+LABELS[mode]+'|'+'|'.join(fmt(m[k],3) for k in ('forecast_nominal_J','actual_power_J','frozen_parameter_bias','clean_total_logodds','observed_total_logodds','affine_predicted_total_logodds'))+'|')
            lines += ['','上述分解只针对历史代表模型库中最强竞争者。真实P/Q/V用于离线解释，不参与规划。J增大可能同时伴随不利的冻结参数偏差；仿射修正的预测方向也需由新日检验。','',
                '|方案|资源类型|日均绝对调节电量/kWh|日均实际激活费/元|','|---|---|---:|---:|']
            resource_notes=[]
            for mode in ('old_gamma2','calibrated'):
                for kind,r in g['strategies'][mode]['resource_delivery_means']['by_kind'].items():
                    lines.append(f"|{LABELS[mode]}|{kind}|{fmt(r['abs_kwh'])}|{fmt(r['fee_yuan'])}|")
                top=sorted(g['strategies'][mode]['resource_delivery_means']['by_node'].items(),key=lambda x:x[1]['abs_kwh'],reverse=True)[:3]
                resource_notes += ['','主要作用节点（'+LABELS[mode]+'）：'+ '；'.join(f"节点{k}，{v['abs_kwh']:.2f}kWh/日" for k,v in top)+'。','']
            lines += resource_notes
        lines += ['','|求解方案|可行／尝试|非零最优界闭合|仅保留初值|无需已验证初值找到可行点|平均相对费用间隙|平均含初值时间/秒|证明模型内不可行|','|---|---:|---:|---:|---:|---:|---:|---:|']
        for mode,s in g['solver'].items():
            lines.append(f"|{LABELS[mode]}|{s['feasible']}/{s['quotes']}|{s['nonzero_global_bound_closed']}/{s['nonzero_feasible']}|{s['initial_retained']}|{s['native_found_without_verified_initial']}|{fmt(None if s['mean_relative_gap_nonzero'] is None else 100*s['mean_relative_gap_nonzero'])}%|{fmt(s['mean_driver_seconds'])}|{s['infeasibility_proved']}|")
        lines += ['','|新模型|声明模型内通过的日数|其中实际完整恢复|其中实际未恢复|其中完整库加拒绝后恢复|模型90%失败次数尾概率|竞争者失败上界和／最大项均值|','|---|---:|---:|---:|---:|---:|---:|']
        for mode in ('nominal','calibrated'):
            s=g['solver'][mode]
            tail='—' if s['model90_binomial_tail'] is None else f"{s['model90_binomial_tail']:.4g}"
            lines.append(f"|{LABELS[mode]}|{s['model_certified_days']}|{s['actual_correct_among_certified']}|{s['actual_nonrecovery_among_certified']}|{s['actual_native_with_rejection_correct_among_certified']}|{tail}|{fmt(s['competitor_conservatism']['mean_sum_to_largest_ratio'])}|")
        lines += ['','|拒绝基准倍数|无服务正确／错误／拒绝|旧增量2|名义恢复约束|均值修正恢复约束|','|---|---:|---:|---:|---:|']
        for factor in ('1.25','2.0','4.0'):
            cells=[]
            for mode in ('no_dfs','old_gamma2','nominal','calibrated'):
                r=g['strategies'][mode]['rejection'][factor];cells.append(f"{r['correct']}/{r['wrong']}/{r['rejected']}")
            lines.append('|'+factor+'|'+'|'.join(cells)+'|')
        lines += ['','上表为代表模型加拒绝的规则。完整参数库按最小SSE判断拒绝、按原strict MSE规则选拓扑，主基准2倍的结果如下：','',
            '|方案|正确／错误／拒绝|','|---|---:|']
        for mode in ('no_dfs','old_gamma2','nominal','calibrated'):
            r=g['strategies'][mode]['native_with_rejection']['2.0']
            lines.append(f"|{LABELS[mode]}|{r['correct']}/{r['wrong']}/{r['rejected']}|")
        lines += ['','|优化相对对应轮换|新增／丢失恢复|平均真权重变化/百分点|权重变化95%区间|同计划费用不一致日|','|---|---:|---:|---:|---:|']
        for mode in ('old_gamma2','nominal','calibrated'):
            p=g['paired'][mode+'_vs_rotation'];wi=p['true_weight_gain_bootstrap95']
            lines.append(f"|{LABELS[mode]}|{p['new_recoveries']}/{p['lost_recoveries']}|{fmt(None if p['mean_true_weight_gain'] is None else 100*p['mean_true_weight_gain'])}|{'—' if wi is None else f'[{100*wi[0]:.2f},{100*wi[1]:.2f}]'}|{g['solver'][mode]['same_planned_fee_failures']}|")
        lines += ['']
    lines += ['## 解释范围','',
        '完整恢复采用原完整参数库的strict MSE规则；代表模型恢复和权重另列。含拒绝规则是额外评价，拒绝仍计入全部计划日分母。缺真候选时没有真权重，目标权重可能很高，须结合完整恢复解释。',
        '', '新约束的90%为给定预测、名义响应及声明正态误差模型内的条件下界；实际AC仍含预测误差、响应偏差、P/Q/V表计误差及冻结参数偏差。模型内通过与实际失败并列，不能用前者替代恢复率。',
        '', '费用界仅针对当前QCQP及其固定风险分配。零动作充分时最小激活费为0；非零费用界闭合另列。求解器证明该充分条件不可行，不能外推为原概率任务或物理系统无可行方案。',
        '', '费用沿用示例激活补偿，未加入完整市场采购与保留容量费用。主拒绝基准为纯表计误差量级的2倍，1.25与4倍为事前锁定的敏感性对照，均未宣称实证校准。',
        '', '权重差区间使用2000次按日配对百分位bootstrap，起始种子1001。恢复率差先对新增恢复率与丢失恢复率分别计算97.5%的Clopper–Pearson精确区间，再以相反端点相减；联合覆盖至少95%，结果较保守，即使没有不一致日也不退化为[0,0]。JSON另外保留恢复率差的经验bootstrap作为对照。',
        '', '每个固定历史条件独立汇报。协议锁定原完整库恢复率为主要结果指标；所有成对统计检验均作探索性分析。p值为精确配对恢复检验，未作多方案校正；不能将三个选定历史条件解释为总体网络样本。',
        '', '模型90%失败次数尾概率为P[Bin(n,0.10)不小于实际未恢复次数]，未恢复包括完整库拒绝及AC失败；只用于含真候选的模型通过日。它诊断所声明误差模型与实际执行是否相符，不能用大p值证明模型已经校准。竞争者失败上界和／最大项反映多比较分摊的程度，真实联合概率还取决于相关性。',
        '', '所有策略的电压越界、设备核验、AC失败、费用有效界、最大时间、全部拒绝敏感性及源哈希见REPORT.json。私人观测、拟合系数和逐日报价保持在忽略的outputs目录。','']
    (destination/'REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    return outcome

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--destination',type=Path,required=True)
    a=p.parse_args();r=report(a.run.resolve(),a.destination.resolve())
    print('groups',len(r['groups']),'verification',r['verification']['passed'])
