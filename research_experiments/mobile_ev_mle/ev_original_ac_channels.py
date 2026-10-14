"""Offline AC evidence by channel from saved true-noiseless joint fits.

No AC solver, fitting routine, design policy or truth generator is called.
Every candidate, failure and independently generated history is retained.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

CHANNELS=('p','q','v','master_p','master_q')
METHODS=('none','constant','quality_fixed','quality_mobile','random_mobile',
         'charge_fixed','v2g_fixed','charge_mobile','v2g_mobile')
CONTRASTS=(('charge_time','charge_fixed','constant'),('fixed_V2G','v2g_fixed','charge_fixed'),
    ('charge_space','charge_mobile','charge_fixed'),('V2G_space','v2g_mobile','v2g_fixed'),
    ('mobile_V2G','v2g_mobile','charge_mobile'),('full_vs_constant','v2g_mobile','constant'),
    ('full_vs_quality','v2g_mobile','quality_mobile'),('full_vs_random','v2g_mobile','random_mobile'))


def read_json(path):
    path=Path(path);payload=path.read_bytes()
    return json.loads(gzip.decompress(payload) if path.suffix=='.gz' else payload)


def require(condition,message):
    if not condition:raise ValueError(message)


def close(a,b,label):
    a,b=np.asarray(a,float),np.asarray(b,float)
    require(a.shape==b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            and np.allclose(a,b,atol=2e-6,rtol=5e-8),label)


def number(value):
    return float(value) if isinstance(value,(int,float,np.integer,np.floating)) and not isinstance(value,bool) and np.isfinite(value) else None


def stats(values):
    values=[number(v) for v in values];values=[v for v in values if v is not None]
    return dict(n=len(values),mean=float(np.mean(values)) if values else None,
        median=float(np.median(values)) if values else None,min=min(values) if values else None,
        max=max(values) if values else None,positive=sum(v>0 for v in values),zero=sum(v==0 for v in values),negative=sum(v<0 for v in values))


def midranks(values):
    """Average ranks for exact ties; equal cached actions retain their ties."""
    values=np.asarray(values,float);order=np.argsort(values,kind='stable');ranks=np.empty(len(values),float)
    first=0
    while first<len(values):
        stop=first+1
        while stop<len(values) and values[order[stop]]==values[order[first]]:stop+=1
        ranks[order[first:stop]]=.5*(first+1+stop);first=stop
    return ranks


def within_history_spearman(rows,predictor,outcome):
    valid=[r for r in rows if number(r.get(predictor)) is not None and number(r.get(outcome)) is not None]
    answer=dict(predictor=predictor,outcome=outcome,planned_methods=len(rows),valid_method_pairs=len(valid),
        valid_methods=[r['method'] for r in valid],missing_methods=[r['method'] for r in rows if r not in valid],
        spearman=None,available=False,reason=None)
    if len(valid)<2:answer['reason']='fewer_than_two_available_method_pairs';return answer
    x=midranks([r[predictor] for r in valid]);y=midranks([r[outcome] for r in valid])
    x-=x.mean();y-=y.mean();sx=float(x@x);sy=float(y@y)
    if sx==0 or sy==0:
        answer['reason']='constant_predictor_and_outcome' if sx==0 and sy==0 else 'constant_predictor' if sx==0 else 'constant_outcome'
        return answer
    answer.update(spearman=float(np.clip((x@y)/np.sqrt(sx*sy),-1,1)),available=True)
    return answer


def candidate_channels(fit,hfit,history,future,hm,fm,base_kw):
    """J=J_H+sum(channel J_N); penalty=J_H-J_H0 when H fit succeeded."""
    if fit is None:return dict(available=False,fit_success=False,historical_baseline_success=False,reason='missing_future_fit')
    H=len(history['p']);SSE_H={};SSE_N={};rms={}
    for key in CHANNELS:
        prediction=np.asarray(fit['predicted'][key],float)
        rh=prediction[:H]-np.asarray(history[key],float)
        rn=prediction[H:]-np.asarray(future[key],float)
        sh=np.asarray(hm[key+'_sigma'],float);sn=np.asarray(fm[key+'_sigma'],float)
        SSE_H[key]=float(np.sum((rh/sh)**2));SSE_N[key]=float(np.sum((rn/sn)**2))
        rms[key]=float(np.sqrt(np.mean(rn*rn)))
    JH=sum(SSE_H.values());JN=sum(SSE_N.values());close(fit['loss'],JH+JN,'channel SSE partition of attained joint fit')
    baseline_ok=hfit is not None and hfit['success'] and number(hfit['loss']) is not None
    JH0=number(hfit['loss']) if baseline_ok else None
    return dict(available=True,fit_success=bool(fit['success']),historical_baseline_success=baseline_ok,
        joint_SSE=JH+JN,historical_SSE=JH,future_SSE=JN,historical_baseline_SSE=JH0,
        historical_penalty=None if JH0 is None else JH-JH0,
        added_half_SSE=None if JH0 is None else .5*(JH+JN-JH0),
        future_V_SSE=SSE_N['v'],future_P_SSE=SSE_N['p'],future_Q_SSE=SSE_N['q'],
        future_master_P_SSE=SSE_N['master_p'],future_master_Q_SSE=SSE_N['master_q'],
        future_V_RMS_pu=rms['v'],future_P_RMS_kw=base_kw*rms['p'],future_Q_RMS_kvar=base_kw*rms['q'],
        future_master_P_RMS_kw=base_kw*rms['master_p'],future_master_Q_RMS_kvar=base_kw*rms['master_q'],
        future_V_white_RMS=float(np.sqrt(SSE_N['v']/np.size(future['v']))),historical_channel_SSE=SSE_H)


def pair_decomposition(own,comp):
    """Exact signed margin, retaining the two fitted historical costs."""
    if not all(r.get('available') and r.get('fit_success') for r in (own,comp)):
        return dict(available=False,reason='required_future_fit_missing_or_failed')
    out=dict(available=True,M=comp['joint_SSE']-own['joint_SSE'],
        historical_difference=comp['historical_SSE']-own['historical_SSE'])
    for key in ('V','P','Q','master_P','master_Q'):
        out['future_'+key+'_difference']=comp['future_'+key+'_SSE']-own['future_'+key+'_SSE']
    close(out['M'],out['historical_difference']+sum(out['future_'+k+'_difference'] for k in ('V','P','Q','master_P','master_Q')),
          'signed pair margin channel identity')
    out['historical_baselines_available']=own['historical_baseline_success'] and comp['historical_baseline_success']
    if out['historical_baselines_available']:
        out['historical_baseline_margin']=comp['historical_baseline_SSE']-own['historical_baseline_SSE']
        out['historical_penalty_difference']=comp['historical_penalty']-own['historical_penalty']
        close(out['M']-out['historical_baseline_margin'],out['historical_penalty_difference']
            +sum(out['future_'+k+'_difference'] for k in ('V','P','Q','master_P','master_Q')),
            'historical penalty plus future channel identity')
    else:out.update(historical_baseline_margin=None,historical_penalty_difference=None)
    return out


def selected_resource(trial,name,row):
    if name=='none':return dict(moves=0,drive_kwh=0.,worker_minutes=0.,remote_throughput_kwh=0.)
    selection=trial['selection'];rec=selection.get('methods',{}).get(name,{})
    if name=='constant' or name in METHODS[-4:] and not (rec.get('guard') or {}).get('accepted'):
        operation=selection.get('constant_operation') or {}
    else:operation=rec.get('operation') or {}
    if operation and row.get('power_kw') is not None:close(operation['u_kw'],row['power_kw'],'selected resource/action consistency')
    p=np.asarray(row['power_kw'],float) if row.get('power_kw') is not None else None
    return dict(moves=operation.get('moves'),drive_kwh=number(operation.get('travel_energy_kwh')),
        worker_minutes=number(operation.get('driver_cost')),
        remote_throughput_kwh=None if p is None else float(trial['config']['dt_hours']*abs(p[:,1:]).sum()))


def crosscheck_csv(candidate_rows,path):
    if path is None:return dict(available=False)
    with Path(path).open(encoding='utf-8-sig',newline='') as file:
        saved={(int(r['seed']),r['method'],r['candidate']):r for r in csv.DictReader(file)}
    expected={(r['seed'],r['method'],r['candidate']) for r in candidate_rows}
    require(set(saved)<=expected,'unplanned row in noiseless candidate CSV')
    checked=0;omitted=[]
    for row in candidate_rows:
        key=row['seed'],row['method'],row['candidate']
        if key not in saved:
            require(not row['available'],'available candidate absent from analysis CSV')
            omitted.append(dict(seed=row['seed'],method=row['method'],candidate=row['candidate'],reason=row.get('reason')))
            continue
        old=saved[key]
        for key in ('joint_SSE','historical_SSE','added_half_SSE','future_V_SSE','future_P_SSE',
                    'future_Q_SSE','future_master_P_SSE','future_master_Q_SSE'):
            value=row.get(key);text=old[key]
            if value is None:require(text in ('','None'),'CSV fills unavailable channel metric')
            else:close(value,float(text),'independent channel arithmetic vs CSV:'+key)
        checked+=1
    return dict(available=True,rows_checked=checked,expected_candidate_rows=len(candidate_rows),available_csv_rows=len(saved),
        omitted_unavailable_rows=omitted,path=str(Path(path).resolve()),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())


def analyze_trial(trial):
    config=trial['config'];family='star' if not trial['truth']['family'] else 'cherry_'+''.join(map(str,trial['truth']['family'][0]))
    candidates=trial['proposal']['bank'];names=[r['name'] for r in candidates]
    wanted=tuple(map(tuple,trial['truth']['family']))
    true=[r['name'] for r in candidates if tuple(map(tuple,r['supports']))==wanted]
    require(len(true)==1,'channel analysis requires unique true reduced candidate');true=true[0]
    hfits={r['name']:r for r in trial['historical_fits']};meters=trial['measurement_models']
    saved={r['method']:r for r in trial['rows']};candidate_rows=[];pair_rows=[];method_rows=[]
    for name in METHODS:
        row=saved[name];mean=row.get('noiseless_AC_refit') or {};fits={r['name']:r for r in mean.get('fits',[])}
        results={}
        for candidate in names:
            detail=candidate_channels(fits.get(candidate),hfits.get(candidate),trial['history'],mean.get('future_mean_observations'),
                meters['hm'],meters['fm'],config['base_kw']) if candidate in fits else dict(available=False,fit_success=False,historical_baseline_success=False,reason='missing_future_fit')
            result=dict(seed=trial['seed'],phase=trial['phase'],family=family,method=name,candidate=candidate,is_true=candidate==true,**detail)
            candidate_rows.append(result);results[candidate]=result
        pairs={}
        for candidate in names:
            if candidate==true:continue
            value=pair_decomposition(results[true],results[candidate]);pairs[candidate]=value
            pair_rows.append(dict(seed=trial['seed'],phase=trial['phase'],family=family,method=name,
                                  true_candidate=true,competitor=candidate,**value))
        complete=all(v.get('available') for v in pairs.values())
        nearest=min(pairs,key=lambda n:(pairs[n]['M'],n)) if complete else None
        closest=pairs.get(nearest,{})
        r=dict(seed=trial['seed'],phase=trial['phase'],family=family,method=name,
            noiseless_complete=complete,noiseless_nearest_wrong=nearest,
            noiseless_M=closest.get('M'),historical_difference=closest.get('historical_difference'),
            historical_penalty_difference=closest.get('historical_penalty_difference'),
            historical_baseline_margin=closest.get('historical_baseline_margin'),
            actual_M=number(row.get('margin')),grid_net_kwh=number(row.get('grid_net_kwh')),
            grid_throughput_kwh=number(row.get('grid_throughput_kwh')),**selected_resource(trial,name,row))
        predicted=trial['selection'].get('predicted_AC',{}).get(name)
        prediction_source=name
        if predicted is None and row.get('power_kw') is not None and saved['constant'].get('power_kw') is not None:
            if np.array_equal(row['power_kw'],saved['constant']['power_kw']):
                predicted=trial['selection'].get('predicted_AC',{}).get('constant');prediction_source='constant_identical_executed_action'
        r.update(mean_predicted_AC=number((predicted or {}).get('mean_added_score')),
            worst_predicted_AC=number((predicted or {}).get('worst_added_score')),
            prediction_source=prediction_source,predicted_complete=bool((predicted or {}).get('success')))
        for channel in ('V','P','Q','master_P','master_Q'):
            r['nearest_future_'+channel+'_difference']=closest.get('future_'+channel+'_difference')
            values=[p.get('future_'+channel+'_difference') for p in pairs.values()]
            r['mean_wrong_future_'+channel+'_difference']=float(np.mean(values)) if complete else None
        r['nearest_wrong_future_V_RMS_pu']=results[nearest].get('future_V_RMS_pu') if nearest else None
        r['true_future_V_RMS_pu']=results[true].get('future_V_RMS_pu')
        method_rows.append(r)
    byname={r['method']:r for r in method_rows};paired=[]
    metrics=('actual_M','noiseless_M','historical_difference','historical_penalty_difference',
        'nearest_future_V_difference','nearest_future_P_difference','nearest_future_Q_difference',
        'nearest_future_master_P_difference','nearest_future_master_Q_difference',
        'mean_wrong_future_V_difference','grid_net_kwh','grid_throughput_kwh','drive_kwh','worker_minutes','remote_throughput_kwh')
    for label,a,b in CONTRASTS:
        left,right=byname[a],byname[b];r=dict(seed=trial['seed'],phase=trial['phase'],family=family,
            contrast=label,a=a,b=b,nearest_competitor_changed=left['noiseless_nearest_wrong']!=right['noiseless_nearest_wrong'])
        for key in metrics:
            l,h=number(left.get(key)),number(right.get(key));r['delta_'+key]=None if l is None or h is None else l-h
        if r['delta_noiseless_M'] is not None:
            close(r['delta_noiseless_M'],r['delta_historical_difference']+sum(r['delta_nearest_future_'+k+'_difference']
                for k in ('V','P','Q','master_P','master_Q')),'paired margin component identity')
        paired.append(r)
    correlations=[]
    for predictor in ('mean_predicted_AC','worst_predicted_AC'):
        for outcome in ('actual_M','noiseless_M'):
            correlations.append(dict(seed=trial['seed'],phase=trial['phase'],family=family,
                **within_history_spearman(method_rows,predictor,outcome)))
    return dict(candidate_rows=candidate_rows,pair_rows=pair_rows,method_rows=method_rows,paired_rows=paired,
                within_history_rank_correlations=correlations)


def analyze_directory(directory,*,noiseless_csv=None):
    directory=Path(directory);freeze=read_json(directory/'freeze.json');trials=[];seen=set();errors=[]
    for path in sorted(directory.glob('trial_*.json.gz')):
        trial=read_json(path);require(trial['seed'] not in seen and trial['seed'] in freeze['seeds'],'independent history denominator')
        require(trial['config']==freeze['config'] and trial['source_sha256']==freeze['source_sha256'],'frozen scientific identity')
        seen.add(trial['seed'])
        try:trials.append(analyze_trial(trial))
        except (KeyError,ValueError,TypeError,IndexError) as error:errors.append(dict(seed=trial['seed'],error=type(error).__name__+':'+str(error)))
    keys=('candidate_rows','pair_rows','method_rows','paired_rows','within_history_rank_correlations');rows={k:[r for t in trials for r in t[k]] for k in keys}
    method_metrics=('actual_M','noiseless_M','historical_penalty_difference','nearest_future_V_difference',
        'mean_wrong_future_V_difference','nearest_wrong_future_V_RMS_pu','true_future_V_RMS_pu',
        'grid_net_kwh','grid_throughput_kwh','drive_kwh','worker_minutes','remote_throughput_kwh')
    family_summary={}
    for family in sorted({r['family'] for r in rows['method_rows']}):
        family_summary[family]={m:dict(independent_histories=sum(r['family']==family and r['method']==m for r in rows['method_rows']),
            metrics={key:stats(r.get(key) for r in rows['method_rows'] if r['family']==family and r['method']==m) for key in method_metrics}) for m in METHODS}
    contrasts={}
    for family in ['all']+sorted(family_summary):
        contrasts[family]={label:dict(independent_histories=sum(r['contrast']==label and (family=='all' or r['family']==family) for r in rows['paired_rows']),
            metrics={key:stats(r[key] for r in rows['paired_rows'] if r['contrast']==label and (family=='all' or r['family']==family))
                for key in (rows['paired_rows'][0] if rows['paired_rows'] else {}) if key.startswith('delta_')}) for label,_,_ in CONTRASTS}
    rank_summary={}
    for predictor in ('mean_predicted_AC','worst_predicted_AC'):
        for outcome in ('actual_M','noiseless_M'):
            group=[r for r in rows['within_history_rank_correlations'] if r['predictor']==predictor and r['outcome']==outcome]
            rank_summary[predictor+'__to__'+outcome]=dict(planned_histories=len(freeze['seeds']),available_histories=len(group),
                correlation=stats(r['spearman'] for r in group),
                unavailable=[dict(seed=r['seed'],reason=r['reason'],valid_method_pairs=r['valid_method_pairs'],missing_methods=r['missing_methods'])
                    for r in group if not r['available']],
                inference='one descriptive correlation per independent history; no pooled method-row p-value')
    result=dict(success=not errors,phase=freeze['phase'],planned_independent_histories=len(freeze['seeds']),
        available_independent_histories=len(seen),analyzed_independent_histories=len(trials),
        missing_histories=[s for s in freeze['seeds'] if s not in seen],analysis_errors=errors,
        source_sha256=freeze['source_sha256'],CSV_crosscheck=crosscheck_csv(rows['candidate_rows'],noiseless_csv),
        family_summary=family_summary,contrast_summary=contrasts,within_history_rank_summary=rank_summary,**rows,
        negative_historical_penalties=sum(r.get('historical_penalty',0.) is not None and r.get('historical_penalty',0.) < -1e-7 for r in rows['candidate_rows']),
        mathematical_meaning=dict(joint='J_b=J_H,b+J_N,V,b+J_N,PQ,b+J_N,master,b',
            penalty='J_H,b minus its own attained historical fit loss; negative values retained',
            signed_margin='M_b=J_b-J_true; components add exactly at the two saved joint fits',
            changing_competitor='nearest wrong candidate may differ by method; every fixed competitor pair is retained',
            scope='local complete-AC joint refits on true noiseless future means; evaluation only, no global MLE or rejection probability'),
        new_AC_fits=0,new_selector_runs=0,formal_risk_calibration_histories=0,
        analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return result


def write_outputs(output,result):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'channel_decomposition.json').write_text(json.dumps(result,sort_keys=True,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    for name in ('candidate_rows','pair_rows','method_rows','paired_rows','within_history_rank_correlations'):
        rows=result[name];columns=list(dict.fromkeys(k for r in rows for k in r if k!='historical_channel_SSE'))
        with (output/(name+'.csv')).open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows)
    def fmt(v):return '—' if v is None else f'{v:.5g}'
    lines=['# 完整AC误差通道与指标排序', '',
        f"已分析 {result['analyzed_independent_histories']}/{result['planned_independent_histories']} 条独立历史。每条历史保留9方法及全部4候选；通道差由保存的历史+真实无噪声未来联合拟合重算。",'',
        'M为最邻近错误候选SSE减真候选SSE；M₀使用真实无噪声未来均值。每个M₀等于两候选历史SSE差与未来V/P/Q/根表SSE差之和。', '',
        '## 每条历史内的排序一致程度', '',
        'Spearman先给方法排位，再比较预测分数与实际区分差的排位。相同分数采用平均排位。下表汇总每条独立历史自己的相关系数。', '',
        '|历史预测分数|评价量|有效历史/已分析|中位数|最小值|最大值|',
        '|---|---|---:|---:|---:|---:|']
    for key,r in result['within_history_rank_summary'].items():
        predictor,outcome=key.split('__to__');v=r['correlation']
        lines.append(f"|{predictor}|{outcome}|{v['n']}/{r['available_histories']}|{fmt(v['median'])}|{fmt(v['min'])}|{fmt(v['max'])}|")
    lines+=['', '## 家族内的配对结果', '',
        '差值为前者减后者。每个家族包含独立历史，方法均在同一历史内配对；无可用结果的配对保留缺失分母。', '',
        '|真结构|比较|有效配对/历史|平均ΔM|ΔM正/零/负|平均ΔM₀|未来V差项对ΔM₀的贡献|历史差项贡献|额外净电量(kWh)|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for family,summary in result['contrast_summary'].items():
        if family=='all':continue
        for label in ('fixed_V2G','charge_space','V2G_space'):
            r=summary[label];v=r['metrics'];m=v['delta_actual_M']
            lines.append(f"|{family}|{label}|{m['n']}/{r['independent_histories']}|{fmt(m['mean'])}|{m['positive']}/{m['zero']}/{m['negative']}|"
                f"{fmt(v['delta_noiseless_M']['mean'])}|{fmt(v['delta_nearest_future_V_difference']['mean'])}|"
                f"{fmt(v['delta_historical_difference']['mean'])}|{fmt(v['delta_grid_net_kwh']['mean'])}|")
    lines+=['', 'fixed_V2G为固定双向相对固定充电；charge_space为移动充电相对固定充电；V2G_space为移动双向相对固定双向。最邻近错误候选可随方法变化，逐对固定候选结果在pair_rows.csv中保留。', '',
        'candidate_rows.csv同时给出电压物理RMS(p.u.)、末端P/Q和根表功率RMS(kW/kvar)、标准化SSE及历史罚损。method_rows.csv记录实际移动、异地吞吐、电量和人员代价。', '',
        f"历史罚损为负的保存状态共 {result['negative_historical_penalties']} 个，保留原值。全常数或缺失方法引起的不可用相关系数保存在within_history_rank_correlations.csv。", '',
        '本分析描述声明模拟分布下、当前局部完整AC联合拟合取得的结果；正式统计风险校准历史数为0。分析没有调用新拟合或选择器。', '']
    (output/'channel_decomposition.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--directory',required=True,type=Path)
    p.add_argument('--noiseless-csv',type=Path);p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    result=analyze_directory(args.directory,noiseless_csv=args.noiseless_csv);write_outputs(args.output,result)
    print(json.dumps({k:result[k] for k in ('success','available_independent_histories','analyzed_independent_histories','analysis_errors','CSV_crosscheck')}))


if __name__=='__main__':main()
