"""Paired parameter-domain diagnostics from frozen saved trials only.

No selector, simulator, AC solve or fit is called. The asset pilot histories
are reused development inputs; the two short-edge histories are a domain
stress test. Neither adds histories to the independent 12-history evaluation.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
from .ev_original_analysis import METHODS,DESIGNS,analyze_trial,stats,read_json,write_json,number,family_key

HERE=Path(__file__).resolve().parent
SPECS=(('pilot_domain','pilot_v2','pilot_numeric_v2',4),
       ('short_edge_domain','stress_asset_v2','stress_numeric_v2',2))
NUMERIC_BOUNDS=dict(r_bounds=[1e-6,.1],x_bounds=[1e-6,.1])


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_pair(asset,numeric):
    if asset.get('domain')!='asset' or numeric.get('domain')!='numeric':raise ValueError('asset/numeric domains required')
    keys=('seed','phase','short_edge','seeds','truth','history','measurement_models','config','source_sha256','source_interface')
    missing=[key for key in keys if key not in asset or key not in numeric]
    if missing:raise ValueError('paired immutable inputs missing: '+','.join(missing))
    checks={key:asset.get(key)==numeric.get(key) for key in keys}
    if not all(checks.values()):raise ValueError('paired immutable inputs differ: '+','.join(k for k,v in checks.items() if not v))
    def structural_bank(trial):return sorted(family_key(r['supports']) for r in trial['proposal']['bank'])
    if structural_bank(asset)!=structural_bank(numeric):raise ValueError('paired structural candidate banks differ')
    arows={r['method']:r for r in asset['rows']};nrows={r['method']:r for r in numeric['rows']}
    if set(arows)!=set(METHODS) or set(nrows)!=set(METHODS):raise ValueError('all nine method/failure denominators required')
    for method in ('none','constant'):
        if arows[method].get('power_kw')!=nrows[method].get('power_kw'):
            raise ValueError('domain-independent control differs: '+method)
        if (arows[method].get('observations') is not None and nrows[method].get('observations') is not None
                and arows[method]['observations']!=nrows[method]['observations']):
            raise ValueError('same-action paired future meters differ: '+method)
    return dict(all_immutable_inputs_equal=True,checks=checks,
        immutable_sha256=identity({key:asset.get(key) for key in keys}),
        structural_banks_equal=True,candidate_order_equal=[r['name'] for r in asset['proposal']['bank']]
            ==[r['name'] for r in numeric['proposal']['bank']],
        all_nine_methods_present=True,constant_and_none_controls_equal=True)


def declared_bounds(trial):
    bounds=trial['config']['asset_bounds'] if trial['domain']=='asset' else NUMERIC_BOUNDS
    result={key:list(map(float,bounds[key])) for key in ('r_bounds','x_bounds')}
    actual=(trial.get('selection',{}).get('context') or {}).get('line_bounds')
    if actual is not None and {key:list(map(float,actual[key])) for key in result}!=result:
        raise ValueError('saved context domain differs from declared fit domain')
    return result


def parameter_coverage(trial):
    truth=trial['truth'];wanted=family_key(truth['family']);bank=trial['proposal']['bank'];bounds=declared_bounds(trial)
    matches=[r['name'] for r in bank if family_key(r['supports'])==wanted]
    supports=[r['support'] for r in truth.get('assets',[])]
    if len(supports)!=len(truth['r']):
        supports=[list(range(3))]+[list(s) for s in sorted(wanted,key=lambda s:(-len(s),s))]+[[j] for j in range(3)]
    if len(truth['r'])!=len(truth['x']) or len(supports)!=len(truth['r']):raise ValueError('generating line/support dimensions differ')
    edges=[]
    for support,r,x in zip(supports,truth['r'],truth['x']):
        ri=bounds['r_bounds'][0]<=r<=bounds['r_bounds'][1];xi=bounds['x_bounds'][0]<=x<=bounds['x_bounds'][1]
        edges.append(dict(support=support,true_R=r,true_X=x,R_in_domain=ri,X_in_domain=xi,
            R_violation=max(0.,bounds['r_bounds'][0]-r,r-bounds['r_bounds'][1]),
            X_violation=max(0.,bounds['x_bounds'][0]-x,x-bounds['x_bounds'][1])))
    covered=all(e['R_in_domain'] and e['X_in_domain'] for e in edges)
    return dict(seed=trial['seed'],domain=trial['domain'],short_edge=bool(trial.get('short_edge')),
        candidate_count=len(bank),structural_candidate_covered=len(matches)==1,
        true_candidate_name=matches[0] if len(matches)==1 else None,true_RX_in_domain=covered,
        structure_and_true_RX_covered=len(matches)==1 and covered,
        domain_status='structure_missing' if len(matches)!=1 else 'structure_and_RX_covered' if covered else 'outside_RX_domain',
        coverage_scope='candidate structure and shared line R/X box; coverage is separate from fit or identification success',
        bounds=bounds,edges=edges,outside_edge_count=sum(not(e['R_in_domain'] and e['X_in_domain']) for e in edges))


def line_error_rows(trial,coverage,group):
    true=coverage['true_candidate_name'];by={r['method']:r for r in trial['rows']};records=[]
    historical=next((f for f in trial['historical_fits'] if f['name']==true),None)
    selections=[('historical',historical)]
    for method in METHODS:selections.append((method,next((f for f in by[method].get('fits',[]) if f['name']==true),None)))
    for method,fit in selections:
        fitted={tuple(sorted(s)):(r,x) for s,r,x in zip((fit or {}).get('supports',[]),(fit or {}).get('r',[]),(fit or {}).get('x',[]))}
        for e in coverage['edges']:
            estimate=fitted.get(tuple(sorted(e['support'])),(None,None));r,x=map(number,estimate)
            records.append(dict(group=group,seed=trial['seed'],domain=trial['domain'],method=method,
                support=','.join(map(str,e['support'])),edge_kind='internal' if 1<len(e['support'])<3 else 'root' if len(e['support'])==3 else 'terminal',
                true_R=e['true_R'],true_X=e['true_X'],estimated_R=r,estimated_X=x,
                R_relative_error=None if r is None or e['true_R']==0 else abs(r-e['true_R'])/abs(e['true_R']),
                X_relative_error=None if x is None or e['true_X']==0 else abs(x-e['true_X'])/abs(e['true_X']),
                R_in_domain=e['R_in_domain'],X_in_domain=e['X_in_domain'],
                true_RX_in_domain=coverage['true_RX_in_domain'],fit_success=bool(fit and fit.get('success')),
                error_scope='within declared parameter domain' if coverage['true_RX_in_domain'] else 'distance to an out-of-domain generating parameter'))
    return records


def describe_domain(trial,group,resolution):
    analyzed=analyze_trial(trial,resolution=resolution);coverage=parameter_coverage(trial)
    by={r['method']:r for r in analyzed['method_rows']};constant=by['constant'];rows=[]
    for method in METHODS:
        row=dict(by[method]);row.update(group=group,true_RX_in_domain=coverage['true_RX_in_domain'],
            structural_candidate_covered=coverage['structural_candidate_covered'],domain_status=coverage['domain_status'],
            margin_scope='candidate-family attained SSE comparison with generating R/X outside allowed domain'
                if not coverage['true_RX_in_domain'] else 'candidate-family attained SSE comparison',
            numerical_actual_unresolved=None if row['actual_half_M'] is None else abs(row['actual_half_M'])<=resolution)
        for key in ('actual_M','actual_half_M','predicted_mean','noiseless_joint_M'):
            a,b=number(row[key]),number(constant[key]);row[key+'_minus_constant']=None if a is None or b is None else a-b
        rows.append(row)
    return coverage,rows,line_error_rows(trial,coverage,group)


def compare_pair(asset,numeric,*,group='pilot_domain',resolution=.01):
    audit=validate_pair(asset,numeric);ac,ar,ae=describe_domain(asset,group,resolution);nc,nr,ne=describe_domain(numeric,group,resolution)
    lookup={r['method']:r for r in nr};comparisons=[];nested=[]
    af={r['method']:r for r in asset['rows']};nf={r['method']:r for r in numeric['rows']}
    fields=('actual_M','actual_half_M','predicted_mean','predicted_worst','noiseless_joint_M',
        'actual_M_minus_constant','actual_half_M_minus_constant','predicted_mean_minus_constant','noiseless_joint_M_minus_constant')
    for a in ar:
        name=a['method'];n=lookup[name];same=af[name].get('power_kw')==nf[name].get('power_kw') and af[name].get('power_kw') is not None
        row=dict(group=group,seed=asset['seed'],family=a['family'],method=name,controls_equal_across_domains=same,
            asset_status=a['status'],numeric_status=n['status'],asset_fallback=a['fallback_to_constant'],numeric_fallback=n['fallback_to_constant'],
            asset_true_RX_in_domain=ac['true_RX_in_domain'],numeric_true_RX_in_domain=nc['true_RX_in_domain'],
            structural_coverage_equal=ac['structural_candidate_covered']==nc['structural_candidate_covered'])
        for key in fields:
            av,nv=number(a[key]),number(n[key]);row['asset_'+key]=av;row['numeric_'+key]=nv
            row['asset_minus_numeric_'+key]=None if av is None or nv is None else av-nv
        comparisons.append(row)
        if same and af[name].get('observations')==nf[name].get('observations'):
            aa={f['name']:f for f in af[name].get('fits',[]) if f.get('success')};nn={f['name']:f for f in nf[name].get('fits',[]) if f.get('success')}
            for candidate in sorted(r['name'] for r in asset['proposal']['bank']):
                narrow,wide=number(aa.get(candidate,{}).get('loss')),number(nn.get(candidate,{}).get('loss'))
                both=narrow is not None and wide is not None
                nested.append(dict(group=group,seed=asset['seed'],method=name,candidate=candidate,
                    asset_attained_SSE=narrow,numeric_attained_SSE=wide,numeric_minus_asset=wide-narrow if both else None,
                    both_costs_available=both,local_attainment_order_violation=wide>narrow+1e-6 if both else None,
                    interpretation='same action/data: exact nested-domain minima obey J_numeric<=J_asset; local attained costs may violate it'))
    return dict(seed=asset['seed'],group=group,audit=audit,coverage=[ac,nc],method_rows=ar+nr,
        paired_rows=comparisons,edge_error_rows=ae+ne,nested_cost_audits=nested)


def summarize(base,*,resolution=.01):
    base=Path(base);results=[];files=[];freezes={};common=None
    for group,asset_name,numeric_name,count in SPECS:
        ad,nd=base/asset_name,base/numeric_name;af,nf=read_json(ad/'freeze.json'),read_json(nd/'freeze.json')
        if af['seeds']!=nf['seeds'] or len(af['seeds'])!=count or af['config']!=nf['config']:
            raise ValueError('paired freeze seed/config mismatch: '+group)
        if af['source_sha256']!=nf['source_sha256']:raise ValueError('paired scientific source differs: '+group)
        if common is not None and af['source_sha256']!=common:raise ValueError('scientific source differs between groups')
        common=af['source_sha256'];freezes[group]=dict(asset=af,numeric=nf)
        for directory,freeze in ((ad,af),(nd,nf)):
            path=directory/'freeze.json';files.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            if len(list(directory.glob('trial_*.json.gz')))!=count:raise ValueError('wait for all predeclared domain trials: '+str(directory))
        for seed in af['seeds']:
            paths=[directory/f'trial_{seed}.json.gz' for directory in (ad,nd)]
            if not all(p.is_file() for p in paths):raise ValueError('missing paired history '+str(seed))
            asset,numeric=map(read_json,paths)
            for trial,freeze in ((asset,af),(numeric,nf)):
                if trial['source_sha256']!=freeze['source_sha256'] or trial['config']!=freeze['config']:raise ValueError('trial/freeze mismatch')
            results.append(compare_pair(asset,numeric,group=group,resolution=resolution))
            files.extend(dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in paths)
    rows=[r for result in results for r in result['method_rows']];pairs=[r for result in results for r in result['paired_rows']]
    grouped={}
    for group,_,_,_ in SPECS:
        grouped[group]={}
        for method in METHODS:
            selected=[r for r in pairs if r['group']==group and r['method']==method]
            grouped[group][method]=dict(paired_histories=len(selected),metrics={key:stats(r[key] for r in selected)
                for key in ('asset_actual_M','numeric_actual_M','asset_actual_M_minus_constant',
                    'numeric_actual_M_minus_constant','asset_minus_numeric_actual_M_minus_constant',
                    'asset_predicted_mean_minus_constant','numeric_predicted_mean_minus_constant')})
    return dict(paired_histories=len(results),domain_trial_conditions=2*len(results),method_domain_conditions=len(rows),
        new_domain_runs=8,reused_asset_pilot_runs=4,additional_independent_evaluation_histories=0,
        scope='paired development-domain ablation and out-of-family short-edge stress; no new rejection threshold',
        resolution=resolution,numerical_unresolved_rule='prediction pair added<=resolution; actual |M/2|<=resolution; no statistical rejection',
        common_scientific_source_sha256=common,input_files=files,freezes=freezes,grouped=grouped,paired_histories_detail=results,
        method_rows=rows,paired_rows=pairs,coverage=[r for x in results for r in x['coverage']],
        edge_error_rows=[r for x in results for r in x['edge_error_rows']],nested_cost_audits=[r for x in results for r in x['nested_cost_audits']],
        analysis_sources={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
            for name in (Path(__file__).name,'ev_original_analysis.py')},new_AC_fits=0,new_selector_runs=0)


def write_csv(path,rows):
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def fmt(value):return '—' if value is None else f'{value:.4g}'


def report(result):
    text=['# 参数域与EV增量的配对对照','',
        '同一历史、真线路、未来噪声、计量及设备条件分别在asset/numeric域运行。4条pilot为开发复用，2条短边为参数域压力；原asset独立评价仍为12条。', '',
        'M为最近错误候选联合SSE减真结构家族联合SSE；G=M方法−M恒充。G先在各参数域内计算，最后比较两域G，分开参数先验共同作用与各域下的EV增量。', '',
        '|组别|方法|历史数|asset平均M|numeric平均M|asset平均G|numeric平均G|两域G差(asset−numeric)|',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for group,methods in result['grouped'].items():
        for name,r in methods.items():
            m=r['metrics'];keys=('asset_actual_M','numeric_actual_M','asset_actual_M_minus_constant','numeric_actual_M_minus_constant','asset_minus_numeric_actual_M_minus_constant')
            text.append('|'+group+'|'+name+'|'+str(r['paired_histories'])+'|'+'|'.join(fmt(m[k]['mean']) for k in keys)+'|')
    text+=['','## 真实参数覆盖','', '|组别/历史|域|结构覆盖|真实R/X入域|域外边数|解释|','|---|---|---|---|---:|---|']
    for result_ in result['paired_histories_detail']:
        for c in result_['coverage']:
            text.append(f"|{result_['group']}/{c['seed']}|{c['domain']}|{c['structural_candidate_covered']}|{c['true_RX_in_domain']}|{c['outside_edge_count']}|{c['domain_status']}|")
    text+=['', '2m内部支路落在asset声明长度族之外。结构仍在候选表中，生成该结构的真实阻抗已在允许域外；对应M用于观察受限家族的损失排序，参数误差标为域外距离。', '',
        '每域每方法的失败、回退、未闭合优化和数值未决计数保存在per_domain_method.csv；逐边R/X误差保存在line_errors.csv。缺失成本保持为空，九方法分母完整。', '',
        '对相同曲线和计量，精确最小成本满足J_numeric≤J_asset。nested_cost_audit.csv保留局部取得成本的反向现象，用于定位初值或数值问题；M作为两个成本之差本身没有单调性。所有结果为配对描述，未新增统计拒绝规则。','']
    return '\n'.join(text)


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--resolution',type=float,default=.01);args=p.parse_args()
    if number(args.resolution) is None or args.resolution<0:raise ValueError('finite nonnegative numerical resolution required')
    result=summarize(args.base,resolution=args.resolution);args.output.mkdir(parents=True,exist_ok=True)
    for name,key in (('per_domain_method.csv','method_rows'),('paired_domains.csv','paired_rows'),
            ('line_errors.csv','edge_error_rows'),('nested_cost_audit.csv','nested_cost_audits')):write_csv(args.output/name,result[key])
    write_json(args.output/'summary.json',result);(args.output/'summary.md').write_text(report(result),encoding='utf-8')
    print(json.dumps(dict(paired_histories=result['paired_histories'],method_domain_conditions=result['method_domain_conditions'],
        all_inputs_paired=True,new_AC_fits=0,additional_independent_evaluation_histories=0)))


if __name__=='__main__':main()
