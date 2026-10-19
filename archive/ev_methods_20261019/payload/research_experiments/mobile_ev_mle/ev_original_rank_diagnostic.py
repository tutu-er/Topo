"""Fine policy-ranking diagnostics from saved per-history analysis only.

Average ranks retain ties. Each history is analyzed separately for the four
optimized policies. Each predeclared contrast is then correlated across its
12 paired histories, without pooling contrasts or computing p values.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from .ev_original_analysis import (DESIGNS,METHODS,CONTRASTS,number,stats,read_json,write_json)

HERE=Path(__file__).resolve().parent


def average_ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);ranks=[0.]*len(values);start=0
    while start<len(order):
        end=start+1
        while end<len(order) and values[order[end]]==values[order[start]]:end+=1
        average=(start+1+end)/2
        for i in order[start:end]:ranks[i]=average
        start=end
    return ranks


def spearman(x,y):
    if len(x)!=len(y):raise ValueError('paired rank vectors must have equal length')
    valid=[(number(a),number(b)) for a,b in zip(x,y) if number(a) is not None and number(b) is not None]
    result=dict(planned_pairs=len(x),valid_pairs=len(valid),missing_pairs=len(x)-len(valid),rho=None,
        x_unique=None,y_unique=None,x_tied_observations=None,y_tied_observations=None,status='insufficient_pairs')
    if len(valid)<2:return result
    vx,vy=map(list,zip(*valid));rx,ry=average_ranks(vx),average_ranks(vy);mx,my=sum(rx)/len(rx),sum(ry)/len(ry)
    xx=sum((a-mx)**2 for a in rx);yy=sum((b-my)**2 for b in ry)
    def tied(values):return sum(values.count(v) for v in set(values) if values.count(v)>1)
    result.update(x_unique=len(set(vx)),y_unique=len(set(vy)),x_tied_observations=tied(vx),
        y_tied_observations=tied(vy),x_range=max(vx)-min(vx),y_range=max(vy)-min(vy))
    if xx==0 or yy==0:result['status']='constant_rank_vector';return result
    result.update(rho=sum((a-mx)*(b-my) for a,b in zip(rx,ry))/math.sqrt(xx*yy),status='complete')
    return result


def analyze(summary):
    if summary.get('phase')!='evaluation' or summary.get('available_independent_histories')!=12 or summary.get('missing_histories'):
        raise ValueError('complete predeclared 12-history evaluation analysis required')
    rows=summary['method_rows'];seeds=sorted({r['seed'] for r in rows})
    if len(seeds)!=12:raise ValueError('twelve distinct historical units required')
    per_history=[];policy_points=[]
    for seed in seeds:
        by={r['method']:r for r in rows if r['seed']==seed}
        if set(by)!=set(METHODS):raise ValueError('all nine methods/failures must remain in saved analysis')
        constant=by['constant']
        for method in DESIGNS:
            r=by[method]
            if r.get('fallback_to_constant') and number(r.get('predicted_mean')) is not None:
                if number(constant.get('predicted_mean')) is None or abs(r['predicted_mean']-constant['predicted_mean'])>1e-8:
                    raise ValueError('fallback prediction is not the actually executed constant action')
            policy_points.append(dict(seed=seed,family=r['family'],method=method,status=r['status'],
                fallback_to_constant=r.get('fallback_to_constant'),predicted_mean=r.get('predicted_mean'),
                actual_M=r.get('actual_M'),noiseless_joint_M=r.get('noiseless_joint_M')))
        for response in ('actual_M','noiseless_joint_M'):
            four=spearman([by[m]['predicted_mean'] for m in DESIGNS],[by[m][response] for m in DESIGNS])
            nine=spearman([by[m]['predicted_mean'] for m in METHODS],[by[m][response] for m in METHODS])
            per_history.append(dict(seed=seed,family=by[DESIGNS[0]]['family'],response=response,
                optimized_policies=list(DESIGNS),optimized=four,all_nine_reference=nine,
                no_pooling='one correlation per history; policy count is not a historical sample size'))
    contrast_rows=[];contrast_points=[]
    for label,a,b in CONTRASTS:
        points=[r for r in summary['paired_rows'] if r['contrast']==label]
        if len(points)!=12 or {r['seed'] for r in points}!=set(seeds):raise ValueError('contrast lost a historical/failure denominator')
        for response in ('delta_actual_M','delta_noiseless_joint_M'):
            correlation=spearman([r.get('delta_predicted_mean') for r in points],[r.get(response) for r in points])
            missing=[]
            for r in points:
                x,y=number(r.get('delta_predicted_mean')),number(r.get(response));available=x is not None and y is not None
                if not available:missing.append(r['seed'])
                contrast_points.append(dict(contrast=label,a=a,b=b,response=response,seed=r['seed'],family=r['family'],
                    delta_predicted_mean=x,delta_response=y,available=available))
            contrast_rows.append(dict(contrast=label,a=a,b=b,response=response,missing_histories=missing,
                **correlation,scope='descriptive correlation across 12 fixed-stratum paired histories for this contrast only'))
    return dict(independent_histories=12,optimized_methods=list(DESIGNS),per_history=per_history,
        policy_points=policy_points,contrast_correlations=contrast_rows,contrast_points=contrast_points,
        within_history_summary={response:dict(optimized=stats(r['optimized']['rho'] for r in per_history if r['response']==response),
            all_nine_reference=stats(r['all_nine_reference']['rho'] for r in per_history if r['response']==response))
            for response in ('actual_M','noiseless_joint_M')},
        rank_method='Pearson correlation of average ranks; exact ties retained; constant ranks are undefined, not zero',
        missing_policy='pairwise missing values retained as unavailable, with planned and valid counts',
        no_p_values=True,no_pooled_method_history_sample=True,formal_risk_calibration=False,
        scope='descriptive fine-selection check; four fixed family strata with three independent histories each')


def flatten_history(records):
    return [dict(seed=r['seed'],family=r['family'],response=r['response'],
        **{'optimized_'+k:v for k,v in r['optimized'].items()},
        **{'all_nine_'+k:v for k,v in r['all_nine_reference'].items()}) for r in records]


def write_csv(path,rows):
    columns=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader();writer.writerows(rows)


def fmt(value):return '—' if value is None else f'{value:.4g}'


def report(result):
    text=['# 优化策略内部的排序诊断','',
        '每条历史只对四种优化策略计算预测平均分数与实际M的Spearman相关，使用实际执行的动作；回退恒充保留其相同分数和并列秩。全部九策略的相关作为对照。M为最近错误候选联合SSE减真结构候选SSE。','',
        '|响应|四优化策略相关的中位数|范围|有效历史|九策略对照中位数|', '|---|---:|---:|---:|---:|']
    for response,groups in result['within_history_summary'].items():
        a,b=groups['optimized'],groups['all_nine_reference']
        text.append(f"|{response}|{fmt(a['median'])}|[{fmt(a['min'])},{fmt(a['max'])}]|{a['n']}/12|{fmt(b['median'])}|")
    text+=['', '|历史|家族|四策略有噪声ρ|四策略无噪声ρ|九策略有噪声ρ|九策略无噪声ρ|','|---|---|---:|---:|---:|---:|']
    by={}
    for r in result['per_history']:by.setdefault(r['seed'],{})[r['response']]=r
    for seed,v in sorted(by.items()):
        a,b=v['actual_M'],v['noiseless_joint_M'];text.append(f"|{seed}|{a['family']}|{fmt(a['optimized']['rho'])}|{fmt(b['optimized']['rho'])}|{fmt(a['all_nine_reference']['rho'])}|{fmt(b['all_nine_reference']['rho'])}|")
    text+=['','## 每个预声明配对跨历史的增量相关','',
        '分别对每个配对计算Δ预测平均分数与Δ实际M的相关。每个配对最多12条历史；配对之间不合并样本。','',
        '|配对|响应|有效/计划历史|Spearman ρ|缺失历史|','|---|---|---:|---:|---|']
    for r in result['contrast_correlations']:
        text.append(f"|{r['contrast']}|{r['response']}|{r['valid_pairs']}/{r['planned_pairs']}|{fmt(r['rho'])}|{r['missing_histories']}|")
    text+=['','Spearman使用平均秩处理相同分数；某列全相同时相关未定义，保存为空。固定家族配额下仅作描述，未计算p值、置信区间或拓扑正确概率。原失败、回退和配对缺失均保留；方法数与候选方向数不增加独立历史数。','']
    return '\n'.join(text)


def main():
    p=argparse.ArgumentParser();p.add_argument('--summary',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    args=p.parse_args();raw=args.summary.read_bytes();summary=json.loads(raw);result=analyze(summary)
    result.update(input_summary=str(args.summary.resolve()),input_summary_sha256=hashlib.sha256(raw).hexdigest(),
        parent_trial_sha256=summary['trial_hashes'],parent_scientific_source_sha256=summary['source_sha256'],
        analysis_sources={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest() for name in (Path(__file__).name,'ev_original_analysis.py')})
    if args.summary.read_bytes()!=raw:raise ValueError('input summary changed during rank diagnostic')
    args.output.mkdir(parents=True,exist_ok=True)
    for name,rows in (('per_history_optimized.csv',flatten_history(result['per_history'])),
            ('optimized_policy_points.csv',result['policy_points']),('contrast_correlations.csv',result['contrast_correlations']),
            ('contrast_points.csv',result['contrast_points'])):write_csv(args.output/name,rows)
    write_json(args.output/'summary.json',result);(args.output/'summary.md').write_text(report(result),encoding='utf-8')
    print(json.dumps(result['within_history_summary']))


if __name__=='__main__':main()
