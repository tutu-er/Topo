"""Compact, transparent exports of independent structural-information studies.

Read saved trials only. Raw physical states and optimizer traces stay locally
in immutable trial archives; manifest hashes locate them. This exporter does
not choose actions or calibrate gates.
"""
from pathlib import Path
import csv,gzip,hashlib,json
from collections import Counter,defaultdict
import numpy as np
from scipy.stats import binomtest

HERE=Path(__file__).resolve().parent

def read(path):
    p=Path(path);b=p.read_bytes()
    return json.loads(gzip.decompress(b) if p.suffix=='.gz' else b)

def write(path,value):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n',encoding='utf-8')

def _csv(path,rows):
    if not rows:return
    names=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=names);w.writeheader()
        for row in rows:w.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(list,dict)) else v for k,v in row.items()})

def _stats(ts,methods):
    n=len(ts);out={}
    for m in methods:
        rows=[next(r for r in t['records'] if r['method']==m) for t in ts]
        good=[r for r in rows if r['status']=='complete']
        out[m]=dict(conditions=n,independent_histories=len({t['history_id'] for t in ts}),complete=len(good),
            unique_correct=sum(r.get('only_truth_retained',False) for r in rows),
            unresolved=sum(r.get('unresolved',False) for r in rows),
            safe_unresolved=sum(r['status']=='complete' and not r.get('false_rejection',False) and r.get('unresolved',False) for r in rows),
            compute_or_physical_failures=len(rows)-len(good),
            false_rejections=sum(r.get('false_rejection',False) for r in rows),
            mean_signed_margin=float(np.mean([r['signed_margin_full_SSE'] for r in good])) if good else None,
            median_signed_margin=float(np.median([r['signed_margin_full_SSE'] for r in good])) if good else None,
            mean_impedance_absolute_rmse_pu=float(np.mean([r['impedance_absolute_rmse_pu'] for r in good])) if good else None,
            median_impedance_relative_rmse=float(np.median([r['impedance_relative_rmse'] for r in good])) if good else None)
    return out

def export_phase(output,phase):
    out=Path(output);d=out/phase;paths=sorted(d.glob('trial_*.json.gz'));ts=[read(p) for p in paths]
    cfg=read(d/'config.json')
    if len(ts)!=cfg['run_count']:raise ValueError('complete phase required')
    methods=[r['method'] for r in ts[0]['records']]
    if any([r['method'] for r in t['records']]!=methods for t in ts):raise ValueError('mixed methods')
    table=d/'tables';table.mkdir(exist_ok=True)
    fitrows=[];methodrows=[];searchrows=[];files=[]
    for p,t in zip(paths,ts):
        common=dict(run_id=t['run_id'],history_id=t['history_id'],case=t['case'],quality=t['quality'],true_name=t['true_name'])
        files.append(dict(file=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size,
            history_id=t['history_id'],policy_id=t['policy_id'],complete=t['complete']))
        groups=[('H',t['historical_fits'])]+[(r['method'],r.get('fits',[])) for r in t['records']]
        for method,rows in groups:
            for f in rows:
                fitrows.append({**common,'method':method,**{k:f.get(k) for k in ('name','supports','success','loss','loglik','r','x','nfev','optimality')},
                    'global_optimality_certified':False})
        for r in t['records']:
            methodrows.append({**common,**{k:r.get(k) for k in ('method','status','winner','signed_margin_full_SSE','true_gap',
                'only_truth_retained','unresolved','false_rejection','wrong_candidates_removed','retained_candidates',
                'impedance_absolute_rmse_pu','impedance_relative_rmse','seconds','ac_calls','fit_calls','equivalent_action_fit_cache_hit')},
                'site':r.get('action',{}).get('site'),'p_pu':r.get('action',{}).get('p'),
                'selection_status':t['selection']['methods'][r['method']]['status']})
        for tag,s in t['selection']['searches'].items():
            for site in s.get('site_records',[]):
                searchrows.append({**common,'search':tag,'search_success':s.get('success'),'site':site['site'],
                    'selected_site':s.get('selected_site'),'score':site.get('selected_score'),'best_seed_score':site.get('best_seed_score'),
                    'gradient_gain_over_best_seed':site.get('gradient_gain_over_best_seed'),'gain_over_site_constant':site.get('gain_over_site_constant'),
                    'search_seconds':s.get('seconds'),'initialization_seconds':s.get('initialization_seconds'),
                    'ac_calls':s.get('total_ac_calls'),'initialization_ac_calls':s.get('initialization_ac_calls'),
                    'anchor_count':t['selection'].get('context',{}).get('anchor_count'),'failures':len(site.get('failures',[])),
                    'criterion':s.get('criterion'),'global_design_optimality_certified':False})
    _csv(table/'fits.csv',fitrows);_csv(table/'methods.csv',methodrows);_csv(table/'searches.csv',searchrows)
    strata={}
    for key in ('case','quality'):
        strata[key]={value:_stats([t for t in ts if t[key]==value],methods) for value in sorted({t[key] for t in ts})}
    pairs=[]
    for baseline in ('constant_fixed','quality_global','quality_local','information_seed','uniform_information','nominal_information','profile_designed'):
        if baseline not in methods:continue
        deltas=defaultdict(list);wins=losses=ties=0;condition_count=0
        for t in ts:
            r={r['method']:r for r in t['records']};a,b=r['information_designed'],r[baseline]
            if phase=='evaluation':
                x,y=a['only_truth_retained'],b['only_truth_retained'];wins+=bool(x and not y);losses+=bool(y and not x);ties+=x==y
            if a['status']=='complete' and b['status']=='complete':
                deltas[t['history_id']].append(a['signed_margin_full_SSE']-b['signed_margin_full_SSE']);condition_count+=1
        v=np.asarray([np.mean(x) for x in deltas.values()]);n=len(v);discord=wins+losses
        pairs.append(dict(baseline=baseline,paired_histories=n,paired_conditions=condition_count,
            mean_margin_difference=float(v.mean()) if n else None,median_margin_difference=float(np.median(v)) if n else None,
            paired_standard_error=float(v.std(ddof=1)/np.sqrt(n)) if n>1 else None,
            positive_history_differences=int((v>1e-6).sum()),negative_history_differences=int((v< -1e-6).sum()),
            binary_paired_histories=len(ts) if phase=='evaluation' else None,
            binary_failure_rule='required fit failure retains all and counts as not uniquely correct',
            unique_correct_discordant_wins=wins,unique_correct_discordant_losses=losses,unique_correct_ties=ties,
            paired_binary_two_sided_p=float(binomtest(wins,discord,.5).pvalue) if phase=='evaluation' and discord else None,
            p_scope='descriptive predeclared paired comparison; no multiplicity adjustment'))
    infos=[sum(s.get('seconds',0)+s.get('initialization_seconds',0) for tag,s in t['selection']['searches'].items() if tag!='profile') for t in ts]
    performance=dict(independent_histories=len({t['history_id'] for t in ts}),conditions=len(ts),methods=_stats(ts,methods),strata=strata,
        paired= pairs,planning=dict(mean_all_selection_seconds=float(np.mean([t['selection']['seconds'] for t in ts])),
            max_all_selection_seconds=max(t['selection']['seconds'] for t in ts),mean_information_search_seconds=float(np.mean(infos)),
            max_information_search_seconds=max(infos),H_gate_activated=sum(t['selection']['activated'] for t in ts),
            context_reason_counts=dict(Counter(t['selection'].get('context',{}).get('reason',t['selection'].get('inactive_reason')) for t in ts)),
            anchor_count_counts=dict(Counter(str(t['selection'].get('context',{}).get('anchor_count',0)) for t in ts)),
            total_AC_calls=sum(t['total_ac_calls'] for t in ts)),
        source_sha256=cfg['source_sha256'],policy_id=cfg['policy_id'],data_scope='declared synthetic mixture; retained all failures',
        exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write(table/'analysis.json',performance);write(table/'raw_manifest.json',dict(raw_trial_archives_kept_locally=True,files=files,
        exact_source_snapshot='../../science_source_snapshot.json.gz',reconstruction='restore exact newline bytes at corresponding stage commit, then run frozen config/seeds'))
    return performance

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--phase',required=True);a=p.parse_args()
    result=export_phase(a.output,a.phase)
    print(json.dumps({k:result[k] for k in ('independent_histories','conditions','policy_id')},sort_keys=True))