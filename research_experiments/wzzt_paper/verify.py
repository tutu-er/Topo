from pathlib import Path
import argparse
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from research_experiments.wzzt_systematic.common import read,write,sha
from research_experiments.rnj.audit_source_ac96_benchmark import physical_truth
from research_experiments.wzzt_fine.evaluate import signature
from .common import jobs,MAIN,AC_METHODS
from .records import find,load,load_trace,load_guard,load_retry,restore

ROOT=Path(__file__).resolve().parents[2]


def audit_mip_arrays(output,destination):
    from scipy.sparse import coo_matrix
    rows=[]
    for folder in sorted((output/'free_domain').glob('*/steps/step*')):
        if not (folder/'model.npz').exists() or not (folder/'result.json').exists():continue
        info=read(folder/'result.json')
        with np.load(folder/'model.npz') as d:raw={k:d[k] for k in d.files}
        A=coo_matrix((raw['constraint_data'],(raw['constraint_row_index'],raw['constraint_col_index'])),
            shape=(len(raw['constraint_lower']),len(raw['objective_coefficients']))).tocsr()
        for name in ('start','primal','selected_primal'):
            file=folder/(name+'.npz')
            if not file.exists():continue
            with np.load(file) as d:x=d['primal']
            activity=A@x;integer=raw['integrality'].astype(bool)
            cv=max(0.,float(np.max(raw['constraint_lower']-activity)),float(np.max(activity-raw['constraint_upper'])))
            bv=max(0.,float(np.max(raw['lower_bounds']-x)),float(np.max(x-raw['upper_bounds'])))
            iv=float(np.max(abs(x[integer]-np.rint(x[integer])))) if integer.any() else 0.
            objective=float(raw['objective_coefficients']@x);expected=info.get('selected_upper') if name=='selected_primal' else info.get('objective') if name=='primal' else None
            required=name!='primal' or info['primal_audit']['feasible']
            passed=not required or (np.isfinite(x).all() and np.isfinite(objective) and
                max(cv,bv,iv)<=1e-7 and (expected is None or abs(objective-expected)<=1e-7))
            rows.append(dict(step=folder.relative_to(output).as_posix(),kind=name,constraint_violation=cv,bound_violation=bv,
                integrality_violation=iv,recomputed_objective=objective,recorded_objective=expected,required_feasible=required,passed=passed))
    pd.DataFrame(rows).to_csv(destination/'free_domain_primal_checks.csv',index=False)
    return rows


def verify(output,destination):
    output=Path(output).resolve();destination=Path(destination).resolve();restore(destination);checks=[]
    def check(name,passed):checks.append(dict(check=name,passed=bool(passed)))
    def hashes(name,root,receipt):
        items=read(receipt);check(name,all((root/p).is_file() and sha(root/p)==h for p,h in items.items()));return items
    frozen=hashes('all frozen source, input and physical data bytes',output,output/'manifest.json')
    hashes('all raw selected, checkpoint, timing and AC records',output,destination/'result_manifest.json')
    analysis=hashes('analysis source bytes',Path(__file__).parent,destination/'analysis_source_manifest.json')
    hashes('independent audit helper bytes',ROOT,destination/'analysis_helper_manifest.json')
    delivery=hashes('delivery bytes before verification',destination,destination/'delivery_manifest.json')
    for name,items in (('producer_source.zip',{p.removeprefix('source/'):h for p,h in frozen.items() if p.startswith('source/')}),('analysis_source.zip',analysis)):
        with zipfile.ZipFile(destination/name) as z:
            check(name+' exact bound source entries',all(p in z.namelist() and hashlib.sha256(z.read(p)).hexdigest()==h for p,h in items.items()))
    with zipfile.ZipFile(destination/'records.zip') as z:
        check('archived records match all local member bytes',all(hashlib.sha256(z.read(p)).hexdigest()==delivery[p] for p in z.namelist()))
    p=read(output/'protocol.json');expected=jobs();f=pd.read_csv(destination/'per_job.csv');rows=find(destination/'catalog.sqlite')
    check('full predeclared 1204-input protocol',p['jobs']==[j['job_id'] for j in expected] and p['planned']==1204)
    check('all planned 13500 rows retained',len(f)==len(rows)==p['planned_model_rows']==13500 and f.job_id.nunique()==1204)
    check('all planned method groups, including failures, present',all(set(f[f.job_id==j['job_id']].variant)==set(j['methods']) for j in expected))
    check('main 640 inputs and ten independent bundles',len(f[(f.panel=='main')&(f.variant=='ridge_union')])==640 and set(f[f.panel=='main'].bundle)==set(range(10)))
    check('stress400 topology144 and case33 20',all(int(((f.panel==panel)&(f.variant=='ridge_union')).sum())==n for panel,n in [('stress',400),('topology',144),('case33',20)]))
    data_diagnostics=pd.read_csv(destination/'data_diagnostics.csv')
    check('all 442 physical/noise regimes retain nine daily data diagnostics',len(data_diagnostics)==3978 and data_diagnostics.groupby(['case','condition','bundle','voltage_sigma']).size().eq(9).all() and data_diagnostics.converged.eq(96).all())
    index={(r['job_id'],r['variant']):r for r in rows};line_count=0;signatures=True
    with (destination/'selected_models.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            m=json.loads(line);signatures&=signature(m['point'])==m['model_signature']==index[(m['job_id'],m['variant'])]['model_signature'];line_count+=1
    check('JSONL and catalog models identical',signatures and line_count==int(f.model_available.sum()))
    truth={};lastjob=None;raw=None;raw_same=True;scores=True;available=0
    for r in rows:
        if not r['model_available']:continue
        if r['job_id']!=lastjob:raw=read(r['result_path']);lastjob=r['job_id']
        point=raw['models'][r['variant']];raw_same&=signature(point)==r['model_signature'];available+=1
        labels=json.loads(r['labels_json'])
        if r['truth_network'] not in truth:truth[r['truth_network']]=physical_truth(Path(r['truth_network']),labels)['clades']
        predicted={frozenset(labels[i] for i in s) for s in point['supports'] if 2<=len(s)<len(labels)};true=truth[r['truth_network']]
        f1=2*len(predicted&true)/(len(predicted)+len(true)) if predicted or true else 1.
        scores&=abs(f1-r['clade_f1'])<1e-12
    check('all available catalog models identical to raw selections',raw_same and available==int(f.model_available.sum()))
    check('all structural scores recomputed from physical graph',scores)
    ac=read(destination/'ac_summary.json');planned=f[f.ac_planned]
    check('predeclared AC scope and 876-row denominator',len(planned)==p['planned_AC_rows']==876 and set(planned.variant)==set(AC_METHODS) and not f[~f.ac_planned].ac_evaluated.any())
    check('all predeclared AC evaluations attempted',int(f.ac_evaluated.sum())==int((planned.model_available).sum()))
    check('AC cache binding count',len(list((output/'ac').glob('*.json')))==ac['unique_records'])
    counts={j['job_id']:j['N'] for j in expected};ac_rows=True;ac_bindings=True
    for file in (output/'ac').glob('*.json'):
        a=read(file);job,model_signature=a['key'].rsplit('__',1)
        ac_bindings&=model_signature==a['model_signature'] and ((f.job_id==job)&(f.model_signature==model_signature)&f.ac_planned).any()
        if a.get('fitting',{}).get('valid'):ac_rows&=a['fitting']['ac_converged_points']==counts[job]
        for part in ('testing','physical'):
            if a.get(part,{}).get('valid'):ac_rows&=a[part]['ac_converged_points']==288
    check('AC cached model signatures match planned model selections',ac_bindings)
    check('actual AC training rows equal N and both testing targets use 288 rows',ac_rows)
    for N in (24,288):
        m=load(destination/'catalog.sqlite',case='flynn16',panel='main',condition='reference',voltage_sigma=.001,bundle=3,N=N,variant='ridge_union',with_observations=True)
        check(f'precise bundle, condition and N={N} observation lookup',len(m['train'][0])==N and len(m['test'][0])==len(m['validation'][0])==288 and m['record']['bundle']==3)
        check(f'N={N} response matrix reconstruction',np.isfinite(m['R']).all() and m['R'].shape==m['X'].shape==(len(m['labels']),)*2)
    trace=load_trace(destination/'catalog.sqlite',case='flynn16',panel='main',condition='reference',voltage_sigma=.001,bundle=3,N=288)
    check('trace reader and no-truth provenance',trace['truth_used_by_estimator'] is False)
    stats=read(destination/'primary_statistics.json')
    check('ten independent blocks in each primary comparison',len(stats)==3 and all(r['independent_blocks']==10 and r['paired_inputs']==640 for r in stats))
    for s in stats:
        reference=s['contrast'].removeprefix('ridge_union vs ');sub=f[(f.panel=='main')&f.variant.isin(['ridge_union',reference])]
        pivot=sub.pivot(index=['job_id','bundle'],columns='variant',values='clade_f1');delta=pivot.ridge_union-pivot[reference]
        check(s['contrast']+' independently recomputed mean',abs(delta.groupby(level='bundle').mean().mean()-s['mean'])<1e-12)
    timing=read(output/'timing/summary.json')
    check('all 64 standalone timing attempts retained',len(timing)==64 and len({(r['job_id'],r['variant']) for r in timing})==64)
    check('all 64 timing replays produced finite estimator measurements',all('error' not in r and np.isfinite(r.get('estimator_seconds',np.nan)) for r in timing))
    check('available numerical/data evidence passed',read(destination/'audit_summary.json')['available_evidence_passed'])
    ref=read(destination/'case33_reference.json');network=output/'networks/case33_terminal/reference'
    buses=pd.read_csv(network/'buses.csv');edges=pd.read_csv(network/'branches.csv')
    check('case33 frozen input matches official nominal loads and ohmic lines',
        np.array_equal(buses.groupby('original_bus_id')[['pd_kw','qd_kvar']].sum().sort_index().to_numpy(),np.column_stack((ref['P_kW'],ref['Q_kvar']))) and
        np.array_equal(edges[edges.branch_type=='backbone'][['from_bus','to_bus','r_ohm','x_ohm']].to_numpy(),ref['closed_branches_ohm']))
    post=pd.read_csv(destination/'postfilter_sensitivity.csv');guard=pd.read_csv(destination/'postfilter_guard.csv')
    check('6020 exploratory floor records and 1204 guarded records',len(post)==6020 and len(guard)==1204)
    fixed=post[post.floor==.16].set_index('job_id');baseline=f[f.variant=='ridge_union'].set_index('job_id')
    check('every sensitivity .16 result matches unchanged primary',np.allclose(fixed.loc[baseline.index].clade_f1,baseline.clade_f1,atol=1e-12,rtol=0))
    aligned=guard.set_index('job_id').loc[fixed.index]
    valid=aligned.model_available.to_numpy() & fixed.model_available.to_numpy()
    check('observation-only filter never worsens validation on same model',np.all(aligned.validation_mae.to_numpy()[valid]<=fixed.validation_mae.to_numpy()[valid]+1e-12))
    import inspect
    from .resolution import guard_filter
    receipt=read(destination/'guard_freeze_receipt.json')
    check('prospective filter rule source unchanged',hashlib.sha256(inspect.getsource(guard_filter).encode()).hexdigest()==receipt['guard_source_sha256'])
    holdout=pd.read_csv(destination/'postfilter_guard_holdout.csv')
    check('prospective filter denominators 132 new-tree plus16 case33',len(holdout)==148 and (holdout.panel=='topology').sum()==132 and (holdout.panel=='case33').sum()==16)
    retry=read(destination/'lp60_retry_verification.json');retry_protocol=read(output/'lp60_retry/protocol.json')
    check('all thirteen fixed-LP retries returned and passed numerical checks',retry['planned_inputs']==retry['returned_inputs']==13 and retry['available_evidence_passed'])
    check('six originally planned AC rows also attempted on LP60 recovered models',retry['AC_planned']==retry['AC_evaluated']==6)
    retry_ac_receipt=read(destination/'lp60_ac_freeze_receipt.json')
    check('recovered AC scope frozen before its results',retry_ac_receipt['existing_AC_results']==0 and retry_ac_receipt['expected_model_rows']==6 and
        retry_ac_receipt['protocol_sha256']==sha(destination/'LP_RETRY_PROTOCOL.md'))
    case33_coverage=pd.read_csv(destination/'case33_recovered_coverage.csv')
    check('separate mixed-budget case33 coverage has 20 unique inputs for each of twelve models',len(case33_coverage)==240 and not case33_coverage.duplicated(['job_id','variant']).any() and case33_coverage.groupby('variant').job_id.nunique().eq(20).all())
    retry_binding=True
    for job in retry_protocol['jobs']:
        result=read(output/'lp60_retry'/job/'result.json')
        retry_binding&=result['original_failure_sha256']==sha(output/'jobs'/job/'error.json') and result['fixed_LP_seconds']==60
    check('LP60 supplements bound to unchanged original failures',retry_binding)
    loaded=load_retry(destination/'catalog.sqlite',case='case33_terminal',panel='case33',condition='reference',voltage_sigma=.001,bundle=0,N=288,with_observations=True)
    check('LP60 reader preserves original failure and exact observation subset',not loaded['original_record']['model_available'] and len(loaded['train'][0])==288 and loaded['record']['fixed_LP_seconds']==60)
    guarded=load_guard(destination/'catalog.sqlite',case='flynn16',panel='main',condition='reference',voltage_sigma=.001,bundle=3,N=24)
    check('guard reader keeps exploratory scope separate from primary',guarded['record']['variant']=='ridge_postfilter_guard' and guarded['original_record']['variant']=='ridge_union')
    joint=read(destination/'joint_extension_verification.json')
    check('secondary two-step extension retains 160 inputs and 480 rows over ten bundles',joint['planned_inputs']==160 and joint['rows']==480 and joint['independent_blocks']==10)
    check('available secondary joint-extension evidence passed',joint['available_evidence_passed'])
    joint_diagnostics=pd.read_csv(destination/'joint_selected_diagnostics.csv')
    check('each available joint proposal selection has a separate mechanism record',len(joint_diagnostics)==joint['available_models']//3 and not joint_diagnostics.job_id.duplicated().any())
    joint_receipt=read(destination/'joint_extension_freeze_receipt.json');joint_protocol=read(output/'joint_extension/protocol.json')
    check('joint extension uses its frozen secondary protocol and worker source',joint_receipt['protocol_sha256']==sha(destination/'JOINT_EXTENSION_PROTOCOL.md') and
        joint_receipt['worker_source_sha256']==joint_protocol['source_sha256']==sha(output/'joint_extension/worker_source.py'))
    free=read(destination/'free_domain_verification.json')
    check('80-input free-support supplement fully retained and audited',free['inputs']==80 and free['rows']==160 and free['all_passed'])
    mip=audit_mip_arrays(output,destination)
    check('all required feasible free-domain MILP primals satisfy original arrays; rejected native primals remain separate',bool(mip) and all(r['passed'] for r in mip))
    overview=read(destination/'overview.json')
    check('all 1204 inputs reached a retained terminal state',overview['completed_inputs']==1204)
    summary=dict(all_passed=all(c['passed'] for c in checks),checks=checks,planned_inputs=1204,model_rows=len(rows),available_models=available,
        AC_evaluated=ac['evaluated_rows'],AC_valid=ac['valid_rows'],timing_structure_differences=sum(r.get('structure_equal_to_shared') is False for r in timing))
    summary['independent_MIP_primal_checks']=len(mip)
    write(destination/'delivery_verification.json',summary)
    write(destination/'delivery_manifest.json',{p.name:sha(p) for p in destination.iterdir() if p.is_file() and p.name!='delivery_manifest.json'})
    print(dict(all_passed=summary['all_passed'],checks=len(checks),failed=[c['check'] for c in checks if not c['passed']]),flush=True)
    if not summary['all_passed']:raise AssertionError('delivery audit failed')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('destination');a=p.parse_args();verify(a.output,a.destination)
