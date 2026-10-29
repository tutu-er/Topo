"""Frozen development grid for contraction and AC-L1 algorithm updates.

These are already-seen development data, never independent confirmation.
Candidate fitting receives only observed scenarios; physical truth is loaded
only after the corresponding methods return, for evaluation.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import hashlib,json,os,pickle
from pathlib import Path
import traceback
import numpy as np
from .candidates_update import enumerate_contractions,fit_with_contractions,write,plain
from .ac_candidate_study import run_ac_candidates
from .zero_leaf_face import audit_zero_leaves
from research_experiments.wzzt_dr_selection.dr_pool_v2.runner import observed_only,subsample_training,support_metrics
from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize,matrices


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return pickle.loads(Path(p).read_bytes())


def sources():
    out=source_hashes()
    from research_experiments.wzzt_dr_selection.dr_pool_v2 import runner
    from research_experiments.wzzt_dr_selection.end_to_end import physical_case
    for module in (runner,physical_case):out[str(Path(module.__file__).resolve())]=sha(module.__file__)
    for name in ('__init__.py','candidates_update.py','ac_family_refit.py','ac_candidate_study.py','zero_leaf_face.py','study.py'):
        p=Path(__file__).parent/name;out[str(p.resolve())]=sha(p)
    return out


def prepare(root,build):
    root,build=Path(root),Path(build);root.mkdir(exist_ok=True,parents=True)
    target=root/'development_protocol.json'
    if target.exists():
        p=read(target)
        if p['source_sha256']!=sources():raise RuntimeError('frozen source changed')
        for name,h in p['input_sha256'].items():
            if sha(name)!=h:raise RuntimeError('frozen development input changed '+name)
        return p
    old=read(build/'existing_development_cases.json');cases=[];hashes={}
    for c in old['cases']:
        src=Path(c['source_run']);summary=read(src/'summary.json')
        paths=[src/'summary.json',Path(c['fit_result'])]
        paths += [src/'history'/(d+'.pkl') for d in summary['wzzt_training_dates']]
        paths += [src/'strategies/no_reward'/n for n in ('action.pkl','validation.pkl')]
        for f in paths:hashes[str(f)]=sha(f)
        original=read(c['fit_result']);count=len(enumerate_contractions(original))
        cases.append(dict(id=f'{c["noise"]}__N{c["N"]}',kind='old_development',N=c['N'],noise=c['noise'],
            source_run=str(src),original_fit=c['fit_result'],planned_families=count))
    packet=build/'strictpast_observed.pkl';fit=root/'strictpast_baseline/result.json'
    hashes[str(packet)]=sha(packet);hashes[str(fit)]=sha(fit)
    cases.append(dict(id='strictpast_48',kind='strict_past',packet=str(packet),original_fit=str(fit),
                      planned_families=len(enumerate_contractions(read(fit)))))
    if max(c['planned_families'] for c in cases)>256:raise RuntimeError('predeclared family cap exceeded')
    p=dict(created_at=datetime.now(timezone.utc).isoformat(),phase='development only; no confirmatory claim',
        cases=cases,source_sha256=sources(),input_sha256=hashes,
        settings=dict(max_families=256,linear_coefficient_domain='nonnegative unbounded',
            AC_outer_iterations=3,AC_line_steps=[1.,.5,.25],AC_objective='observed squared-voltage-drop L1',
            selection='same3% plus support-count and original0.16 contraction; AC rescored after contraction',
            base_kv=.4,base_mva=.2,physical_DR_budget_spent=0.),
        fair_baselines=['complete RNJ jointly refit','RNJ plus same contractions','RNJ plus same AC treatment'],
        no_true_support_targets=True,no_future_price_design=True,
        stop='finish exactly five inputs; independent dates required before any main superiority claim',
        next_DR_contract=dict(price=[0,1,2],participation=[0,.5,.75],response_kw=.2,response_kvar=.07,
            four_event_total_budget=1.6,node_capacity_kw=45.,price_currency='generic; not old1000NOK'))
    write(target,p);return p


def inputs(c):
    if c['kind']=='strict_past':
        data=load(c['packet']);return data['training'],data['validation']
    src=Path(c['source_run']);summary=read(src/'summary.json')
    past=[observed_only(load(src/'history'/(d+'.pkl'))) for d in summary['wzzt_training_dates']]
    action=load(src/'strategies/no_reward/action.pkl');val=load(src/'strategies/no_reward/validation.pkl')
    return subsample_training(past+[observed_only(action['simulation'])],c['N']),[observed_only(val['simulation'])]


def evaluate(models,labels,training=None,validation=None):
    # Physical reference appears ONLY here, after fitting and selection.
    from research_experiments.wzzt_dr_selection.end_to_end.physical_case import build_reference_case,effective_truth_supports
    from rnj_wzzt.models.lin_distflow import build_reduced_sensitivity_matrices
    net=build_reference_case('paper15');truth=effective_truth_supports(net)
    rr,xx=build_reduced_sensitivity_matrices(net,observed_terminals=labels,voltage_model='squared-voltage')
    denom=float(np.sum(rr*rr)+np.sum(xx*xx));out={}
    if training is not None:
        from .ac_family_refit import predict_ac
        train=normalize(training,'training')[0];valid=normalize(validation,'validation')[0]
    for name,point in models.items():
        m=support_metrics(point,truth,labels)
        m['all_block_errors']=len(m['extra_effective_supports'])+len(m['missing_effective_supports'])
        r,x=matrices(len(labels),point['supports'],point['r'],point['x'])
        m.update(kernel_relative_RMSE=float(np.sqrt((np.sum((r-rr)**2)+np.sum((x-xx)**2))/denom)),
                 validation_mae=point.get('validation_mae'),train_mae=point.get('train_mae',point.get('objective')),
                 truth_role='postfit evaluation only; not used for candidates, parameters or selection')
        if training is not None:
            try:
                m['AC_training_mae']=predict_ac(point,train,base_kv=.4,base_mva=.2)['mae']
                m['AC_validation_mae']=predict_ac(point,valid,base_kv=.4,base_mva=.2)['mae']
            except Exception as exc:m['AC_evaluation_error']=repr(exc)
        out[name]=m
    return out


def run(root,build,max_cases=1,ac=True):
    root=Path(root);p=prepare(root,build);lock=root/'update.run.lock'
    if lock.exists():raise RuntimeError('existing update run lock')
    write(lock,dict(pid=os.getpid(),created_at=datetime.now(timezone.utc).isoformat()))
    completed=0
    try:
        for c in p['cases']:
            folder=root/'development'/c['id'];folder.mkdir(parents=True,exist_ok=True)
            done=folder/'case_result.json'
            if done.exists():continue
            if completed>=max_cases:break
            completed+=1;train,val=inputs(c);original=read(c['original_fit'])
            record=dict(id=c['id'],status='running',scope='development input, not independent test',original_sha256=sha(c['original_fit']))
            try:
                lfile=folder/'linear.json'
                if lfile.exists():
                    linear=read(lfile)
                    if linear['status'] not in ('complete','completed_with_failures'):raise RuntimeError('partial linear stage retained; no implicit restart')
                else:linear=fit_with_contractions(train,val,original,bound=float('inf'),max_families=256,output_path=lfile)
                record['linear_status']=linear['status'];record['linear_evaluation']=evaluate(linear['models'],original['terminal_labels'],train,val)
                record['original_evaluation']=evaluate(original['models'],original['terminal_labels'],train,val)
                if ac:
                    afile=folder/'ac.json'
                    if afile.exists():
                        aa=read(afile)
                        if aa['status'] not in ('complete','completed_with_failures'):raise RuntimeError('partial AC stage retained; no implicit restart')
                    else:aa=run_ac_candidates(train,val,original,linear,base_kv=.4,base_mva=.2,
                        outer_iterations=3,max_families=256,output_path=afile)
                    record['AC_status']=aa['status'];record['AC_failures']=len(aa['failures'])
                    record['AC_evaluation']=evaluate(aa['models'],original['terminal_labels'],train,val)
                zfile=folder/'zero_leaf_face.json'
                if zfile.exists():zz=read(zfile)
                else:zz=audit_zero_leaves(train,original,bound=float('inf'),output_path=zfile)
                record['zero_leaf_audit_status']=zz.get('status')
                record['status']='complete' if record.get('AC_status','complete')=='complete' and linear['status']=='complete' else 'completed_with_failures'
            except Exception as exc:record.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
            write(done,record)
            print(json.dumps(dict(case=c['id'],status=record['status'],linear=record.get('linear_evaluation'),AC=record.get('AC_evaluation'))),flush=True)
    finally:lock.unlink(missing_ok=True)
    rows=[read(root/'development'/c['id']/'case_result.json') for c in p['cases'] if (root/'development'/c['id']/'case_result.json').exists()]
    summary=dict(status='complete' if len(rows)==len(p['cases']) else 'partial',planned_cases=len(p['cases']),completed_cases=len(rows),
        failures=sum(r['status']!='complete' for r in rows),records=rows)
    write(root/'development_summary.json',summary);return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--build',type=Path,required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--max-cases',type=int,default=1)
    a=p.parse_args();r=prepare(a.root,a.build) if a.prepare_only else run(a.root,a.build,a.max_cases)
    print(json.dumps({k:v for k,v in r.items() if k not in ('records','source_sha256','input_sha256','cases')},ensure_ascii=False))
