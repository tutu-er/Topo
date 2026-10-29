"""Frozen two-output checks and fifteen old development model records."""
from __future__ import annotations
import argparse,hashlib,json,os,sys,platform,traceback
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from .guard import choose_output,clean_point
from research_experiments.wzzt_dr_selection.dr_pool_v2.adapter import source_hashes
from research_experiments.wzzt_dr_selection.dr_pool_v2.observations import normalize,input_hash,matrices
from research_experiments.wzzt_dr_selection.dr_paper15_update.study import inputs

DEFAULT_PARENT=Path(r'C:\Users\23761\.codex\visualizations\2026\10\25\01a1726e-8ec7-7bc1-9f18-391f9cf79a1a\wzzt_dr_paper15_update_v1')
MODELS=('rnj_bank','baseline25','multistart_original_rule')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def identity(point,obs_hashes,cal):
    raw=dict(point=point,observations=obs_hashes,calibration=cal)
    return hashlib.sha256(json.dumps(raw,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def sources():
    out=source_hashes()
    from research_experiments.wzzt_dr_selection.dr_paper15_update import study as old
    from research_experiments.wzzt_dr_selection.dr_pool_v2 import runner
    from research_experiments.wzzt_dr_selection.end_to_end import physical_case
    for module in (old,runner,physical_case):out[str(Path(module.__file__).resolve())]=sha(module.__file__)
    for name in ('__init__.py','guard.py','study.py','test_guard.py'):
        p=Path(__file__).parent/name;out[str(p.resolve())]=sha(p)
    return out


def prepare(root,parent_root=DEFAULT_PARENT):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);target=root/'protocol.json';current=sources()
    if target.exists():
        p=read(target)
        if p['source_sha256']!=current:raise RuntimeError('frozen output-guard sources changed')
        for f,h in p['input_sha256'].items():
            if sha(f)!=h:raise RuntimeError('frozen development input changed '+f)
        return p
    parent=read(Path(parent_root)/'development_protocol.json');cases=parent['cases']
    if len(cases)!=5:raise RuntimeError('expected exactly five frozen old cases')
    hashes=dict(parent['input_sha256']);hashes[str(Path(parent_root)/'development_protocol.json')]=sha(Path(parent_root)/'development_protocol.json')
    for case in cases:
        original=read(case['original_fit'])
        for name in MODELS:
            if 'selected_uncontracted' not in original['models'][name]:raise RuntimeError('raw input missing')
    p=dict(created_at=datetime.now(timezone.utc).isoformat(),stage='bounded output-layer check; already-seen development, not DR',
        source_sha256=current,input_sha256=hashes,cases=cases,models=list(MODELS),
        planned_development_records=15,planned_mechanism_checks=2,
        mechanism_parameters=dict(leaf_r=.1,leaf_x=.1,common=[.01,0.],
            training_input='.01*(I4+J4)',validation_input='.013*(I4+J4)',root_voltage=1.,
            scope='given exact fitted point; postprocessing unit check only, no fitting or AC'),
        settings=dict(validation_fraction=.03,absolute_tolerance=1e-12,geometry_factor=.16,
            compare=['given fitted raw','original contract of that raw'],final_contraction=False,
            new_LP_calls=0,new_AC_calls=0,new_fits=0,new_future_data=False,old_development_observations_reused=True),
        fairness='same output handling for RNJ, primary and multistart',
        stop='exactly2 deterministic mechanisms +15 model records with duplicate cache; no expansion',
        truth_role='physical network read only after all 15 choices return, for evaluation',
        environment=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,platform=platform.platform()))
    write(target,p);return p


def mechanism(name):
    from research_experiments.rnj.rx75_edge_floor import calibrate_rx75_floor
    c=.01 if name=='positive_weak' else 0.
    point=dict(supports=[[0],[1],[0,1]],r=[.1,.1,c],x=[.1,.1,c])
    r,x=matrices(2,point['supports'],point['r'],point['x']);cal=calibrate_rx75_floor(r,x)
    def obs(scale,date):
        u=scale*(np.eye(4)+np.ones((4,4)));p,q=u[:,:2],u[:,2:]
        y=p@r.T+q@x.T;ix=pd.date_range(date,periods=4,freq='h');labels=[101,102]
        return (pd.DataFrame(p,index=ix,columns=labels),pd.DataFrame(q,index=ix,columns=labels),
                pd.DataFrame(np.sqrt(1-y),index=ix,columns=labels),pd.Series(np.ones(4),index=ix))
    return point,obs(.01,'2021-01-01'),obs(.013,'2021-01-02'),cal


def run(root,parent_root=DEFAULT_PARENT):
    root=Path(root);p=prepare(root,parent_root);target=root/'summary.json'
    if target.exists():return read(target)
    lock=root/'run.lock'
    if lock.exists():raise RuntimeError('existing output-guard lock')
    write(lock,dict(pid=os.getpid(),created_at=datetime.now(timezone.utc).isoformat()))
    records=[];checks=[];cache={}
    try:
        for name in ('positive_weak','zero_common'):
            f=root/('mechanism_'+name+'.json')
            if f.exists():row=read(f)
            else:
                try:
                    point,tr,va,cal=mechanism(name);choice=choose_output(point,tr,va,cal)
                    expected=0 if name=='positive_weak' else 1
                    if choice['chosen_index']!=expected:raise RuntimeError('mechanism expected output mismatch')
                    row=dict(name=name,status='passed',choice=choice,calibration=cal,expected_chosen_index=expected,
                        input_rank=4,scope='given exact point, no search/fit/AC/independent DR effect')
                except Exception as exc:row=dict(name=name,status='failed',error=repr(exc),traceback=traceback.format_exc())
                write(f,row)
            checks.append(row)
        for case in p['cases']:
            train,val=inputs(case);tr,days,_=normalize(train,'training');va,vdays,_=normalize(val,'validation')
            original=read(case['original_fit']);obs_hashes=dict(training=input_hash(tr,days),validation=input_hash(va,vdays))
            if obs_hashes!=original['input_sha256']:raise RuntimeError('old observation fingerprint mismatch')
            for model in p['models']:
                f=root/(case['id']+'__'+model+'.json')
                if f.exists():
                    row=read(f)
                    if row['status']=='complete':cache.setdefault(row['identity'],row['choice'])
                    records.append(row);continue
                row=dict(case=case['id'],model=model,status='running',original_fit_sha256=sha(case['original_fit']),input_sha256=obs_hashes)
                try:
                    old=original['models'][model];raw=clean_point(old['selected_uncontracted'],len(original['terminal_labels']));cal=original['calibrations']['raw']
                    key=identity(raw,obs_hashes,cal);reused=key in cache
                    choice=cache[key] if reused else choose_output(raw,tr,va,cal)
                    cache[key]=choice
                    contracted=choice['candidate_points'][1]
                    if clean_point(old,len(original['terminal_labels']))!=clean_point(contracted,len(original['terminal_labels'])):
                        raise RuntimeError('recomputed contract differs from old output')
                    row.update(status='complete',identity=key,reused_choice=reused,choice=choice,
                        changed_output=clean_point(choice['point'],len(original['terminal_labels']))!=clean_point(contracted,len(original['terminal_labels'])),
                        validation_change=choice['point']['validation_mae']-contracted['validation_mae'])
                except Exception as exc:row.update(status='failed',error=repr(exc),traceback=traceback.format_exc())
                write(f,row);records.append(row)
        # All choices are frozen before the physical reference enters evaluation.
        from research_experiments.wzzt_dr_selection.end_to_end.physical_case import build_reference_case,effective_truth_supports
        from research_experiments.wzzt_dr_selection.dr_pool_v2.runner import support_metrics
        truth=effective_truth_supports(build_reference_case('paper15'));evaluations=[]
        for row in records:
            if row['status']!='complete':continue
            original=read(next(c['original_fit'] for c in p['cases'] if c['id']==row['case']))
            metrics={}
            for name,point in [('raw',row['choice']['candidate_points'][0]),('old_contract',row['choice']['candidate_points'][1]),('new',row['choice']['point'])]:
                m=support_metrics(point,truth,original['terminal_labels'])
                m['all_block_errors']=len(m['missing_effective_supports'])+len(m['extra_effective_supports'])
                metrics[name]=m
            evaluations.append(dict(case=row['case'],model=row['model'],metrics=metrics))
        summary=dict(status='complete' if len(records)==15 and all(x['status']=='complete' for x in records) and all(x['status']=='passed' for x in checks) else 'completed_with_failures',
            planned=15,finished=len(records),complete=sum(r['status']=='complete' for r in records),
            failed=sum(r['status']!='complete' for r in records),unique_choices=len(cache),
            changed_outputs=sum(r.get('changed_output',False) for r in records),mechanism_checks=checks,
            records=records,evaluations=evaluations,new_LP_calls=0,new_AC_calls=0,new_fits=0,new_future_data=False,
            role='already-seen output postprocessing only; not independent recovery or DR evidence',protocol_sha256=sha(root/'protocol.json'))
        write(target,summary);print(json.dumps({k:summary[k] for k in ('status','planned','finished','failed','unique_choices','changed_outputs')}),flush=True)
        return summary
    finally:lock.unlink(missing_ok=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,required=True);a.add_argument('--parent',type=Path,default=DEFAULT_PARENT);a.add_argument('--prepare-only',action='store_true');x=a.parse_args()
    p=prepare(x.root,x.parent) if x.prepare_only else run(x.root,x.parent)
    if x.prepare_only:print(json.dumps(dict(planned=p['planned_development_records'],mechanisms=p['planned_mechanism_checks'],frozen=True)))
