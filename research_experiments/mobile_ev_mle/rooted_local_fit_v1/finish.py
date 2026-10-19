"""Read-only scientific replay, paired statistics, and compact evidence."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
from pathlib import Path
import shutil
import zipfile
import numpy as np
from scipy.linalg import helmert
from ..reference_benchmark.representation import physical_tree_ac
from .experiment import ROOT,load,save,sha
from .model import prepare_case,line_map,project


def independent_response(candidate,payload,r,x,covariance=None):
    obs=payload['observed'];sigma=payload['sigma'];n=candidate.n;t=len(obs['p'])
    vp=payload['voltage_positions'];variance=4*np.asarray(sigma['v'])**2+2*np.asarray(sigma['v'])**4
    rr=(candidate.M*np.asarray(r))@candidate.M.T;xx=(candidate.M*np.asarray(x))@candidate.M.T
    if payload['root_meter'] is None:
        d=helmert(n,full=False);h=-2*np.c_[d@rr,d@xx]
        y=(obs['v']**2-np.asarray(sigma['v'])**2)@d.T
        cy=(d*variance)@d.T
    else:
        b,bs=payload['root_meter'];h=2*np.c_[rr[vp],xx[vp]]
        y=(np.asarray(b)**2-bs**2)[:,None]-obs['v'][:,vp]**2+np.asarray(sigma['v'])[vp]**2
        cy=np.diag(variance[vp])+(4*bs**2+2*bs**4)
    sx=np.r_[sigma['p'],sigma['q']]**2;data=np.c_[obs['p'],obs['q']]
    residual=y-data@h.T
    omega=cy+(h*sx)@h.T if covariance is None else np.asarray(covariance)[0]
    weighted=np.linalg.solve(omega,residual.T).T;value=float(np.sum(residual*weighted))
    latent=data+(weighted@h)*sx
    return value,h,weighted,latent


def independent_ac(candidate,payload,fitted):
    # Construct a general physical tree from the proposed supports and fitted
    # R/X, then use the independent node/branch sweep, never candidate.B/zbus.
    supports=list(candidate.supports);branches=[]
    for i,support in enumerate(supports):
        ancestors=[j for j,s in enumerate(supports) if set(support)<set(s)]
        parent=min(ancestors,key=lambda j:len(supports[j]))+1 if ancestors else 0
        branches.append(dict(from_bus=parent,to_bus=i+1,r_ohm=.8*fitted['r'][i],x_ohm=.8*fitted['x'][i],status=True))
    ids=[supports.index((j,))+1 for j in range(candidate.n)]
    physical=physical_tree_ac(branches,ids,0,.4,200.,np.asarray(fitted['p'])*200,np.asarray(fitted['q'])*200,0,
                              root_v=fitted['root_v'])
    sig=payload['sigma'];obs=payload['observed'];vp=payload['voltage_positions']
    terms=[(np.asarray(fitted['p'])-obs['p'])/sig['p'],(np.asarray(fitted['q'])-obs['q'])/sig['q'],
           (physical['v'][:,vp]-obs['v'][:,vp])/np.asarray(sig['v'])[vp]]
    if payload['root_meter'] is not None:
        b,bs=payload['root_meter'];terms.append(((np.asarray(fitted['root_v'])-b)/bs)[:,None])
    loss=float(sum(np.sum(z*z) for z in terms))
    return loss,float(np.max(abs(physical['v']-fitted['v'])))


def order(values,tolerance=1e-5):
    if len(values)!=4 or any(v is None or not np.isfinite(v) for v in values.values()):return None
    minimum=min(values.values());best=sorted(k for k,v in values.items() if v<=minimum+tolerance)
    return dict(best=best,full_order=sorted(values,key=lambda k:(values[k],k)),values=values)


def compare(a,b):
    if a is None or b is None:return None
    names=sorted(a['values']);agree=0
    for j,x in enumerate(names):
        for y in names[:j]:
            sa=np.sign(a['values'][x]-a['values'][y]) if abs(a['values'][x]-a['values'][y])>1e-5 else 0
            sb=np.sign(b['values'][x]-b['values'][y]) if abs(b['values'][x]-b['values'][y])>1e-5 else 0
            agree+=sa==sb
    return dict(first_agrees=a['best']==b['best'],pair_agrees=int(agree),pairs=6)


def finish(folder,evidence):
    folder=Path(folder);evidence=Path(evidence)
    if evidence.exists():raise FileExistsError('fresh evidence folder required')
    config=load(folder/'configuration.json');result=load(folder/'results.json');issues=[]
    for relative,expected in config['source_sha256'].items():
        if sha(ROOT/relative.replace('\\','/'))!=expected:issues.append('changed source '+relative)
    restored=folder/'audit_source_restore';restored.mkdir(exist_ok=False)
    with zipfile.ZipFile(folder/'scientific_sources.zip') as z:
        for member in z.namelist():
            target=(restored/member).resolve()
            if not target.is_relative_to(restored.resolve()):raise ValueError('unsafe archive member')
            target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(member))
        for relative,expected in config['source_sha256'].items():
            if sha(restored/relative.replace('\\','/'))!=expected:issues.append('archive recovery '+relative)
    audit=dict(ac_replays=0,response_replays=0,qp_bounds_checked=0,input_records_checked=0,
               max_AC_loss_difference=0.,max_AC_voltage_difference=0.,max_response_loss_difference=0.,
               max_QP_bound_difference=0.,source_members_restored=len(config['source_sha256']),
               inference_seals_checked=0,nondecreasing_AC_traces=0,wrong_heldout_lines=0)
    allrows=[];rankrecords=[];hashes={}
    for task in result['results']:
        if task.get('status')=='worker_failed':continue
        spec=task['spec'];key=f"{spec['name']}_{spec['seed']}_{spec['N']}_{spec['context']}"
        payload,hold,truth=prepare_case(spec['name'],spec['seed'],spec['N'],spec['context'])
        inference=load(folder/'inference'/f'{key}.json.gz');record=load(folder/'records'/f'{key}.json.gz')
        filehash=sha(folder/'inference'/f'{key}.json.gz')
        if filehash!=record['inference_sha256']:issues.append('inference changed '+key)
        audit['inference_seals_checked']+=1
        with np.load(folder/'inputs'/f'{key}.npz') as raw:
            for ch in ('p','q','v'):
                if not np.array_equal(raw[ch],payload['observed'][ch]):issues.append('input changed '+key+'/'+ch)
        audit['input_records_checked']+=1;bank={c.name:c for c in payload['candidates']}
        for label,fitted in inference['fits'].items():
            name,method=label.split('/');candidate=bank[name]
            if 'r' not in fitted:continue
            if method=='AC':
                actual,volts=independent_ac(candidate,payload,fitted)
                audit['max_AC_loss_difference']=max(audit['max_AC_loss_difference'],abs(actual-fitted['loss']))
                audit['max_AC_voltage_difference']=max(audit['max_AC_voltage_difference'],volts);audit['ac_replays']+=1
                for attempt in fitted['attempts']:
                    if any(np.diff(attempt.get('trace',[]))>=0):audit['nondecreasing_AC_traces']+=1
            else:
                covariance=fitted.get('covariance') if method=='QP' else None
                value,h,weighted,latent=independent_response(candidate,payload,fitted['r'],fitted['x'],covariance)
                audit['max_response_loss_difference']=max(audit['max_response_loss_difference'],abs(value-fitted['loss']))
                audit['response_replays']+=1
                if method=='QP':
                    transform,active=line_map(candidate,payload);width=transform.shape[1]
                    # Differentiate the explicit H model along each ray basis.
                    gradient=[]
                    for j in range(width):
                        rr,xx=np.split(transform[:,j],2)
                        dr=(candidate.M*rr)@candidate.M.T;dx=(candidate.M*xx)@candidate.M.T
                        direction=-helmert(candidate.n,full=False) if payload['root_meter'] is None else np.eye(candidate.n)[payload['voltage_positions']]
                        dh=2*np.c_[direction@dr,direction@dx]
                        dy=np.c_[payload['observed']['p'],payload['observed']['q']]@dh.T
                        gradient.append(-2*float(np.sum(weighted*dy)))
                    gradient=np.array(gradient);d=width//2;theta=np.asarray(fitted['theta'])
                    lower=value+np.minimum(0,np.minimum(gradient[:d],gradient[d:])).sum()-gradient@theta
                    audit['max_QP_bound_difference']=max(audit['max_QP_bound_difference'],abs(lower-fitted['lower_bound']))
                    audit['qp_bounds_checked']+=1
            if label in record['common_ac_holdout']:
                held=record['common_ac_holdout'][label];actual,volts=independent_ac(candidate,hold,held)
                audit['max_AC_loss_difference']=max(audit['max_AC_loss_difference'],abs(actual-held['loss']))
                audit['max_AC_voltage_difference']=max(audit['max_AC_voltage_difference'],volts);audit['ac_replays']+=1
                for ch in ('r','x'):
                    if not np.array_equal(held[ch],fitted[ch]):audit['wrong_heldout_lines']+=1
        # Additional post-seal diagnostics use truth only to explain error
        # scale; they never change fits, weights, candidates or rankings.
        for row in record['rows']:
            if row['candidate']!='base':continue
            fitted=inference['fits']['base/'+row['method']]
            if 'r' not in fitted:continue
            candidate=bank['base'];focus=truth['focus_positions']
            leaf=[i for i,s in enumerate(candidate.supports) if len(s)==1 and s[0] in focus]
            for ch in ('r','x'):
                true=np.asarray(truth[ch])[leaf];estimated=np.asarray(fitted[ch])[leaf]
                row['leaf_'+ch.upper()+'_MAE_ohm']=float(np.mean(abs(estimated-true))*.8)
                row['leaf_'+ch.upper()+'_true_mean_ohm']=float(np.mean(true)*.8)
            rr=(candidate.M*truth['r'])@candidate.M.T;xx=(candidate.M*truth['x'])@candidate.M.T
            count=spec['N'];w=truth['root_v'][:count,None]**2-2*(truth['p'][:count]@rr.T+truth['q'][:count]@xx.T)
            row['true_lines_lossless_V_error_RMSE_pu']=float(np.sqrt(np.mean((np.sqrt(np.maximum(w,.01))-truth['v'][:count])**2)))
        allrows.extend(record['rows'])
        if spec['context'] in ('FULL','CUT'):
            ranks={}
            for method in ('AC','RESP','QP'):
                rows=[r for r in record['rows'] if r['method']==method]
                for phase,column in [('train','train_J'),('holdout','holdout_J'),('common_AC','common_AC_holdout_J')]:
                    qualified=(len(rows)==4 and all(r.get('train_qualified') and
                        (phase=='train' or r.get('holdout_qualified' if phase=='holdout' else 'common_AC_holdout_qualified')) for r in rows))
                    ranks[method+'/'+phase]=dict(qualified=qualified,rank=order({r['candidate']:r.get(column) for r in rows}))
            cross=order({label.split('/')[0]:z['J'] for label,z in record['ac_lines_response_holdout'].items()})
            rankrecords.append(dict(spec,ranks=ranks,AC_parameters_RESP_holdout=cross))
        hashes[key]=dict(inference=filehash,record=sha(folder/'records'/f'{key}.json.gz'),input=sha(folder/'inputs'/f'{key}.npz'))
    if audit['max_AC_loss_difference']>1e-6:issues.append('AC loss replay')
    if audit['max_AC_voltage_difference']>1e-9:issues.append('independent AC voltage replay')
    if audit['max_response_loss_difference']>1e-6:issues.append('response loss replay')
    if audit['max_QP_bound_difference']>1e-4:issues.append('QP bound replay')
    for key in ('wrong_heldout_lines','nondecreasing_AC_traces'):
        if audit[key]:issues.append(key)
    groups=defaultdict(list)
    for row in allrows:groups[row['name'],row['N'],row['context'],row['method']].append(row)
    aggregate=[]
    for (name,n,context,method),rows in sorted(groups.items()):
        base=[r for r in rows if r['candidate']=='base'];good=[r for r in base if r.get('train_qualified') and r.get('status')=='completed']
        metrics={}
        for key in set().union(*(r.keys() for r in good)) if good else []:
            values=[float(r[key]) for r in good if isinstance(r.get(key),(int,float)) and not isinstance(r[key],bool)]
            if values:metrics[key]=dict(mean=float(np.mean(values)),median=float(np.median(values)),minimum=min(values),maximum=max(values))
        aggregate.append(dict(name=name,N=n,context=context,method=method,total_slots=len(rows),
            train_qualified=sum(r.get('train_qualified',False) for r in rows),
            holdout_qualified=sum(r.get('holdout_qualified',False) for r in rows),
            common_AC_qualified=sum(r.get('common_AC_holdout_qualified',False) for r in rows),
            evaluation_failures=sum(r.get('status')!='completed' for r in rows),base_total=len(base),base_qualified=len(good),metrics=metrics))
    ranking_summary=[]
    for context in ('FULL','CUT'):
        for n in (288,672):
            records=[r for r in rankrecords if r['context']==context and r['N']==n]
            for method in ('AC','RESP','QP'):
                entry=dict(context=context,N=n,method=method,total=len(records))
                for phase in ('train','holdout','common_AC'):
                    qualified=[r['ranks'][method+'/'+phase]['rank'] for r in records if r['ranks'][method+'/'+phase]['qualified']]
                    qualified=[q for q in qualified if q]
                    entry[phase]=dict(usable=len(qualified),base_in_best=sum('base' in q['best'] for q in qualified),
                        unique_base_best=sum(q['best']==['base'] for q in qualified),ties=sum(len(q['best'])>1 for q in qualified))
                common=[r for r in records if r['ranks']['AC/holdout']['qualified'] and r['ranks'][method+'/holdout']['qualified']]
                comparisons=[compare(r['ranks']['AC/holdout']['rank'],r['ranks'][method+'/holdout']['rank']) for r in common]
                entry['vs_AC_holdout']=dict(paired=len(common),first_agrees=sum(c['first_agrees'] for c in comparisons),
                    pair_agrees=sum(c['pair_agrees'] for c in comparisons),pairs=6*len(common))
                ranking_summary.append(entry)
    audit.update(issues=issues,passed=not issues,total_conditions=len(config['specifications']),rows=len(allrows),
                 worker_failures=sum(r.get('status')=='worker_failed' for r in result['results']),
                 training_qualified=sum(r.get('train_qualified',False) for r in allrows),
                 qualified_AC=sum(r['method']=='AC' and r.get('train_qualified',False) for r in allrows),
                 main_wall_seconds=result['seconds'],configuration_sha256=sha(folder/'configuration.json'))
    evidence.mkdir(parents=True)
    for name in ('configuration.json','completed.json','scientific_sources.zip'):shutil.copyfile(folder/name,evidence/name)
    save(evidence/'audit.json',audit);save(evidence/'aggregate.json',aggregate)
    save(evidence/'rankings.json',rankrecords);save(evidence/'ranking_summary.json',ranking_summary);save(evidence/'record_hashes.json',hashes)
    scalarrows=[{k:v for k,v in row.items() if not isinstance(v,(list,dict))} for row in allrows]
    fields=sorted(set().union(*(r.keys() for r in scalarrows)))
    with (evidence/'all_rows.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(scalarrows)
    for name in ('paper15','paper11'):
        key=f'{name}_36001_672_FULL'
        for source in (folder/'inference'/f'{key}.json.gz',folder/'records'/f'{key}.json.gz',folder/'inputs'/f'{key}.npz'):
            if source.exists():
                target=evidence/'representative'/source.parent.name/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
    print(audit,flush=True)
    if issues:raise AssertionError('audit issues retained in evidence')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--input',required=True);parser.add_argument('--evidence',required=True)
    args=parser.parse_args();finish(args.input,args.evidence)
