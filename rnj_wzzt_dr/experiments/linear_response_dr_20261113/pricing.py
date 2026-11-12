"""Continuous ADMM proposals, original-actuator replay, then frozen model checks."""
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_key]='1'
from pathlib import Path
from time import perf_counter
import argparse,shutil,traceback
import numpy as np
from scipy.special import logsumexp
from ... import io
from ..unified_posterior_dr_20261112.history_scan import ROOT,require,sha,utc
from ..unified_posterior_dr_20261112.prospective import ForecastLaw,stream
from ..unified_posterior_dr_20261112.confirmation import rotation
from ..expected_logodds_admm_20261113.run import REFERENCE,load_target
from ..expected_logodds_admm_20261113.admm import run as admm
from ..expected_logodds_admm_20261113.confirm import log_odds_samples,means_interval,BETAS
from .models import ContinuousStep


def prepare(out):
    require((out/'screen_complete.json').exists(),'Screen all targets first')
    require(not (out/'pricing_protocol.json').exists(),'Preserve existing pricing')
    base=io.read(out/'protocol.json');screen=io.read(out/'screen_results.json')
    ranks={}
    for r in screen:
        if r['mode']!='linear_response' or not r['screen']['qualified']:continue
        gain=min(r['screen']['projected_gains'])
        if gain>=base['gain_required']-1e-8:ranks[r['id']]=max(ranks.get(r['id'],0),gain)
    selected=sorted(ranks,key=lambda k:(-ranks[k],k))[:2]
    targets=[r for r in base['targets'] if r['id'] in selected]
    require(len(targets)<=base['ADMM_targets_max'],'Excess targets')
    sources={}
    for folder in (Path(__file__).parent,ROOT/'rnj_wzzt_dr/experiments/expected_logodds_admm_20261113',ROOT/'rnj_wzzt_dr/experiments/unified_posterior_dr_20261112',ROOT/'rnj_wzzt_dr/dr'):
        for p in folder.glob('*.py'):
            rel=p.relative_to(ROOT);dest=out/'pricing_source_snapshot'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(p,dest);sources[rel.as_posix()]=io.ref(p)
    protocol=dict(created_utc=utc(),screen=io.ref(out/'screen_results.json'),parent=io.ref(out/'protocol.json'),
        targets=targets,selected=selected,rank_by_original_projected_minimum_gain=ranks,
        selection_detail='Achieved gain means after original controller replay. Choose the two inputs with the largest history-only projected minimum gain; retain both target directions. This is fixed before any new model sample.',
        sources=sources,seconds_per_target_mode=base['ADMM_seconds_per_target_mode'],
        seed_cost='Screening seed construction is reported separately and added to workflow time; the ADMM phase has its own 30-second budget including branch setup.',
        modes=base['modes'],gain_required=base['gain_required'],rho=10.,iterations=400,
        starts='zero for storage relaxation; zero and precomputed supporting-plane seed for linear response',
        model_paths=base['model_paths'],model_seed=base['model_seed'],model_namespace=base['model_namespace'],
        future_opened=False,new_fits=0)
    io.write(out/'pricing_protocol.json',protocol);return protocol


def solve(data,mode,directory,protocol,screen_row,seed_path):
    import gurobipy as gp
    directory.mkdir(parents=True);o=data['original'];e=data['evidence'];began=perf_counter()
    starts=[('zero',np.zeros((o.R,24)))]
    if mode=='linear_response' and screen_row['screen']['qualified']:
        with np.load(seed_path,allow_pickle=False) as a:starts.append(('screen_QP',a['x'].copy()))
    original_best=None;plant_best=None;attempts=[]
    # The original actuator is used as an independent repair/evaluation map.
    # It is not part of the relaxed QP constraints or an ADMM convergence proof.
    def retain(u,origin,x):
        nonlocal original_best,plant_best
        gains=e.gains(u);cost=o.fees(u)
        if e.feasible(u) and (plant_best is None or cost<plant_best['fee']):
            plant_best=dict(fee=cost,gains=gains.copy(),origin=origin,u=u.copy(),x=x.copy())
        replay=o.nominal(x);g=e.gains(replay);fee=o.fees(replay)
        if e.feasible(replay) and (original_best is None or fee<original_best['fee']):
            require(o.audit(replay)['passed'],'Original replay infeasible')
            original_best=dict(fee=fee,gains=g.copy(),origin=origin,u=replay.copy(),x=x.copy(),raw_gains=gains.copy())
    # Keep a successfully replayed linear proposal even if ADMM never improves it.
    if screen_row['mode']==mode and screen_row['screen']['qualified']:
        with np.load(seed_path,allow_pickle=False) as a:retain(a['u'], 'input_screen_QP', a['x'])
    for label,x0 in starts:
        steps=[];start=perf_counter();branch=directory/label;branch.mkdir()
        with gp.Env(empty=True) as env:
            env.setParam('OutputFlag',0);env.start()
            try:
                for r in range(o.R):
                    steps.append(ContinuousStep(data['problem'],o,r,mode,data['fee_scale'],env,x0[r])
                                 if e.active[:,r].any() else None)
                initial=o.scale*np.repeat(x0,4,axis=1)
                def fee(u):
                    requests=np.vstack([s.requests if s is not None else np.zeros((1,24)) for s in steps])
                    retain(u, label+'_ADMM_or_initial', requests)
                    return o.fees(u)
                result=admm(e,steps,initial,fee,seconds=protocol['seconds_per_target_mode']/len(starts),
                    iterations=protocol['iterations'],rho=protocol['rho'],setup_seconds=perf_counter()-start)
                io.write(branch/'trace.json',result.pop('trace'));best=result.pop('best')
                result.update(start=label,retained_iteration=None if best is None else best['iteration'])
                io.write(branch/'summary.json',result);attempts.append(result)
            finally:
                for step in steps:
                    if step is not None:step.close()
    def save(name,p):
        if p is None:return None
        np.savez_compressed(directory/(name+'.npz'),u=p['u'],x=p['x'])
        return {k:v for k,v in p.items() if k not in ('u','x')}
    result=dict(mode=mode,ADMM_seconds=perf_counter()-began,screen_seed_seconds=screen_row['screen']['seconds'],
        original_qualified=original_best is not None,plant_qualified=plant_best is not None,
        original=save('original_quote',original_best),plant=save('plant_quote',plant_best),attempts=attempts,
        global_optimality_proved=False,converged_starts=sum(r['status']=='residuals_small_and_original_gain_feasible' for r in attempts))
    io.write(directory/'summary.json',result);return result


def plan(out):
    protocol=prepare(out);screens={(r['id'],r['target'],r['mode']):r for r in io.read(out/'screen_results.json')};rows=[]
    for item in protocol['targets']:
        for mode in protocol['modes']:
            path=out/'pricing'/(item['id']+f"__target{item['target']}")/mode
            try:
                data=load_target(item,protocol['gain_required']);key=(item['id'],item['target'],mode);s=screens[key]
                result=solve(data,mode,path,protocol,s,out/'screen'/(item['id']+f"__target{item['target']}")/mode/'quote.npz')
                rows.append(dict(**item,status='complete',result=result));print(item['id'],item['target'],mode,'original',result['original_qualified'],'plant',result['plant_qualified'],flush=True)
            except Exception as exc:
                rows.append(dict(**item,mode=mode,status='failed',error=repr(exc),traceback=traceback.format_exc()))
                print('retained failure',item['id'],mode,repr(exc),flush=True)
            io.write(out/'pricing_results.json',rows)
    io.write(out/'pricing_complete.json',dict(status='complete',results=io.ref(out/'pricing_results.json'),failures=sum(r['status']!='complete' for r in rows)))


def confirm(out):
    require((out/'pricing_complete.json').exists(),'Finish all planning first');directory=out/'confirmation'
    require(not directory.exists(),'Preserve all confirmation samples');directory.mkdir()
    protocol=io.read(out/'pricing_protocol.json');rows=[];seal={};policies=[]
    results=io.read(out/'pricing_results.json')
    for item in protocol['targets']:
        data=load_target(item,protocol['gain_required']);o=data['original']
        controls={'no_dr':np.zeros_like(o.scale)}
        seedpath=out/'screen'/(item['id']+f"__target{item['target']}")/'linear_response/quote.npz'
        with np.load(seedpath,allow_pickle=False) as a:controls['linear_QP_replayed']=o.nominal(a['x'])
        for r in results:
            if r['id']==item['id'] and r['target']==item['target'] and r['status']=='complete' and r['result']['original_qualified']:
                mode=r['result']['mode'];path=out/'pricing'/(item['id']+f"__target{item['target']}")/mode/'original_quote.npz'
                with np.load(path,allow_pickle=False) as a:controls[mode+'_ADMM_replayed']=a['u'].copy()
        cost=o.fees(controls['linear_QP_replayed']);controls['rotation_same_QP_fee'],matching=rotation(o,cost)
        for name,u in controls.items():
            require(o.audit(u)['passed'],'Confirmed policy violates original device domain')
            path=directory/(item['id']+f"__target{item['target']}__{name}.npz");np.savez_compressed(path,u=u)
            seal[str(path)]=sha(path)
        policies.append(dict(item=item,controls=list(controls),rotation_matching=matching))
    frozen=dict(created_utc=utc(),policies=policies,policies_sha256=seal,protocol=io.ref(out/'pricing_protocol.json'),
        planning=io.ref(out/'pricing_results.json'),source=io.ref(__file__),new_model_paths_per_target=protocol['model_paths'],new_AC_paths=0)
    io.write(directory/'protocol.json',frozen)
    # Only now open fresh paths, shared across policies within each conditional target.
    truth={r['id']:r for r in io.read(REFERENCE/'evaluation.json')}
    for entry in policies:
        item=entry['item'];data=load_target(item,protocol['gain_required']);score=data['score'];H=data['H'];o=data['original']
        rng=stream(protocol['model_namespace'],item['id'],item['target'],protocol['model_seed'])
        count=protocol['model_paths'];power=ForecastLaw(data['forecast']).sample(rng,count)
        error=np.sqrt(score['sigma2'])*rng.normal(size=(count,96,H.shape[1]));history=np.array([m['log_probability'] for m in score['models']])
        baseline=log_odds_samples(H,history,item['target'],power,error,score['sigma2']);saved={}
        for name in entry['controls']:
            path=directory/(item['id']+f"__target{item['target']}__{name}.npz");require(sha(path)==seal[str(path)],'Policy changed')
            with np.load(path,allow_pickle=False) as a:u=a['u']
            dp,dq=o.aggregate(u);shift=np.column_stack((dp,dq))/data['base']
            odds=log_odds_samples(H,history,item['target'],power+shift[None],error,score['sigma2'])
            probability=np.exp(-logsumexp(-odds,axis=1));gains=odds[:,item['competitors']]-baseline[:,item['competitors']]
            # Independent direct likelihood on first 16 paths.
            y=(power[:16]+shift[None])@H[item['target']].T+error[:16]
            loss=np.column_stack([np.sum((y-(power[:16]+shift[None])@h.T)**2,axis=(1,2)) for h in H])
            ll=history[None]-loss/(2*score['sigma2']);expected=np.exp(ll[:,item['target']]-logsumexp(ll,axis=1))
            require(np.max(abs(expected-probability[:16]))<1e-8,'Direct posterior mismatch')
            rows.append(dict(id=item['id'],target=item['target'],case=item['case'],strategy=name,
                target_correct=score['models'][item['target']]['topology_id']==truth[item['id']]['truth_id'],
                paths=count,mean_posterior=float(probability.mean()),pass_rates={str(b):float(np.mean(probability>=b)) for b in BETAS},
                analytic_gains=data['evidence'].gains(u),paired_gains=[means_interval(g) for g in gains.T],
                normalized_fee=o.fees(u)/data['fee_scale']))
            saved[name+'_posterior']=probability;saved[name+'_odds']=odds
        np.savez_compressed(directory/(item['id']+f"__target{item['target']}_samples.npz"),**saved)
        io.write(directory/'results.json',rows)
    io.write(directory/'complete.json',dict(status='passed',conditional_targets=len(policies),new_AC_paths=0,results_sha256=sha(directory/'results.json'),direct_likelihood_checks=16*len(rows)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--stage',choices=['plan','confirm'],required=True)
    a=p.parse_args();(plan if a.stage=='plan' else confirm)(a.run.resolve())
