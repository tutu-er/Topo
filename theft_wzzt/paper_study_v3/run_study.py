"""Fresh, frozen follow-up study. Three processes handle disjoint dataset files."""
import argparse
import collections
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy.stats import binom
from paper_study.pilot import entry,ROOT
from paper_study.run_study import tree_path,CASES
from paper_study.profile import fixed_profile
from paper_study.lp_profile import lp_profile
from paper_study_v2.regularized import regularized_profile
from theft_wzzt.data.paper_style_case_bank import CASE_BUILDERS
from theft_wzzt.theft.identified_tree import load_identified,to_theft_tree,region_label
from theft_wzzt.theft.theft_simulation import TheftSpec,simulate_theft_scenarios
from theft_wzzt.theft.loss_models import estimate_loss
from theft_wzzt.theft.theft_model import data_from_scenario,TheftData
OUT=ROOT/'outputs/theft_paper_v3'
OUT.mkdir(exist_ok=True)
CONDITIONS=('nominal','drift10','hetero10','loss_under25')


def jobs():
    for case in CASES:
        buses=[int(b) for b in CASE_BUILDERS[case]().buses.query("bus_type != 'root'").bus_id]
        terminal={'paper15':108,'soumalas11':108,'flynn16':208}[case]
        for condition in (CONDITIONS if case=='paper15' else ('nominal',)):
            for role,seeds in [('null_cal',range(20000,20199) if condition=='nominal' else range(20200,20399)),('null_test',range(21000,21100))]:
                for rep in seeds:yield dict(case=case,condition=condition,role=role,replicate=rep,bus=None,amp_kw=0,start=45)
            for bus in [2,terminal]:
                for amp in ([2,4,8] if condition=='nominal' else [8]):
                    for rep in range(24000,24020):yield dict(case=case,condition=condition,role='grid_test',replicate=rep,bus=bus,amp_kw=amp,start=45)
            if condition!='nominal':continue
            for role,seeds in [('event_cal',range(22000,22199)),('event_test',range(23000,23100))]:
                for rep in seeds:
                    rng=np.random.default_rng(900000+rep+10000*CASES.index(case))
                    yield dict(case=case,condition=condition,role=role,replicate=rep,bus=int(rng.choice(buses)),amp_kw=int(rng.choice([2,4,8])),start=int(rng.integers(36,55)))


def freeze():
    files=sorted((ROOT/'outputs/theft_paper_v2').glob('regdev_*.json'))
    if len(files)!=90:raise RuntimeError('Require all 90 new regularization development datasets before choosing strength')
    groups=collections.defaultdict(list)
    for f in files:
        d=json.loads(f.read_text())
        for r in d['results']:groups[r['strength']].append(r['correct'])
    strength=max(groups,key=lambda k:(sum(groups[k]),-k))
    development={str(k):dict(correct=sum(v),n=len(v)) for k,v in groups.items()}
    protocol=dict(version=3,created_utc=datetime.now(timezone.utc).isoformat(),window=[32,64],strength=strength,
        selection_rule='max total point accuracy on 90 new balanced development datasets; tie choose smaller penalty',development=development,
        primary='shrinkage',prior='strength * sum(abs(w-rho*w0)/max(r0,x0,1e-6)) with common rho in[0.5,2]',
        detection='gain strictly greater than the maximum of all 199 calibration scores in every declared null bank; ties conservative; no true-condition selection',
        location_and_amplitude='maximum of 199 independent labelled calibration scores; marginal coverage >=99.5% under exchangeability; targets >=95% distribution content with confidence >=1-.95**199 under iid',
        set_alpha=.05,calibration_failure_probability=.95**199,order_statistic=199,
        scope='single stable intermittent source; three synthetic case-bank networks; per-network marginal mixture; no guarantee for unknown shifts or conditional on detection',
        independence='All V3 seeds differ from V1 and V2. Model and strength10 are unchanged. V2 calibration failures motivated the fixed maximum-score rule. No V3 outcomes used in design.',jobs=list(jobs()))
    source=[ROOT/'paper_study_v2/regularized.py',ROOT/'paper_study_v3/run_study.py',ROOT/'paper_study/profile.py',ROOT/'paper_study/lp_profile.py',ROOT/'detect_simple.py']+[tree_path(c) for c in CASES]
    protocol['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source}
    path=OUT/'frozen_protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['source_sha256']!=protocol['source_sha256'] or old['jobs']!=protocol['jobs'] or old['strength']!=strength:raise RuntimeError('V2 protocol changed')
        return old
    path.write_text(json.dumps(protocol,indent=2),encoding='utf-8');return protocol


def run_one(job,identified,strength):
    started=perf_counter();amp=np.zeros(96)
    if job['bus'] is not None:amp[job['start']:job['start']+6]=job['amp_kw']
    thefts=None if job['bus'] is None else (TheftSpec(job['bus'],tuple(amp)),)
    condition=job['condition'];options={'impedance_scale':1.1} if condition=='drift10' else {}
    if condition=='hetero10':
        def modify(net):
            rng=np.random.default_rng(job['replicate']+500000);f=rng.uniform(.9,1.1,len(net.branches))
            net.branches.loc[:,['r_ohm','x_ohm']]=net.branches[['r_ohm','x_ohm']].to_numpy()*f[:,None]
            return net
        options['case_modifier']=modify
    net,s=simulate_theft_scenarios(job['case'],thefts,replicate=job['replicate'],**options)
    measured={k:s[k] for k in entry.MATRIX_CHANNELS+entry.SERIES_CHANNELS+('scenario_settings',)}
    tree,data,bounds,envelope,signature=entry.prepare(measured,identified,(32,64));weights=to_theft_tree(identified)[1]
    if condition=='loss_under25':
        lp,lq=estimate_loss(measured,weights,tree,model='L1');full=data_from_scenario(measured,loss_p_estimate=.75*lp,loss_q_estimate=.75*lq)
        data=TheftData(*(getattr(full,k)[32:64] for k in ('p','q','y','balance','amplitude','extra_q')),full.voltage_scale,full.balance_scale)
    results=[regularized_profile(tree,data,weights,bounds,strength=strength),fixed_profile(tree,data,weights,stable=True)]
    truth=None if job['bus'] is None else region_label(job['bus'],net,identified)
    atrue=amp[32:64]/(net.base_mva*1000);active=atrue>0
    for fit in results:
        minimum=min(r['loss'] for r in fit['profile']);fit['profile_candidate_min']=minimum
        fit['truth_profile_gap']=None if truth is None else next((r['loss']-minimum for r in fit['profile'] if r['location']==truth),None)
        fit['point_correct']=None if truth is None else fit['best_location']==truth
        fit['active_time_correct']=None if not active.any() else float(np.mean([fit['locations'][i]==truth for i in np.flatnonzero(active)]))
    mae=float(np.mean(np.abs(data.amplitude[active]-atrue[active]))) if active.any() else None
    return dict(**job,truth_region=truth,protocol_signature=signature,base_mva=net.base_mva,
        amplitude_input_pu=data.amplitude.tolist(),true_amplitude_pu=atrue.tolist(),amplitude_max_abs_error_pu=float(np.max(np.abs(data.amplitude-atrue))),
        amplitude_active_mae_pu=mae,amplitude_active_relative_mae=mae/atrue[active].mean() if active.any() else None,
        old_loss_only_envelope_active_coverage=float(np.mean(((envelope['lower'].to_numpy()<=atrue)&(atrue<=envelope['upper'].to_numpy()))[active])) if active.any() else None,
        balance_only=float(np.maximum(data.balance,0).sum()/data.balance_scale),voltage_gain=results[0]['gain_voltage'],results=results,seconds=perf_counter()-started)


def worker(payload):
    worker_id,joblist,strength=payload
    trees={c:load_identified(tree_path(c)) for c in CASES};dest=OUT/'rows';dest.mkdir(exist_ok=True)
    failures=0
    for i,job in enumerate(joblist):
        name=f"{job['case']}_{job['condition']}_{job['role']}_{job['replicate']}_{job['bus']}_{job['amp_kw']}"
        target=dest/f'{name}.json'
        if target.exists():continue
        try:result=run_one(job,trees[job['case']],strength)
        except Exception as error:result=dict(**job,error=repr(error));failures+=1
        target.write_text(json.dumps(result,indent=1),encoding='utf-8')
        if i%20==0 or 'error' in result:print(worker_id,i+1,len(joblist),name,result.get('error','ok'),flush=True)
    return dict(worker=worker_id,failures=failures)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=3);args=parser.parse_args()
    protocol=freeze();n=args.workers
    if not 1<=n<=3:raise ValueError('1..3 workers supported')
    payloads=[(i,protocol['jobs'][i::n],protocol['strength']) for i in range(n)]
    print('FROZEN strength',protocol['strength'],'order statistic',protocol['order_statistic'],'datasets',len(protocol['jobs']),flush=True)
    if n==1:print(worker(payloads[0]))
    else:
        with ProcessPoolExecutor(max_workers=n) as pool:
            for result in pool.map(worker,payloads):print(result,flush=True)
if __name__=='__main__':main()
