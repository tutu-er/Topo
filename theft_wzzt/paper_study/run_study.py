"""Frozen-protocol simulation study; one JSON checkpoint per generated dataset.
Truth is read exclusively for evaluation after measured-only inference.
"""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from paper_study.pilot import entry, ROOT, OUT
from paper_study.profile import fixed_profile
from paper_study.lp_profile import lp_profile
from theft_wzzt.data.paper_style_case_bank import CASE_BUILDERS
from theft_wzzt.theft.identified_tree import load_identified,to_theft_tree,region_label
from theft_wzzt.theft.theft_simulation import TheftSpec,simulate_theft_scenarios
from theft_wzzt.theft.loss_models import estimate_loss
from theft_wzzt.theft.theft_model import data_from_scenario,TheftData

CASES=('paper15','soumalas11','flynn16')
CONDITIONS=('nominal','drift10','loss_under25','noise2')
WINDOW=(32,64)


def tree_path(case):
    return ROOT/'outputs/theft/identified_tree.json' if case=='paper15' else OUT/f'identified_{case}.json'


def jobs():
    for case in CASES:
        buses=[int(b) for b in CASE_BUILDERS[case]().buses.query("bus_type != 'root'").bus_id]
        terminal={'paper15':108,'soumalas11':108,'flynn16':208}[case]
        for condition in (CONDITIONS if case=='paper15' else ('nominal',)):
            for role,seeds in [('null_cal',range(1000,1199) if condition=='nominal' else range(1200,1299)),('null_test',range(2000,2100))]:
                for rep in seeds:
                    yield dict(case=case,condition=condition,role=role,replicate=rep,bus=None,amp_kw=0,start=45)
            for bus in [2,terminal]:
                for amp in ([2,4,8] if condition=='nominal' else [8]):
                    for rep in range(3000,3020):
                        yield dict(case=case,condition=condition,role='grid_test',replicate=rep,bus=bus,amp_kw=amp,start=45)
            if condition!='nominal':continue
            for role,seeds in [('event_cal',range(4000,4199)),('event_test',range(5000,5100))]:
                for rep in seeds:
                    rng=np.random.default_rng(900000+rep+10000*CASES.index(case))
                    yield dict(case=case,condition=condition,role=role,replicate=rep,bus=int(rng.choice(buses)),amp_kw=int(rng.choice([2,4,8])),start=int(rng.integers(36,55)))


def freeze():
    protocol=dict(version=1,created_utc=datetime.now(timezone.utc).isoformat(),window=WINDOW,
        primary_method='stable_positive_lp',activity='u=1[fixed L1 P amplitude > 0], same location throughout the window',
        alternatives=['fixed_stable','fixed_free','balance_only','voltage_gain'],
        decision='empirical rank <=0.05; each method and condition calibrated separately; also report nominal-calibration transfer to stress',
        localization='candidate profile loss minus minimum candidate loss; 190th order statistic of 199 labelled simulation calibration scores; report all candidates within cutoff',
        amplitude='window maximum absolute amplitude error; 190th order statistic of 199 labelled simulation calibration scores; simultaneous across 32 samples',
        exchangeability='claims only marginal under matching frozen simulation law; no field-data guarantee; grid subgroup coverage is descriptive',
        selection_evidence='development rep200..204 only: 32-point refit 26/30 nominal; 10/10 drift10 versus fixed-RX 0/10; retain fixed-RX baseline',
        scope='single stationary positive source within window; generated balanced AC scenarios; synthetic case-bank networks, not official feeder reproductions',
        jobs=list(jobs()))
    files=list(p for p in (ROOT/'paper_study').glob('*.py') if not p.name.startswith('test_'))+[ROOT/'detect_simple.py']+[tree_path(c) for c in CASES]
    protocol['source_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    path=OUT/'frozen_protocol.json'
    if path.exists():
        old=json.loads(path.read_text())
        if old['source_sha256']!=protocol['source_sha256'] or old['jobs']!=protocol['jobs']:
            raise RuntimeError('Frozen protocol or source changed; create a separate study rather than overwriting it')
        return old
    path.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    return protocol


def run_one(job,identified):
    started=perf_counter();amp=np.zeros(96)
    if job['bus'] is not None:amp[job['start']:job['start']+6]=job['amp_kw']
    thefts=None if job['bus'] is None else (TheftSpec(job['bus'],tuple(amp)),)
    condition=job['condition']
    kwargs={'impedance_scale':1.1} if condition=='drift10' else {}
    if condition=='noise2':kwargs.update(pq_noise_rel=.01,v_noise_rel=.0004,master_noise_rel=.004)
    net,scenario=simulate_theft_scenarios(job['case'],thefts,replicate=job['replicate'],**kwargs)
    # Strip all truth channels before any inference.
    measured={k:scenario[k] for k in entry.MATRIX_CHANNELS+entry.SERIES_CHANNELS+('scenario_settings',)}
    tree,data,bounds,envelope,signature=entry.prepare(measured,identified,WINDOW)
    weights=to_theft_tree(identified)[1]
    if condition=='loss_under25':
        lp,lq=estimate_loss(measured,weights,tree,model='L1')
        full=data_from_scenario(measured,loss_p_estimate=.75*lp,loss_q_estimate=.75*lq)
        data=TheftData(*(getattr(full,k)[32:64] for k in ('p','q','y','balance','amplitude','extra_q')),full.voltage_scale,full.balance_scale)
    results=[lp_profile(tree,data,bounds),fixed_profile(tree,data,weights,stable=True),fixed_profile(tree,data,weights,stable=False)]
    # Evaluation only from this point onward.
    truth=None if job['bus'] is None else region_label(job['bus'],net,identified)
    true_amplitude=amp[32:64]/(net.base_mva*1000)
    active=true_amplitude>0
    for r in results:
        minimum=min(row['loss'] for row in r['profile'])
        r['profile_candidate_min']=minimum
        r['truth_profile_gap']=None if truth is None else next((row['loss']-minimum for row in r['profile'] if row['location']==truth),None)
        r['point_correct']=None if truth is None else r['best_location']==truth
        r['active_time_correct']=None if not active.any() else float(np.mean([r['locations'][i]==truth for i in np.flatnonzero(active)]))
    amplitude_mae=float(np.abs(data.amplitude[active]-true_amplitude[active]).mean()) if active.any() else None
    amp_envelope_covered=(envelope['lower'].to_numpy()<=true_amplitude)&(true_amplitude<=envelope['upper'].to_numpy())
    return dict(**job,truth_region=truth,protocol_signature=signature,base_mva=net.base_mva,
        amplitude_max_abs_error_pu=float(np.max(np.abs(data.amplitude-true_amplitude))),
        amplitude_active_mae_pu=amplitude_mae,
        amplitude_active_relative_mae=amplitude_mae/true_amplitude[active].mean() if active.any() else None,
        old_loss_only_envelope_active_coverage=float(amp_envelope_covered[active].mean()) if active.any() else None,
        amplitude_input_pu=data.amplitude.tolist(),true_amplitude_pu=true_amplitude.tolist(),
        balance_only=float(np.maximum(data.balance,0).sum()/data.balance_scale),
        voltage_gain=results[0]['gain_voltage'],results=results,seconds=perf_counter()-started)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--case',choices=CASES);parser.add_argument('--condition',choices=CONDITIONS)
    args=parser.parse_args();protocol=freeze()
    rows=OUT/'rows';rows.mkdir(exist_ok=True)
    trees={case:load_identified(tree_path(case)) for case in CASES}
    for i,job in enumerate(protocol['jobs']):
        if args.case and job['case']!=args.case:continue
        if args.condition and job['condition']!=args.condition:continue
        key=f"{job['case']}_{job['condition']}_{job['role']}_{job['replicate']}_{job['bus']}_{job['amp_kw']}"
        target=rows/f'{key}.json'
        if target.exists():continue
        try:
            result=run_one(job,trees[job['case']])
        except Exception as error:
            result=dict(**job,error=repr(error))
        target.write_text(json.dumps(result,indent=1),encoding='utf-8')
        if i%20==0 or 'error' in result:print(i+1,len(protocol['jobs']),key,round(result.get('seconds',0),2),result.get('error','ok'),flush=True)
if __name__=='__main__':main()
