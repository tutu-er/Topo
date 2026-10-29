from pathlib import Path
from dataclasses import replace
from datetime import datetime,timezone
import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
import sys
import traceback
import numpy as np
from research_experiments.wzzt_systematic.common import read,write,sha
from .common import jobs,MAIN,AC_METHODS,noise_errors,indices
from .networks import register

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'outputs/wzzt_fine_20261028_full'


def settings(condition,sigma):
    return dict(suite='tap_step' if condition=='tap_step' else 'reference',
        v_sigma=sigma,pq_sigma=0. if condition=='ideal_ac' else .02 if condition=='high_pq_noise' else .005,
        root_meter_sigma=.001 if condition=='root_noise' else sigma,
        load_factor=1.5 if condition=='higher_load' else .5 if condition=='light_load' else 1.,
        low_xr=condition=='low_xr',excitation_mix=.9 if condition=='weak_excitation' else 0.,
        ar1=.8 if condition=='ar1_errors' else 0.,outlier_fraction=.01 if condition=='outliers' else 0.,outlier_size=8.)


def measurements(sim,context,base,p,q,v,config):
    data=sim._observe_terminal_channels(context,base,p,q,v,pq_noise_rel=config['pq_sigma'],v_noise_rel=config['v_sigma'])
    result={k:data[name].to_numpy() for k,name in zip(('P','Q','V','root'),('p_measured','q_measured','voltage_measured','root_observed'))}
    if config['ar1'] or config['outlier_fraction']:
        for i,(key,true,sigma) in enumerate(zip(('P','Q','V','root'),(p,q,v,base.root_true),
            (config['pq_sigma'],config['pq_sigma'],config['v_sigma'],config['root_meter_sigma']))):
            true=true.to_numpy();scale=sigma*np.maximum(abs(true),1e-12);z=(result[key]-true)/scale
            if config['ar1']:z=noise_errors(z,config['ar1'])
            if config['outlier_fraction']:
                rng=np.random.default_rng(base.shifted_seed+70001+1000*i)
                mask=rng.random(z.shape)<config['outlier_fraction'];z=z+mask*rng.choice([-1.,1.],z.shape)*config['outlier_size']
            result[key]=true+scale*z
    return result


def prepare(output,only=None):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    frozen=read(PARENT/'manifest.json')
    for name,h in frozen.items():
        if name.startswith('source/') and sha(PARENT/name)!=h:raise ValueError('parent producer changed: '+name)
    shutil.copytree(PARENT/'source',output/'source',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(Path(__file__).parent,output/'source/research_experiments/wzzt_paper',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for p in (ROOT/'terminal_case33').rglob('*.py'):
        to=output/'source'/p.relative_to(ROOT);to.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,to)
    sys.path.insert(0,str(output/'source'))
    from rnj_wzzt.scenario import simulation as sim
    rows=jobs();rows=[p for p in rows if only is None or any(s in p['job_id'] for s in only)]
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),jobs=[r['job_id'] for r in rows],planned=len(rows),
        planned_full_inputs=1204,full_matrix=len(rows)==1204,methods=MAIN,AC_methods=AC_METHODS,
        planned_model_rows=sum(len(p['methods']) for p in rows),planned_AC_rows=3*sum(p['ac_planned'] for p in rows),
        panels={'main':'4 fixed networks x 10 independent bundles x 4 V-noise levels x 4 sample counts = 640',
                'stress':'4 fixed networks x 5 bundles x 10 predeclared conditions x N48/288 = 400',
                'topology':'12 unseen random trees (3 each n8/16/24/32) x 3 bundles x 2 V-noise levels x N48/288 = 144',
                'case33':'hybrid-leaf modified case33bw x 5 bundles x 2 V-noise levels x N48/288 = 20'},
        seed_policy='bundle b: train replicate 100+10*b; validation 101+10*b; test 102+10*b; previous data replicates excluded',
        independent_units='10 bundle blocks for main; 5 stress bundles; 12 random topology blocks; 5 case33 bundles; conditions within blocks paired',
        physical_days=3,physical_points_per_day=96,validation_points=288,test_points=288,
        samples_per_day={N:indices(N).tolist() for N in (24,48,96,288)},
        shared_noise='same physical days and Gaussian standard draws across meter levels; nested N sets',
        noise_zero_label='main V sigma 0 retains PQ sigma .005; stress ideal_ac sets every meter sigma to 0',
        budgets=dict(workers=4,solver_threads=1,worker_hard_seconds=900,fixed_LP_seconds=15,path_seconds=25,long_path_seconds=100),
        frozen_parameters=dict(from_strategy='20261028 ridge_union',bounds=[0,2],validation_fraction=.03,resolution=.16,
            ridge_strengths=[.001,.01],training_gain_absolute=1e-10,training_gain_relative=.001,max_extra_starts=2),
        guard='one backward round, only proposals with validation MAE <= current pre-contraction chosen MAE; original final selection unchanged',
        lambda_ablation='main bundles 0..2, sigma>0, N24/96; fresh independent LP per extension',
        budget_ablation='main bundles 0..4, sigma .001, N96/288; 25 vs100 from same multistart chosen model; same union LP columns, domain restricted',
        pair_ablation='main bundle0, sigma .0005/.001, N48/288; width5 two-step after ridge and optionally guard0',
        AC_scope='NJ/multistart/ridge_union; main sigma .0002/.001 N48/288 all bundles; other panels bundle0 all conditions',
        runtime_scope='standalone replays separate from shared experiment timings; no truth/test access',
        statistics=dict(primary_panel='main',primary_contrasts=['ridge_union vs NJ','ridge_union vs multistart','ridge_union vs restart_control'],
            resampling='paired 10 bundle means, 20000 percentile bootstrap resamples, seed20261029; exact two-sided sign tests; Holm for three primary contrasts',
            topology_resampling='12 random-topology means; conditions/data repeats nested within topology',
            interpretation='conditional simulation evidence; intervals describe fixed generator/data regimes, not universal topology correctness'),
        data_domains='balanced radial AC, zero hidden injection, synchronous terminal P/Q/V and measured root; no field-data or unbalanced-three-phase claim',
        truth_tuning=False,all_failures_retained=True,
        environment=dict(python=sys.version,platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ('numpy','scipy','pandas','gurobipy','matplotlib')}))
    write(output/'protocol.json',protocol)
    datasets={}
    for p in rows:datasets.setdefault((p['case'],p['bundle'],p['condition'],p['voltage_sigma']),[]).append(p)
    contexts={};physical={};data_errors=[]
    for (case,bundle,condition,sigma),selected in datasets.items():
        config=settings(condition,sigma);key=(case,condition)
        if key not in contexts:
            register(case,sim)
            ctx=sim._prepare_case_context(case,96,scenario_suite=config['suite'],root_observation='noisy',root_meter_noise_rel=config['root_meter_sigma'],root_sigma=.003,impedance_scale=1.)
            if case=='case33_terminal':
                # The shared profile builder indexes resources by original bus,
                # while generic case-bank assignments are keyed by meter bus.
                ctx.assignment.loc[:,'bus_id']=ctx.assignment.bus_id.map(ctx.net.terminal_to_original)
                if ctx.assignment.bus_id.isna().any():raise ValueError('case33 resource mapping incomplete')
            if config['low_xr']:
                service=ctx.net.branches.branch_type.str.contains('service',case=False)
                ctx.net.branches.loc[:,'x_ohm']=ctx.net.branches.r_ohm*np.where(service,.06,.30)
            network=output/'networks'/case/condition;network.mkdir(parents=True)
            ctx.net.buses.to_csv(network/'buses.csv',index=False);ctx.net.branches.to_csv(network/'branches.csv',index=False)
            write(network/'metadata.json',dict(root_bus=int(ctx.net.root_bus),base_kv=float(ctx.net.base_kv),base_mva=float(ctx.net.base_mva),source=ctx.net.metadata))
            contexts[key]=ctx
        ctx=replace(contexts[key],settings=dict(contexts[key].settings,root_meter_noise_rel=config['root_meter_sigma']))
        folder=output/'data'/case/f'b{bundle}'/condition/f'v{round(sigma*10000):04d}';folder.mkdir(parents=True)
        error=None;diagnostics=[]
        try:
            for split,replicate in (('train',100+10*bundle),('validation',101+10*bundle),('test',102+10*bundle)):
                cache_key=(case,bundle,condition,split)
                if cache_key not in physical:
                    days=[]
                    for day in range(3):
                        base=sim._build_base_trajectory(ctx,day,replicate)
                        base=replace(base,p_base=base.p_base*config['load_factor'],q_base=base.q_base*config['load_factor'])
                        if config['excitation_mix']:
                            def mix(f):
                                means=f.mean();common=(f/np.maximum(abs(means),1e-12)).mean(axis=1)
                                return (1-config['excitation_mix'])*f+config['excitation_mix']*np.outer(common,means)
                            base=replace(base,p_base=mix(base.p_base),q_base=mix(base.q_base))
                        ac,v=sim._solve_physical_ac(ctx,base.p_base,base.q_base,base.root_true)
                        days.append((base,base.p_base.loc[:,ctx.terminals],base.q_base.loc[:,ctx.terminals],v,ac))
                    physical[cache_key]=days
                truth={k:[] for k in ('P','Q','V','root')};measured={k:[] for k in truth}
                for day,(base,p,q,v,ac) in enumerate(physical[cache_key]):
                    obs=measurements(sim,ctx,base,p,q,v,config)
                    for k,value in zip(truth,(p,q,v,base.root_true)):truth[k].append(value.to_numpy());measured[k].append(obs[k])
                    drop=base.root_true.to_numpy()[:,None]-v.to_numpy()
                    design=np.c_[p.to_numpy(),q.to_numpy()];centered=design-design.mean(axis=0)
                    signal=base.root_true.to_numpy()[:,None]**2-v.to_numpy()**2
                    noisy=obs['root'][:,None]**2-obs['V']**2
                    diagnostics.append(dict(split=split,day=day,replicate=replicate,seed=base.shifted_seed,profile=base.profile,
                        converged=int(ac['converged'].sum()),voltage_min=float(v.to_numpy().min()),voltage_max=float(v.to_numpy().max()),
                        mean_drop=float(drop.mean()),max_drop=float(drop.max()),min_drop=float(drop.min()),
                        design_rank=int(np.linalg.matrix_rank(centered)),design_columns=design.shape[1],condition_number=float(np.linalg.cond(centered)),
                        dynamic_signal_rms=float(np.sqrt(np.mean((signal-signal.mean(axis=0))**2))),
                        dynamic_noise_rms=float(np.sqrt(np.mean(((noisy-signal)-(noisy-signal).mean(axis=0))**2))),
                        relative_noise_std={k:float(np.std((obs[k]-value.to_numpy())/np.maximum(abs(value.to_numpy()),1e-12))) for k,value in zip(truth,(p,q,v,base.root_true))}))
                np.savez_compressed(folder/f'{split}_truth.npz',labels=ctx.terminals,**{k:np.stack(v) for k,v in truth.items()})
                np.savez_compressed(folder/f'{split}_observed.npz',labels=ctx.terminals,**{k:np.stack(v) for k,v in measured.items()})
        except Exception as e:
            error=f'{type(e).__name__}: {e}';data_errors.append(dict(case=case,bundle=bundle,condition=condition,sigma=sigma,error=error,traceback=traceback.format_exc()))
        write(folder/'diagnostics.json',dict(config=config,records=diagnostics,data_error=error))
        for p in selected:
            d=output/'jobs'/p['job_id'];d.mkdir(parents=True)
            payload=dict(p,pq_sigma=config['pq_sigma'],config=config,truth_network=str(output/'networks'/case/condition),data_error=error)
            for split in ('train','validation','test'):payload[split]=str(folder/f'{split}_observed.npz');payload[split+'_truth']=str(folder/f'{split}_truth.npz')
            write(d/'payload.json',payload)
        print('PREPARED',case,bundle,condition,sigma,error or 'OK',flush=True)
    write(output/'data_errors.json',data_errors)
    write(output/'manifest.json',{p.relative_to(output).as_posix():sha(p) for p in output.rglob('*') if p.is_file()})
    print('FROZEN',len(rows),'inputs',protocol['planned_model_rows'],'model rows','data errors',len(data_errors),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');p.add_argument('--only',nargs='+');a=p.parse_args();prepare(a.output,a.only)
