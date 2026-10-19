"""Mocked full data flow. No project import, actual H, AC, or fit is run."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict
from datetime import datetime,timezone
import numpy as np
import pytest
try:
    from . import paired_runner as r
    from . import paired_contracts as c
    from .test_paired_contracts import config,manual_plan,first_fits
except ImportError:
    import paired_runner as r
    import paired_contracts as c
    from test_paired_contracts import config,manual_plan,first_fits


@pytest.fixture(autouse=True)
def authorize_only_pytest_temporary_outputs(tmp_path, monkeypatch):
    """Mock tests may write their isolated tmp_path; production guard stays intact."""
    production_guard = r.ensure_staging
    def guard(path):
        resolved = Path(path).resolve()
        if resolved.is_relative_to(tmp_path.resolve()):
            return resolved
        return production_guard(path)
    monkeypatch.setattr(r, 'ensure_staging', guard)


class FakeProject:
    def __init__(self,first_failure=False,final_failure=False,mismatch=False):
        self.original=dict(calibration=dict(source_sha256={'mock':'not-real-code'},runtime=r.runtime(),
            policy={'frozen':'mock'},initial_ev_config=config(),population={'mock_population':True}))
        self.first_failure=first_failure;self.final_failure=final_failure;self.mismatch=mismatch
        self.fresh_calls=[];self.fit_calls=[];self.observe_calls=[]
    def original_source_hashes(self):return {'mock':'not-real-code'}
    def path(self,seed,policy):
        # These are deterministic test arrays, not generated physical history.
        p=.12+np.arange(60,dtype=float).reshape(20,3)*1e-5
        return SimpleNamespace(p=p,q=np.full((20,3),.04),noise=np.arange(20,dtype=float)*1e-6,
                               r=np.full(4,.012),x=np.full(4,.006)),42,0,'strong'
    def action(self,name,u,sites):return dict(name=name,p=np.asarray(u)/20.,q=np.zeros_like(u),sites=sites)
    def observe(self,path,rows,action=None):
        self.observe_calls.append((rows.start,rows.stop,None if action is None else action['name']))
        p=path.p[rows].copy();q=path.q[rows].copy()
        if action is not None:p+=action['p'];q+=action['q']
        n=len(p);v=.995-.01*p;noise=path.noise[rows]
        obs=SimpleNamespace(p=p+noise[:,None],q=q-noise[:,None],v=v+noise[:,None],
            master_p=p.sum(axis=1)+.1+noise,master_q=q.sum(axis=1)+.035-noise,root_v=np.ones(n))
        return obs,dict(v=v)
    def concat(self,h,y):return SimpleNamespace(**{k:np.concatenate([getattr(h,k),getattr(y,k)]) for k in r.CHANNELS})
    def validate_operations(self,pair,configuration,trace):
        a=c.operation_audit(pair['openloop'],configuration);b=c.operation_audit(pair['rolling'],configuration)
        assert a['prefix']==b['prefix']
        return dict(mock_operation_validation=True,prefix=a['prefix'])
    def fresh_trace(self,seed,phase,policy,folder):
        self.fresh_calls.append(seed);folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        first=manual_plan();actual=manual_plan(True);path,_,_,_=self.path(seed,policy)
        H,_=self.observe(path,slice(0,12))
        op=c.operation_audit(actual,config())
        Y1,_=self.observe(path,slice(12,14),self.action('old_first',np.asarray(actual['u_kw'])[:2],op['sites'][:2]))
        Y2,_=self.observe(path,slice(14,20),self.action('old_tail',np.asarray(actual['u_kw'])[2:],op['sites'][2:]))
        H14=self.concat(H,Y1);H20=self.concat(H14,Y2)
        fitrows=first_fits()
        if self.first_failure:fitrows[1]['success']=False
        firstbank=dict(round=1,losses=[1.,2.,3.,4.],success=[f['success'] for f in fitrows],fits=fitrows)
        trace=dict(seed=seed,history_id=str(seed),case='strong',true_index=0,truth_topology=0,policy_seed=42,
            selections=[dict(actual_plan=first,status='complete'),dict(status='complete')],realized_plan=actual,
            records=[firstbank,dict(round=2)],seconds=.1,execution_failure=False,
            selection={'selector_total_seconds':.08},ac_call_metrics={'whole_trial_ac_calls':10})
        arrays={}
        for label,obs in [('history',H),('round_1_observed',Y1),('round_2_observed',Y2),('round_1_joint',H14),('round_2_joint',H20)]:
            arrays.update({label+'_'+key:value for key,value in r.observation_arrays(obs).items()})
        if self.mismatch:arrays['round_2_joint_v']=arrays['round_2_joint_v']+.001
        arrays['realized_u_kw']=np.asarray(actual['u_kw'])
        archive=folder/f'rolling_{seed}_observations.npz';np.savez_compressed(archive,**arrays)
        r.dump(folder/f'trace_{seed}.json',trace)
        return trace,archive
    def fit_bank(self,H20,initial,seed):
        assert all(set(v)=={'r','x'} for v in initial.values())
        assert H20.p.shape==(20,3)
        self.fit_calls.append(dict(seed=seed,initial=deepcopy(initial),observations=H20))
        loss=float(np.sum(H20.p**2));rows=first_fits()
        success=[True]*4
        if self.final_failure and len(self.fit_calls)==1:success[2]=False
        for j,f in enumerate(rows):f['loss']=loss+j;f['success']=success[j]
        latent={n:dict(p=H20.p.copy(),q=H20.q.copy(),source_p=np.full(20,.1),source_q=np.full(20,.035)) for n in c.FAMILIES}
        return dict(losses=[loss+j for j in range(4)],success=success,fits=rows,fit_failures={},latent=latent,
                    seconds=.02,ac_calls=5,counter_includes_failed_entries=True)


def manifest(tmp_path,project):
    return r.freeze(tmp_path/'frozen.json',project)


def test_mocked_fresh_pipeline_common_noise_prefix_seed_and_no_reselection(tmp_path):
    p=FakeProject();m=manifest(tmp_path,p);row=r.run_history(941101,'pilot',tmp_path/'pilot',m,p)
    assert row['score_record']['complete'] and row['actual_actions_not_modified']
    assert not row['new_selection_performed_after_fresh_trace']
    assert [z['seed'] for z in p.fit_calls]==[2242,2242]
    assert all(z['exact_values'] for z in row['replay_checks'].values())
    a,b=p.fit_calls[0]['observations'],p.fit_calls[1]['observations']
    assert np.array_equal(a.p[:14],b.p[:14]) and not np.array_equal(a.p[14:],b.p[14:])
    assert np.array_equal(a.q,b.q) # EV has zero Q and both see identical fake noise.
    assert row['operation_costs'][c.METHODS[0]]['driving_energy_kwh']==pytest.approx(.4)
    assert row['operation_costs'][c.METHODS[1]]['driving_energy_kwh']==pytest.approx(.2)


def test_first_fit_failure_abstains_both_without_new_default_initial(tmp_path):
    p=FakeProject(first_failure=True);m=manifest(tmp_path,p)
    row=r.run_history(941101,'pilot',tmp_path/'pilot',m,p)
    assert row['initialization_failure'] and not row['score_record']['complete'] and not p.fit_calls
    assert row['score_record']['relative'] is None
    assert all(z['operational_voltage_ok'] for z in row['records'])
    # Mathematical infinity appears as explicit null + complete=False in JSON.
    assert all(not any(z['success']) for z in row['records'])


def test_final_required_fit_failure_stays_in_pair_score(tmp_path):
    p=FakeProject(final_failure=True);m=manifest(tmp_path,p)
    row=r.run_history(941101,'pilot',tmp_path/'pilot',m,p)
    assert not row['score_record']['complete'] and np.isinf(row['score_record']['relative'])
    assert row['score_record']['failed_fit_count']==1 and len(p.fit_calls)==2


def test_replay_mismatch_stops_before_common_fit_instead_of_score_zero(tmp_path):
    p=FakeProject(mismatch=True);m=manifest(tmp_path,p)
    with pytest.raises(r.ReplayMismatch,match='exceeds tolerance'):
        r.run_history(941101,'pilot',tmp_path/'pilot',m,p)
    assert not p.fit_calls


def test_replay_channels_do_not_confuse_master_P_with_terminal_P():
    tol=dict(exact=['p','q','root_v'],absolute_other=2e-10)
    measured={'round_2_joint_master_p':np.array([1.]),'round_2_joint_p':np.array([[1.,2.,3.]])}
    rebuilt={k:v.copy() for k,v in measured.items()};rebuilt['round_2_joint_master_p']+=1e-12
    checks=r.compare_replay(rebuilt,measured,tol)
    assert checks['round_2_joint_master_p']['tolerance']==2e-10
    assert checks['round_2_joint_p']['tolerance']==0
    rebuilt['round_2_joint_p'][0,0]+=1e-12
    with pytest.raises(r.ReplayMismatch,match='exceeds tolerance'):r.compare_replay(rebuilt,measured,tol)


def test_deadline_does_not_begin_new_mock_history(tmp_path):
    p=FakeProject();m=manifest(tmp_path,p)
    summary=r.run_batch('pilot',tmp_path/'pilot',tmp_path/'frozen.json',p,
        now=lambda:datetime(2026,10,13,3,1,tzinfo=timezone.utc))
    assert not p.fresh_calls and summary['history_count']==0
    assert not summary['all_requested_histories_attempted'] and not summary['formal_schedule_eligible']


def test_mock_pilot_complete_resume_does_not_regenerate_and_checks_archive_identity(tmp_path):
    p=FakeProject();m=manifest(tmp_path,p)
    before_cutoff=lambda:datetime(2026,10,12,23,tzinfo=timezone.utc)
    first=r.run_batch('pilot',tmp_path/'pilot',tmp_path/'frozen.json',p,now=before_cutoff)
    assert first['history_count']==2 and first['formal_schedule_eligible']
    calls=len(p.fresh_calls)
    r.run_batch('pilot',tmp_path/'pilot',tmp_path/'frozen.json',p,now=before_cutoff)
    assert len(p.fresh_calls)==calls
    archive=tmp_path/'pilot'/'trial_941101_paired_observations.npz'
    archive.write_bytes(b'changed mock archive')
    with pytest.raises(ValueError,match='archive changed'):
        r.run_batch('pilot',tmp_path/'pilot',tmp_path/'frozen.json',p,now=before_cutoff)


def test_outputs_cannot_write_into_original_D_directory(tmp_path):
    p=FakeProject();m=manifest(tmp_path,p)
    with pytest.raises(ValueError,match='approved'):
        r.run_history(941101,'pilot',r.FROZEN_D.parent,m,p)
    assert not p.fresh_calls


def test_source_drift_saves_pending_new_fit_records_before_stopping(tmp_path):
    class Drift(FakeProject):
        def original_source_hashes(self):
            return {'mock':'changed'} if len(self.fit_calls)>=2 else {'mock':'not-real-code'}
    p=Drift();m=manifest(tmp_path,p);out=tmp_path/'pilot'
    with pytest.raises(ValueError,match='source changed'):r.run_history(941101,'pilot',out,m,p)
    pending=r.read(out/'pending_source_drift_941101.json')
    assert pending['score_record']['complete'] and len(pending['records'])==2


def test_freeze_is_immutable_and_mutation_of_manifest_is_detected(tmp_path):
    p=FakeProject();m=manifest(tmp_path,p)
    with pytest.raises(ValueError,match='already exists'):manifest(tmp_path,p)
    m['max_nfev']=301
    with pytest.raises(ValueError,match='identity'):r.verify_manifest(m,p)


def write_mock_batches(tmp_path,base,m,cal_failure=False):
    cal=tmp_path/'calibration';ev=tmp_path/'evaluation'
    for folder,phase,count,start in [(cal,'calibration',32,940001),(ev,'evaluation',48,950001)]:
        seeds=list(range(start,start+count));r.dump(folder/'config.json',dict(phase=phase,seeds=seeds,manifest=m))
        r.dump(folder/'summary.json',dict(all_requested_histories_attempted=True))
        for seed in seeds:
            row=deepcopy(base);row.update(seed=seed,history_id=str(seed))
            if cal_failure and seed==940001:
                row['records'][0]['success'][1]=False
                banks=[row['first_bank']]+row['records']
                score=c.pair_score(*[z['losses'] for z in banks],0,success=[z['success'] for z in banks])
                row['score_record']=asdict(score)
            r.dump(folder/f'trial_{seed}.json',row)
    return cal,ev


def test_mocked_collector_keeps32_48_denominators_and_infinite_calibration(tmp_path):
    p=FakeProject();m=manifest(tmp_path,p);base=r.run_history(941101,'pilot',tmp_path/'toy_trace',m,p)
    cal,ev=write_mock_batches(tmp_path,base,m,cal_failure=True)
    x=r.collect(cal,ev,tmp_path/'audit',replay=False)
    assert x['calibration_failure_count']==1 and np.isinf(x['calibration']['relative']['threshold'])
    assert x['calibration']['relative']['sample_count']==32 and x['calibration']['relative']['rank']==32
    assert x['whole_pair_false_rejection']['history_count']==48
    assert all(z['algorithm_unique_correct']['count']==0 for z in x['methods'].values())
    assert all(z['unresolved']['count']==48 for z in x['methods'].values())
    assert x['paired']['false_rejection']['event_direction']=='positive change is worse'
    assert 'event_0_to_1' in x['paired']['false_rejection']
    assert 'gain_event_count' in x['paired']['algorithm_unique_correct']


def test_mocked_collector_reconstructs_scores_and_rejects_tampered_summary(tmp_path):
    p=FakeProject();m=manifest(tmp_path,p);base=r.run_history(941101,'pilot',tmp_path/'toy_trace',m,p)
    cal,ev=write_mock_batches(tmp_path,base,m)
    path=cal/'trial_940001.json';row=r.read(path);row['score_record']['absolute']+=1;r.dump(path,row)
    with pytest.raises(ValueError,match='score differs'):r.collect(cal,ev,tmp_path/'audit',replay=False)
