"""Pure wrapper-schema tests: optimizer, physical worker and identifier not run."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_file = Path(__file__).with_name('actual_bridge.py')
_spec = importlib.util.spec_from_file_location('validation_actual_bridge_tested',_file)
bridge=importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bridge)


@pytest.fixture
def fake_inputs(tmp_path,monkeypatch):
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.effect_test import prepare
    from research_experiments.wzzt_dr_selection.dr_conic_pricing.model import PriceProblem
    old=tmp_path/'old';old.mkdir();planning=tmp_path/'planning';planning.mkdir()
    for name in ['history','metadata','baseline','problem']:
        bridge._write(old/(name+'.json'),{})
    vector=np.zeros(16);vector[2]=.1
    zero=dict(resources_by_event=[[]]*4,modules_by_event=[[{}]*8]*4,
              response_factors_by_event=[[[.15,.05]]*8]*4,global_expected_amplitudes=np.zeros(16).tolist())
    rotation=dict(zero,global_expected_amplitudes=vector.tolist())
    bridge._write(old/'zero.json',zero);bridge._write(old/'rotation.json',rotation)
    alg={'frozen_algorithm':'identical_hash'}
    c=dict(status='frozen',repository=str(Path.cwd()),case='flynn16',case_index=0,
           hard_four_event_budget=1.6,original_history_rows=[288,288],future_rows=[480,480],
           source_root=str(old/'producer'),source_metadata=bridge._ref(old/'metadata.json'),
           history_packet=bridge._ref(old/'history.json'),historical_baseline_result=bridge._ref(old/'baseline.json'),
           problem=bridge._ref(old/'problem.json'),algorithm_source_sha256=alg,
           plans={'zero':bridge._ref(old/'zero.json'),'rotation':bridge._ref(old/'rotation.json')},
           target_labels=[201,202,203],labels=list(range(201,217)),root_label=1,base_kw=200.,
           voltage_sigma=.0005,pq_sigma=.005,search_seconds=25.,lp_method=2)
    bridge._write(old/'protocol.json',c)
    bridge._write(planning/'protocol.json',dict(status='frozen'))
    bridge._write(planning/'summary.json',dict(status='complete',selected_refit_sample90_feasible=False))
    problem=SimpleNamespace(n_contracts=16,budget=1.6,contract_ids=[f'c{j}' for j in range(16)],
                            exclusive_groups=[tuple(range(k,k+4)) for k in [0,4,8,12]])
    monkeypatch.setattr(PriceProblem,'from_dict',lambda value:problem)
    monkeypatch.setattr(prepare,'_sources',lambda repo,cases,extras:({_file.as_posix():bridge._sha(_file)},alg))
    def fake_plan(problem,resources,modules,local_d,amplitudes,policy,threshold):
        return (dict(policy=policy,excitation_threshold=threshold),
                dict(expected_cost=float(np.sum(amplitudes**2)),hard_budget_bound=float(np.sum(amplitudes**2)),reliable_benefit=0.))
    monkeypatch.setattr(prepare,'_plan',fake_plan)
    return old,planning,vector,tmp_path


def test_freeze_aliases_and_diagnostic_status_without_actual_execution(fake_inputs):
    old,planning,rotation,tmp=fake_inputs
    selected=np.zeros(16);selected[0]=.1
    result=bridge.freeze_actual(old,planning,selected,rotation,tmp/'actual')
    c=bridge._read(tmp/'actual/protocol.json')
    assert c['policies']==['zero','rotation','continuous_validation']
    assert c['policy_aliases']=={'historical_frozen':'rotation'}
    assert c['quote_feasibility_status']=='diagnostic_phaseI_not90_feasible'
    assert c['test_paths']==[1200,1201] and c['replicate_base']==470000
    assert c['response_seed']==2026110133 and c['maximum_unique_fits']==6
    assert result['new_AC']==result['new_identifier']==0
    assert bridge._checked(result['protocol'])==tmp/'actual/protocol.json'


def test_distinct_frozen_guidance_is_separate_policy(fake_inputs):
    old,planning,rotation,tmp=fake_inputs
    selected=np.zeros(16);selected[0]=.1
    frozen=np.zeros(16);frozen[8]=.1
    result=bridge.freeze_actual(old,planning,selected,frozen,tmp/'actual')
    c=bridge._read(tmp/'actual/protocol.json')
    assert c['policies'][-1]=='historical_frozen' and c['maximum_unique_fits']==8
    assert c['policy_aliases']=={}
    assert result['prices_frozen_before_actual']


def test_exact_inactive_type_zero_required_before_freeze(fake_inputs):
    old,planning,rotation,tmp=fake_inputs
    selected=np.zeros(16);selected[0]=.1;selected[1]=1e-12
    with pytest.raises(ValueError,match='exactly zero'):
        bridge.freeze_actual(old,planning,selected,None,tmp/'actual')
    assert not (tmp/'actual').exists()
