"""Small, independent accounting fixtures; no generated future is read."""
from copy import deepcopy
import gzip
import json
from . import ev_mobile_v2_analysis_audit as audit
from .test_ev_reliability_v2 import row


def fixture(seed=1,phase='calibration'):
    record=row(seed,phase);template=deepcopy(record['methods'][0]);record['methods']=[]
    config=dict(methods=list(audit.EXPECTED_METHODS),selector_uses_truth=False,
                selector_uses_validation=False,PQ_direct_inputs=True,root_and_bias_fixed=True)
    record.update(source_unchanged=True,line_parameters_unchanged_during_planning=True,
        common_cost=dict(training_seconds=2.,fixed_history_seconds=.4,joint_history_seconds=.6,
                         combined_adapter_preparation_seconds=1.02))
    own={'none':0.,'constant':0.,'highlow':0.,'square_fixed':.1,'square_mobile':.2,
         'scalar_fixed':.3,'scalar_mobile':.4,'cap_fixed':.5,'cap_mobile':.6,
         'jointgrad_fixed':.7,'jointgrad_mobile':.8}
    deps={'none':[],'constant':[],'highlow':[],'square_fixed':[],
          'square_mobile':['square_fixed'],'scalar_fixed':['square_fixed'],
          'scalar_mobile':['square_fixed','square_mobile','scalar_fixed'],
          'cap_fixed':['square_fixed','scalar_fixed'],
          'cap_mobile':['square_fixed','square_mobile','scalar_mobile','cap_fixed'],
          'jointgrad_fixed':['square_fixed','scalar_fixed'],
          'jointgrad_mobile':['square_fixed','square_mobile','scalar_mobile','jointgrad_fixed']}
    metadata={}
    for name in audit.EXPECTED_METHODS:
        ancestors=set()
        def visit(key):
            if key in ancestors:return
            ancestors.add(key)
            for parent in deps[key]:visit(parent)
        for parent in deps[name]:visit(parent)
        planning=own[name]+sum(own[key] for key in ancestors)+(.05 if name.startswith(('scalar_','cap_','jointgrad_')) else 0)
        metadata[name]=dict(planning_seconds=planning,planning_actual_seconds=own[name],dependencies=deps[name],standalone_ancestors=sorted(ancestors))
        method=deepcopy(template);method.update(name=name,planning_seconds=planning,
            planning_actual_seconds=own[name],evaluation_seconds=.5,total_seconds=2.+planning+.5+1.02,
            planning_failure=None,operation=dict(success=True,driver_cost=0.,audit=dict(moves=0)))
        for mode,value in method['inference'].items():
            value['seconds']=.9 if mode=='joint_total' else .2
            value['total_seconds']=2.+planning+value['seconds']+.4+(.6 if mode=='joint_total' else 0)
            value['joint_pays_fixed_source_witness_dependencies']=mode=='joint_total'
        record['methods'].append(method)
    record['selection']=dict(methods=metadata,context_seconds=.05,uses_truth=False,uses_validation=False,uses_future_noise=False)
    phase_row=dict(records=[record],phase=phase,methods=list(audit.EXPECTED_METHODS),manifest=dict(config=config),
                   files=[],identity=dict(source_id='source-v1',config_id='config-v1'))
    return record,phase_row


def test_complete_eleven_method_costs_and_unique_ancestors_are_verified():
    record,phase=fixture()
    checked=audit.audit_phase(phase,expected_count=1)
    assert checked['success']
    assert checked['checks']['inference_total_cost']==33
    assert checked['checks']['joint_witness_dependency_flag']==33
    assert checked['checks']['all_inference_total_cost']==11
    raw=audit._cost_record(record,raw=record)
    assert raw['success'] and raw['checks']['planning_ancestor_cost']==11
    assert record['selection']['methods']['scalar_mobile']['planning_seconds']==1.05


def test_mode_total_and_dependency_flag_tampering_are_separate_errors():
    record,phase=fixture()
    method=record['methods'][-1]
    method['inference']['joint_total']['total_seconds']-=.6
    method['inference']['joint_total']['joint_pays_fixed_source_witness_dependencies']=False
    checked=audit.audit_phase(phase,expected_count=1)
    assert not checked['success']
    errors=checked['histories'][0]['errors']
    assert any('mode total cost arithmetic mismatch' in e for e in errors)
    assert any('witness dependency flag' in e for e in errors)


def test_raw_dependency_replay_and_forbidden_selection_fields(tmp_path):
    record,phase=fixture();directory=tmp_path/'raw';directory.mkdir()
    path=directory/'case_000_1.json.gz'
    with gzip.open(path,'wt',encoding='utf-8') as stream:json.dump(record,stream)
    checked=audit.audit_phase(phase,expected_count=1,raw_directory=directory)
    assert checked['success'] and checked['checks']['planning_ancestor_cost']==11
    assert checked['checks']['flag_uses_truth']==1 and len(checked['raw_files'])==1
    record['selection']['methods']['scalar_mobile']['planning_seconds']-=.1
    record['selection']['uses_future_noise']=True
    record['selection']['truth']={'r':[1.]}
    with gzip.open(path,'wt',encoding='utf-8') as stream:json.dump(record,stream)
    checked=audit.audit_phase(phase,expected_count=1,raw_directory=directory)
    assert not checked['success']
    errors=checked['histories'][0]['errors']
    assert any('ancestor/context' in e for e in errors)
    assert any('future_noise must be false' in e for e in errors)
    assert any('forbidden input' in e for e in errors)


def test_phase_flags_count_bank_and_pair_seed_checks():
    _,cal=fixture(1);_,ev=fixture(2,'evaluation')
    assert audit.audit_pair(cal,ev,calibration_count=1,evaluation_count=1)['success']
    assert not audit.audit_pair(cal,ev,calibration_count=96,evaluation_count=48)['success']
    ev['records'][0]['seed']=1
    assert not audit.audit_pair(cal,ev,calibration_count=1,evaluation_count=1)['success']
    cal['manifest']['config']['selector_uses_truth']=True
    assert not audit.audit_phase(cal,expected_count=1)['success']


def test_failed_training_is_kept_as_a_denominator_not_cost_fabrication():
    record,phase=fixture();record['status']='training_failure'
    record['common_cost']={'training_seconds':.05}
    record.pop('selection')
    for method in record['methods']:
        method.update(planning_seconds=None,planning_actual_seconds=None,planning_failure='training',
                      evaluation_seconds=None,total_seconds=None,operation=dict(success=False))
        for value in method['inference'].values():value.update(complete=False,seconds=None,total_seconds=None)
    checked=audit.audit_phase(phase,expected_count=1)
    assert checked['success']
    assert checked['method_failures']['jointgrad_mobile']['histories']==1
    assert checked['method_failures']['jointgrad_mobile']['planning_failure']==1
    assert checked['method_failures']['jointgrad_mobile']['incomplete_joint_total']==1
