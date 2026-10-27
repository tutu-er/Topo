"""Mainline forwarding contracts plus a bounded real four-terminal call."""
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_experiments.wzzt_dr_selection.baseline_alignment import mainline_adapter as adapter
pipeline=adapter.pipeline
core=adapter.core


def scenario(name,seed,n=3):
    rng=np.random.default_rng(seed)
    labels=pd.Index(range(101,101+n))
    index=pd.date_range('2021-01-01',periods=32,freq='h')
    p=pd.DataFrame(rng.uniform(.01,.04,(32,n)),index=index,columns=labels)
    q=pd.DataFrame(rng.uniform(.005,.025,(32,n)),index=index,columns=labels)
    r=.2*np.eye(n);x=.05*np.eye(n)
    r[:2,:2]+=.03;x[:2,:2]+=.01
    root=pd.Series(1.02,index=index)
    v=pd.DataFrame(np.sqrt(root.to_numpy()[:,None]**2-p.to_numpy()@r.T-q.to_numpy()@x.T),index=index,columns=labels)
    return adapter.observed.make_observed_scenario(p,q,v,root,name)


def selection():
    pair=frozenset((101,102))
    return pipeline._RnjSelection(full_clades={pair},full_cherries={pair},confidence={pair:1.},
        selected=[pair],r2_score=1.,condition=2.,elapsed=0.)


def fitted(train,valid,supports,stop='no_significant_one_atom_gain'):
    labels=tuple(train[0]['P_terminal'].columns)
    rs=np.asarray([.2 if len(s)==1 else .03 for s in supports])
    xs=np.asarray([.05 if len(s)==1 else .01 for s in supports])
    r,x=core.build_matrices_from_atoms(len(labels),supports,rs,xs)
    tm,_,_=core.evaluate_l1_matrices(train,r,x)
    vm,vs,_=core.evaluate_l1_matrices(valid,r,x)
    diag=core.SolverDiagnostics(status=0,success=True,message='explicit stub',objective=tm,
        dual_bound=None,mip_gap=None,node_count=None,runtime_seconds=0.,variable_count=5,
        binary_variable_count=0,constraint_count=5)
    label_supports=tuple(tuple(labels[i] for i in s) for s in supports)
    point=core.LaminarL1PathPoint(iteration=0,support_indices=tuple(supports),support_labels=label_supports,
        r_values=rs,x_values=xs,r_matrix=r,x_matrix=x,train_mae=tm,validation_mae=vm,
        validation_se=vs,accepted_gain=None,solver=diag)
    return core.LaminarL1Result(terminal_labels=labels,support_indices=tuple(supports),support_labels=label_supports,
        r_values=rs,x_values=xs,r_matrix=r,x_matrix=x,train_mae=tm,validation_mae=vm,
        selected_path_index=0,path=(point,),attempted_extensions=(),stop_reason=stop,
        r_upper_bound=2.,x_upper_bound=2.,bound_expansions=0)


def test_defaults_are_read_from_pipeline_signature():
    defaults=adapter.mainline_defaults()
    for key,value in defaults.items():
        assert value==inspect.signature(pipeline.run).parameters[key].default
    assert defaults['time_limit']==1800. and defaults['milp_solver']=='highs'
    assert defaults['bootstrap_replicates']==100


def test_forwarded_milp_arguments_match_production_variant(monkeypatch,tmp_path):
    train=[scenario('train',1)];valid=[scenario('valid',2)]
    selected=selection();calls=[];selection_calls=[]
    def select(case,**kw):
        selection_calls.append((case,kw));return selected
    def fit(scenarios,**kw):
        calls.append((scenarios,kw))
        return fitted(scenarios,kw['validation_scenarios'],kw['initial_supports'])
    monkeypatch.setattr(pipeline,'_select_case_rnj',select)
    monkeypatch.setattr(pipeline,'fit_laminar_l1_sensitivity',fit)
    case=pipeline._CaseData('reference',train,valid,[101,102,103],1,set(),set())
    list(pipeline._fit_milp_variants(case,selected,run_baseline=False,
        coefficient_bound=2.,time_limit=1800.,milp_solver='highs'))
    result=adapter.fit_mainline_observed(train,valid,1,tmp_path)
    production=calls[0][1];external=calls[1][1].copy()
    assert callable(external.pop('progress_callback'))
    assert production.keys()==external.keys()
    for key in production:
        if key!='validation_scenarios':assert production[key]==external[key]
    assert external['initial_supports']==[(0,),(1,),(2,),(0,1)]
    assert 'max_atoms' not in external and 'validation_blocks' not in external
    assert selection_calls[0][1]==dict(bootstrap_replicates=100,block_length=4,
        confidence_threshold=.75,maximum_candidate_count=2,tolerance_factor=.16,seed=20260903)
    assert result['status']=='complete' and result['rnj_stage_called']
    assert json.loads((tmp_path/'wzzt.json').read_text())['rnj']['selected']==[[101,102]]


def test_truth_is_stripped_before_both_stages(monkeypatch,tmp_path):
    train=scenario('train',1);valid=scenario('valid',2)
    train['truth_supports']=[[101,103]];valid['true_net']='must not enter estimator'
    def select(case,**kw):
        assert case.truth_clades==case.truth_cherries==set()
        assert not any('truth_supports' in s or 'true_net' in s for s in case.training+case.validation)
        return selection()
    def fit(s,**kw):
        assert not any('truth_supports' in x or 'true_net' in x for x in s+kw['validation_scenarios'])
        return fitted(s,kw['validation_scenarios'],kw['initial_supports'])
    monkeypatch.setattr(pipeline,'_select_case_rnj',select)
    monkeypatch.setattr(pipeline,'fit_laminar_l1_sensitivity',fit)
    result=adapter.fit_mainline_observed([train],[valid],1,tmp_path)
    evaluation=adapter.observed.evaluate_effective_supports(result,[[101],[102],[103],[101,102]])
    assert evaluation['full_effective_family_recovered']


@pytest.mark.parametrize('override',[{'solver':'gurobi'},{'truth_supports':[[101,102]]},{'bootstrap_replicates':0}])
def test_unknown_or_invalid_configuration_rejected_before_execution(tmp_path,override):
    with pytest.raises(ValueError):
        adapter.fit_mainline_observed([scenario('train',1)],[scenario('valid',2)],1,tmp_path,override)
    assert not (tmp_path/'wzzt.json').exists()


def test_units_and_output_replacement_are_rejected(monkeypatch,tmp_path):
    train=scenario('train',1);valid=scenario('valid',2)
    train['units']['P_terminal']='kW'
    with pytest.raises(ValueError,match='units'):
        adapter.fit_mainline_observed([train],[valid],1,tmp_path)
    (tmp_path/'wzzt.json').write_text('{}')
    with pytest.raises(FileExistsError):
        adapter.fit_mainline_observed([scenario('train',1)],[valid],1,tmp_path)


def test_exception_keeps_certified_progress(monkeypatch,tmp_path):
    monkeypatch.setattr(pipeline,'_select_case_rnj',lambda *a,**k:selection())
    def fail(s,**kw):
        result=fitted(s,kw['validation_scenarios'],kw['initial_supports'])
        kw['progress_callback']({'event':'initialized','path':result.path,'attempted_extensions':(),
            'bound_expansions':0,'stop_reason':'in_progress'})
        raise RuntimeError('controlled failure')
    monkeypatch.setattr(pipeline,'fit_laminar_l1_sensitivity',fail)
    result=adapter.fit_mainline_observed([scenario('train',1)],[scenario('valid',2)],1,tmp_path)
    assert result['status']=='failed' and not result['has_model']
    assert len(result['last_certified_path'])==1 and result['exception']=='controlled failure'


def test_actual_four_terminal_full_chain_with_declared_small_budget(tmp_path):
    result=adapter.fit_mainline_observed([scenario('train',1,n=4)],[scenario('valid',2,n=4)],1,tmp_path,
        {'bootstrap_replicates':2,'time_limit':10.})
    assert result['status']=='complete',result.get('exception',result.get('summary'))
    assert result['rnj_stage_called'] and result['has_model']
    assert result['explicit_overrides']=={'bootstrap_replicates':2,'time_limit':10.}
    assert result['mainline_options']['milp_solver']=='highs'
    evaluation=adapter.observed.evaluate_effective_supports(result,[[101],[102],[103],[104],[101,102]])
    assert evaluation['full_effective_family_recovered']
