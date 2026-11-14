import ast
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import numpy as np
import pytest
from . import recover_partial as r


def test_trace_is_not_an_exact_stop_time(tmp_path):
    prior=tmp_path/'shared_initial_receipt.json';prior.write_text('{}');os.utime(prior,(1000,1000))
    d=tmp_path/'normal_fixed_diagnostics';d.mkdir()
    (d/'native.log').write_text('still solving')
    (d/'progress.jsonl').write_text(json.dumps(dict(wall_seconds=100.,bound=.8))+'\n{"partial":')
    state=r.budget_state(tmp_path,'normal_fixed',1500.2)
    assert state['prior_measured_wall_lower_bound']==100.
    assert state['prior_budget_charge_seconds']==501.
    assert state['remaining_native_budget_seconds']==99.
    assert state['prior_bound']==.8
    assert not state['prior_time_exact']


def test_finished_quote_gets_no_new_native_budget(tmp_path):
    (tmp_path/'normal_fixed_plan.json').write_text(json.dumps(dict(seconds=600.2,cost_yuan=2.1,lower_bound_yuan=1.2)))
    s=r.budget_state(tmp_path,'normal_fixed',2000.)
    assert s['remaining_native_budget_seconds']==0.
    assert s['prior_budget_charge_seconds']==600.
    assert s['prior_measured_wall_lower_bound']==600.2
    assert s['kind']=='completed_quote_action_unavailable'
    assert 'point' not in s


def test_unstarted_mode_keeps_original_budget(tmp_path):
    s=r.budget_state(tmp_path,'mgf_fixed',2000.)
    assert s['kind']=='unstarted' and s['remaining_native_budget_seconds']==600.
    assert s['prior_budget_charge_seconds']==0.


def test_inherited_bound_is_same_problem_lower_not_lost_upper():
    assert r.merge_bounds(30.,1.2,2.)==(2.,28./30.,False)
    assert r.merge_bounds(1.,2.,None)==(0.,1.,True)
    assert r.merge_bounds(None,2.,float('inf'))==(2.,None,False)
    assert r.merge_bounds(3.,-2.,float('nan'))==(0.,1.,False)


def test_saved_pool_selection_rejects_cheaper_infeasible_point(monkeypatch):
    class Case:
        array=staticmethod(np.asarray)
        fee=staticmethod(lambda u:float(abs(u).sum()))
    monkeypatch.setattr(r.solver,'direct_audit',lambda c,m,u:dict(passed_acceptance=bool(u.min()>=1)))
    point,receipt=r.select_saved_seed(Case(),None,dict(zero=np.array([0.]),expensive=np.array([4.]),valid=np.array([2.])))
    assert np.array_equal(point,np.array([2.]))
    assert receipt['name']=='valid' and receipt['cost_yuan']==2.


def test_action_checkpoint_is_exact(tmp_path):
    u=np.array([[0.,-2.5],[3.25,0.]])
    ref=r.atomic_action(tmp_path/'action.npz',u)
    with np.load(ref['path']) as z:assert np.array_equal(z['u'],u)
    assert not (tmp_path/'action.npz.tmp').exists()


def test_interrupted_recovery_cannot_receive_budget_again(tmp_path,monkeypatch):
    (tmp_path/'mgf_forecast_resume01_diagnostics').mkdir()
    monkeypatch.setattr(r.evidence,'DistributionEvidence',lambda *a,**k:object())
    def forbidden(*a,**k):raise AssertionError('Must not launch another search')
    monkeypatch.setattr(r.solver,'plan',forbidden)
    with pytest.raises(RuntimeError,match='itself interrupted'):
        r.recover_quote(None,dict(score=None,val=None,base=200.),None,{},None,0.,'mgf_forecast',
                        dict(remaining_native_budget_seconds=305.),dict(solver_seconds=600.),tmp_path)


def test_recovery_information_boundary():
    tree=ast.parse(Path(r.__file__).read_text(encoding='utf8'))
    functions={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
    calls=[n for n in ast.walk(functions['run_forecast']) if isinstance(n,ast.Call)]
    contexts=[n for n in calls if ast.unparse(n.func)=='old.context']
    seals=[n for n in calls if ast.unparse(n.func)=='io.write' and n.args and ast.unparse(n.args[0])=="directory / 'seal.json'"]
    assert len(contexts)==len(seals)==1 and contexts[0].lineno>seals[0].lineno
    solves=[n for n in ast.walk(functions['recover_quote']) if isinstance(n,ast.Call) and ast.unparse(n.func)=='solver.plan']
    assert len(solves)==1
    assert [ast.unparse(x) for x in solves[0].args]==['case',"planning['score']","planning['val']",'prediction',"planning['base']"]
    assert not any(isinstance(n,ast.Call) and ast.unparse(n.func)=='old.context' for n in ast.walk(functions['recover_quote']))
