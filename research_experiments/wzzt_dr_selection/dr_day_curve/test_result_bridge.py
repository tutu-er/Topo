import json
from .test_day_model import small_problem,small_scenarios
from . import result_bridge


def test_adapter_failure_preserves_raw_solution_and_receipt(tmp_path,monkeypatch):
    def fail(*args,**kwargs):raise ValueError('deliberate DTO rejection')
    monkeypatch.setattr(result_bridge,'build_plan',fail)
    raw=dict(status='target_feasible_cost_incumbent',mean_u=[[-.2,.2]],prices=[.4],primary={'objective':0.})
    r=result_bridge.persist_and_adapt(tmp_path,raw,small_problem(),small_scenarios(),[],[1],'toy')
    assert r['status']=='adapter_failed'
    assert r['optimization_status']=='target_feasible_cost_incumbent'
    assert r['training']['passed']==1
    assert r['raw_solver_retained']
    assert json.loads((tmp_path/'raw_solver.json').read_text())==raw
