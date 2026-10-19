"""Physical initialization and planning/evaluation information boundary."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_gradient_flow_study as study
from . import likelihood as lk


def test_public_saturated_curve_matches_task_and_known_square_optimum():
    curves = study.public_curves(6, 3)
    for name in ('constant', 'highlow', 'analytic_square'):
        u = curves[name]
        op = study.dynamic.operation_record(u, discharge=name=='analytic_square')
        assert op['success'], op
        assert op['energy_kwh'][-1] == pytest.approx(26.85, abs=1e-7)
    assert np.sum(curves['analytic_square']**2) == pytest.approx(157.2383160982642)
    assert np.sum(curves['constant']) == pytest.approx(12.)
    assert np.any(curves['analytic_square'] < 0)


def test_freeze_requires_three_disjoint_history_groups():
    cfg = deepcopy(study.CONFIG)
    study.validate_config(cfg)
    cfg['evaluation_seeds'][0] = cfg['development_seeds'][0]
    with pytest.raises(ValueError, match='disjoint'):
        study.validate_config(cfg)


def test_pair_forecast_keeps_training_path_and_selects_only_named_sources():
    f = dict(p=[[.1]], q=[[.02]], root=[1.], metadata=dict(training_only=True),
        source_by_candidate={name:dict(source_p=[.01], source_q=[.02]) for name in ('a','b','c')})
    g = study._subset_forecast(f, ['a', 'c'])
    assert set(g['source_by_candidate']) == {'a', 'c'}
    assert g['p'] == f['p'] and g['q'] == f['q']
    assert 'planning_candidate_subset' not in f['metadata']


def test_simulator_truth_validation_and_future_noise_do_not_reach_planning(monkeypatch):
    cfg = deepcopy(study.CONFIG); cfg['training_points'] = 8
    first = study.dynamic.make_case(993901, 1, cfg)
    changed = deepcopy(first)
    start = changed['future_start']
    changed['p'][start:] += .3; changed['q'][start:] += .1
    changed['white']['v'][start:] *= -3
    obs = changed['validation']
    changed['validation'] = lk.Observations(p=obs.p+.1, q=obs.q,
        v=obs.v+.001, root_v=obs.root_v, master_p=obs.master_p, master_q=obs.master_q)
    families = study.dynamic.base.FAMILIES
    bank = [lk.RadialCandidate.from_supports('star' if not f else 'clade_'+''.join(map(str,f[0])), 3, f)
            for f in families]
    fits = {c.name:SimpleNamespace(r=np.full(c.m,.004),x=np.full(c.m,.001),
        source_p=np.full(8,.1),source_q=np.full(8,.03),success=True,loss=float(i))
        for i,c in enumerate(bank)}
    fit_rows = [dict(candidate=c.name, name=c.name, success=True, loss=1., r=fits[c.name].r.tolist(),
        x=fits[c.name].x.tolist()) for c in bank]
    block = dict(all_complete=True, fits=fit_rows, failures={}, seconds=.01, cost={})
    monkeypatch.setattr(study, 'source_hashes', lambda:{'fixture':'constant'})
    monkeypatch.setattr(study, 'propose', lambda *a,**kw:(bank, {}))
    monkeypatch.setattr(study.rx, 'train_bank', lambda *a,**kw:(fits, deepcopy(block)))
    monkeypatch.setattr(study.old, 'fit_bank', lambda *a,**kw:(fits, deepcopy(block)))
    calls = []
    current = [first]

    def planner(training, candidates, trained, tm, fm, forecast, config, *, seed, counter):
        assert training is current[0]['train'] and trained is fits
        calls.append(dict(p=training.p.copy(), forecast=deepcopy(forecast)))
        u = study.public_curves(6, 3)['constant']
        return {name:u.copy() for name in study.METHODS}, dict(methods={},
            training_top2_names=[bank[0].name,bank[1].name], timing={name:dict(
                planning_actual_seconds=0.,planning_standalone_seconds=0.,dependency_seconds=0.,
                soft_budget_overrun_seconds=0.,actual_calls={}) for name in study.METHODS})

    monkeypatch.setattr(study, 'select_plans', planner)
    monkeypatch.setattr(study.dynamic, 'operation_record', lambda *a,**kw:dict(success=True))
    monkeypatch.setattr(study.dynamic, 'actual_curve', lambda *a,**kw:dict(
        status='complete',M=0.,noisy=dict(seconds=.01,cost={}),noiseless=dict(seconds=.01)))
    monkeypatch.setattr(study.dynamic, 'make_case', lambda *a,**kw:current[0])
    a = study.run_trial(993901, 1, cfg)
    current[0] = changed
    b = study.run_trial(993901, 1, cfg)
    assert a['status'] == b['status'] == 'complete'
    assert np.array_equal(calls[0]['p'], calls[1]['p'])
    assert study.io.serial(calls[0]['forecast']) == study.io.serial(calls[1]['forecast'])
    assert len(a['methods']) == len(b['methods']) == 13
    assert a['training_top2']['names'] == b['training_top2']['names']
