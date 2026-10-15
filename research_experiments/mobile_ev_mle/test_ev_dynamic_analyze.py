"""Analytic saved-record checks for paired arithmetic and read-only figures."""
from copy import deepcopy
import csv
import hashlib
import json
import math
from pathlib import Path

import pytest

from . import ev_dynamic_analyze as analyze
from . import ev_dynamic_collect as collect
from .test_ev_dynamic_collect import _freeze,_save


def _method(family,M):
    return dict(status='failure' if M is None else 'complete',raw_status='selection_failure' if M is None else 'complete',
        M=M,losses={} if M is None else {family:10.,'wrong':10.+M},
        failure_reasons=['selection_failure'] if M is None else [],
        energy=dict(grid_energy_kwh=3.,throughput_kwh=3.,moves=0),
        origin=dict(category='quality',raw='quality_charge_fixed'))


def _report():
    records=[]
    for seed,family,a,b in ((1,'star',12.,10.),(2,'clade_01',20.00005,20.),(3,'clade_02',None,30.)):
        records.append(dict(seed=str(seed),family=family,candidate_miss=False,
            training_parameter_error=dict(R=.1,X=.3),seconds=7.,
            methods=dict(design_charge_fixed=_method(family,a),constant=_method(family,b))))
    return dict(policy_id='fixture',source_manifest_sha256='science',
        methods=['constant','design_charge_fixed'],records=records)


def test_pairing_uses_histories_and_preserves_failures_and_numerical_ties():
    result=analyze.analyze(_report())
    pair=result['paired_contrasts']['charge_fixed_vs_constant']['delta_M']
    assert pair['total']==3 and pair['valid']==2 and pair['missing_or_failed']==1
    assert pair['mean']==pytest.approx(1.000025)
    assert pair['median']==pytest.approx(1.000025)
    assert (pair['positive'],pair['zero'],pair['negative'])==(1,1,0)
    assert result['independent_histories']==3
    assert result['method_summary']['design_charge_fixed']['M']['valid']==2
    assert len(result['history_methods'])==6 and len(result['history_contrasts'])==3
    assert result['analysis_manifest']['new_fit_calls']==result['analysis_manifest']['new_selector_calls']==0
    json.dumps(result,allow_nan=False)


def test_negative_changes_and_invalid_margin_or_duplicate_history_are_not_hidden():
    saved=_report();saved['records'][0]['methods']['design_charge_fixed']=_method('star',8.)
    result=analyze.analyze(saved)
    assert result['paired_contrasts']['charge_fixed_vs_constant']['delta_M']['negative']==1
    bad=deepcopy(saved);bad['records'][0]['methods']['constant']['M']=99.
    with pytest.raises(ValueError,match='saved M differs'):
        analyze.analyze(bad)
    duplicate=deepcopy(saved);duplicate['records'].append(deepcopy(duplicate['records'][0]))
    with pytest.raises(ValueError,match='duplicate independent'):
        analyze.analyze(duplicate)


def test_forecast_RMSE_uses_units_last_horizon_and_only_offline_truth():
    trial=dict(true_name='star',background=dict(p_kw=[[99.,99.],[2.,4.],[5.,6.]],
        q_kvar=[[99.,99.],[1.,2.],[2.,3.]],source_p_kw=[99.,2.,5.],source_q_kvar=[99.,1.,2.]),
        forecast=dict(p=[[.1,.2],[.2,.3]],q=[[.05,.1],[.1,.15]],
            source_by_candidate=dict(star=dict(source_p=[.1,.2],source_q=[.05,.1])),
            metadata=dict(horizon=2,base_kw=20.)))
    result=analyze._forecast_errors(trial)['ar']
    assert result['P_RMSE_kw']==pytest.approx(.5)
    assert result['Q_RMSE_kvar']==pytest.approx(0.)
    assert result['true_source_P_RMSE_kw']==pytest.approx(math.sqrt(.5))
    trial['forecast_error']=dict(P_RMSE_kw=99.)
    with pytest.raises(ValueError,match='RMSE mismatch'):
        analyze._forecast_errors(trial)


def test_new_flag_is_separate_from_material_gain_and_proxy_gap_is_recomputed():
    saved=_report();raw={}
    for row in saved['records']:
        raw[row['seed']]=dict(selection=dict(methods=dict(charge_fixed=dict(
            selected_origin='MILP',selected_is_new_MILP_proposal=True,
            selected_score=5.00005,initial_best_score=5.,raw_proxy_gate=False,
            raw=dict(objective_lower=4.,objective_upper=4.2,gap=.2,completed=True,
                unfinished=False,target_gap=.3,iterations=[dict(master={})])))))
    result=analyze.analyze(saved,raw)
    method=result['method_summary']['design_charge_fixed']
    assert method['recorded_new']==3 and method['recorded_new_with_material_predictive_gain']==0
    assert result['proxy_summary']['conditions']==3 and result['proxy_summary']['completed']==3
    assert result['proxy_summary']['gate_passed']==0
    raw['1']['selection']['methods']['charge_fixed']['raw']['gap']=.9
    with pytest.raises(ValueError,match='proxy gap'):
        analyze.analyze(saved,raw)


def test_raw_and_collection_inputs_match_and_figures_do_not_touch_science(tmp_path):
    data=tmp_path/'data';frozen=_freeze(data)
    _save(data,frozen,200,0,1.)
    _save(data,frozen,201,1,2.,failed='design_charge_fixed')
    report=collect.collect(data);collect.write_json(tmp_path/'summary.json',report)
    here=Path(__file__).parent
    files=[here/name for name in ('ev_dynamic_study.py','ev_fixed_train_design.py','ev_fixed_train_fit.py')]
    before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    result=analyze.run_analysis(report=tmp_path/'summary.json',raw_directory=data,output=tmp_path/'figures')
    assert result['independent_histories']==2 and result['raw_diagnostic_histories']==2
    assert result['paired_contrasts']['charge_fixed_vs_constant']['delta_M']['missing_or_failed']==1
    assert len(result['figures'])==4
    for filename in result['figures']:
        assert (tmp_path/'figures'/filename).stat().st_size>1000
    with (tmp_path/'figures'/'paired_contrasts.csv').open(encoding='utf-8-sig',newline='') as handle:
        values=list(csv.DictReader(handle))
    assert next(v for v in values if v['contrast']=='charge_fixed_vs_constant')['total']=='2'
    assert {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}==before
    tampered=deepcopy(report);tampered['records'][0]['seconds']=999.
    collect.write_json(tmp_path/'bad.json',tampered)
    with pytest.raises(ValueError,match='report/raw supplement mismatch'):
        analyze.load_saved(report=tmp_path/'bad.json',raw_directory=data)
