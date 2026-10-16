"""Pure postprocessing checks, including label-conditional finite sample ranks."""
from copy import deepcopy
from pathlib import Path
import json
import gzip
import pytest

from . import ev_reliability_v2 as rel
from . import ev_systematic_analyze as base
from . import ev_reliability_v2_audit as audit
from .test_ev_systematic_analyze import phase, record


def row(seed, kind='calibration', true='star', star=10., cherry=20., H=8):
    value=record(seed,kind,true=true)
    value['candidates']=[dict(name='star',supports=[[0,1,2],[0],[1],[2]]),
                         dict(name='cherry',supports=[[0,1,2],[0,1],[0],[1],[2]])]
    value['candidate_names']=['star','cherry'];value['config']={'training_points':H}
    for method in value['methods']:
        for mode in method['inference'].values():
            mode.update(losses={'star':star,'cherry':cherry},
                        M=(cherry-star if true=='star' else star-cherry))
    return value


def test_complexity_uses_actual_supports_and_names_are_opaque():
    value=row(1);pars,error=rel.candidate_parameters(value)
    assert error is None and pars['star']['line_parameters']==8
    assert pars['cherry']['line_parameters']==10
    value['candidates'][0]['name']='z';value['candidate_names'][0]='z'
    for method in value['methods']:
        for mode in method['inference'].values():mode['losses']['z']=mode['losses'].pop('star')
    pars,error=rel.candidate_parameters(value)
    assert error is None and pars['z']['line_parameters']==8


def test_aic_penalty_is_exact_and_changes_gap_by_four_for_one_extra_edge():
    value=row(1,star=11.,cherry=10.)
    original=rel.scored_mode(value,'none','joint_total','global_R')
    penalized=rel.scored_mode(value,'none','joint_total','global_AIC')
    assert original['relative']=={'star':1.,'cherry':0.}
    assert penalized['scores']=={'star':27.,'cherry':30.}
    assert penalized['relative']=={'star':0.,'cherry':3.}


@pytest.mark.parametrize('mutation',[
    lambda r:r.pop('candidates'),
    lambda r:r['candidates'][0].pop('supports'),
    lambda r:r['candidates'][0]['supports'].append([0]),
    lambda r:r['candidates'][0]['supports'].append([True]),
    lambda r:r['candidates'][0]['supports'].append([]),
])
def test_unknown_structure_keeps_all_in_every_rule(mutation):
    value=row(1,'evaluation');mutation(value)
    for rule in rel.RULES:
        dec=rel.decision(value,'none','joint_total',rule,{})
        assert not dec['complete'] and dec['category']=='failed_keep_all'
        assert dec['retained']==['cherry','star'] and dec['false_reject'] is False


def test_label_calibration_uses_only_corresponding_true_labels_and_exact_rank(tmp_path):
    rows=[row(i,true='star',star=10.+i,cherry=10.) for i in range(19)]
    rows += [row(100+i,true='cherry',star=10.,cherry=10.+2*i) for i in range(19)]
    t=rel.calibrate(phase(rows),tmp_path)
    groups=t['rules']['label_R']['none']['joint_total']
    star=rel.group_key(rows[0],'label_R','star');cherry=rel.group_key(rows[0],'label_R','cherry')
    assert groups[star]['n']==groups[cherry]['n']==19
    assert groups[star]['order_rank']==18
    assert groups[star]['threshold']==17. and groups[cherry]['threshold']==34.
    assert t['rules']['global_R']['none']['joint_total']['all']['n']==38
    assert t['rules']['H_R']['none']['joint_total']['32']['infinite']
    assert rel.calibrate(phase(rows),tmp_path)==t
    with pytest.raises(ValueError,match='existing thresholds differ'):
        rel.calibrate(phase(rows),tmp_path,alpha=.2)


def test_missing_label_and_small_group_are_kept(tmp_path):
    t=rel.calibrate(phase([row(1)]),tmp_path)
    for group in t['rules']['label_R']['none']['joint_total'].values():
        assert group['infinite']
    dec=rel.decision(row(3,'evaluation',star=1000.,cherry=1.),'none','joint_total','label_R',
                     t['rules']['label_R']['none']['joint_total'])
    assert dec['retained']==['cherry','star']
    assert dec['false_reject'] is False


def test_computation_failure_is_zero_calibration_score_and_keeps_all(tmp_path):
    rows=[row(i,star=10.+i,cherry=10.) for i in range(19)]
    rows[-1]['methods'][0]['inference']['joint_total']['complete']=False
    t=rel.calibrate(phase(rows),tmp_path)
    g=t['rules']['global_R']['none']['joint_total']['all']
    assert g['scores'][-1]==0. and g['incomplete_calibration']==1
    assert g['threshold']==16.
    dec=rel.decision(rows[-1],'none','joint_total','global_AIC',{})
    assert dec['category']=='failed_keep_all'


def test_formal_io_rejects_overlap_source_drift_changed_labels_and_undurable_thresholds(tmp_path):
    rows=[row(i) for i in range(19)];t=rel.calibrate(phase(rows),tmp_path)
    ev=phase([row(100,'evaluation')])
    result=rel.evaluate(ev,t,tmp_path,expected_count=1)
    assert result['overall']['global_R']['none']['joint_total']['unique_correct']==1
    assert result['overall']['global_AIC']['none']['joint_total']['unique_correct']==1
    assert result['by_truth']['star']['label_R']['none']['joint_total']['denominator']==1
    assert result['by_H']['8']['label_R']['none']['joint_total']['denominator']==1
    overlap=phase([row(1,'evaluation')])
    with pytest.raises(ValueError,match='overlap'):rel.evaluate(overlap,t,tmp_path)
    changed=deepcopy(ev);changed['identity']['source_id']='other'
    with pytest.raises(ValueError,match='source_id differs'):rel.evaluate(changed,t,tmp_path)
    changed=deepcopy(ev);changed['records'][0]['candidates'][1]['supports'][1]=[1,2]
    with pytest.raises(ValueError,match='structural candidate banks differ'):rel.evaluate(changed,t,tmp_path)
    with pytest.raises(ValueError,match='durably saved'):rel.evaluate(ev,t,tmp_path/'other')
    json.loads((tmp_path/'evaluation_reliability_v2.json').read_text(encoding='utf-8'))


def test_source_record_and_inference_do_not_mutate(tmp_path):
    rows=[row(i) for i in range(19)];original=deepcopy(rows)
    t=rel.calibrate(phase(rows),tmp_path)
    ev=phase([row(100,'evaluation')]);old_ev=deepcopy(ev)
    rel.evaluate(ev,t,tmp_path)
    assert rows==original and ev==old_ev


def test_global_R_exactly_matches_existing_primary_decision(tmp_path):
    rows=[row(i,star=10.+i,cherry=10.) for i in range(19)]
    t=rel.calibrate(phase(rows),tmp_path/'new')
    old=base.calibrate(phase(rows),tmp_path/'old')
    for difference in (0.,16.,17.,17.0000001,100.):
        value=row(100,'evaluation',star=10.+difference,cherry=10.)
        a=rel.decision(value,'none','joint_total','global_R',t['rules']['global_R']['none']['joint_total'])
        b=base.decision(value,'none','joint_total',old['methods']['none']['joint_total'])
        assert (a['retained'],a['false_reject'],a['category'])==(b['retained'],b['false_reject'],b['category'])


def test_missing_truth_prohibited_for_formal_and_retained_in_pressure_diagnostics(tmp_path):
    t=rel.calibrate(phase([row(i) for i in range(19)]),tmp_path)
    ev=phase([row(100,'evaluation')]);ev['records'][0]['candidate_miss']=True
    with pytest.raises(ValueError,match='candidate miss'):rel.evaluate(ev,t,tmp_path)
    result=rel.evaluate(ev,t,tmp_path,development=True)
    assert result['overall']['label_R']['none']['joint_total']['candidate_miss']==1


def test_group_key_never_uses_observed_loss_for_H_or_unknown_label():
    value=row(1,H=32)
    assert rel.group_key(value,'H_R','star')=='32'
    assert rel.group_key(value,'label_R','star')=='{"n":3,"supports":[[0],[0,1,2],[1],[2]]}'
    value['config']['training_points']=33
    assert rel.group_key(value,'H_R','star')=='unknown'
    dec=rel.decision(value,'none','joint_total','H_R',{})
    assert dec['retained']==['cherry','star']


def test_independent_arithmetic_checks_every_rule_and_detects_tampering(tmp_path):
    cal=phase([row(i,true='star' if i%2 else 'cherry',star=10.+i,cherry=10.) for i in range(40)])
    ev=phase([row(100,'evaluation'),row(101,'evaluation',true='cherry',star=20.,cherry=10.)])
    thresholds=rel.calibrate(cal,tmp_path)
    result=rel.evaluate(ev,thresholds,tmp_path)
    checked=audit.independent_verify(cal,ev,thresholds,result)
    assert checked['success'] and checked['decision_checks']==2*5*10*3
    tampered=deepcopy(result)
    tampered['by_history'][0]['rules']['global_R']['none']['joint_total']['retained']=[]
    checked=audit.independent_verify(cal,ev,thresholds,tampered)
    assert not checked['success'] and len(checked['errors'])==1


def test_audit_measures_physical_curves_separately_from_decisions(tmp_path):
    cal=phase([row(i) for i in range(19)]);t=base.calibrate(cal,tmp_path)
    ev=phase([row(100,'evaluation')]);methods=base._methods(ev['records'][0])
    methods['square_fixed']['power_kw']=[[1.]]
    methods['scalar_fixed']['power_kw']=[[2.]]
    result=audit.audit(ev,t)
    value=result['pairwise']['square_fixed__scalar_fixed']
    assert value['same_curve_1e_6_kw']==0 and value['same_raw_argmin']==1
    assert value['same_retained']==1


def test_loader_obeys_declared_eleven_methods_instead_of_old_ten(tmp_path):
    methods=['none','constant','highlow','square_fixed','square_mobile','scalar_fixed',
             'scalar_mobile','cap_fixed','cap_mobile','jointgrad_fixed','jointgrad_mobile']
    value=row(1)
    template=deepcopy(value['methods'][0]);value['methods']=[]
    for name in methods:
        item=deepcopy(template);item['name']=name;value['methods'].append(item)
    directory=tmp_path/'input';directory.mkdir()
    (directory/'freeze.json').write_text(json.dumps({'config':{'methods':methods}}),encoding='utf-8')
    with gzip.open(directory/'case_000.json.gz','wt',encoding='utf-8') as stream:json.dump(value,stream)
    loaded=rel.read_phase(directory,expected_phase='calibration')
    assert loaded['methods']==methods
    thresholds=rel.calibrate(loaded,tmp_path/'out')
    assert list(thresholds['rules']['label_AIC'])==methods
    value['methods'].pop()
    with gzip.open(directory/'case_000.json.gz','wt',encoding='utf-8') as stream:json.dump(value,stream)
    with pytest.raises(ValueError,match='exactly the declared methods'):rel.read_phase(directory)


def test_structural_label_separates_n3_and_n4_star_without_name_guess():
    a=row(1);b=deepcopy(a);b['seed']=2
    for item in b['candidates']:
        item['supports'][0]=[0,1,2,3];item['supports'].append([3])
    assert rel.group_key(a,'label_R','star')!=rel.group_key(b,'label_R','star')
    pa,_=rel.candidate_parameters(a);pb,_=rel.candidate_parameters(b)
    assert pa['star']['n']==3 and pb['star']['n']==4
    assert pb['star']['line_parameters']-pa['star']['line_parameters']==2
    with pytest.raises(ValueError,match='mixes structural candidate banks'):
        rel.calibrate(phase([a,b]),Path('unused_output'))
