"""Independent selector contracts, physical compensation, and information use."""
from copy import deepcopy
import inspect
from types import SimpleNamespace
import numpy as np
import pytest

from . import ev_rx_design as design
from . import ev_rx_study as study


def test_nearest_means_each_references_hardest_rival():
    rows=[dict(a=a,b=b,added=v,success=True) for a,b,v in (
        ('A','B',1),('A','C',100),('B','A',4),('B','C',5),('C','A',0),('C','B',8))]
    p=dict(success=True,pairs=rows)
    assert design.ac_score(p,'nearest')==pytest.approx(5/3)
    assert design.ac_score(p,'mean')==pytest.approx(118/6)
    p['pairs'][1]['success']=False
    assert design.ac_score(p,'nearest') is None


def test_compensation_equal_history_different_active_response():
    k=.25;R=.006;X=.001;deltaX=.001;deltaR=-k*deltaX
    P=np.array([.1,.15,.08]);Q=k*P
    assert np.allclose(R*P+X*Q,(R+deltaR)*P+(X+deltaX)*Q)
    for u in (-.1,0.,.2):
        actual=R*u;fitted=(R+deltaR)*u
        assert fitted-actual==pytest.approx(-k*deltaX*u)


def test_selector_has_no_evaluation_inputs():
    parameters=inspect.signature(design.select_controls).parameters
    assert not {'truth','validation','future_noise','case'} & set(parameters)


def test_baselines_survive_failed_AC_context(monkeypatch):
    meter=SimpleNamespace(v_sigma=np.full((6,3),.001))
    train=SimpleNamespace(n=3)
    events=[]
    def square(domain,meter,**kwargs):
        events.append('baseline');return dict(success=True,power_kw=kwargs['initial_powers'][0])
    def broken(*args,**kwargs):
        assert events==['baseline']*4
        return dict(success=False,pairs=[],diagnostics=dict(failure='physical context failure'))
    monkeypatch.setattr(design,'solve_squared_precision',square)
    monkeypatch.setattr(design.old,'build_fixed_contexts',broken)
    choices,record=design.select_controls(train,[],{},None,meter,forecast={},seed=1)
    assert record['failures']['context']=='physical context failure'
    assert all(choices['square_'+d] is not None for d in design.old.DOMAIN_FLAGS)
    assert all(choices['nearest_'+d] is None for d in design.old.DOMAIN_FLAGS)


def test_failure_preserved_without_hidden_retry(monkeypatch):
    monkeypatch.setattr(study,'source_hashes',lambda:{'file':'unchanged'})
    def broken(*args):raise ValueError('known failure')
    monkeypatch.setattr(study,'_run_trial',broken)
    result=study.run_trial(992999,0)
    assert result['status']=='driver_failure' and result['source_unchanged']
    assert result['failure']=='ValueError:known failure'


def test_data_generations_paired_between_training_policies():
    box=deepcopy(study.CONFIG);cone=deepcopy(box);box['training_domain']='box'
    a=study.dynamic.make_case(992998,1,box);b=study.dynamic.make_case(992998,1,cone)
    for key in ('p','q','source_p','source_q','root_v'):
        assert np.array_equal(a[key],b[key])
    for key in a['white']:assert np.array_equal(a['white'][key],b['white'][key])


def test_held_out_data_changes_do_not_change_selector_arguments(monkeypatch):
    cfg=deepcopy(study.CONFIG);cfg['training_points']=8
    first=study.dynamic.make_case(992997,1,cfg);second=deepcopy(first)
    second['p'][12:]+=.03;second['white']['v'][12:]*=-2
    second['validation']=study.lk.Observations(**{**study.io.observations_record(second['validation']),
        'v':second['validation'].v+.002})
    bank=[study.base.RadialCandidate.from_supports('star',3,())]
    f=SimpleNamespace(r=np.full(3,.004),x=np.full(3,.001),source_p=np.full(8,.1),source_q=np.full(8,.03))
    fitted={'star':f};seen=[];current=[first]
    result=dict(all_complete=True,fits=[dict(candidate='star',loss=1.)],cost={})
    monkeypatch.setattr(study,'source_hashes',lambda:{'file':'fixed'})
    monkeypatch.setattr(study.dynamic,'make_case',lambda *args:current[0])
    monkeypatch.setattr(study,'propose',lambda *args,**kwargs:(bank,{}))
    monkeypatch.setattr(study,'train_bank',lambda *args,**kwargs:(fitted,result))
    monkeypatch.setattr(study.old,'fit_bank',lambda *args,**kwargs:(fitted,result))
    monkeypatch.setattr(study.old,'PQ_error_scale',lambda *args:{})
    def select(training,candidates,models,train_meter,future_meter,**kwargs):
        assert training is current[0]['train'] and models is fitted
        seen.append(study.io.serial(kwargs['forecast']))
        assert not {'validation','truth','future_noise','case'} & set(kwargs)
        return {'none':np.zeros((6,3))},dict(methods={})
    monkeypatch.setattr(study,'select_controls',select)
    monkeypatch.setattr(study.dynamic,'actual_curve',lambda *args,**kwargs:dict(status='complete',M=None))
    a=study.run_trial(992997,1,cfg);current[0]=second;b=study.run_trial(992997,1,cfg)
    assert a['status']==b['status']=='complete'
    assert seen[0]==seen[1]
    assert a['methods'][0]['power_kw']==b['methods'][0]['power_kw']
