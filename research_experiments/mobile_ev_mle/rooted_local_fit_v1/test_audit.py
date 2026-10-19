import numpy as np
from .model import prepare_case
from .finish import independent_response,independent_ac,order,compare
from .response import prepare
from ..local_response_score.scoring import _profile_h


def test_general_tree_ac_audit_and_raw_meter_loss():
    p,_,t=prepare_case('paper11',35991,4);c=p['candidates'][0]
    fitted={k:t[k][:4] for k in ('p','q','v','root_v')};fitted.update(r=t['r'],x=t['x'])
    loss,error=independent_ac(c,p,fitted)
    expected=sum(np.sum(((t[k][:4]-p['observed'][k])/p['sigma'][k])**2) for k in ('p','q','v'))
    expected+=np.sum(((t['root_v'][:4]-p['root_meter'][0])/.0005)**2)
    assert abs(loss-expected)<1e-8 and error<1e-12


def test_independent_root_and_cut_profile_replay():
    for context in ('FULL','CUT'):
        p,_,t=prepare_case('paper15',35991,4,context);c=p['candidates'][0]
        actual,h,_,_=independent_response(c,p,t['r'],t['x'])
        expected=_profile_h(prepare(c,p),h,derivatives=False)['loss']
        assert abs(actual-expected)<1e-8


def test_rank_ties_and_failed_denominator():
    assert order(dict(base=1,pair01=2,pair02=3)) is None
    a=order(dict(base=1,pair01=1+1e-7,pair02=2,pair12=3))
    b=order(dict(base=1,pair01=2,pair02=3,pair12=4))
    assert a['best']==['base','pair01']
    assert compare(a,b)==dict(first_agrees=False,pair_agrees=5,pairs=6)
