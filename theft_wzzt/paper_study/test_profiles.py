import numpy as np
from theft_wzzt.theft.theft_model import TheftTree,TheftData,fit_theft_milp,predict
from paper_study.profile import fixed_profile,stable_profile
from paper_study.lp_profile import fit_location_lp,lp_profile

def fixture():
    tree=TheftTree(((0,1),(1,2),(1,3)),(2,3),(1,2,3))
    rng=np.random.default_rng(8)
    p=rng.uniform(.1,.5,(6,2));q=.4*p
    a=np.array([0,.03,.06,.08,0,.03]);bal=a.copy();bal[0]=-.001
    weights=(np.array([.02,.03,.04]),np.array([.01,.015,.02]))
    data=TheftData(p,q,np.zeros_like(p),bal,a,.4*a,.0003,.001)
    selection=np.zeros((6,3));selection[:,0]=a>0
    y=predict(tree,data,*weights,selection)+rng.normal(0,.00003,p.shape)
    data=TheftData(p,q,y,bal,a,.4*a,.0003,.001)
    return tree,data,weights,(np.full(3,.005),np.full(3,.1))

def test_fixed_enumeration_matches_existing_milp():
    tree,data,weights,bounds=fixture()
    fit=fit_theft_milp(tree,data,fixed_rx=weights,weight_bounds=bounds)
    result=fixed_profile(tree,data,weights)
    assert abs(fit.objective-result['h1_loss'])<1e-6
    assert fit.selected_locations(tree)==result['locations']

def test_compact_lp_matches_existing_fixed_location_milp():
    tree,data,_,bounds=fixture()
    for label in [None,*tree.candidates]:
        loc={t:tree.candidates.index(label) if a>0 else -1 for t,a in enumerate(data.amplitude)} if label is not None else None
        fit=fit_theft_milp(tree,data,allow_theft=label is not None,fixed_locations=loc,weight_bounds=bounds)
        compact=fit_location_lp(tree,data,bounds,label)
        assert abs(fit.objective-compact['loss'])<1e-6
        assert compact['diagnostics']['primal_dual_gap']<1e-6

def test_global_union_nested_and_equivalent():
    tree,data,_,bounds=fixture()
    old=stable_profile(tree,data,bounds,optional_activity=False)
    compact=lp_profile(tree,data,bounds)
    assert compact['gain']>=0
    assert abs(old['h1_loss']-compact['h1_loss'])<1e-6
    assert all(abs(a['loss']-b['loss'])<1e-6 for a,b in zip(old['profile'],compact['profile']))
