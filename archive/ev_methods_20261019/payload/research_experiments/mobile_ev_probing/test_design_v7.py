"""Independent properties of the small historical-set design, not AC tests."""
from fractions import Fraction as F
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
import design_v7 as d


def history():
    obs,_=d.simulate("cherry12",d.PROFILES["symmetric"],601,0,"constant")
    return d.observation_slice(obs,slice(0,12))


def test_shared_interval_elimination_matches_direct_definition():
    obs=history()
    rng=np.random.default_rng(803)
    for h in d.NAMES:
        poly=d.build_history_polytope(obs,h)
        points=list(rng.uniform(0,1,(30,5)))
        rep=d.solve_qp(poly,d.midpoint_objective())
        points.append(np.asarray(rep["x"]))
        for x in points:
            th=[l+w*d.f(v) for l,w,v in zip(poly.offset,poly.width,x)]
            direct=True
            for t,p in enumerate(obs["P_total"]):
                residual=[d.f(obs["Y_drop"][t,i])-sum((c*v for c,v in zip(d.prediction_coeff(p,.12,h,i),th)),F()) for i in range(3)]
                eps=[d.f(v) for v in obs["drop_error_bound"][t]]
                source=d.f(obs["source_error_bound"][t])*th[0]
                lower=max([-source]+[r-e for r,e in zip(residual,eps)])
                upper=min([source]+[r+e for r,e in zip(residual,eps)])
                direct &= lower<=upper
            assert poly.exact_feasible(x)==direct


def test_pairwise_projector_identity():
    rng=np.random.default_rng(902)
    c=np.array([[1.,-1.,0.],[1.,1.,-2.]])/np.array([[np.sqrt(2.)],[np.sqrt(6.)]])
    for v in rng.normal(size=(20,3)):
        pair=sum((v[i]-v[j])**2 for i,j in ((0,1),(0,2),(1,2)))/3
        assert abs(pair-np.linalg.norm(c@v)**2)<1e-12


def test_endpoint_dominance_after_full_parameter_refit():
    polys,reps,_=d.history_representatives(history())
    ref=reps["cherry12"]
    for site in range(3):
        p_const,_=d.action_inputs(site,"constant")
        p_end,_=d.action_inputs(site,"endpoint")
        p_mid=np.full((24,3),.2); p_mid[:,site]+=np.tile([.05,.25],12)
        for h in ("cherry13","cherry23"):
            fits=[d.solve_qp(polys[h],d.design_objective(polys[h],ref,p),reps[h]["x"],scale=1e4)
                  for p in (p_const,p_mid,p_end)]
            assert all(q["gap"]<1e-9 for q in fits)
            assert fits[1]["upper"]+1e-10>=fits[0]["lower"]
            assert fits[2]["upper"]+1e-10>=fits[1]["lower"]


def test_permutations_are_the_same_quadratic_and_score():
    polys,reps,_=d.history_representatives(history())
    p,_=d.action_inputs(1,"endpoint")
    q=d.design_objective(polys["cherry13"],reps["cherry12"],p)
    for order in (np.argsort(p[:,1]),np.random.default_rng(805).permutation(24)):
        r=d.design_objective(polys["cherry13"],reps["cherry12"],p[order])
        assert q==r


def test_sum_min_not_min_sum_scenario_counterexample():
    # Separate mutually exclusive futures +/-1 each fit their own c exactly.
    separate=0.
    together=min(.5*(1-c)**2+.5*(-1-c)**2 for c in np.linspace(-2,2,401))
    assert separate==0 and together==1


def test_observed_history_is_identical_across_actions_and_truth_is_separate():
    reference=None
    for site in range(3):
        for policy in ("constant","endpoint"):
            obs,truth=d.simulate("cherry13",d.PROFILES["asymmetric"],901,site,policy)
            hist=d.observation_slice(obs,slice(0,12))
            assert "topology" not in hist and "theta" not in hist
            if reference is None:
                reference=hist
            for k in hist:
                np.testing.assert_array_equal(hist[k],reference[k])
            assert all(truth["checks"].values())
            assert abs(truth["energy_kwh"]-9)<1e-12
