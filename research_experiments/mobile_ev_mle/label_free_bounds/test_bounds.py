"""Independent analytic and exact radial-AC checks of label-free bounds."""
from dataclasses import fields, replace

import numpy as np
import pytest
from scipy.stats import chi2

from research_experiments.mobile_ev_mle.label_free_bounds.bounds import (
    Measurements, PhysicalDomain, TreeSpec, certify_candidate,
    linear_voltage_matrix, path_support, qp_with_support_bound,
    _outer_constraints, _physical_theta)


def exact_ac(tree, r, x, p, q, source=0j, root=1.):
    """Independent current-injection fixed point; includes internal loads."""
    nodes = [child for _, child in tree.edges]
    all_tree = TreeSpec(tree.root, tree.edges, nodes)
    a = path_support(all_tree)
    z = (a * (np.asarray(r)+1j*np.asarray(x))) @ a.T
    p, q = np.asarray(p), np.asarray(q)
    power = np.zeros((len(p),len(nodes)), dtype=complex)
    for j, node in enumerate(tree.observed_nodes):
        power[:,nodes.index(node)] += p[:,j]+1j*q[:,j]
    if tree.source_node is not None:
        power[:,nodes.index(tree.source_node)] += source
    voltage = np.full(power.shape,complex(root))
    for _ in range(1000):
        nxt = root - np.conj(power/voltage) @ z.T
        if np.max(np.abs(nxt-voltage)) < 2e-14:
            voltage = nxt
            break
        voltage = nxt
    else:
        raise AssertionError("independent test AC did not converge")
    assert np.max(np.abs(voltage-root+np.conj(power/voltage)@z.T)) < 1e-12
    return np.abs(voltage[:,[nodes.index(j) for j in tree.observed_nodes]])


def test_scalar_qp_primal_and_independent_dual_lower():
    result = qp_with_support_bound(np.array([[1.,0.]]),np.array([2.]),
                                  np.array([[1.,1.]]),np.array([.2]),
                                  np.zeros(2),np.ones(2))
    assert result["primal"] == pytest.approx(1.,abs=1e-7)
    assert 1.-1e-6 <= result["support_lower"] <= 1.
    assert result["guard"] > 0


def test_scipy_fallback_still_has_independent_lower():
    result = qp_with_support_bound(np.eye(2),np.array([2.,-1.]),
                                  np.array([[1.,1.]]),np.array([.5]),
                                  np.zeros(2),np.ones(2),solver="scipy")
    assert result["primal"] == pytest.approx(2.,abs=1e-7)
    assert 2.-1e-5 <= result["support_lower"] <= 2.


def test_internal_observed_node_path_matrix():
    tree=TreeSpec(0,[(0,1),(1,2),(1,3)],[1,2,3])
    h=linear_voltage_matrix(tree,[1.,2.,3.],[0.,0.,0.])
    np.testing.assert_allclose(h[:,:3],[[2.,2.,2.],[2.,6.,2.],[2.,2.,8.]])
    np.testing.assert_allclose(h[:,3:],0.)


def test_positive_impedance_does_not_identify_symmetric_leaf_grouping():
    # Two cherries with z per edge and a star with 3z per edge have the
    # same exact AC port response when all four port powers are identical.
    grouped=TreeSpec(0,[(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)],[3,4,5,6])
    star=TreeSpec(0,[(0,3),(0,4),(0,5),(0,6)],[3,4,5,6])
    p=np.broadcast_to(np.array([.02,.04,.06])[:,None],(3,4))
    q=.35*p
    vg=exact_ac(grouped,np.full(6,.01),np.full(6,.005),p,q)
    vs=exact_ac(star,np.full(4,.03),np.full(4,.015),p,q)
    np.testing.assert_allclose(vg,vs,atol=2e-14)


def test_exact_ac_remainder_covers_internal_and_reverse_injections():
    tree=TreeSpec(0,[(0,1),(1,2),(1,3),(2,4)],[1,2,3,4])
    r=np.array([.012,.018,.014,.009]); x=.45*r
    p=np.array([[.06,.03,-.02,.04],[.025,-.015,.035,.02]])
    q=np.array([[.02,.01,-.006,.013],[.008,-.003,.012,.007]])
    v=exact_ac(tree,r,x,p,q)
    domain=PhysicalDomain(.005,.025,.9,1.1,[.08,.05,.05,.06])
    obs=Measurements(p,q,v,1.,1e-4,1e-4,1e-4)
    result=certify_candidate(tree,obs,domain)
    h=linear_voltage_matrix(tree,r,x)
    eta=1-v*v-np.c_[p,q]@h.T
    assert eta.min() > -1e-12
    assert np.all(eta <= np.asarray(result.remainder_per_node)+1e-12)
    assert result.ac_sse_lower < 1e-12
    assert not result.reject


@pytest.mark.parametrize("seed",range(4))
def test_bound_never_exceeds_raw_distance_to_a_valid_ac_point(seed):
    tree=TreeSpec(0,[(0,1),(1,2),(1,3)],[1,2,3])
    r=np.array([.015,.02,.025]); x=.5*r
    rng=np.random.default_rng(100+seed)
    p=rng.uniform(.015,.07,(4,3)); q=.3*p
    v=exact_ac(tree,r,x,p,q)
    sigma_p,sigma_q,sigma_v=2e-3,1e-3,2e-4
    noise=rng.normal(size=(3,4,3))
    obs=Measurements(p+sigma_p*noise[0],q+sigma_q*noise[1],
                     v+sigma_v*noise[2],1.,sigma_p,sigma_q,sigma_v)
    domain=PhysicalDomain(.005,.035,.9,1.1,[.1,.1,.1],ratio_min=.2,ratio_max=1.)
    result=certify_candidate(tree,obs,domain)
    assert result.ac_sse_lower <= float(np.sum(noise**2))+1e-7
    assert result.qp_support_lower <= result.qp_primal+1e-10
    assert result.raw_noise_dimension==36
    assert result.chi_square_threshold==pytest.approx(chi2.isf(.05,36))


def test_common_source_projection_removes_exact_common_path_and_losses():
    tree=TreeSpec(0,[(0,1),(1,2),(1,3)],[1,2,3],source_node=1)
    r=np.array([.03,.02,.025]); x=.6*r
    p=np.array([[.025,.06,.04],[.03,.045,.05]])
    q=.3*p
    source=np.array([.12+.04j,-.10-.03j])
    v=exact_ac(tree,r,x,p,q,source=source)
    obs=Measurements(p,q,v,1.,1e-4,1e-4,1e-4)
    domain=PhysicalDomain(.005,.04,.85,1.1,[.05,.09,.08],source_power_cap=.2)
    result=certify_candidate(tree,obs,domain,source_projection=True)
    assert result.common_prefix_edges_removed==[0]
    assert result.remainder_per_node[0]==0
    assert result.ac_sse_lower < 1e-12
    projection=np.eye(3)-np.ones((3,3))/3
    h=linear_voltage_matrix(tree,r,x)
    residual=(1-v*v-np.c_[p,q]@h.T)@projection.T
    assert np.linalg.norm(residual) <= result.remainder_norm+1e-12
    # Source is at the virtual root: increasing its cap cannot affect any
    # downstream current bound or the projected exact AC remainder.
    huge=certify_candidate(tree,obs,replace(domain,source_power_cap=100.),
                           source_projection=True)
    assert huge.remainder_norm==result.remainder_norm
    # Upstream impedance is also absent from projected signal/remainder.
    maxes=np.array([4.,.04,.04])
    huge_stem=certify_candidate(tree,obs,replace(domain,z_max=maxes),
                                source_projection=True)
    assert huge_stem.remainder_norm==result.remainder_norm
    assert huge_stem.lipschitz_k==result.lipschitz_k


def test_noncommon_source_is_preserved_as_unsupported_not_silently_ignored():
    tree=TreeSpec(0,[(0,1),(0,2)],[1,2],source_node=1)
    obs=Measurements(np.ones((1,2))*.02,np.zeros((1,2)),np.ones((1,2)),
                     1.,.001,.001,.001)
    domain=PhysicalDomain(.005,.03,.9,1.1,[.05,.05],source_power_cap=.05)
    with pytest.raises(ValueError,match="ancestor"):
        certify_candidate(tree,obs,domain,source_projection=True)
    with pytest.raises(ValueError,match="projection"):
        certify_candidate(tree,obs,domain)


def test_observed_voltage_outside_operating_domain_keeps_lipschitz_valid():
    tree=TreeSpec(0,[(0,1)],[1])
    p,q=np.array([[.03]]),np.array([[.01]])
    vtrue=exact_ac(tree,[.02],[.01],p,q)
    vobs=np.array([[4.]])
    obs=Measurements(p,q,vobs,1.,1e-5,1e-5,1e-3)
    domain=PhysicalDomain(.01,.04,.9,1.1,[.05])
    result=certify_candidate(tree,obs,domain)
    actual=float(np.sum(((vobs-vtrue)/.001)**2))
    assert result.ac_sse_lower <= actual+1e-6
    assert result.lipschitz_k >= (4.+1.1)*.001


def test_noisy_root_count_and_raw_distance():
    tree=TreeSpec(0,[(0,1)],[1])
    p,q=np.array([[.03],[.04]]),np.array([[.01],[.012]])
    v=exact_ac(tree,[.02],[.01],p,q)
    obs=Measurements(p,q,v,np.array([1.001,.999]),1e-4,1e-4,1e-4,sigma_root=1e-3)
    result=certify_candidate(tree,obs,PhysicalDomain(.01,.03,.9,1.1,[.06]))
    assert result.raw_noise_dimension==8
    assert result.ac_sse_lower <= 2.+1e-6


def test_minimum_norm_outer_relaxation_and_physical_backscore_point():
    tree=TreeSpec(0,[(0,1)],[1])
    # Equal P/Q permits a relaxed optimum r=x=zmin/2, below zmin in norm.
    obs=Measurements(np.array([[.1]]),np.array([[.1]]),np.array([[np.sqrt(.98)]]),
                     1.,1e-4,1e-4,1e-4)
    result=certify_candidate(tree,obs,PhysicalDomain(.1,.2,.8,1.2,[.2],
                                                       ratio_min=.5,ratio_max=2.))
    r,x=result.theta
    assert .1-1e-12 <= np.hypot(r,x) <= .2+1e-12
    assert .5-1e-12 <= x/r <= 2+1e-12
    assert sum(result.qp_theta)>=.1-1e-8
    assert result.qp_support_lower <= result.qp_primal+1e-9


def test_misspecified_candidate_can_be_rejected_without_a_truth_input():
    tree=TreeSpec(0,[(0,1)],[1])
    obs=Measurements(np.array([[.1]]),np.array([[.03]]),np.array([[.99]]),
                     1.,.001,.001,.001)
    result=certify_candidate(tree,obs,PhysicalDomain(.01,.05,.9,1.1,[.2]))
    assert result.reject
    assert result.ac_sse_lower > result.chi_square_threshold
    assert result.qp_support_lower > result.qp_rejection_threshold
    names={f.name for cls in (TreeSpec,PhysicalDomain,Measurements) for f in fields(cls)}
    assert not any("truth" in name or "true_" in name for name in names)


def test_unobserved_internal_load_is_not_free_compensation():
    tree=TreeSpec(0,[(0,1),(1,2)],[2])
    obs=Measurements(np.array([[.03]]),np.array([[.01]]),np.array([[.99]]),
                     1.,.001,.001,.001)
    with pytest.raises(ValueError,match="unobserved injections"):
        certify_candidate(tree,obs,PhysicalDomain(.01,.03,.9,1.1,{1:.03,2:.05}))


def test_residual_chi_rank_uses_output_projection_not_fitted_parameter_count():
    tree=TreeSpec(0,[(0,1),(1,2),(1,3)],[1,2,3],source_node=1)
    p=np.full((8,3),.02); q=.3*p
    v=exact_ac(tree,[.01,.015,.02],[.005,.007,.01],p,q,source=.04+.01j)
    result=certify_candidate(tree,Measurements(p,q,v,1.,1e-4,1e-4,1e-4),
                             PhysicalDomain(.005,.025,.9,1.1,[.04]*3,
                                            source_power_cap=.1),
                             source_projection=True)
    assert result.residual_chi_df==2*8
    assert result.raw_noise_dimension==3*3*8
    assert sum(result.residual_chi_alpha_split)==pytest.approx(.05)
    assert result.residual_chi_threshold < result.qp_rejection_threshold
    assert not result.residual_chi_reject


def test_residual_chi_expansion_has_correct_pq_and_voltage_square_terms():
    tree=TreeSpec(0,[(0,1),(1,2),(1,3)],[1,2,3])
    r=np.array([.012,.018,.022]); x=.4*r
    p=np.array([[.02,.04,.03],[.04,.02,.025]]); q=.3*p
    v=exact_ac(tree,r,x,p,q)
    rng=np.random.default_rng(885)
    ep,eq,ev=rng.normal(size=(3,2,3))*np.array([2e-4,1e-4,1e-4])[:,None,None]
    h=linear_voltage_matrix(tree,r,x)
    eta=1-v*v-np.c_[p,q]@h.T
    measured=1-(v+ev)**2-np.c_[p+ep,q+eq]@h.T
    expansion=eta-2*v*ev-np.c_[ep,eq]@h.T-ev**2
    np.testing.assert_allclose(measured,expansion,atol=3e-16)
    result=certify_candidate(tree,Measurements(p+ep,q+eq,v+ev,1.,2e-4,1e-4,1e-4),
                             PhysicalDomain(.005,.03,.9,1.1,[.08]*3))
    assert result.qp_support_lower <= float(np.sum(measured**2))+1e-10
    assert result.residual_chi_quadratic_radius > 0


def test_projected_scalar_has_no_information_and_never_rejects():
    tree=TreeSpec(0,[(0,1)],[1],source_node=1)
    obs=Measurements(np.array([[.1]]),np.array([[.02]]),np.array([[.9]]),
                     1.,.001,.001,.001)
    result=certify_candidate(tree,obs,PhysicalDomain(.01,.04,.8,1.1,[.2],
                                                    source_power_cap=.2),
                             source_projection=True)
    assert result.residual_chi_df==0
    assert result.residual_chi_threshold==0
    assert result.ac_sse_lower==0
    assert not result.residual_chi_reject


def test_residual_chi_noisy_root_square_correction_is_included():
    tree=TreeSpec(0,[(0,1)],[1])
    p,q=np.array([[.03]]),np.array([[.01]])
    v=exact_ac(tree,[.02],[.01],p,q)
    fixed=certify_candidate(tree,Measurements(p,q,v,1.,1e-4,1e-4,1e-4),
                            PhysicalDomain(.01,.03,.9,1.1,[.06]))
    noisy=certify_candidate(tree,Measurements(p,q,v,1.,1e-4,1e-4,1e-4,
                                              sigma_root=1e-3),
                            PhysicalDomain(.01,.03,.9,1.1,[.06]))
    assert noisy.residual_chi_linear_radius > fixed.residual_chi_linear_radius
    assert noisy.residual_chi_quadratic_radius > fixed.residual_chi_quadratic_radius


def test_resistance_floor_zero_preserves_default_numeric_results():
    tree=TreeSpec(0,[(0,1),(1,2)],[1,2])
    obs=Measurements(np.array([[.03,.04],[.04,.02]]),
                     np.array([[.01,.014],[.012,.008]]),
                     np.array([[.998,.996],[.997,.996]]),1.,1e-4,1e-4,1e-4)
    domain=PhysicalDomain(.005,.03,.9,1.1,[.07,.07],ratio_min=.2,ratio_max=1.)
    default=certify_candidate(tree,obs,domain,solver="scipy")
    explicit=certify_candidate(tree,obs,replace(domain,r_min=np.zeros(2)),solver="scipy")
    for name in ("qp_primal","qp_support_lower","ac_sse_lower","remainder_norm",
                 "lipschitz_k","qp_rejection_threshold","residual_chi_threshold"):
        assert getattr(default,name)==getattr(explicit,name)
    assert default.theta==explicit.theta
    assert not default.diagnostics["physical_resistance_min_active"]


def test_positive_resistance_floor_has_valid_qp_lower_bound():
    # min (r-.02)^2+(x-.01)^2 with r>=.08 has optimum .0036.
    c,d,lower,upper=_outer_constraints(np.array([.02]),np.array([.2]),
                                      .1,1.,r_min=.08)
    result=qp_with_support_bound(np.eye(2),np.array([.02,.01]),c,d,lower,upper,
                                 solver="scipy")
    assert result["primal"]==pytest.approx(.0036,abs=1e-10)
    assert .0036-1e-8 <= result["support_lower"] <= .0036
    assert result["theta"][0]>=.08-1e-9
    # An independently declared feasible physical point also upper-bounds it.
    feasible=np.array([.09,.03])
    assert result["support_lower"] <= float(np.sum((feasible-[.02,.01])**2))


@pytest.mark.parametrize("theta",[
    [0.,0.,0.,0.], [.01,.6,.9,.001], [-1.,-2.,-3.,-4.], [.3,.3,.3,.3]])
def test_resistance_floor_physical_projection_satisfies_all_domains(theta):
    zmin=np.array([.05,.07]); zmax=np.array([.2,.3]); rmin=np.array([.12,.2])
    point=_physical_theta(np.array(theta),zmin,zmax,.4,1.2,rmin)
    r,x=point[:2],point[2:]
    assert np.all(r>=rmin-1e-14)
    assert np.all(np.hypot(r,x)>=zmin-1e-14)
    assert np.all(np.hypot(r,x)<=zmax+1e-14)
    assert np.all(x/r>=.4-1e-14)
    assert np.all(x/r<=1.2+1e-14)


def test_resistance_floor_does_not_change_ac_remainder_or_noise_envelopes():
    tree=TreeSpec(0,[(0,1),(1,2)],[1,2])
    obs=Measurements(np.array([[.03,.04]]),np.array([[.01,.014]]),
                     np.array([[.998,.996]]),1.,1e-4,1e-4,1e-4)
    domain=PhysicalDomain(.005,.03,.9,1.1,[.07,.07],ratio_min=.2,ratio_max=1.)
    before=certify_candidate(tree,obs,domain,solver="scipy")
    after=certify_candidate(tree,obs,replace(domain,r_min=[.012,.02]),solver="scipy")
    assert after.remainder_norm==before.remainder_norm
    assert after.lipschitz_k==before.lipschitz_k
    assert after.residual_chi_threshold==before.residual_chi_threshold
    assert after.qp_rejection_threshold==before.qp_rejection_threshold
    assert after.diagnostics["physical_resistance_min"]==[.012,.02]
    assert after.diagnostics["physical_resistance_min_active"]
    assert np.all(np.asarray(after.theta)[:2]>=np.array([.012,.02])-1e-12)


@pytest.mark.parametrize("changes",[
    {"r_min":.031},
    {"r_min":.025,"ratio_min":1.},
    {"r_min":-.001},
    {"r_min":np.inf},
    {"r_min":[.01,.02]},
])
def test_conflicting_or_invalid_resistance_floor_fails_explicitly(changes):
    tree=TreeSpec(0,[(0,1)],[1])
    obs=Measurements(np.array([[.03]]),np.array([[.01]]),np.array([[.99]]),
                     1.,.001,.001,.001)
    domain=PhysicalDomain(.005,.03,.9,1.1,[.07])
    with pytest.raises(ValueError,match="r_min|physical parameter domain"):
        certify_candidate(tree,obs,replace(domain,**changes),solver="scipy")
