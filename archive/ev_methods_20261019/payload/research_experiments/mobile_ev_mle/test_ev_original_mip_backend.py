"""Independent linear MIP translation, status, start and bound checks."""
import numpy as np
import pytest
from scipy.optimize import Bounds,LinearConstraint
from scipy.sparse import csr_matrix
from . import ev_original_mip_backend as backend


@pytest.fixture(scope='module')
def licensed_gurobi():
    probe=backend.gurobi_available()
    if not probe['available']:pytest.skip('Gurobi runtime/license unavailable: '+probe['reason'])
    return probe


def check_solution(r,c,bounds,constraints):
    assert r.status==0 and r.success and r.x is not None
    assert r.fun==pytest.approx(np.asarray(c)@r.x,abs=1e-8)
    assert np.all(r.x>=np.asarray(bounds.lb)-1e-8) and np.all(r.x<=np.asarray(bounds.ub)+1e-8)
    for con in constraints if isinstance(constraints,list) else [constraints]:
        if con is None:continue
        value=con.A@r.x
        assert np.all(value>=np.asarray(con.lb)-1e-8) and np.all(value<=np.asarray(con.ub)+1e-8)
    assert r.mip_dual_bound<=r.fun+1e-8
    assert r.mip_dual_bound==pytest.approx(r.fun,abs=1e-8)


@pytest.mark.parametrize('c,integrality,bounds,constraints,expected',[
    ([1.,1.],[1,1],Bounds([0,0],[2,2]),LinearConstraint([[1,1]],1.5,np.inf),2.),
    ([-2.,-1.],[1,1],Bounds([0,0],[1,1]),LinearConstraint(csr_matrix([[1,1]]),-np.inf,1),-2.),
    ([1.,-1.],[1,1],Bounds([-3,-1],[3,2]),LinearConstraint([[1,2]],1,1),-5.),
    ([1.,1.],[0,0],Bounds([-np.inf,-np.inf],[np.inf,np.inf]),
        [LinearConstraint([[1,0]],1,1),LinearConstraint([[0,1]],2,np.inf)],3.),
    ([1.,-1.,2.],[1,0,1],Bounds([0,0,0],[3,3,3]),
        LinearConstraint(csr_matrix([[1,1,0],[0,0,1]]),[1,0],[2,1]),-2.),
])
def test_gurobi_vs_independent_scipy(licensed_gurobi,c,integrality,bounds,constraints,expected):
    g=backend.solve_linear_mip(c,integrality,bounds,constraints,3.,0.,backend='gurobi')
    s=backend.solve_linear_mip(c,integrality,bounds,constraints,3.,0.,backend='scipy')
    check_solution(g,c,bounds,constraints);check_solution(s,c,bounds,constraints)
    assert g.fun==pytest.approx(s.fun,abs=1e-8)==pytest.approx(expected,abs=1e-8)
    assert g.backend_used=='gurobi' and not g.fallback_used
    assert g.threads==1 and g.seed==0 and g.objective_sense=='min' and g.objective_transform=='none'


@pytest.mark.parametrize('kind,lower,expected',[(2,2.,2.),(3,2.1,3.)])
def test_native_semi_domains(licensed_gurobi,kind,lower,expected):
    b=Bounds([lower],[5.9]);con=LinearConstraint([[1]],1,np.inf)
    g=backend.solve_linear_mip([1.],[kind],b,con,3.,0.)
    s=backend.solve_linear_mip([1.],[kind],b,con,3.,0.,backend='scipy')
    assert g.status==0 and s.status==0
    assert g.fun==pytest.approx(expected) and g.fun==pytest.approx(s.fun)


def test_full_and_partial_starts_preserve_problem_and_remain_silent(licensed_gurobi,capsys):
    args=([-2.,-1.],[1,1],Bounds([0,0],[1,1]),LinearConstraint([[1,1]],-np.inf,1),3.,0.)
    full=backend.solve_linear_mip(*args,start=[0.,1.]);partial=backend.solve_linear_mip(*args,start=[1.,np.nan])
    assert full.fun==partial.fun==-2. and full.status==partial.status==0
    assert full.start_used and full.start_provided_count==2
    assert partial.start_used and partial.start_provided_count==1
    captured=capsys.readouterr();assert captured.out=='' and captured.err==''


def test_zero_time_limit_is_limit_not_optimality(licensed_gurobi):
    r=backend.solve_linear_mip(-np.ones(20),np.ones(20),Bounds(np.zeros(20),np.ones(20)),
        LinearConstraint(np.ones((1,20)),-np.inf,9.5),0.,0.,start=np.r_[np.ones(9),np.zeros(11)])
    assert r.status==1 and not r.success and r.gurobi_status==9
    if r.x is not None:assert r.fun==pytest.approx(-r.x.sum())
    if r.fun is not None and r.mip_dual_bound is not None:assert r.mip_dual_bound<=r.fun+1e-8


def test_infeasible_and_unbounded_are_distinct(licensed_gurobi):
    infeasible=backend.solve_linear_mip([1.],[1],Bounds([0],[1]),LinearConstraint([[1]],2,np.inf),3.,0.)
    unbounded=backend.solve_linear_mip([-1.],[0],Bounds([0],[np.inf]),None,3.,0.)
    assert infeasible.status==2 and not infeasible.success and infeasible.x is None
    assert unbounded.status==3 and not unbounded.success and unbounded.x is None


def test_scipy_explicit_start_policy_and_continuous_bound():
    r=backend.solve_linear_mip([-1.],0,Bounds([0],[1]),None,3.,0.,start=[1.],backend='scipy')
    assert r.status==0 and r.fun==r.mip_dual_bound==-1.
    assert r.start_supplied and not r.start_used and r.backend_used=='scipy'
    assert 'explicitly ignored' in r.start_policy


def test_solver_failure_fallback_requires_explicit_request(monkeypatch):
    def unavailable(*args,**kwargs):raise RuntimeError('fixture license unavailable')
    monkeypatch.setattr(backend,'_gurobi',unavailable)
    args=([-1.],[1],Bounds([0],[1]),None,3.,0.)
    no=backend.solve_linear_mip(*args,backend='gurobi')
    yes=backend.solve_linear_mip(*args,backend='auto')
    assert no.status==4 and not no.success and no.x is None and not no.fallback_used
    assert yes.status==0 and yes.fun==-1 and yes.fallback_used and yes.backend_used=='scipy'
    assert 'fixture license unavailable' in yes.fallback_reason


@pytest.mark.parametrize('change',[
    dict(c=[np.nan]),dict(integrality=[1.5]),dict(start=[np.inf]),dict(start=[0.,1.]),
    dict(time_limit=-1.),dict(mip_rel_gap=-.1),dict(backend='unknown'),
    dict(constraints=LinearConstraint([[np.nan]],0,1)),
])
def test_invalid_inputs_fail_before_solver(change):
    args=dict(c=[1.],integrality=[1],bounds=Bounds([0],[1]),constraints=None,
        time_limit=1.,mip_rel_gap=0.,backend='scipy');args.update(change)
    with pytest.raises((ValueError,TypeError)):backend.solve_linear_mip(**args)
