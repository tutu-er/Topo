"""Pure/mock tests only: no actual historical generation, AC, or fits."""
import ast
from copy import deepcopy
import inspect
from pathlib import Path
import numpy as np
import pytest
try:
    from . import paired_contracts as c
except ImportError:
    import paired_contracts as c


def config():
    access=np.ones((8,3),bool);access[:2,2]=False
    return dict(n_slots=8,n_sites=3,dt_hours=.25,p_max_kw=6.,allow_discharge=False,
                travel_slots=1,travel_energy_kwh=.2,access=access.tolist(),
                initial_soc=.4,battery_capacity_kwh=60.,soc_min=.1,soc_max=.9,
                eta_charge=.95,eta_discharge=.95,initial_site=0,net_grid_energy_kwh=6.,
                require_terminal_connection=True)


def manual_plan(second_move=False):
    u=np.zeros((8,3));u[0,0]=4.
    arcs=[dict(departure=0,arrival=1,origin=0,destination=0,travel_energy_kwh=0.),
          dict(departure=1,arrival=2,origin=0,destination=1,travel_energy_kwh=.2)]
    if second_move:
        u[2,1]=4.;u[4:,2]=4.
        arcs.append(dict(departure=2,arrival=3,origin=1,destination=1,travel_energy_kwh=0.))
        arcs.append(dict(departure=3,arrival=4,origin=1,destination=2,travel_energy_kwh=.2))
        arcs += [dict(departure=t,arrival=t+1,origin=2,destination=2,travel_energy_kwh=0.) for t in range(4,8)]
        energy=[24.,24.95,24.75,25.7,25.5,26.45,27.4,28.35,29.3]
    else:
        u[2:,1]=10/3
        arcs += [dict(departure=t,arrival=t+1,origin=1,destination=1,travel_energy_kwh=0.) for t in range(2,8)]
        energy=[24.,24.95,24.75]+[24.75+(k+1)*.95*(10/3)*.25 for k in range(6)]
    return dict(arcs=arcs,p_plus_kw=u.tolist(),p_minus_kw=np.zeros_like(u).tolist(),
                u_kw=u.tolist(),energy_kwh=energy)


def first_fits():
    return [dict(name=n,success=True,loss=10.,r_pu=[.012]*(4 if n=='star' else 5),
                 x_pu=[.006]*(4 if n=='star' else 5),p=[[999]],source_p=[999]) for n in c.FAMILIES]


def test_max_statistic_includes_shared_first_and_both_new_finals_once():
    s=c.pair_score([9,2,3,4],[4,3,1,2],[8,3,2,1],0,success=np.ones((3,4),bool))
    assert s.relative==7 and s.absolute==9 and s.required_fit_count==12


@pytest.mark.parametrize('bank,candidate',[(0,2),(1,3),(2,1)])
def test_any_required_failure_is_infinite_and_not_deleted(bank,candidate):
    success=np.ones((3,4),bool);success[bank,candidate]=False
    s=c.pair_score([1,2,3,4],[2,3,4,5],[3,4,5,6],0,success=success)
    assert not s.complete and s.failed_fit_count==1 and np.isinf(s.relative) and np.isinf(s.absolute)


def test_rank32_preserves_single_infinite_failure_and_original_sample_count():
    q=c.upper_quantile(list(range(31))+[np.inf])
    assert q['sample_count']==32 and q['rank']==32 and np.isinf(q['threshold'])
    assert q['rank_tail_bound']==pytest.approx(1/33)


def test_new_quantile_is_sample_max_and_strict_ties_do_not_reject():
    q=c.upper_quantile([5.]*32)
    assert q['threshold']==5 and not 5>q['threshold']


def test_common_initial_only_R_X_no_latent_or_truth_warm_state():
    initial=c.common_initial(first_fits())
    assert set(initial)==set(c.FAMILIES)
    assert all(set(v)=={'r','x'} for v in initial.values())
    bad=first_fits();bad[1]['success']=False
    with pytest.raises(ValueError,match='abstention'):c.common_initial(bad)


def test_saved_actions_have_equal_prefix_but_allow_feedback_suffix():
    pair=c.saved_action_pair(manual_plan(),manual_plan(True))
    assert not np.array_equal(pair['rolling']['u_kw'],pair['openloop']['u_kw'])
    assert set(inspect.signature(c.saved_action_pair).parameters)=={'first_actual_plan','realized_plan'}


def test_prefix_route_power_and_SOC_mismatch_is_not_repaired():
    a=manual_plan();b=manual_plan(True);b['u_kw'][0][0]=5
    with pytest.raises(ValueError,match='prefix'):c.saved_action_pair(a,b)
    b=manual_plan(True);b['energy_kwh'][2]=24
    with pytest.raises(ValueError,match='SOC'):c.saved_action_pair(a,b)


def test_manual_two_routes_share_first_state_and_task_but_different_driving_cost():
    a=c.operation_audit(manual_plan(),config());b=c.operation_audit(manual_plan(True),config())
    assert a['grid_energy_kwh']==pytest.approx(6) and b['grid_energy_kwh']==pytest.approx(6)
    assert a['prefix']==b['prefix']==dict(grid_energy_kwh=1.,energy_kwh=24.75,site=1)
    assert a['driving_energy_kwh']==pytest.approx(.2) and b['driving_energy_kwh']==pytest.approx(.4)


def test_driving_deduction_twice_or_SOC_reset_is_rejected():
    p=manual_plan();p['energy_kwh'][2]=24.55
    with pytest.raises(ValueError,match='recurrence'):c.operation_audit(p,config())
    p=manual_plan();p['u_kw'][1][0]=1;p['p_plus_kw'][1][0]=1
    with pytest.raises(ValueError,match='transit'):c.operation_audit(p,config())


def test_seed_sets_are_fresh_disjoint_and_pilot_excluded():
    cal=set(range(940001,940033));evaluation=set(range(950001,950049));pilot=set(c.SCHEDULE['pilot_seeds'])
    old=set(range(920001,920065))|set(range(930001,930097))|set(range(720001,720129))|set(range(730001,730257))
    assert not cal&evaluation and not (cal|evaluation)&pilot and not (cal|evaluation|pilot)&old


def test_source_observe_reads_fixed_noise_without_drawing_RNG_static_contract():
    source=Path(r'C:\Users\23761\.codex\worktrees\36a1\Topo\research_experiments\mobile_ev_mle\simulation.py')
    tree=ast.parse(source.read_text(encoding='utf-8'))
    observe=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='observe')
    attributes=[node.attr for node in ast.walk(observe) if isinstance(node,ast.Attribute)]
    assert 'noise' in attributes
    assert not {'normal','uniform','random','default_rng','generate_state','spawn'}&set(attributes)
    assert any(isinstance(node,ast.Subscript) and isinstance(node.value,ast.Attribute) and node.value.attr=='noise' for node in ast.walk(observe))
