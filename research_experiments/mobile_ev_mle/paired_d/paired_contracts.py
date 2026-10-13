"""Pure contracts for a future fresh paired D runner.

No project imports, generator, AC, optimizer, selector, or file mutation.
This is a reviewable schema/acceptance layer, not an implemented experiment.
"""
from copy import deepcopy
from dataclasses import dataclass
import math
import numpy as np

METHODS = ('rolling_common_initial', 'openloop_common_initial')
FAMILIES = ('star','clade_01','clade_02','clade_12')
SCHEDULE = dict(pilot_seeds=[941101,941102],pilot_seconds_threshold=45.,
                calibration_count=32,calibration_start_seed=940001,
                evaluation_count=48,evaluation_start_seed=950001,
                no_new_history_after_utc='2026-10-13T03:00:00+00:00',
                total_checkpoint_utc='2026-10-13T03:00:00+00:00',
                alpha_total=.1,relative_fraction=.5)


@dataclass(frozen=True)
class PairScore:
    relative: float
    absolute: float
    complete: bool
    required_fit_count: int
    failed_fit_count: int


def pair_score(first_losses,rolling_losses,openloop_losses,truth_index,*,success):
    """Evaluator-only; max over shared H14 and two common-initial H20 fits."""
    a=np.asarray([first_losses,rolling_losses,openloop_losses],float)
    ok=np.asarray(success,bool)
    if a.shape!=(3,4) or ok.shape!=a.shape:
        raise ValueError('one shared first round plus two final banks of four required')
    valid=ok & np.isfinite(a) & (a>=0)
    failures=int((~valid).sum())
    if truth_index is None:
        return PairScore(math.inf,math.inf,False,12,failures)
    if isinstance(truth_index,bool) or not isinstance(truth_index,(int,np.integer)) or not 0<=truth_index<4:
        raise ValueError('truth index must label the declared full audit bank')
    if failures:
        return PairScore(math.inf,math.inf,False,12,failures)
    return PairScore(float(np.max(a[:,truth_index]-a.min(axis=1))),
                     float(np.max(a[:,truth_index])),True,12,0)


def upper_quantile(values,alpha=.05):
    a=np.asarray(values,float)
    if a.ndim!=1 or np.isnan(a).any() or np.isneginf(a).any():
        raise ValueError('scores must preserve positive infinities; no NaN or negative infinity')
    if not 0<alpha<1:raise ValueError('alpha out of range')
    n=len(a);rank=math.ceil((n+1)*(1-alpha))
    return dict(threshold=float(np.sort(a)[rank-1]) if rank<=n else math.inf,
                rank=rank,sample_count=n,rank_tail_bound=max(0.,(n+1-rank)/(n+1)))


def common_initial(first_fit_rows):
    """No truth R/X or final-fit warm states are accepted by this interface."""
    by_name={f['name']:f for f in first_fit_rows}
    if set(by_name)!=set(FAMILIES) or len(first_fit_rows)!=4:
        raise ValueError('missing or duplicated shared first fit')
    out={}
    for name in FAMILIES:
        f=by_name[name]
        if not f.get('success') or f.get('loss') is None or not math.isfinite(f['loss']) or f['loss']<0:
            raise ValueError('shared first fit unavailable: paired abstention required')
        r,x=np.asarray(f['r_pu'],float),np.asarray(f['x_pu'],float)
        edge_count=4 if name=='star' else 5
        if r.shape!=(edge_count,) or x.shape!=r.shape or not np.isfinite(r).all() or not np.isfinite(x).all():
            raise ValueError('invalid canonical R/X shape or values')
        if not ((r>=1e-6).all() and (r<=.1).all() and (x>=1e-6).all() and (x<=.1).all()):
            raise ValueError('R/X initial outside frozen physical bounds')
        out[name]={'r':r.copy(),'x':x.copy()}
    return out


def saved_action_pair(first_actual_plan,realized_plan):
    """Only already selected plan payloads; no truth/path/future-data argument."""
    a,b=deepcopy(first_actual_plan),deepcopy(realized_plan)
    for p in [a,b]:
        for key,shape in [('p_plus_kw',(8,3)),('p_minus_kw',(8,3)),('u_kw',(8,3)),('energy_kwh',(9,))]:
            if np.asarray(p[key],float).shape!=shape or not np.isfinite(p[key]).all():
                raise ValueError('invalid saved plan schema')
        if not isinstance(p.get('arcs'),list):raise ValueError('saved selected arcs required')
    for key in ['p_plus_kw','p_minus_kw','u_kw']:
        if not np.array_equal(np.asarray(a[key])[:2],np.asarray(b[key])[:2]):
            raise ValueError('first actual prefix differs')
    if not np.allclose(np.asarray(a['energy_kwh'])[:3],np.asarray(b['energy_kwh'])[:3],rtol=0,atol=2e-6):
        raise ValueError('first prefix SOC differs')
    prefix=lambda p:[arc for arc in p['arcs'] if arc['departure']<2]
    if prefix(a)!=prefix(b):raise ValueError('first prefix route differs')
    return {'openloop':a,'rolling':b}


def operation_audit(plan,config,tolerance=2e-6):
    """Independent recurrence for the frozen charge-only 8-slot public task."""
    n=config['n_slots'];m=config['n_sites'];dt=config['dt_hours']
    if (n,m)!=(8,3) or config['allow_discharge'] or config['travel_slots']!=1:
        raise ValueError('only frozen charge-only one-slot-travel task supported')
    pp=np.asarray(plan['p_plus_kw'],float);pm=np.asarray(plan['p_minus_kw'],float);u=np.asarray(plan['u_kw'],float)
    if pp.shape!=(n,m) or pm.shape!=pp.shape or u.shape!=pp.shape:raise ValueError('plan shape')
    if (pp<-tolerance).any() or abs(pm).max()>tolerance or abs(u-pp+pm).max()>tolerance:
        raise ValueError('charge/discharge/net input inconsistency')
    if (pp>config['p_max_kw']+tolerance).any():raise ValueError('power maximum')
    access=np.asarray(config['access'],bool)
    sites=np.full(n,-1,int);drive=np.zeros(n);cursor=0;site=config['initial_site']
    for arc in sorted(plan['arcs'],key=lambda a:a['departure']):
        d,e,o,v=arc['departure'],arc['arrival'],arc['origin'],arc['destination']
        if d!=cursor or o!=site or e!=d+1 or not 0<=o<m or not 0<=v<m:
            raise ValueError('route continuity or travel length')
        travel=o!=v
        expected_energy=config['travel_energy_kwh'] if travel else 0.
        if abs(arc['travel_energy_kwh']-expected_energy)>tolerance:raise ValueError('driving energy')
        if travel:
            if abs(u[d]).max()>tolerance:raise ValueError('charging in transit')
            drive[d]=expected_energy
        else:
            sites[d]=o
            if not access[d,o] or np.sum(pp[d])-pp[d,o]>tolerance:raise ValueError('site/access')
        cursor=e;site=v
    if cursor!=n:raise ValueError('route does not span complete horizon')
    E0=config['initial_soc']*config['battery_capacity_kwh']
    increment=dt*(config['eta_charge']*pp.sum(axis=1)-pm.sum(axis=1)/config['eta_discharge'])-drive
    energy=np.r_[E0,E0+np.cumsum(increment)]
    if abs(energy-np.asarray(plan['energy_kwh'],float)).max()>tolerance:raise ValueError('SOC recurrence')
    lo=config['soc_min']*config['battery_capacity_kwh'];hi=config['soc_max']*config['battery_capacity_kwh']
    before_charge=energy[:-1]-drive
    if energy.min()<lo-tolerance or energy.max()>hi+tolerance or before_charge.min()<lo-tolerance:
        raise ValueError('SOC bounds')
    grid=dt*u.sum()
    if abs(grid-config['net_grid_energy_kwh'])>tolerance:raise ValueError('grid task reset or mismatch')
    if config.get('max_throughput_kwh') is not None and dt*(pp+pm).sum()>config['max_throughput_kwh']+tolerance:
        raise ValueError('throughput')
    if config.get('terminal_site') is not None and site!=config['terminal_site']:raise ValueError('terminal site')
    if config.get('terminal_energy_kwh') is not None and energy[-1]<config['terminal_energy_kwh']-tolerance:
        raise ValueError('terminal energy')
    if config.get('require_terminal_connection',True) and sites[-1]<0:raise ValueError('terminal connection')
    return dict(grid_energy_kwh=float(grid),driving_energy_kwh=float(drive.sum()),sites=sites.tolist(),
                prefix=dict(grid_energy_kwh=float(dt*u[:2].sum()),energy_kwh=float(energy[2]),
                            site=next(a['destination'] for a in plan['arcs'] if a['arrival']==2)))
