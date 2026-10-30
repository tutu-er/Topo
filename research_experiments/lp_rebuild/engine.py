"""Enumerate legal local hierarchies; solve one continuous LP per family.

The normalized response floor applies during every fit, including final output
refits. It is a resolution assumption, not an inferred physical line impedance.
"""
from __future__ import annotations

from time import perf_counter
from functools import lru_cache
import numpy as np

from research_experiments.rebuild_benchmark.engine import Bank, SCALE, loss, regions
from research_experiments.local_rebuild.model import canonical, compatible, build_model
from research_experiments.local_rebuild.matched import all_hierarchies
from research_experiments.wzzt_retune.engine import PoolLP


class InfeasibleFamily(RuntimeError):pass


def signal_coefficients(p,q,s,calibration):
    """Fixed unit projection: g_r*r+g_x*x lower-bounds response RMS.

    Center time within this training scenario. The direction is fixed before
    fitting. Cauchy--Schwarz guarantees RMS >= this directional projection;
    other supports may still explain the same signal, so it is not significance.
    """
    pp=p[:,s].sum(axis=1);qq=q[:,s].sum(axis=1);pp=pp-pp.mean();qq=qq-qq.mean()
    direction=calibration['a']*pp+calibration['b']*qq
    if np.linalg.norm(direction)<1e-12:direction=pp if np.linalg.norm(pp)>=np.linalg.norm(qq) else qq
    norm=np.linalg.norm(direction)
    if norm<1e-12:return 0.,0.
    direction=direction/norm;scale=np.sqrt(len(s)/(len(p)*p.shape[1]))
    return float(scale*(direction@pp)),float(scale*(direction@qq))


class FloorLP(PoolLP):
    def __init__(self, data, pool, calibration, floor, seconds=5.,mode='depth',reference=1.):
        super().__init__(None,pool,'raw',bound=2.,data=data)
        self.model.Params.TimeLimit=seconds
        self.calibration=calibration;self.floor=float(floor);self.mode=mode
        if not np.isfinite(floor) or floor<0:raise ValueError('nonnegative finite floor required')
        self.floor_indices=[i for i,s in enumerate(self.pool) if 1<len(s)<self.n]
        k=len(self.pool);a,b=calibration['a'],calibration['b']
        if min(a,b,calibration['median_root_depth'])<=0:raise ValueError('positive response calibration required')
        if mode=='depth':coefficients=[(a/calibration['median_root_depth'],b/calibration['median_root_depth']) for i in self.floor_indices]
        elif mode=='signal' and reference>0:
            coefficients=[tuple(v/reference for v in signal_coefficients(self.p/SCALE,self.q/SCALE,self.pool[i],calibration)) for i in self.floor_indices]
        else:raise ValueError('depth or signal mode and positive reference required')
        self.coefficients=np.asarray(coefficients).reshape(-1,2)
        self.floor_rows=self.model.addConstr(self.coefficients[:,0]*self.weights[self.floor_indices]+self.coefficients[:,1]*self.weights[np.asarray(self.floor_indices,dtype=int)+k]>=0.) if floor>0 else None

    def fit(self,family):
        family=canonical(family)
        if family in self.cache:return self.cache[family]
        chosen=set(family)
        if self.floor_rows is not None:self.floor_rows.RHS=np.asarray([self.floor if self.pool[i] in chosen else 0. for i in self.floor_indices])
        try:q=super().fit(family)
        except RuntimeError:
            if self.model.Status==self.gp.GRB.INFEASIBLE:
                self.solves.append(dict(family=family,status=3,normalized_floor=self.floor,floor_mode=self.mode,infeasible=True))
                raise InfeasibleFamily('fixed family violates the chosen response floor')
            raise
        lookup={self.pool[i]:self.coefficients[k] for k,i in enumerate(self.floor_indices)}
        strength=[float(lookup[tuple(s)]@np.asarray([r,x])) for s,r,x in zip(q['supports'],q['r'],q['x']) if tuple(s) in lookup]
        violation=max([self.floor-v for v in strength]+[0.]) if self.floor>0 else 0.
        self.solves[-1]['response_floor_violation']=violation
        self.solves[-1]['normalized_floor']=self.floor
        self.solves[-1]['floor_mode']=self.mode
        if violation>1e-7:raise RuntimeError('response floor lost during fixed-family fit')
        return q


class FloorBank(Bank):
    def __init__(self,data,pool,calibration,*,floor=0.,seconds=5.,mode='depth',reference=1.):
        self.data=data;self.calibration=calibration;self.floor=float(floor);self.seconds=seconds
        self.mode=mode;self.reference=reference
        from rnj_wzzt.estimation import laminar_l1_milp as core
        self.scaled=core._PreparedScenarios(data.labels,tuple(p*SCALE for p in data.p),tuple(q*SCALE for q in data.q),tuple(y*SCALE for y in data.target))
        self.lp=self._make(pool);self.failures=[]

    def _make(self,pool):
        return FloorLP((self.scaled.p[0],self.scaled.q[0],self.scaled.target[0]),pool,self.calibration,self.floor,self.seconds,self.mode,self.reference)

    def extend(self,supports):
        missing=set(canonical(supports))-set(self.lp.pool)
        if not missing:return
        cache,solves=self.lp.cache,self.lp.solves
        other=self._make([*self.lp.pool,*missing]);self.lp.close();self.lp=other
        self.lp.cache.update(cache);self.lp.solves=solves


def internal(point,n):return sum(1<len(s)<n for s in point['supports'])


def criterion(point,n,penalty):return point['objective']+penalty*internal(point,n)


@lru_cache(maxsize=8)
def hierarchy_choices(b):return tuple(all_hierarchies(b))


def local_families(bank,family,region,blocks,virtual):
    """26 families for four blocks, optionally times two for a new region edge."""
    augmented=canonical([*family,region]) if virtual and region not in family else family
    _,meta=build_model(bank.scaled,augmented,region,blocks,bound=2.,cuts='subset')
    keep=meta['keep'];optional=virtual and region not in family and len(region)<bank.data.n
    if optional:keep=canonical(s for s in keep if s!=region)
    families=[]
    for h in hierarchy_choices(len(blocks)):
        unions=[tuple(sorted(i for a in s for i in blocks[a])) for s in h]
        for add_region in ((False,True) if optional else (False,)):
            proposed=canonical([*keep,*unions,*([region] if add_region else [])])
            if all(compatible(s,t) for i,s in enumerate(proposed) for t in proposed[i+1:]):families.append(proposed)
    return sorted(set(families),key=lambda f:(len(f),f)),meta


def pool_phase(bank,current,*,penalty,relative,deadline):
    path=[];events=[];reason='step_limit'
    for iteration in range(3*bank.data.n):
        best=None;complete=True;checked=0
        for c in tuple(bank.lp.pool):
            if len(c)<2 or c in canonical(current['supports']):continue
            if perf_counter()>=deadline:complete=False;break
            checked+=1
            try:q=bank.fit([*[s for s in current['supports'] if compatible(s,c)],c])
            except InfeasibleFamily:continue
            key=(criterion(q,bank.data.n,penalty),internal(q,bank.data.n),canonical(q['supports']))
            if best is None or key<best[0]:best=(key,q)
        gain=0. if best is None else criterion(current,bank.data.n,penalty)-best[0][0]
        gate=1e-10+relative*current['objective'];accept=best is not None and gain>gate
        events.append(dict(kind='pool',complete=complete,checked=checked,gain=gain,threshold=gate,accepted=accept))
        if accept:current=best[1];path.append(current)
        else:reason='pool_stationary' if complete else 'wall_limit';break
        if not complete:reason='wall_limit';break
    return current,path,events,reason


def enumerate_search(bank,initial,kernel,*,seconds=20.,relative=.001,penalty=0.,strategy='joint',b=4):
    if penalty>0 and bank.floor<=0:raise ValueError('exact family-count penalty requires a positive activation floor')
    then=perf_counter();deadline=then+seconds
    current=bank.fit(initial);path=[current];events=[];tried=set();reason='wall_limit'
    if strategy in ('pool','pool_joint'):
        current,p,e,reason=pool_phase(bank,current,penalty=penalty,relative=relative,deadline=deadline)
        path+=p;events+=e
        if strategy=='pool':return dict(path=path,point=current,attempts=events,seconds=perf_counter()-then,stop_reason=reason)
    for iteration in range(3*bank.data.n):
        if perf_counter()>=deadline:reason='wall_limit';break
        family=canonical(current['supports'])
        todo=[q for q in regions(family,bank.data.n,b,kernel) if (family,q) not in tried]
        if not todo:reason='neighborhood_scan_complete';break
        region,blocks,virtual=todo[0];tried.add((family,todo[0]))
        candidates,meta=local_families(bank,family,region,blocks,virtual)
        best=(criterion(current,bank.data.n,penalty),internal(current,bank.data.n),canonical(current['supports'])),current
        complete=True;before=len(bank.lp.solves);checked=0
        for proposed in candidates:
            if perf_counter()>=deadline:complete=False;break
            checked+=1
            try:q=bank.fit(proposed)
            except InfeasibleFamily:continue
            key=(criterion(q,bank.data.n,penalty),internal(q,bank.data.n),canonical(q['supports']))
            if key<best[0]:best=key,q
        score=criterion(current,bank.data.n,penalty);gain=score-best[0][0]
        gate=1e-10+relative*current['objective'];accepted=gain>gate
        events.append(dict(kind='joint',region=region,blocks=blocks,virtual=virtual,
            candidate_families=len(candidates),checked=checked,complete=complete,new_LPs=len(bank.lp.solves)-before,
            before_score=score,after_score=best[0][0],gain=gain,threshold=gate,accepted=accepted,
            lower_bound=best[0][0] if complete else None,scope='enumerated local family domain'))
        if accepted:current=best[1];path.append(current)
        if not complete:reason='wall_limit';break
    else:reason='step_limit'
    return dict(path=path,point=current,attempts=events,seconds=perf_counter()-then,stop_reason=reason)


def select_output(bank,points,validation,*,resolution=.16):
    candidates=[];cal=bank.calibration
    for source_index,q in enumerate(points):
        family=[s for s,r,x in zip(q['supports'],q['r'],q['x']) if len(s)==1 or cal['a']*r+cal['b']*x+1e-10>=resolution*cal['median_root_depth']]
        try:fitted=bank.fit(family)
        except InfeasibleFamily:continue
        candidates.append(dict(point=fitted,validation=loss(validation,fitted),internal=internal(fitted,bank.data.n),source_index=source_index))
    if not candidates:raise InfeasibleFamily('no feasible output candidate')
    best=min(q['validation'] for q in candidates)
    i,q=min(((i,q) for i,q in enumerate(candidates) if q['validation']<=best+1e-12),key=lambda t:(len(t[1]['point']['supports']),t[1]['validation'],t[0]))
    return dict(q,index=q['source_index'])
