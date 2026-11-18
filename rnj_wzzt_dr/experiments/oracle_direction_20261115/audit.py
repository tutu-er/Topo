"""Independent oracle-direction verification from saved planning arrays.

The known physical response is used openly as an oracle. No AC solve, noisy
future observation, topology fit or optimizer is called by this verifier.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import traceback

import numpy as np

from ..distribution_panel_20261115 import audit as panel

base=panel.base
ROOT=panel.ROOT


def array_hash(value):
    array=np.ascontiguousarray(value,dtype=np.float64)
    return hashlib.sha256(str(array.shape).encode('ascii')+array.tobytes()).hexdigest()


def resource_box(config,labels,forecast,base_kw):
    """Original per-hour bounds, with energy equations relaxed only for a bound."""
    lower=np.zeros((24,len(config['resources'])));upper=lower.copy();clock=(8+np.arange(24)+.5)%24
    for col,(r,unit) in enumerate(zip(config['resources'],config['action_units'],strict=True)):
        if unit=='kvar':
            cap=float(r['capacity_kvar']);lo=np.full(24,-cap);hi=-lo
            hours=r.get('available_clock_hours')
            if hours is not None and hours!=list(range(24)):
                aa,bb=hours;valid=(clock>=aa)&(clock<bb) if aa<bb else (clock>=aa)|(clock<bb)
                lo[~valid]=hi[~valid]=0.
        else:
            cap=float(r['contract_kw'])
            if r['kind']=='pv':
                export=np.maximum(0.,-forecast[:,labels.index(int(r['node_label']))]*base_kw).reshape(24,4).min(1)
                lo=np.zeros(24);hi=np.where((clock>=10)&(clock<14),np.minimum(cap,export),0.)
            else:
                if r['kind'] not in ('ev','thermal'):raise ValueError('Unknown P contract')
                valid=(clock>=18)|(clock<7) if r['kind']=='ev' else np.ones(24,bool)
                lo=np.where(valid,-cap,0.);hi=-lo
        lower[:,col],upper[:,col]=lo,hi
    return lower,upper


def interval_quadratic_upper(Q,a,lower,upper):
    """Independent termwise outer bound for u'Qu+a'u (a includes factor two)."""
    Q,a,lo,hi=(np.asarray(x,np.longdouble) for x in (Q,a,lower,upper))
    values=[]
    values.extend(np.maximum(a*lo,a*hi).ravel())
    for i in range(lo.shape[1]):
        squares=np.stack((lo[:,i]**2,hi[:,i]**2,np.where((lo[:,i]<=0)&(hi[:,i]>=0),0.,lo[:,i]**2)))
        values.extend(np.max(Q[:,i,i]*squares,axis=0))
        for j in range(i+1,lo.shape[1]):
            products=np.stack((lo[:,i]*lo[:,j],lo[:,i]*hi[:,j],hi[:,i]*lo[:,j],hi[:,i]*hi[:,j]))
            values.extend(np.max((Q[:,i,j]+Q[:,j,i])*products,axis=0))
    raw=np.sum(values,dtype=np.longdouble);absolute=np.sum(np.abs(values),dtype=np.longdouble)
    return dict(upper=float(raw)+1e-10*max(1.,float(absolute)),padding=1e-10*max(1.,float(absolute)))


def maximization_history(log_path,trace_path,verified_z):
    """A previous MAX upper bound below a later feasible z is contradictory."""
    callback=[];printed=[];terminal=None
    if trace_path is not None:
        for number,line in enumerate(Path(trace_path).read_text(encoding='utf-8-sig').splitlines(),1):
            if not line.strip():continue
            row=json.loads(line)
            if row.get('event')=='mip' and row.get('bound') is not None and np.isfinite(row['bound']):
                callback.append(dict(line=number,bound=float(row['bound']),seconds=row.get('native_runtime_seconds')))
            if row.get('event')=='native_end':terminal=row.get('attributes')
    if log_path is not None:
        for number,line in enumerate(Path(log_path).read_text(encoding='utf-8-sig').splitlines(),1):
            tokens=line.split()
            for ix,token in enumerate(tokens):
                if ix>=2 and re.fullmatch(r'\d+(?:\.\d+)?%',token):
                    try:value=float(tokens[ix-1])
                    except ValueError:continue
                    if np.isfinite(value):printed.append(dict(line=number,bound=value,text=line.strip()))
    tolerance=1e-7*max(1.,abs(verified_z))
    conflicts=[r for r in callback if r['bound']<verified_z-tolerance]
    displayed=[r for r in printed if r['bound']<verified_z-max(tolerance,1e-5)]
    final=(terminal or {}).get('ObjBound')
    final_conflict=final is not None and np.isfinite(final) and final<verified_z-tolerance
    return dict(callback_count=len(callback),printed_count=len(printed),
        minimum_callback_upper=min(callback,key=lambda r:r['bound']) if callback else None,
        minimum_printed_upper=min(printed,key=lambda r:r['bound']) if printed else None,
        callback_conflicts=conflicts,printed_conflicts=displayed,terminal_conflict=bool(final_conflict),
        conflict=bool(conflicts or displayed or final_conflict),terminal_native_attributes=terminal,
        interpretation='Maximum objective: saved solver upper bounds are compared to this verified feasible z. A conflict withdraws the native certificate, not the feasible action.')


def planning_boundary(path):
    tree=ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    runner=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_case')
    calls=[n for n in ast.walk(runner) if isinstance(n,ast.Call)]
    seals=[n for n in calls if ast.unparse(n.func)=='io.write' and n.args and isinstance(n.args[0],ast.BinOp)
        and isinstance(n.args[0].right,ast.Constant) and n.args[0].right.value=='seal.json']
    eta_reads=[n for n in calls if ast.unparse(n.func)=='np.load']
    forbidden=[n for n in calls if ast.unparse(n.func) in ('old.context','previous.old.context','previous.context','physical_state')]
    solves=[n for n in calls if ast.unparse(n.func)=='solver.plan']
    passed=(len(seals)==1 and len(eta_reads)==1 and not forbidden and len(solves)==1
        and solves[0].lineno<seals[0].lineno<eta_reads[0].lineno
        and [ast.unparse(a) for a in solves[0].args]==['case','dict(g, mode=mode)','seeds'])
    return dict(passed=passed,source=base.ref(path),seal_line=seals[0].lineno if seals else None,
        eta_read_line=eta_reads[0].lineno if eta_reads else None,
        scope='Static runner check: solver receives oracle geometry, resource case and old initial pool; execution eta is opened after sealing both actions. No actual future P/Q/V context call in this runner.')


def physical_matrix(metadata, branches, labels):
    """Build first-order squared-voltage H* from original ohmic branches."""
    labels=list(map(int,labels));positions={node:i for i,node in enumerate(labels)}
    if len(positions)!=len(labels):raise ValueError('Duplicate measured labels')
    root=int(metadata['root_bus']);children=defaultdict(list);parents={};edges=[]
    for row in branches:
        if str(row['status']).lower() not in ('true','1'):continue
        u,v=int(row['from_bus']),int(row['to_bus'])
        if v in parents:raise ValueError('Physical graph is not a rooted tree')
        parents[v]=u;children[u].append(v);edges.append(row)
    if root in parents:raise ValueError('Root has an incoming branch')
    visited=set();active=set();desc={}
    def visit(node):
        if node in active:raise ValueError('Cycle in physical graph')
        active.add(node);group={positions[node]} if node in positions else set()
        for child in children[node]:group|=visit(child)
        active.remove(node);visited.add(node);desc[node]=group;return group
    visit(root)
    if not set(labels)<=visited or not set(parents)<=visited:raise ValueError('Disconnected physical graph')
    n=len(labels);H=np.zeros((n,2*n));truth=set()
    zbase=float(metadata['base_kv'])**2/float(metadata['base_mva'])
    if not np.isfinite(zbase) or zbase<=0:raise ValueError('Invalid impedance base')
    for row in edges:
        group=desc[int(row['to_bus'])]
        if not group:continue
        truth.add(tuple(sorted(group)));mask=np.array([i in group for i in range(n)],float)
        block=np.outer(mask,mask)
        H[:,:n]+=2*float(row['r_ohm'])/zbase*block
        H[:,n:]+=2*float(row['x_ohm'])/zbase*block
    return H,truth


def direction_metrics(physical,H,power,sigma2,target,competitors,history=None):
    """Direct norms and exact linear-model score differences, no expanded QP."""
    physical,H,power=np.asarray(physical,float),np.asarray(H,float),np.asarray(power,float)
    if physical.shape!=H.shape[1:] or power.shape[1]!=physical.shape[1] or sigma2<=0:
        raise ValueError('Invalid response/power dimensions or score scale')
    y=power@physical.T;pred=np.einsum('tc,knc->ktn',power,H)
    residual=y-pred[target]
    e=float(np.square(residual).sum()/(2*sigma2))
    sse=np.square(y[None]-pred).sum(axis=(1,2))
    history=np.zeros(len(H)) if history is None else np.asarray(history,float)
    rows=[]
    for j in competitors:
        d=pred[target]-pred[j]
        J=float(np.square(d).sum()/(2*sigma2))
        signed=float(np.sum(residual*d)/sigma2)
        h=float((history[j]-history[target])/(2*sigma2))
        direct=float(h+(sse[j]-sse[target])/(2*sigma2))
        if not np.isclose(direct,h+J+signed,atol=1e-8,rtol=1e-12):
            raise ArithmeticError('Direct linear likelihood identity differs')
        rows.append(dict(competitor_native=int(j),J=J,bias=signed,h=h,linear_logodds=direct,
            cauchy_lower_bound=float(h+J-2*np.sqrt(max(0.,e*J)))))
    return dict(e=e,J=[r['J'] for r in rows],rows=rows,native_new_sse=sse,
                target=int(target),competitors=list(map(int,competitors)))


def achievable_z(current,baseline,mode):
    if mode not in ('protected','distance_only'):raise ValueError('Unknown oracle direction mode')
    if current['competitors']!=baseline['competitors']:raise ValueError('Competitor order changed')
    J,J0=np.asarray(current['J']),np.asarray(baseline['J'])
    if not J.size:raise ValueError('No foreign-family native competitors')
    distance=(J-J0)/np.maximum(1.,J0)
    distance_limit=float(distance.min());error_limit=None
    zero_error=baseline['e']==0.
    if mode=='protected':
        if zero_error:
            feasible=current['e']<=1e-14
            return dict(z=distance_limit if feasible else None,distance_limit=distance_limit,error_limit=None,
                zero_baseline_error=True,zero_error_requirement_passed=feasible,
                strictly_decreased_error=False)
        error_limit=float(1-current['e']/baseline['e'])
    return dict(z=distance_limit if error_limit is None else min(distance_limit,error_limit),
                distance_limit=distance_limit,error_limit=error_limit,zero_baseline_error=zero_error,
                strictly_decreased_error=bool(current['e']<baseline['e']))


def direct_constraints(current,baseline,mode,z,tolerance=1e-7):
    if not np.isfinite(z):raise ValueError('Nonfinite common gain variable')
    z=float(z);J,J0=np.asarray(current['J']),np.asarray(baseline['J'])
    margins=J-J0-z*np.maximum(1.,J0)
    error_slack=float((1-z)*baseline['e']-current['e']) if mode=='protected' else None
    if mode not in ('protected','distance_only'):raise ValueError('Unknown oracle mode')
    feasible=z>=-tolerance and bool(np.all(margins>=-tolerance)) and (error_slack is None or error_slack>=-tolerance)
    return dict(passed=feasible,z=z,minimum_distance_margin=float(margins.min()),
        distance_margins=margins.tolist(),protected_error_slack=error_slack,
        achievable=achievable_z(current,baseline,mode),e=current['e'],e0=baseline['e'],
        error_constraint_scope='Full forecast response residual energy under known first-order physical H*, not AC or sampled measurement error')


def linear_selection(score,H,power,physical,internal,active,truth):
    """Unchanged historical score plus noiseless first-order synthetic new day."""
    y=np.asarray(power)@np.asarray(physical).T
    losses=np.array([np.square(y-power@h.T).sum() for h in H])
    history=np.array([r['history_sse'] for r in score['native_bank']])
    total=history+losses;scaled=total/(score['cells']+y.size)/score['reference_mse']
    eligible=np.flatnonzero(scaled<=scaled.min()+score['validation_tie'])
    chosen=min(eligible,key=lambda i:(internal[i],scaled[i],i))
    proper=lambda groups:{g for g in groups if 1<len(g)<len(score['labels'])}
    correct=[i for i,s in enumerate(active) if proper(s)==proper(truth)]
    reps=[m['index'] for m in score['models']]
    logp=-(total[reps]-min(total[reps]))/(2*score['sigma2']);logp-=np.log(np.exp(logp).sum())
    weights=np.exp(logp);ptrue=float(sum(weights[k] for k,i in enumerate(reps) if i in correct))
    best=int(np.argmax(weights))
    return dict(native_selected_index=int(chosen),native_map_correct=bool(chosen in correct),
        native_full_support_exact=bool(active[chosen]==truth),p_true=ptrue,
        wrong_high=bool(reps[best] not in correct and weights[best]>=.9),true_high=ptrue>=.9,
        native_history_sse=history,native_new_sse=losses,native_total_sse=total,
        representative_log_probability=logp,
        interpretation='Original measured historical validation SSE plus noiseless oracle-linear new-day SSE; not AC recovery')


class Audit(panel.Audit):
    """Only saved history/forecast/contracts/network/eta; no old.context call."""
    def context_history(self,entry):
        score=base.read(self.checked(entry['score']));native=base.read(self.checked(entry['native_copy']))
        val=base.native.arrays(self.checked(entry['files']['validation.npz']))
        labels=list(map(int,val['labels']));n=len(labels);H=[];active=[];internal=[]
        for row in native['frozen_candidates']:
            point=row['point'];matrix=np.zeros((n,2*n));supports=set()
            for group,r,x in zip(point['supports'],point['r'],point['x'],strict=True):
                block=np.ix_(group,group);matrix[:,:n][block]+=r;matrix[:,n:][block]+=x
                if max(r,x)>native['config']['zero_tol']:supports.add(tuple(group))
            H.append(matrix);active.append(supports)
            internal.append(sum(1<len(g)<n for g in point['supports']))
        H=np.array(H);history=np.sum((val['Y'][None]-np.einsum('tc,knc->ktn',np.c_[val['P'],val['Q']],H))**2,axis=(1,2))
        self.same('full_native_H',[r['H'] for r in score['native_bank']],H,atol=1e-13,rtol=1e-13)
        self.same('frozen_history_sse',[r['history_sse'] for r in score['native_bank']],history,atol=1e-14)
        self.require(score['cells']==val['Y'].size,'History cell count differs')
        directory=ROOT/'rnj_wzzt_dr'/'physical'/'cases'/entry['case']
        metadata=base.read(self.file(directory/'metadata.json'))
        with self.file(directory/'branches.csv').open(encoding='utf-8-sig',newline='') as stream:branches=list(csv.DictReader(stream))
        physical,truth=physical_matrix(metadata,branches,labels)
        proper=lambda groups:{g for g in groups if 1<len(g)<n}
        correct=[k for k,s in enumerate(active) if proper(s)==proper(truth)]
        self.require(bool(correct),'Oracle correct-internal candidate absent')
        target=min(correct,key=lambda k:(history[k],k))
        competitors=[k for k,s in enumerate(active) if proper(s)!=proper(truth)]
        self.require(entry['oracle_target']==target and entry['oracle_true_indices']==correct and entry['oracle_competitors']==competitors,'Oracle target/foreign full bank selection differs')
        self.same('physical_base_kw',entry['base_kw'],1000*metadata['base_mva'],atol=0,rtol=0)
        return dict(score=score,H=H,history=history,labels=labels,n=n,physical=physical,truth=truth,
                    active=active,internal=internal,target=target,competitors=competitors,correct=correct)

    def values(self,ctx,power):
        return direction_metrics(ctx['physical'],ctx['H'],power,ctx['score']['sigma2'],ctx['target'],ctx['competitors'],ctx['history'])

    def linear(self,ctx,power,saved):
        values=self.values(ctx,power)
        selection=linear_selection(ctx['score'],ctx['H'],power,ctx['physical'],ctx['internal'],ctx['active'],ctx['truth'])
        logp=-selection['native_total_sse']/(2*ctx['score']['sigma2']);logp-=logp.max();weights=np.exp(logp);weights/=weights.sum()
        logodds=np.array([r['linear_logodds'] for r in values['rows']])
        for key,value in dict(J=values['J'],e=values['e'],h=[r['h'] for r in values['rows']],
            bias=[r['bias'] for r in values['rows']],logodds=logodds,native_total_sse=selection['native_total_sse'],
            minimum_logodds=float(logodds.min()),true_native_weight=float(weights[ctx['correct']].sum())).items():
            self.same('linear_'+key,saved[key],value,atol=1e-7,rtol=1e-10)
        self.require(saved['native_selected_index']==selection['native_selected_index'] and saved['internal_correct']==selection['native_map_correct'],'Exact linear native selection differs')
        self.require(saved['strongest_foreign_native']==ctx['competitors'][int(logodds.argmin())],'Strongest foreign point differs')
        return dict(**selection,metrics=values,true_native_weight=float(weights[ctx['correct']].sum()))

    def action(self,ctx,config,forecast,action,entry,saved_cost=None,saved_quantities=None,request=None):
        devices,violations,dp,dq=panel.resource_values(config,action,ctx['labels'],forecast[:,:ctx['n']],entry['base_kw'])
        request=action if request is None else request
        bill=base.procurement(config,request,action,.5,.5)
        if saved_cost is not None:
            for key,value in bill['settled'].items():self.same('price_'+key,saved_cost[key],value,atol=1e-8,rtol=1e-10)
            self.same('planned_activation_price',saved_cost['planned_activation_yuan'],bill['planned']['activation_yuan'])
            self.same('planned_total_price',saved_cost['planned_total_yuan'],bill['planned']['total_yuan'])
        units=np.array(config['action_units']);quantities=dict(active_abs_kwh=float(abs(action[:,units=='kW']).sum()),reactive_abs_kvarh=float(abs(action[:,units=='kvar']).sum()))
        if saved_quantities is not None:
            for key,value in quantities.items():self.same('amount_'+key,saved_quantities[key],value)
        return forecast+np.c_[dp,dq],dict(devices=devices,violations=violations,bill=bill,quantities=quantities)

    def saved_direct(self,saved,current,baseline,mode):
        achieved=achievable_z(current,baseline,mode)
        for key,value in dict(J=current['J'],J0=baseline['J'],e=current['e'],e0=baseline['e'],direct_z=achieved['z'],distance_only_z=achieved['distance_limit']).items():
            self.same('direct_'+key,saved[key],value,atol=1e-7,rtol=1e-10)
        return achieved

    def case(self,source,entry,protocol):
        folder=source/'cases'/entry['id'];self.where=str(folder)
        completed=base.read(self.file(folder/'complete.json'));self.references(completed)
        self.require(completed['status']=='complete','Incomplete oracle case')
        seal=base.read(self.file(folder/'seal.json'));self.references(seal)
        self.require(seal['actual_new_day_opened'] is False,'Actual future data opened')
        ctx=self.context_history(entry);config=base.read(self.checked(seal['resource_case']))
        self.catalogue(entry,config,ctx)
        planner=base.native.arrays(self.checked(seal['planner']));forecast=planner['forecast']
        prior=base.read(self.checked(entry['prior_seal']));old_planner=base.native.arrays(self.checked(prior['planner']))
        self.same('original_forecast',forecast,old_planner['forecast'],atol=0,rtol=0)
        self.require(config==base.read(self.checked(prior['resource_case'])),'Original resource contract changed')
        geometry=base.read(self.checked(seal['geometry']));self.references(geometry)
        g=base.native.arrays(self.checked(geometry['arrays']))
        self.same('physical_Hstar',g['Hstar'],ctx['physical'],atol=1e-13,rtol=1e-13)
        self.same('Hc',g['Hc'],ctx['H'][ctx['target']],atol=0,rtol=0)
        self.same('D',g['D'],ctx['H'][ctx['target']][None]-ctx['H'][ctx['competitors']],atol=0,rtol=0)
        self.same('geometry_forecast',g['forecast'],forecast,atol=0,rtol=0)
        self.same('geometry_target',g['target_native'],ctx['target'],atol=0,rtol=0)
        self.same('geometry_competitors',g['competitor_native_indices'],ctx['competitors'],atol=0,rtol=0)
        self.same('score_sigma2',g['sigma2'],ctx['score']['sigma2'],atol=0,rtol=0)
        self.same('geometry_base',g['base_kw'],entry['base_kw'],atol=0,rtol=0)
        self.same('common_distance_scale',g['j_scales'],np.maximum(1.,g['J_c']),atol=0,rtol=0)
        B=np.zeros((2*ctx['n'],len(config['resources'])))
        for col,(r,unit) in enumerate(zip(config['resources'],config['action_units'],strict=True)):
            node=ctx['labels'].index(int(r['node_label']))
            if unit=='kW':B[node,col]=1/entry['base_kw'];B[ctx['n']+node,col]=r.get('q_over_p',0.)/entry['base_kw']
            else:B[ctx['n']+node,col]=1/entry['base_kw']
        self.same('control_units_map',g['control_map'],B,atol=0,rtol=0)
        for label,responses in [('J',g['D']),('e',(ctx['physical']-ctx['H'][ctx['target']])[None])]:
            expected=[]
            for response in responses:
                response0=forecast@response.T;effect=response@B
                Q=np.repeat((4*effect.T@effect/(2*ctx['score']['sigma2']))[None],24,axis=0)
                a=2*np.einsum('htn,nr->hr',response0.reshape(24,4,ctx['n']),effect)/(2*ctx['score']['sigma2'])
                c=np.sum(response0**2)/(2*ctx['score']['sigma2']);expected.append((Q,a,c))
            for ix,suffix in enumerate(('q','a','c')):
                value=np.array([r[ix] for r in expected]);value=value if label=='J' else value[0]
                self.same('expanded_'+label+'_'+suffix,g[label+'_'+suffix],value,atol=1e-8,rtol=1e-11)
        baseline=self.values(ctx,forecast);self.linear(ctx,forecast,base.read(self.file(folder/'zero.json')))
        lower,upper=resource_box(config,ctx['labels'],forecast,entry['base_kw'])
        interval_bounds=[interval_quadratic_upper(Q,a,lower,upper)['upper']/scale
                         for Q,a,scale in zip(g['J_q'],g['J_a'],g['j_scales'],strict=True)]
        amplitude=np.maximum(abs(lower),abs(upper))
        radius=np.repeat(np.einsum('knr,hr->khn',abs(g['D']@B),amplitude),4,axis=1)
        box_J=np.sum((abs(np.einsum('tc,knc->ktn',forecast,g['D']))+radius)**2,axis=(1,2))/(2*ctx['score']['sigma2'])
        self.same('full_response_box_upper',g['J_box_upper'],box_J,atol=1e-8,rtol=1e-11)
        self.same('caller_z_box_upper',g['z_box_upper'],max(0.,float(np.min((box_J-g['J_c'])/g['j_scales']))),atol=1e-8,rtol=1e-11)
        plans=base.read(self.checked(seal['plans']));self.require(set(plans)==set(protocol['modes']),'Quote mode ledger differs')
        seed_arrays=base.native.arrays(self.checked(seal['initial_points']))
        original_seeds=base.native.arrays(self.checked(entry['initial_points']))
        self.require(set(seed_arrays)==set(original_seeds),'Initial portfolio keys differ')
        for key in seed_arrays:self.same('original_seed_'+key,seed_arrays[key],original_seeds[key],atol=0,rtol=0)
        seed_arrays.update(baseline_zero=np.zeros((24,len(config['resources']))),explicit_zero=np.zeros((24,len(config['resources']))))
        for mode,quote in plans.items():
            self.references(quote);action=planner[mode]
            self.same('stored_action',base.native.arrays(self.file(folder/(mode+'_action.npz')))['action'],action,atol=0,rtol=0)
            power,physical=self.action(ctx,config,forecast,action,entry,quote['cost'],quote['quantities'])
            current=self.values(ctx,power);achieved=self.saved_direct(quote['direct'],current,baseline,mode)
            selection=self.linear(ctx,power,quote['linear'])
            self.require(achieved['z']>=-1e-7,'Returned action has negative achievable common improvement')
            direct=direct_constraints(current,baseline,mode,float(quote.get('z',0.)))
            self.require(direct['passed'],'Returned z violates directly computed oracle constraints')
            self.require(bool(quote['positive_direction_found'])==(achieved['z']>1e-6),'Positive-direction flag differs')
            if quote.get('bounds'):
                bounds=quote['bounds'];z=float(quote['z'])
                self.require(quote['action_sha256']==array_hash(action),'Final action hash differs')
                analytic=min(interval_bounds)
                if mode=='protected' and baseline['e']>0:analytic=min(analytic,1.)
                self.same('independent_original_domain_box_bound',bounds['analytic_box_upper_bound'],analytic,atol=1e-8,rtol=1e-10)
                self.same('verified_z_lower_bound',bounds['lower_bound'],z,atol=1e-9,rtol=1e-10)
                self.require(bounds['upper_bound']>=z-1e-7,'MAX upper bound below verified feasible z')
                self.require(quote['objective_sense']=='maximize' and not quote['cost_is_objective'],'Wrong bound objective')
                self.require(not quote['infeasibility_proved'] and not quote['global_optimality_verified'],'Unsupported global/infeasibility claim')
                self.same('reported_activation_cost',quote['cost_yuan'],physical['bill']['planned']['activation_yuan'])
                self.require(quote['z']>=quote['initial']['z']-1e-7,'Worse objective than feasible retained initial')
                initial=quote['initial'];self.require(initial['name'] in seed_arrays,'Selected initial missing from original common pool')
                self.require(initial['action_sha256']==array_hash(seed_arrays[initial['name']]),'Selected initial hash differs')
                self.require({r['name'] for r in initial['candidates']}==set(seed_arrays),'Missing or added initial candidate')
                feasible_seed_z=[]
                for seed in initial['candidates']:
                    name=seed['name'];seedpoint=seed_arrays[name]
                    if seed.get('action_sha256'):self.require(seed['action_sha256']==array_hash(seedpoint),'Original initial action changed')
                    spower,sphysical=self.action(ctx,config,forecast,seedpoint,entry)
                    sd=self.values(ctx,spower)
                    if seed.get('feasible'):
                        check=direct_constraints(sd,baseline,mode,seed['z'])
                        self.require(check['passed'],'Claimed feasible initial fails direct original equations')
                        self.same('initial_cost',seed['cost_yuan'],sphysical['bill']['planned']['activation_yuan'])
                        feasible_seed_z.append(seed['z'])
                    elif seed.get('direct_audit'):
                        self.require(not direct_constraints(sd,baseline,mode,0.,tolerance=1e-9)['passed'] or
                            abs(achievable_z(sd,baseline,mode)['z'])<=1e-8,'Rejected initial disagrees with original feasible equations')
                self.same('best_verified_initial_z',initial['z'],max(feasible_seed_z),atol=1e-10,rtol=1e-10)
                if quote['retained_verified_initial']:self.same('retained_initial_action',action,seed_arrays[initial['name']],atol=0,rtol=0)
                traces=quote.get('diagnostic_files',{})
                history=maximization_history(self.checked(traces['native_log']) if 'native_log' in traces else None,
                    self.checked(traces['progress']) if 'progress' in traces else None,z)
                terminal=(history['terminal_native_attributes'] or {}).get('ObjBound')
                if terminal is not None and bounds['raw_native_upper_bound'] is not None:
                    self.same('raw_terminal_max_upper',bounds['raw_native_upper_bound'],terminal,atol=0,rtol=0)
                if bounds['native_upper_bound_withdrawn']:
                    self.same('withdrawn_native_effective_upper',bounds['upper_bound'],analytic,atol=1e-8)
                effective=analytic if history['conflict'] else bounds['upper_bound']
                history.update(audit_effective_upper_bound=effective,audit_relative_gap=(effective-z)/max(abs(z),1e-10),
                    additional_native_withdrawal=bool(history['conflict'] and not bounds['native_upper_bound_withdrawn']))
            else:history=None
            self.quote_records.append(dict(id=entry['id'],mode=mode,status=quote['status'],direct=direct,achievable=achieved,
                linear=selection,physical=physical,bounds=quote.get('bounds'),source=quote.get('source'),seconds=quote.get('seconds'),
                native_bound_history=history,source_quote=base.ref(folder/(mode+'_plan.json'))))
        deliveries=base.read(self.file(folder/'deliveries.json'))
        self.require(len(deliveries)==len(plans)*protocol['repeats'],'Delivery denominator differs')
        expected={(m,r) for m in plans for r in range(protocol['repeats'])}
        self.require({(r['mode'],r['repeat']) for r in deliveries}==expected,'Duplicate or missing paired delivery')
        for row in deliveries:
            self.references(row)
            if row['status']!='complete':self.failed_executions.append(dict(id=entry['id'],**row));continue
            mode,repeat=row['mode'],row['repeat'];action=planner[mode]
            rd=folder/f'execution{repeat:02d}';eta=base.native.arrays(self.file(rd/'eta.npz'))['eta']
            prior_eta=base.native.arrays(self.checked(row['eta_source']))['eta']
            self.same('original_eta',eta,prior_eta,atol=0,rtol=0)
            self.require(eta.shape==(len(config['resources']),) and np.all((eta>=0)&(eta<=1)),'Invalid common resource response')
            expected_delivery=action*eta[None]
            for j,r in enumerate(config['resources']):
                if r['kind']=='pv':
                    node=ctx['labels'].index(int(r['node_label']));cap=np.minimum(r['contract_kw'],np.maximum(0.,-forecast[:,node]*entry['base_kw']).reshape(24,4).min(1))
                    expected_delivery[:,j]=np.minimum(expected_delivery[:,j],cap)
            delivered=base.native.arrays(self.file(rd/(mode+'.npz')))['delivered']
            self.same('forecast_based_delivery',delivered,expected_delivery,atol=0,rtol=0)
            power,physical=self.action(ctx,config,forecast,delivered,entry,row['cost'],row['quantities'],request=action)
            current=self.values(ctx,power);achieved=self.saved_direct(row['direct'],current,baseline,mode)
            selection=self.linear(ctx,power,row['linear'])
            self.records.append(dict(id=entry['id'],mode=mode,repeat=repeat,achievable=achieved,linear=selection,physical=physical))
        old=base.read(self.file(folder/'old_action_linear_references.json'))
        for name,saved in old.items():
            power,_=self.action(ctx,config,forecast,old_planner[name],entry,saved['cost'])
            self.linear(ctx,power,saved['linear']);self.saved_direct(saved['direct'],self.values(ctx,power),baseline,'distance_only')
        return dict(id=entry['id'],quotes=len(plans),delivery_rows=len(deliveries),true_native_indices=ctx['correct'],
            full_support_exact_native_indices=[k for k,s in enumerate(ctx['active']) if s==ctx['truth']],
            source_completion=base.ref(folder/'complete.json'),source_seal=base.ref(folder/'seal.json'),
            old_action_linear_references=base.ref(folder/'old_action_linear_references.json'),zero=base.ref(folder/'zero.json'))


def run(source,destination):
    source,destination=Path(source).resolve(),Path(destination).resolve()
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir(parents=True);audit=Audit();cases=[]
    protocol=base.read(audit.file(source/'protocol.json'));audit.references(protocol)
    audit.require(protocol['oracle'] and set(protocol['modes'])=={'distance_only','protected'},'Unknown oracle scope/modes')
    for reference in protocol['sources']:
        path=Path(reference['path']);snapshot=source/'source_snapshot'/path.relative_to(ROOT)
        audit.file(snapshot,reference['sha256'])
    boundary=planning_boundary(source/'source_snapshot'/Path(__file__).relative_to(ROOT).parent/'run.py')
    audit.require(boundary['passed'],'Runner information boundary differs')
    completed=base.read(audit.file(source/'complete.json'))
    audit.require(completed['status']=='complete' and completed['source_unchanged'],'Incomplete or changed source run')
    entries=base.read(audit.checked(protocol['inputs']))
    for entry in entries:
        try:cases.append(audit.case(source,entry,protocol))
        except Exception as exc:audit.errors.append(dict(id=entry['id'],error=repr(exc),traceback=traceback.format_exc()))
    expected_quotes=len(entries)*len(protocol['modes']);expected_rows=expected_quotes*protocol['repeats']
    for name,rows in [('quotes',audit.quote_records),('records',audit.records)]:
        with (destination/(name+'.jsonl')).open('w',encoding='utf-8') as stream:
            for row in rows:panel.json_line(stream,row)
    source_copy=destination/'audit_source.py';shutil.copyfile(__file__,source_copy)
    passed=(not audit.errors and not audit.failed_executions and len(audit.quote_records)==expected_quotes and len(audit.records)==expected_rows)
    result=dict(status='passed' if passed else 'failed',source_protocol=base.ref(source/'protocol.json'),source_complete=base.ref(source/'complete.json'),
        source_inputs=[dict(path=p,sha256=h) for p,h in sorted(audit.hashes.items())],audit_source=base.ref(source_copy),
        expected_quotes=expected_quotes,audited_quotes=len(audit.quote_records),expected_delivery_rows=expected_rows,audited_delivery_rows=len(audit.records),
        errors=audit.errors,failed_executions=audit.failed_executions,cases=cases,maxima=dict(audit.maxima),
        planning_boundary=boundary,native_bound_conflicts=sum(bool((q.get('native_bound_history') or {}).get('conflict')) for q in audit.quote_records),
        quotes=base.ref(destination/'quotes.jsonl'),records=base.ref(destination/'records.jsonl'),
        scope='Saved forecast, original measured history, known physical first-order response and original eta. No new AC, noise draws, topology fit or optimization. Exact linear recovery is distinct from AC recovery.')
    base.write(destination/'report.json',result);base.write(destination/'complete.json',dict(status=result['status'],report=base.ref(destination/'report.json')))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();result=run(args.run,args.out)
    print(json.dumps({k:result[k] for k in ('status','expected_quotes','audited_quotes','expected_delivery_rows','audited_delivery_rows','errors')},ensure_ascii=False))
    if result['status']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
