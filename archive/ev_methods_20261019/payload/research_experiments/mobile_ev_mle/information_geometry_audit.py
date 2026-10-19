"""Independent one-shot nuisance projection of saved AC information trials.

Only saved H-only centers, actual H/N meters and saved power plans are used.
The analytic AC derivative kernel is shared with likelihood, but row assembly
is history_white_jacobian and the full nuisance projection is independent of
StructuredInformationEvaluator. No search/evaluate method is called. Results
validate local geometry arithmetic, not nonlinear or global identification.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from .information_context import history_white_jacobian, meter_from_record
from .likelihood import Observations, RadialCandidate

HERE = Path(__file__).resolve().parent
SUPPORTS = {(0, 1), (0, 2), (1, 2)}
BASE_KW = 20.
# Independent channel-major versus time-major assembly, then two-stage versus
# one-shot projection have different floating point operations. Relative 1e-7
# covers roundoff in small residuals after removing large whitened columns.
# The output preserves observed errors and rank-cutoff separation for review.
MATRIX_RTOL, MATRIX_ATOL = 1e-7, 1e-7
SCORE_RTOL, SCORE_ATOL = 1e-7, 1e-9


def read_json(path):
    path = Path(path)
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == '.gz' else raw)


def _serial(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k): _serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_serial(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def write_json(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(_serial(record), sort_keys=True, allow_nan=False)+'\n').encode()
    path.write_bytes(gzip.compress(raw, mtime=0) if path.suffix == '.gz' else raw)


def _normalized_svd(b, rcond, guard=3.):
    if not (b.ndim == 2 and np.isfinite(b).all() and 0 < rcond < 1):
        raise ValueError('finite nuisance matrix and valid rcond required')
    scale = np.linalg.norm(b, axis=0)
    scale[scale == 0] = 1.
    u, singular, _ = np.linalg.svd(b/scale, full_matrices=False)
    cutoff = rcond*singular[0] if singular.size else 0.
    rank = int(np.count_nonzero(singular > cutoff))
    near = singular[(singular >= cutoff/guard) & (singular <= cutoff*guard)] if cutoff else np.empty(0)
    record = dict(rank=rank, columns=b.shape[1], rows=b.shape[0], cutoff=float(cutoff),
        largest_singular=float(singular[0]) if singular.size else 0.,
        smallest_retained=float(singular[rank-1]) if rank else None,
        largest_omitted=float(singular[rank]) if rank < len(singular) else None,
        retained_to_cutoff=float(singular[rank-1]/cutoff) if rank and cutoff else None,
        near_cutoff_singular=near.tolist(), zero_column_count=int(np.count_nonzero(np.linalg.norm(b, axis=0) == 0)),
        column_normalization=True, rcond=float(rcond), rank_guard_factor=guard)
    return u[:, :rank], record


def full_project(signal, nuisance, *, rcond=1e-10):
    """One SVD on all other shared lines AND all H/N latent states."""
    a, b = np.asarray(signal, float), np.asarray(nuisance, float)
    if a.ndim != 2 or a.shape[0] != b.shape[0] or not np.isfinite(a).all():
        raise ValueError('finite matching signal/nuisance arrays required')
    basis, record = _normalized_svd(b, rcond)
    residual = a-basis@(basis.T@a)
    scales = np.linalg.norm(b, axis=0); scales[scales == 0] = 1.
    record['normalized_nuisance_orthogonality_error'] = float(np.linalg.norm((b/scales).T@residual))
    record['rank_guard_passed'] = not record['near_cutoff_singular']
    info = residual.T@residual
    return dict(information=.5*(info+info.T), rank_record=record)


def joined_geometry(history_jacobian, future_jacobian, shared_width, target_columns, *, rcond=1e-10):
    """Shared beta columns stack; H/N state columns have disjoint supports."""
    h, n = np.asarray(history_jacobian), np.asarray(future_jacobian)
    if h.shape[1] < shared_width or n.shape[1] < shared_width:
        raise ValueError('shared line width exceeds full Jacobian width')
    other = [j for j in range(shared_width) if j not in target_columns]
    a = np.vstack((h[:, target_columns], n[:, target_columns]))
    b = np.zeros((len(h)+len(n), len(other)+h.shape[1]+n.shape[1]-2*shared_width))
    b[:, :len(other)] = np.vstack((h[:, other], n[:, other]))
    b[:len(h), len(other):len(other)+h.shape[1]-shared_width] = h[:, shared_width:]
    b[len(h):, len(other)+h.shape[1]-shared_width:] = n[:, shared_width:]
    return full_project(a, b, rcond=rcond)


def _observations(record):
    return Observations(**{k: None if v is None else np.asarray(v, float) for k, v in record.items()})


def _compare(actual, saved, label, errors, comparisons, *, scalar=False, rtol=None, atol=None):
    try:
        a, b = np.asarray(actual, float), np.asarray(saved, float)
        valid = a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
        difference = float(np.linalg.norm((a-b).ravel())) if valid else None
        scale = max(float(np.linalg.norm(a.ravel())), float(np.linalg.norm(b.ravel()))) if valid else None
        default_rtol, default_atol = (SCORE_RTOL, SCORE_ATOL) if scalar else (MATRIX_RTOL, MATRIX_ATOL)
        rtol = default_rtol if rtol is None else rtol
        atol = default_atol if atol is None else atol
        passed = bool(valid and difference <= atol+rtol*scale)
    except (TypeError, ValueError):
        difference, scale, passed = None, None, False
    comparisons.append(dict(label=label, absolute_error=difference, reference_norm=scale,
        relative_error=None if not scale or difference is None else difference/scale, passed=passed))
    if not passed: errors.append(label)


def _boundaries(center, physical):
    candidate = center['candidate']; e = candidate.supports.index(tuple(center['target_support']))
    active = dict(target_geometry_lower_boundary=bool(center['r'][e] <= 1e-12 or center['x'][e] <= 1e-12))
    valid = True
    for name in ('r', 'x'):
        values = np.delete(center[name], e); lo, hi = 1e-6, .1; tol=1e-7*(hi-lo)
        active[name+'_other_lower'] = int(np.count_nonzero(values-lo <= tol))
        active[name+'_other_upper'] = int(np.count_nonzero(hi-values <= tol))
        valid &= bool(np.all(values >= lo) and np.all(values <= hi) and 0 <= center[name][e] <= hi)
    for name in ('p', 'q', 'source_p', 'source_q'):
        lo, hi = (-.5, .5) if name.startswith('source') else (-1., 1.)
        values = np.asarray(physical[name]); tol=1e-7*(hi-lo)
        active[name+'_lower'] = int(np.count_nonzero(values-lo <= tol))
        active[name+'_upper'] = int(np.count_nonzero(hi-values <= tol))
        valid &= bool(np.all(values >= lo) and np.all(values <= hi))
    v = abs(physical['voltage'])
    valid &= bool(np.all(v >= .9) and np.all(v <= 1.1))
    return dict(active=active, physical_domain_passed=valid, voltage_min=float(v.min()), voltage_max=float(v.max()),
        ac_equation_residual=float(physical['ac_equation_residual']),
        power_balance_residual=float(physical['power_balance_residual']))


class _TrialGeometry:
    def __init__(self, trial):
        self.trial=trial; self.h=_observations(trial['history_observations'])
        self.hm=meter_from_record(trial['measurement_models']['history'])
        self.nm=meter_from_record(trial['measurement_models']['future'])
        self.rcond=float(trial['config']['information']['rcond'])
        self.ratio=float(trial['device_constraints']['q_over_p'])
        self.omega=.5*np.diag(np.asarray(trial['config']['information']['scales_pu'],float)**2)
        self.centers={}; self.cache={}; self.hcache={}; self.ac_calls=0
        for saved in trial['selection'].get('centres',[]):
            c={**saved}; c['r']=np.asarray(c['r'],float); c['x']=np.asarray(c['x'],float)
            c['candidate']=RadialCandidate.from_supports(c['name'],self.h.n,c['supports'])
            if c['candidate'].supports != tuple(tuple(s) for s in c['supports']):
                raise ValueError('saved canonical support order differs')
            if c['candidate'].supports[0] != tuple(range(self.h.n)) or c['candidate'].parents[0] != -1:
                raise ValueError('source must use node1 after the full-support edge')
            if c['id'] in self.centers: raise ValueError('duplicate center id')
            self.centers[c['id']]=c

    def geometry(self, center, power, site, *, history=False):
        if history:
            p,q=(np.array(center['history_'+name],float) for name in ('p','q'))
            sp,sq=(np.array(center['history_'+name],float) for name in ('source_p','source_q'))
            root=np.array(center['history_root_v'],float); obs=self.h; model=self.hm
            if not np.array_equal(root,obs.root_v): raise ValueError('center root differs from H calibration')
        else:
            f=center['future']; p,q=(np.array(f[name],float) for name in ('p','q'))
            sp,sq=(np.array(f[name],float) for name in ('source_p','source_q'))
            root=np.array(f['root_v'],float); model=self.nm
            # EV enters terminal TOTAL P/Q exactly once; source remains at the
            # same physical common junction, with freely projected states.
            p[:,site] += np.asarray(power)/BASE_KW
            q[:,site] += self.ratio*np.asarray(power)/BASE_KW
            obs=Observations(p=np.zeros_like(p),q=np.zeros_like(q),v=np.ones_like(p),root_v=root,
                master_p=np.zeros(len(p)),master_q=np.zeros(len(p)),
                source_p=None if self.h.source_p is None else np.zeros(len(p)),
                source_q=None if self.h.source_q is None else np.zeros(len(p)))
        fit=SimpleNamespace(r=center['r'],x=center['x'],p=p,q=q,source_p=sp,source_q=sq)
        self.ac_calls+=1
        jac,physical=history_white_jacobian(center['candidate'],fit,obs,model,estimate_source=True)
        bounds=_boundaries(center,physical)
        return jac,bounds

    def rebuild(self, center_id, power, site):
        key=(center_id,tuple(np.asarray(power,float)),int(site))
        if key in self.cache:return self.cache[key]
        c=self.centers[center_id]; m=c['candidate'].m
        e=c['candidate'].supports.index(tuple(c['target_support'])); target=(e,m+e)
        other=[j for j in range(2*m) if j not in target]
        if center_id not in self.hcache:
            h,hb=self.geometry(c,power,site,history=True)
            hi=full_project(h[:,target],h[:,other+list(range(2*m,h.shape[1]))],rcond=self.rcond)
            _,hs=_normalized_svd(h[:,2*m:],self.rcond)
            z,zb=self.geometry(c,np.zeros(len(power)),site)
            zi=joined_geometry(h,z,2*m,target,rcond=self.rcond)
            _,zs=_normalized_svd(z[:,2*m:],self.rcond)
            self.hcache[center_id]=(h,hb,hi,hs,zi,zs,zb)
        h,hb,hi,hs,zi,zs,zb=self.hcache[center_id]
        n,nb=self.geometry(c,power,site)
        ni=joined_geometry(h,n,2*m,target,rcond=self.rcond)
        _,ns=_normalized_svd(n[:,2*m:],self.rcond)
        result=dict(history=hi,joint=ni,zero=zi,history_state_rank=hs['rank'],future_state_rank=ns['rank'],
            zero_state_rank=zs['rank'],history_bounds=hb,future_bounds=nb,zero_bounds=zb,target_columns=target,
            shared_line_count=2*m,H_latent_state_count=h.shape[1]-2*m,N_latent_state_count=n.shape[1]-2*m)
        self.cache[key]=result
        return result


def _evaluation_audit(geometry, tag, label, evaluation, plan, expected, errors, comparisons):
    site=int(plan['site']); p,q=np.asarray(plan['p'],float),np.asarray(plan['q'],float)
    if p.shape != (4,3) or q.shape != p.shape or not 0 <= site < 3: raise ValueError('invalid saved plan shape/site')
    if np.any(np.delete(p,site,axis=1) != 0) or not np.allclose(q,geometry.ratio*p,rtol=0,atol=1e-12):
        errors.append(label+':EV_single_site_QP_interface')
    rows=[]; scores={}; task={}
    for target in evaluation.get('targets',[]):
        cid=target['id']; prefix=label+':'+cid
        try:
            if cid not in expected or cid in scores: raise ValueError('unexpected or duplicate saved target')
            c=geometry.centers[cid]; actual=geometry.rebuild(cid,BASE_KW*p[:,site],site)
            ih=actual['history']['information']; ij=actual['joint']['information']; iz=actual['zero']['information']
            delta=ij-iz; gain=ij-ih; score=.5*float(np.trace(geometry.omega@delta))
            for key,value in (('information_history',ih),('information_joint',ij),('information_zero_EV',iz),
                ('information_EV_delta',delta),('information_future_gain',gain)):
                _compare(value,target.get(key),prefix+':'+key,errors,comparisons)
            _compare(geometry.omega,target.get('weight'),prefix+':weight',errors,comparisons,rtol=1e-12,atol=1e-15)
            for key,value in (('score',score),('future_gain_min_eigenvalue',float(np.linalg.eigvalsh(gain).min()))):
                _compare(value,target.get(key),prefix+':'+key,errors,comparisons,scalar=True)
            _compare(.5*np.diag(geometry.omega)*np.diag(ij),target['axes_weighted_half_distance'],prefix+':axes_joint',errors,comparisons)
            _compare(.5*np.diag(geometry.omega)*np.diag(delta),target['axes_EV_delta'],prefix+':axes_delta',errors,comparisons)
            _compare(np.linalg.eigvalsh(delta),target['EV_delta_eigenvalues'],prefix+':EV_delta_eigenvalues',errors,comparisons)
            ranks=dict(history_state_rank=actual['history_state_rank'],future_state_rank=actual['future_state_rank'],
                history_line_nuisance_rank=actual['history']['rank_record']['rank']-actual['history_state_rank'],
                joint_line_nuisance_rank=actual['joint']['rank_record']['rank']-actual['history_state_rank']-actual['future_state_rank'])
            for key,value in ranks.items():
                if target.get(key) != value: errors.append(prefix+':'+key)
            zero_signature=[actual['zero_state_rank'],actual['zero']['rank_record']['rank']-actual['history_state_rank']-actual['zero_state_rank']]
            if target.get('rank_signature_zero_EV') != zero_signature: errors.append(prefix+':zero_rank_signature')
            if target.get('rank_change_relative_to_zero_EV') != ([ranks['future_state_rank'],ranks['joint_line_nuisance_rank']] != zero_signature):
                errors.append(prefix+':rank_change_relative_to_zero')
            for key,bound in (('active_boundaries',actual['future_bounds']),('history_active_boundaries',actual['history_bounds'])):
                if target.get(key) != bound['active']: errors.append(prefix+':'+key)
                if not bound['physical_domain_passed']: errors.append(prefix+':physical_domain')
            if any(not actual[k]['rank_record']['rank_guard_passed'] for k in ('history','joint','zero')):
                errors.append(prefix+':full_nuisance_rank_guard')
            if target['target_columns'] != list(actual['target_columns']) or target['target_support'] != c['target_support'] or target['anchor_id'] != c['anchor_id'] or target.get('candidate') != c['name']:
                errors.append(prefix+':target_interface')
            if target.get('metadata',{}) != c.get('metadata',{}): errors.append(prefix+':target_metadata')
            weight=float(target.get('target_task_weight',1/3))
            if not np.isfinite(weight) or weight <= 0: errors.append(prefix+':positive_task_weight')
            expected_weight=1/3 if tag=='uniform' or 'target_weight_record' not in geometry.trial['selection'] else next(
                r['weight'] for r in geometry.trial['selection']['target_weight_record']['target_records'] if r['target_support']==c['target_support'])
            _compare(weight,expected_weight,prefix+':target_task_weight',errors,comparisons,scalar=True,rtol=1e-12,atol=1e-15)
            scores[cid]=score; task[cid]=weight
            rows.append(dict(id=cid,anchor_id=c['anchor_id'],support=c['target_support'],score=score,target_task_weight=weight,
                anchor_kind=c.get('metadata',{}).get('historical_anchor_kind','v1_saved_H_fit'),
                information_history=ih,information_joint=ij,information_zero_EV=iz,
                ranks={k:actual[k]['rank_record'] for k in ('history','joint','zero')},
                latent_state_counts=[actual['H_latent_state_count'],actual['N_latent_state_count']],
                shared_line_count=actual['shared_line_count'],history_bounds=actual['history_bounds'],future_bounds=actual['future_bounds']))
        except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,KeyError,StopIteration) as exc:
            errors.append(prefix+':rebuild:'+type(exc).__name__+':'+str(exc))
    complete=len(scores)==len(expected) and not evaluation.get('failures')
    if bool(evaluation.get('success')) != complete: errors.append(label+':required_target_completeness')
    anchors=[]
    if complete:
        for aid in dict.fromkeys(geometry.centers[cid]['anchor_id'] for cid in expected):
            ids=[cid for cid in expected if geometry.centers[cid]['anchor_id']==aid]
            if len(ids)!=3: errors.append(label+':all_three_supports')
            _compare(sum(task[cid] for cid in ids),1.,label+':task_weight_sum:'+aid,errors,comparisons,scalar=True,rtol=1e-12,atol=1e-15)
            score=float(sum(task[cid]*scores[cid] for cid in ids))
            saved=next((r for r in evaluation.get('pairs',[]) if r['anchor_id']==aid),None)
            if saved is None:errors.append(label+':missing_anchor:'+aid)
            else:
                _compare(score,saved['score'],label+':anchor_score:'+aid,errors,comparisons,scalar=True)
                if set(saved['target_ids']) != set(ids):errors.append(label+':anchor_target_ids:'+aid)
                if 'target_task_weights' in saved:
                    _compare([task[cid] for cid in saved['target_ids']],saved['target_task_weights'],label+':anchor_weights:'+aid,errors,comparisons)
            anchors.append(dict(anchor_id=aid,score=score))
        _compare(min(a['score'] for a in anchors),evaluation['score'],label+':worst_anchor_score',errors,comparisons,scalar=True)
    elif evaluation.get('score') is not None or evaluation.get('pairs'):
        errors.append(label+':failed_evaluation_aggregate')
    return dict(label=label,site=site,power_kw=BASE_KW*p[:,site],power_pu=p[:,site],saved_success=bool(evaluation.get('success')),
        saved_failures=evaluation.get('failures',[]),expected_target_count=len(expected),rebuilt_target_count=len(rows),
        targets=rows,anchors=anchors,gradient_check='not_in_this_matrix_arithmetic_audit')


def _search_diagnostics(search):
    bounds=[]
    def visit(value,path=''):
        if isinstance(value,dict):
            selected={k:v for k,v in value.items() if k in ('incumbent','best_bound','raw_incumbent','raw_best_bound',
                'absolute_gap','mip_gap','bound_scope','objective_scope','bound_consistent','global_design_optimality_certified')}
            if any(k in selected for k in ('incumbent','best_bound','raw_incumbent','raw_best_bound')):
                bounds.append(dict(path=path,record=selected))
            for key,item in value.items():
                if key not in ('targets','evaluations','site_evaluations','evaluation','best_constant_evaluation'):
                    visit(item,path+'/'+str(key))
        elif isinstance(value,list):
            for index,item in enumerate(value): visit(item,path+'/'+str(index))
    visit(search)
    return dict(saved_success=search.get('success'),failures=search.get('failures',[]),
        bound_scope=search.get('bound_scope'),global_design_optimality_certified=search.get('global_design_optimality_certified'),
        operational_bounds=bounds)


def audit_trial_geometry(trial):
    """Audit one saved v1/v2 trial with independent full H+N nuisance blocks."""
    errors=[];comparisons=[];evaluations=[];failed_searches=[]
    try:
        g=_TrialGeometry(trial);selection=trial['selection'];centers=g.centers
        first=next(iter(centers.values()))['anchor_id'] if centers else None
        for aid in dict.fromkeys(c['anchor_id'] for c in centers.values()):
            group=[c for c in centers.values() if c['anchor_id']==aid]
            if len(group)!=3 or {tuple(c['target_support']) for c in group} != SUPPORTS:
                errors.append('anchor_complete_supports:'+aid)
            for name in ('history_p','history_q','history_source_p','history_source_q','history_root_v','future'):
                if any(group[0][name] != c[name] for c in group[1:]):errors.append('same_anchor_shared_state:'+aid+':'+name)
        for tag in ('seeds','robust','nominal','uniform'):
            search=selection.get('searches',{}).get(tag)
            if search is None:
                required=tag!='uniform' or 'target_weight_record' in selection
                if centers and required and not selection.get('information_not_started_reason'):
                    errors.append(tag+':missing_required_search')
                continue
            expected=[cid for cid,c in centers.items() if tag!='nominal' or c['anchor_id']==first]
            if not search.get('site_evaluations'):
                if search.get('success') is True:
                    errors.append(tag+':successful_search_missing_saved_evaluations')
                elif not (search.get('error') or search.get('failures') or search.get('failed_initialization_record')):
                    errors.append(tag+':unexplained_search_missing_saved_evaluations')
                failed_searches.append(dict(tag=tag,status=search.get('success'),error=search.get('error'),
                    failures=search.get('failures'),failed_initialization_record=search.get('failed_initialization_record')))
                continue
            for site,ev in search['site_evaluations'].items():
                plan=search.get('site_plans',{}).get(str(site))
                if plan is None:errors.append(tag+':site_missing_plan:'+str(site));continue
                evaluations.append(_evaluation_audit(g,tag,tag+':site:'+str(site),ev,plan,expected,errors,comparisons))
            for evkey,plkey in (('evaluation','plan'),('best_constant_evaluation','best_constant_plan')):
                if search.get(evkey) is not None and search.get(plkey) is not None:
                    evaluations.append(_evaluation_audit(g,tag,tag+':'+evkey,search[evkey],search[plkey],expected,errors,comparisons))
            if search.get('success'):
                _compare(search['score'],search['evaluation']['score'],tag+':selected_score',errors,comparisons,scalar=True)
                if int(search['plan']['site']) != int(search['selected_site']):errors.append(tag+':selected_site')
                _compare(search['plan']['p'],search['site_plans'][str(search['selected_site'])]['p'],tag+':selected_site_plan',errors,comparisons)
        return _serial(dict(success=not errors,run_id=trial.get('run_id'),history_id=trial.get('history_id'),seed=trial.get('seed'),quality=trial.get('quality'),
            errors=errors,activated=bool(selection.get('activated')),center_count=len(centers),evaluations=evaluations,
            information_not_started_reason=selection.get('information_not_started_reason'),
            target_weight_record=selection.get('target_weight_record'),
            geometry_coverage_status='inactive' if not centers else 'partially_unavailable' if not evaluations or selection.get('information_not_started_reason') or failed_searches or any(not e['saved_success'] for e in evaluations) else 'complete_saved_evaluations',
            saved_evaluation_success_count=sum(e['saved_success'] for e in evaluations),
            saved_evaluation_failure_count=sum(not e['saved_success'] for e in evaluations),
            search_failure_and_bound_records={tag:_search_diagnostics(search) for tag,search in selection.get('searches',{}).items() if tag in ('seeds','robust','nominal','uniform')},
            failed_searches=failed_searches,method_failure_records={k:v for k,v in selection.get('methods',{}).items() if 'failure' in v.get('status','')},
            comparisons_count=len(comparisons),failed_comparisons=[c for c in comparisons if not c['passed']],
            maximum_matrix_relative_error=max((c['relative_error'] or 0 for c in comparisons if 'information_' in c['label']),default=0.),
            maximum_score_absolute_error=max((c['absolute_error'] or 0 for c in comparisons if 'score' in c['label']),default=0.),
            independent_AC_rebuild_calls=g.ac_calls,rank_policy='single_column_normalized_SVD_on_full_nuisance_B',
            assembly='history_white_jacobian_time_major; shared_HN_RX; distinct_H_and_N_PQ_source_states',
            source_physical_interface='node1_after_full_support_edge',EV_rule='total_P_plus_power_kW/20_once; total_Q_plus_recorded_ratio*power_kW/20_once',
            scope='saved_local_information_geometry_and_aggregation; not_nonlinear_global_or_probability_evidence'))
    except (ValueError,RuntimeError,FloatingPointError,np.linalg.LinAlgError,KeyError,TypeError) as exc:
        return dict(success=False,run_id=trial.get('run_id'),errors=errors+[type(exc).__name__+':'+str(exc)])


def audit_paths(paths):
    rows=[];files=[]
    for raw in paths:
        path=Path(raw);row=audit_trial_geometry(read_json(path));rows.append(row)
        files.append(dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(success=all(r['success'] for r in rows),created_utc=datetime.now(timezone.utc).isoformat(),
        trial_count=len(rows),independent_history_count=len({r.get('history_id') for r in rows}),
        failed_trial_count=sum(not r['success'] for r in rows),saved_failed_search_count=sum(len(r.get('failed_searches',[])) for r in rows),
        rebuilt_target_count=sum(len(ev['targets']) for r in rows for ev in r.get('evaluations',[])),
        source_inputs=files,auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        tolerances=dict(matrix_rtol=MATRIX_RTOL,matrix_atol=MATRIX_ATOL,score_rtol=SCORE_RTOL,score_atol=SCORE_ATOL,task_weight_rtol=1e-12,task_weight_atol=1e-15,
            rationale='independent_time_major_AC_assembly_and_one_shot_vs_two_stage_SVD_operations; measured_errors_and_rank_cutoff_separation_retained'),
        rows=rows)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('inputs',nargs='+');parser.add_argument('--output',required=True)
    args=parser.parse_args();paths=[]
    for item in args.inputs:
        path=Path(item);paths.extend(sorted(path.glob('trial*.json.gz')) if path.is_dir() else [path])
    if not paths:raise ValueError('no saved trial files')
    result=audit_paths(paths);write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','source_inputs','tolerances')}))
    if not result['success']:raise SystemExit(1)


if __name__ == '__main__':main()
