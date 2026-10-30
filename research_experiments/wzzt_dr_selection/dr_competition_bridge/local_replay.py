"""Frozen-history local WZZT price replay, not a fresh dense-R/X tree search.

Only a training participation mask creates LPs. Validation masks score already
fitted points. A completed finite response table is integrated over prices by
pricing_tables. No future physical trajectory or true topology is accepted.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter
import traceback

import numpy as np

from physical_bridge import CONTRACT, forecast_mean, apply_response
from pricing_tables import (enumerate_menus, unique_training_masks, masks_from_key,
                            score_menus, expected_loss, payment)


class ReplayUnresolved(RuntimeError):
    """Budget or unfinished scan; never translated to a zero conditional loss."""


def _plain(value):
    if isinstance(value, dict):
        return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value, (list,tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _write(path, value):
    Path(path).write_text(json.dumps(_plain(value), ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')


def _hash(value):
    return hashlib.sha256(json.dumps(_plain(value),sort_keys=True,separators=(',',':'),
                                    allow_nan=False).encode()).hexdigest()


def effective_family(point):
    """Positive effective blocks; strict positivity preserves arbitrarily weak ones."""
    supports, r, x = point['supports'], point['r'], point['x']
    if len(supports) != len(r) or len(r) != len(x):
        raise ValueError('support/weight lengths disagree')
    result = set()
    for support, rv, xv in zip(supports,r,x):
        if not np.isfinite(rv) or not np.isfinite(xv) or rv < 0 or xv < 0:
            raise ValueError('hypothetical model weights must be finite nonnegative')
        if rv > 0 or xv > 0:
            result.add(tuple(sorted(int(j) for j in support)))
    return result


def structural_loss(point, target_family):
    return len(effective_family(point).symmetric_difference(set(target_family)))


def point_matrices(point, n):
    r, x = np.zeros((n,n)), np.zeros((n,n))
    if len(point['supports']) != len(point['r']) or len(point['r']) != len(point['x']):
        raise ValueError('point weights misaligned')
    for s, rv, xv in zip(point['supports'],point['r'],point['x']):
        if not s or min(s) < 0 or max(s) >= n or len(set(s)) != len(s):
            raise ValueError('invalid support indices')
        r[np.ix_(s,s)] += rv
        x[np.ix_(s,s)] += xv
    if not np.isfinite(r).all() or not np.isfinite(x).all():
        raise ValueError('nonfinite model kernel')
    return r,x


def history_arrays(scenes):
    p, q, y = [], [], []
    for scene in scenes:
        pp, qq = (scene[k].to_numpy(dtype=float) for k in ('P_terminal','Q_terminal'))
        v = scene['V_terminal'].to_numpy(dtype=float)
        root = scene['V_root_observed'].to_numpy(dtype=float)
        if pp.shape[0] != 96 or qq.shape != pp.shape or v.shape != pp.shape or root.shape != (96,):
            raise ValueError('complete 96-point aligned historical days required')
        target = root[:,None]**2-v**2
        if not all(np.isfinite(a).all() for a in (pp,qq,target)):
            raise ValueError('nonfinite historical channels')
        p.append(pp); q.append(qq); y.append(target)
    return tuple(np.concatenate(parts) for parts in (p,q,y))


def future_inputs(forecast, active_positions):
    """Price-free power trajectories indexed by one-event active-node bits."""
    n, k = len(forecast['labels']), len(active_positions)
    if k != len(set(active_positions)) or any(j < 0 or j >= n for j in active_positions):
        raise ValueError('invalid active positions')
    prices = np.zeros(n, dtype=int)
    prices[active_positions] = 1  # trajectory placeholder only; never used as actual price/payment
    result = {}
    for key in range(1 << k):
        z = np.zeros(n, dtype=bool)
        for j, position in enumerate(active_positions):
            z[position] = bool((key >> j)&1)
        p,q,checks = apply_response(forecast['P_pu'],forecast['Q_pu'],prices,z,
                                  base_kw=forecast['base_kw'])
        if checks['nonfinite_values'] or not checks['energy_return_pass']:
            raise ValueError('invalid predicted whole-event response')
        if not checks['capacity_pass']:
            raise ReplayUnresolved('predicted response outside original45kW contract; no clipping')
        result[key] = (p,q)
    return result


class ValidationScorer:
    """Cache old-validation residuals and each of the four one-day outcomes."""
    def __init__(self, old_validation, future, hypothesis):
        self.old = old_validation
        self.future = future
        self.n = old_validation[0].shape[1]
        self.target_r,self.target_x = point_matrices(hypothesis,self.n)
        self.future_targets = {key:p@self.target_r.T+q@self.target_x.T for key,(p,q) in future.items()}
        self.cache = {}
        self.denominator = (len(old_validation[0])+2*96)*self.n

    def _components(self, point):
        key = (tuple(tuple(s) for s in point['supports']),tuple(point['r']),tuple(point['x']))
        if key not in self.cache:
            r,x = point_matrices(point,self.n)
            p,q,y = self.old
            past = float(np.abs(p@r.T+q@x.T-y).sum())
            future = {mask:float(np.abs(pp@r.T+qq@x.T-self.future_targets[mask]).sum())
                      for mask,(pp,qq) in self.future.items()}
            self.cache[key] = (past,future)
        return self.cache[key]

    def mae(self, point, validation_key):
        past,future = self._components(point)
        return (past+future[validation_key[0]]+future[validation_key[1]])/self.denominator

    def score(self, points, validation_key):
        return [dict(point,validation_mae=self.mae(point,validation_key)) for point in points]


class _TimedLP:
    """Preserve original LP fit semantics while respecting the case wall budget."""
    def __init__(self, lp, deadline, case_budget, max_calls):
        self.lp, self.deadline = lp, deadline
        self.case_budget,self.max_calls = case_budget,max_calls
        self.pool, self.n = lp.pool, lp.n

    def fit(self, family):
        remaining = self.deadline-perf_counter()
        if remaining <= .001:
            raise ReplayUnresolved('fixed-family replay wall budget exhausted')
        key=tuple(sorted(set(tuple(sorted(s)) for s in family),key=lambda s:(len(s),s)))
        if key not in self.lp.cache and self.case_budget['calls'] >= self.max_calls:
            raise ReplayUnresolved('case6000 real-LP call budget exhausted')
        self.lp.model.Params.TimeLimit = min(15.,remaining)
        before=len(self.lp.solves)
        try:
            return self.lp.fit(family)
        finally:
            self.case_budget['calls']+=len(self.lp.solves)-before


def proposal_pool(full_pool, hypotheses, scope):
    """Restrict only newly proposed supports; all LP columns remain available.

    Existing native-RNJ/start blocks outside this proposal pool may retain
    positive weights, be jointly reweighted, or be pruned at zero. They are not
    newly proposed after removal unless present in the declared proposal pool.
    """
    canonical=lambda supports:tuple(sorted(set(tuple(sorted(s)) for s in supports),
                                            key=lambda s:(len(s),s)))
    full=canonical(full_pool)
    if scope=='full_old_pool':
        return full
    if scope!='historical_representative_union':
        raise ValueError('unknown proposal scope')
    selected=canonical([s for h in hypotheses for s in h['supports']])
    if not set(selected)<=set(full):
        raise ValueError('proposal support missing from full LP columns')
    return selected


class _ProposalPoolView:
    """A search-only view; fit always delegates to the full-column TimedLP."""
    def __init__(self, timed_lp, supports):
        self.lp=timed_lp
        self.pool=tuple(supports)
        self.n=timed_lp.n

    def fit(self, family):
        return self.lp.fit(family)


def _crossing_accepts(trajectory):
    accepted = []
    for iteration,decision in enumerate(trajectory.get('decisions',[])):
        if decision['gain'] <= 1e-10+.001*decision['current']:
            continue
        best = None
        for item in decision['candidates']:
            if best is None or item['objective'] < best['objective']-1e-12:
                best = item
        accepted.append(dict(iteration=iteration,candidate=best['candidate'],removed=best['removed'],
                             crosses=bool(best['removed']),gain=decision['gain'],
                             full_sweep=decision['full_sweep']))
    return accepted


def _initial_families(old_result, indices):
    bank,points = old_result['rnj_bank'],old_result['base_points']
    primary = next(p for p in bank if p['config'] == 'raw_RX75_0.16')
    starts = [('old_primary_RNJ_native',primary['native_family']),
              (f'old_current_raw_{indices[0]}',points[indices[0]]['supports'])]
    if len(indices)>1:
        alternative=indices[1]  # Frozen in the case protocol, not reselected under validation masks.
        starts.append((f'old_alternative_raw_{alternative}',points[alternative]['supports']))
    unique,seen = [],set()
    for name,family in starts:
        key = tuple(sorted(tuple(sorted(s)) for s in family))
        if key not in seen:
            unique.append((name,[list(s) for s in key]));seen.add(key)
    return unique


def _expand_menu(menu,positions,n):
    expanded = np.zeros((2,n),dtype=np.uint8)
    expanded[:,positions] = menu
    return expanded.tolist()


def plan_case(history, old_result, representative_indices, active_labels, out_dir,
              wall_seconds=600, *, tie_seed=20261030, path_seconds=15.,max_LP_calls=6000,
              resume=False,proposal_scope='historical_representative_union'):
    """Build one bounded two-node conditional decision table and select prices.

    All old training288 and validation288 rows remain in their original roles.
    Prediction uses their six-day mean. Three past-derived starts and the full
    old RNJ native bank are frozen; dense R/X and RNJ are not rebuilt in this
    planning heuristic. Actual evaluation must use the full observed pipeline.
    """
    from research_experiments.wzzt_dr_selection.dr_pool_v2.engine import PoolLP,search,canonical
    from research_experiments.wzzt_dr_selection.dr_pool_v2.selection import select
    from research_experiments.wzzt_dr_selection.dr_pool_v2 import engine,selection,observations

    started = perf_counter()
    if not np.isfinite(wall_seconds) or wall_seconds <= 0:
        raise ValueError('wall_seconds must be finite and positive')
    if not np.isfinite(path_seconds) or path_seconds <= 0 or isinstance(max_LP_calls,bool) or int(max_LP_calls)!=max_LP_calls or max_LP_calls<1:
        raise ValueError('positive path seconds and positive integer LP budget required')
    max_LP_calls=int(max_LP_calls)
    labels = list(history['labels'])
    if labels != old_result['labels'] or len(active_labels) != 2 or len(set(active_labels)) != 2:
        raise ValueError('first bridge requires two distinct active labels and unchanged terminal order')
    positions = [labels.index(label) for label in active_labels]
    indices = [int(i) for i in representative_indices]
    if not indices or len(indices) != len(set(indices)) or any(i<0 or i>=len(old_result['base_points']) for i in indices):
        raise ValueError('invalid unique historical representatives')
    if len(old_result['rnj_bank']) != 15 or any('native_family' not in p for p in old_result['rnj_bank']):
        raise ValueError('fifteen complete historical RNJ native families required')
    train,valid = history_arrays(history['training']),history_arrays(history['validation'])
    for scene in history['training']+history['validation']:
        if any(scene[key].columns.tolist()!=labels for key in ('P_terminal','Q_terminal','V_terminal')):
            raise ValueError('historical terminal columns are not in the declared order')
    if len(train[0]) != 288 or len(valid[0]) != 288:
        raise ValueError('complete288-point old training and validation required')
    forecast = forecast_mean(history)
    future = future_inputs(forecast,positions)
    hypotheses = [old_result['base_points'][i] for i in indices]
    for h in hypotheses:effective_family(h)
    starts = _initial_families(old_result,indices)
    pool = canonical(old_result['pool']+[s for p in old_result['rnj_bank'] for s in p['native_family']]
                     +[s for h in hypotheses for s in h['supports']])
    proposals=proposal_pool(pool,hypotheses,proposal_scope)
    menus = enumerate_menus(2)
    training_keys = unique_training_masks(menus,'local')
    assert len(menus)==50 and len(training_keys)==16
    expected = len(hypotheses)*len(training_keys)
    identity = dict(job_id=old_result.get('job_id'),historical_source=history.get('receipt'),
                    representative_indices=indices,active_labels=list(active_labels),wall_seconds=float(wall_seconds),
                    tie_seed=tie_seed,path_seconds=float(path_seconds),max_LP_calls=max_LP_calls,
                    proposal_scope=proposal_scope,proposal_pool=[list(s) for s in proposals],
                    LP_pool=[list(s) for s in pool],
                    old_candidates_hash=_hash(dict(base_points=old_result['base_points'],
                       rnj_bank=old_result['rnj_bank'],pool=old_result['pool'],calibration=old_result['calibration'])),
                    historical_arrays_hash=[hashlib.sha256(np.asarray(a,dtype='<f8').tobytes()).hexdigest() for a in (*train,*valid)])
    source_files = [Path(__file__),Path(__file__).with_name('pricing_tables.py'),Path(__file__).with_name('physical_bridge.py'),
                    Path(engine.__file__),Path(selection.__file__),Path(observations.__file__)]
    identity['source_sha256'] = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    fingerprint = _hash(identity)
    out = Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    result_path,checkpoint = out/'plan_result.json',out/'checkpoint.json'
    if result_path.exists():
        old=json.loads(result_path.read_text(encoding='utf-8'))
        if old.get('fingerprint') != fingerprint:raise RuntimeError('existing result identity differs')
        if not resume or old.get('status')!='complete':
            raise FileExistsError('existing result retained; only --resume of unchanged completed result may reuse it')
        return old
    if checkpoint.exists() or (out/'plan.lock').exists():
        raise RuntimeError('existing interrupted replay retained; inspect without restarting')
    _write(out/'plan_protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),fingerprint=fingerprint,
        identity=identity,scope='two-node full price domain with frozen-start finite-pool local replay',
        training_pairs=expected,menu_count=50,training_masks=16,validation_masks=16,
        coefficient_domain='all continuous R/X >=0 with no upper bound',
        price_objective='equal arithmetic mean of conditional complete effective-block errors; not a posterior',
        prediction='six-day observed mean P/Q and separate historical raw R/X generators; no new future noise',
        RNJ_reference='all15 historical native RNJ families jointly refitted under each training mask',
        starts=starts,pool=[list(s) for s in pool],LP_pool=[list(s) for s in pool],
        proposal_pool=[list(s) for s in proposals],proposal_scope=proposal_scope,
        proposal_scope_note='new-block proposal restriction only; full native/start blocks remain fitted and zero-prunable',
        calibration='frozen original historical calibration',
        per_training_pair_fit_seconds=90.,per_path_seconds=float(path_seconds),solver_seconds_per_LP=15.,
        max_real_LP_calls=max_LP_calls,
        response_contract=CONTRACT,zero_and_weak_weights_retained=True,full_parameter_coverage=False,
        future_or_true_topology_used=False,unresolved_rule='retain; never fill missing loss with zero'))
    _write(out/'plan.lock',dict(pid=__import__('os').getpid(),fingerprint=fingerprint))
    result=dict(status='running',fingerprint=fingerprint,plans={},scores={},training_pairs_planned=expected,
                training_pairs_completed=0,LP_calls=0,accepted_moves=0,accepted_crossing_moves=0,
                records=[],identity=identity,local_replay_only=True,truth_probability_claim=False,
                future_physical_data_read=False,global_tree_optimality_claim=False,
                proposal_scope=proposal_scope,proposal_pool=[list(s) for s in proposals],LP_pool=[list(s) for s in pool])
    _write(checkpoint,result)
    deadline = started+wall_seconds
    table_wzzt,table_rnj = [],[]
    active_lp = None
    fit_context = None
    context_counted = False
    case_budget={'calls':0}
    try:
        record_dir=out/'training_masks';record_dir.mkdir()
        for model_serial,(index,hypothesis) in enumerate(zip(indices,hypotheses)):
            target = effective_family(hypothesis)
            rh,xh = point_matrices(hypothesis,len(labels))
            scorer = ValidationScorer(valid,future,hypothesis)
            wt,rt = {},{}
            for training_key in training_keys:
                if perf_counter()>=deadline:raise ReplayUnresolved('case wall budget exhausted before next training mask')
                pair_started=perf_counter()
                p=np.concatenate([train[0],future[training_key[0]][0],future[training_key[1]][0]])
                q=np.concatenate([train[1],future[training_key[0]][1],future[training_key[1]][1]])
                y=np.concatenate([train[2],future[training_key[0]][0]@rh.T+future[training_key[0]][1]@xh.T,
                                  future[training_key[1]][0]@rh.T+future[training_key[1]][1]@xh.T])
                record_path=record_dir/f'm{model_serial}_u{training_key[0]}_{training_key[1]}.json'
                if record_path.exists():raise RuntimeError('training mask record already exists; do not rerun')
                fit_context=dict(status='running',model_serial=model_serial,raw_index=index,
                    training_mask_key=list(training_key),training_rows=len(p),validation_rows=480,
                    target_raw_positive_family=[list(s) for s in sorted(target)],paths={},
                    RNJ_native_families=15,coefficient_upper_bound=None,
                    proposal_scope=proposal_scope,proposal_count=len(proposals),
                    nonleaf_proposals=sum(len(s)>1 for s in proposals),LP_pool_count=len(pool))
                context_counted=False
                _write(record_path,fit_context)
                active_lp=PoolLP(None,pool,'raw',bound=np.inf,data=(p,q,y))
                timed=_TimedLP(active_lp,min(deadline,pair_started+90.),case_budget,max_LP_calls)
                search_view=_ProposalPoolView(timed,proposals)
                bank=[]
                for source in old_result['rnj_bank']:
                    fit=timed.fit(source['native_family'])
                    bank.append(dict(fit,origin='tree_bank',config=source['config'],native_family=source['native_family']))
                candidates=list(bank)
                for name,family in starts:
                    remaining=min(path_seconds,timed.deadline-perf_counter())
                    if remaining <= .001:raise ReplayUnresolved('pair wall budget exhausted before search')
                    fit_context['paths'][name]=dict(reason='started',path=[],decisions=[])
                    def progress(trajectory, path_name=name):
                        fit_context['paths'][path_name]=dict(trajectory,accepted_moves=_crossing_accepts(trajectory))
                        _write(record_path,fit_context)
                    trajectory=search(search_view,family,repair=True,seconds=remaining,max_steps=3*len(labels),
                                      progress_callback=progress)
                    accepted=_crossing_accepts(trajectory)
                    fit_context['paths'][name]=dict(trajectory,accepted_moves=accepted)
                    if trajectory['reason']!='pool_local_optimum':
                        raise ReplayUnresolved('local support scan incomplete: '+trajectory['reason'])
                    candidates.extend(dict(point,origin=name,path_index=i) for i,point in enumerate(trajectory['path']))
                fit_context['LP_solves']=active_lp.solves
                fit_context['LP_count']=len(active_lp.solves)
                fit_context['candidate_count']=len(candidates)
                fit_context['candidate_points']=candidates
                fit_context['bank_points']=bank
                fit_context['maximum_weight']=max(v for point in active_lp.cache.values() for key in ('r','x') for v in point[key])
                active_lp.close();active_lp=None
                validation_records=[]
                for vk in training_keys:
                    if perf_counter()>=deadline:raise ReplayUnresolved('case wall budget exhausted during validation scoring')
                    scored=scorer.score(candidates,vk)
                    selected=select(scored,old_result['calibration'])
                    rnj_selected=select(scored[:len(bank)],old_result['calibration'])
                    fullkey=tuple(training_key)+tuple(vk)
                    wt[fullkey]=structural_loss(selected,target)
                    rt[fullkey]=structural_loss(rnj_selected,target)
                    detail=dict(validation_key=list(vk),full_event_key=list(fullkey),
                        wzzt_loss=wt[fullkey],RNJ_loss=rt[fullkey],
                        wzzt_selected_index=selected['selection_index'],wzzt_origin=selected.get('origin'),
                        RNJ_selected_index=rnj_selected['selection_index'],RNJ_origin=rnj_selected.get('origin'),
                        wzzt_supports=selected['supports'],RNJ_supports=rnj_selected['supports'],
                        wzzt_effective_supports=[list(s) for s in sorted(effective_family(selected))],
                        RNJ_effective_supports=[list(s) for s in sorted(effective_family(rnj_selected))],
                        wzzt_raw_selection_mae=selected['selection_validation_mae'],
                        RNJ_raw_selection_mae=rnj_selected['selection_validation_mae'],
                        wzzt_postcontract_mae=scorer.mae(selected,vk),RNJ_postcontract_mae=scorer.mae(rnj_selected,vk),
                        selected_wzzt=selected,selected_RNJ=rnj_selected)
                    validation_records.append(detail)
                fit_context.update(status='complete',validation_results=validation_records,
                                   elapsed_seconds=perf_counter()-pair_started)
                _write(record_path,fit_context)
                counts=[a for path in fit_context['paths'].values() for a in path['accepted_moves']]
                result['records'].append(dict(model_serial=model_serial,raw_index=index,training_key=list(training_key),
                    path=str(record_path),LP_count=fit_context['LP_count'],status='complete'))
                result['training_pairs_completed']+=1
                result['LP_calls']+=fit_context['LP_count']
                result['accepted_moves']+=len(counts)
                result['accepted_crossing_moves']+=sum(a['crosses'] for a in counts)
                context_counted=True
                result['elapsed_seconds']=perf_counter()-started
                _write(checkpoint,result)
                fit_context=None
            table_wzzt.append(wt);table_rnj.append(rt)
        keys=set(table_wzzt[0])
        if any(set(t)!=keys for t in table_wzzt+table_rnj) or len(keys)!=256:
            raise RuntimeError('conditional response table incomplete')
        mean_w={k:sum(t[k] for t in table_wzzt)/len(hypotheses) for k in keys}
        mean_r={k:sum(t[k] for t in table_rnj)/len(hypotheses) for k in keys}
        result['response_table']=[dict(key=list(k),wzzt_by_hypothesis=[t[k] for t in table_wzzt],
            RNJ_by_hypothesis=[t[k] for t in table_rnj],wzzt_mean=mean_w[k],RNJ_mean=mean_r[k]) for k in sorted(keys)]
        for law in ('local','common'):
            scored=score_menus(menus,mean_w,law,tie_seed=tie_seed)
            rnj_scored=score_menus(menus,mean_r,law,tie_seed=tie_seed)
            result['scores'][law]=dict(scored,selected_menu=scored['selected_menu'].tolist(),
                per_model_menu_loss=[[expected_loss(table,menu,law) for menu in menus] for table in table_wzzt],
                RNJ_menu_records=rnj_scored['records'])
            result['plans']['wzzt_'+law]=_expand_menu(scored['selected_menu'],positions,len(labels))
            singles=np.array([m for m in menus if np.count_nonzero(np.any(m>0,axis=0))<=1])
            single=score_menus(singles,mean_w,law,tie_seed=tie_seed)
            result['plans']['best_single_within_pair_'+law]=_expand_menu(single['selected_menu'],positions,len(labels))
        result['plans'].update(zero=_expand_menu(np.zeros((2,2),dtype=np.uint8),positions,len(labels)),
            synchronous_pair=_expand_menu(np.ones((2,2),dtype=np.uint8),positions,len(labels)),
            split_pair=_expand_menu(np.array([[2,0],[0,2]],dtype=np.uint8),positions,len(labels)))
        result.update(status='complete',elapsed_seconds=perf_counter()-started,
            candidate_menu_count=50,training_mask_count=16,validation_mask_count=16,
            conditional_states_per_hypothesis=256,hypothesis_count=len(hypotheses),
            final_structure_recovery_claim=False,response_amplitude_price_independent=True,
            proposal_scope=proposal_scope,proposal_pool=[list(s) for s in proposals],LP_pool=[list(s) for s in pool])
    except Exception as exc:
        status='unresolved' if isinstance(exc,ReplayUnresolved) else 'failed'
        if active_lp is not None:
            statuses=[r.get('status') for r in active_lp.solves]
            if statuses and statuses[-1] in (9,11):status='unresolved'
            if fit_context is not None:
                fit_context['LP_solves']=active_lp.solves
        if fit_context is not None:
            if not context_counted:
                partial_LPs=len(active_lp.solves) if active_lp is not None else fit_context.get('LP_count',0)
                result['LP_calls']+=partial_LPs
                result['partial_training_record_LP_calls']=partial_LPs
                counts=[a for path in fit_context['paths'].values() for a in path.get('accepted_moves',[])]
                result['accepted_moves']+=len(counts)
                result['accepted_crossing_moves']+=sum(a['crosses'] for a in counts)
            fit_context.update(status=status,exception_type=type(exc).__name__,exception=str(exc))
            _write(record_path,fit_context)
        result.update(status=status,exception_type=type(exc).__name__,exception=str(exc),
                      traceback=traceback.format_exc(),elapsed_seconds=perf_counter()-started,
                      missing_losses_not_filled=True)
    finally:
        if active_lp is not None:active_lp.close()
        result['LP_calls']=case_budget['calls']
        _write(result_path,result);_write(checkpoint,result)
        (out/'plan.lock').unlink(missing_ok=True)
    return _plain(result)
