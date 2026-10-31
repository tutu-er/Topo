"""Independent saved-result audit for the budget-reference diagnostic.

prepare reads protocol and price locks only. run requires ALL phase endings,
fit-freeze markers, global oracle selection and aggregate before opening any
observed packet or evaluator truth. No identification, LP, AC or future data is
generated. Full parameter and trajectory files remain private.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import pickle
from random import Random
import sys
import traceback

EXPECTED_PROTOCOL = 'ed29e9c44f8f18c8623674164bad7cde1011727de32f24bcf889a7a644b764a0'
METHODS = ('multistart_original_rule', 'rnj_bank', 'baseline25',
           'geometric_RNJ_RX75_0.16', 'geometric_NJ_RX75_0.16')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=True, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def utc(value):
    return datetime.fromisoformat(value.replace('Z', '+00:00'))


def family(supports):
    rows = [tuple(sorted(int(x) for x in s)) for s in supports]
    if any(not s for s in rows) or len(rows) != len(set(rows)):
        raise ValueError('empty or duplicate reported support')
    return set(rows)


def prediction(point, labels):
    if 'support_labels' in point:
        return family(point['support_labels'])
    indices = point.get('support_indices', point.get('supports'))
    if indices is None:
        raise ValueError('prediction lacks final supports')
    return family([[labels[i] for i in s] for s in indices])


def errors(predicted, truth):
    missing, extra = truth-predicted, predicted-truth
    return dict(full_errors=len(missing)+len(extra), full_recovery=not missing and not extra,
                missing=sorted(missing), extra=sorted(extra),
                predicted_blocks=len(predicted), truth_blocks=len(truth))


def prepare(root, out, expected_protocol):
    root, out = Path(root).resolve(), Path(out).resolve()
    if sha(root/'protocol.json') != expected_protocol:
        raise ValueError('main protocol is not the declared frozen revision')
    cfg = read(root/'protocol.json')
    if out.exists() and (out/'audit_protocol.json').exists():
        raise FileExistsError('audit protocol already retained')
    out.mkdir(parents=True, exist_ok=True)
    plans = {}
    for key, case in cfg['cases'].items():
        path = Path(case['plan_lock']['path'])
        if sha(path) != case['plan_lock']['sha256']:
            raise ValueError('price lock drift before audit preparation')
        plans[key] = sha(path)
    protocol = dict(id='budget_reference_independent_audit_v1', created_utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=sha(__file__), root=str(root), main_protocol_sha256=expected_protocol,
                    price_lock_sha256=plans, cases=list(cfg['cases']), policies=list(cfg['policies']),
                    planning_paths=list(cfg['planning_paths']), test_paths=list(cfg['test_paths']),
                    guard='all phase_ended and fits_frozen plus oracle and aggregate before packets/truth',
                    new_identifier_calls=0, new_LP_calls=0, new_AC_calls=0,
                    bounded_expected_future_records=64, bounded_expected_historical_baselines=2,
                    tolerance=dict(physical=1e-10, standardized_noise=5e-10, losses=1e-10),
                    scope='saved-data and graph-score replay only; no AC equation re-solve or statistical population claim')
    write(out/'audit_protocol.json', protocol)
    print(json.dumps(dict(status='audit_prepared', source_sha256=protocol['source_sha256'])))


class Audit:
    def __init__(self, root, out):
        self.root, self.out = Path(root).resolve(), Path(out).resolve()
        self.p = read(self.out/'audit_protocol.json')
        if self.p['source_sha256'] != sha(__file__) or self.p['main_protocol_sha256'] != sha(self.root/'protocol.json'):
            raise ValueError('audit or main protocol changed')
        self.cfg = read(self.root/'protocol.json')
        self.checks, self.files, self.markers = [], {}, {}
        # These metadata existence/hash checks precede every future/truth load.
        required = [self.root/'oracle_selected.json', self.root/'oracle_selected.sha256.json', self.root/'aggregate.json']
        for key in self.cfg['cases']:
            for phase in ('planning', 'test'):
                required += [self.root/key/phase/'phase_ended.json', self.root/key/phase/'fits_frozen.json']
        if not all(path.is_file() for path in required):
            raise RuntimeError('not all phases have ended and frozen fits; no packets or truth read')
        self.markers = {str(path):sha(path) for path in required}
        self.maxima = dict(physical_kw=0., physical_kvar=0., energy_kWh=0.,
                           standardized_noise=0., fit_loss=0.)
        self.record_counts = Counter()
        self.natural_reference, self.noise_reference, self.root_reference = {}, {}, {}
        self.natural_path_hashes = {}
        self.phase_data = {}
        self.full_score_rows = []
        self.independent_historical_stop = {}

    def guard(self):
        if any(not Path(path).is_file() or sha(path) != expected for path, expected in self.markers.items()):
            raise RuntimeError('freeze/selection/aggregate marker changed')

    def check(self, name, good, details=None):
        self.checks.append(dict(name=name, passed=bool(good), details=details))

    def same(self, name, a, b, tol=1e-10):
        import numpy as np
        a, b = np.asarray(a), np.asarray(b)
        error = float(np.max(abs(a-b), initial=0.)) if a.shape == b.shape else None
        self.check(name, a.shape == b.shape and error <= tol, error)
        return error if error is not None else 0.

    def load(self, path, expected):
        self.guard()
        path = Path(path)
        digest = sha(path)
        if digest != expected:
            raise ValueError('private artifact hash mismatch: '+str(path))
        self.files[str(path)] = digest
        with path.open('rb') as stream:
            return pickle.load(stream)

    def input_frames(self, scenes):
        import pandas as pd
        fields = ('P_terminal', 'Q_terminal', 'V_terminal', 'V_root_observed')
        obs = tuple(pd.concat([scene[key] for scene in scenes], axis=0) for key in fields)
        index = obs[0].index
        if not isinstance(index, pd.DatetimeIndex) or not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError('current source needs unique chronological datetime rows')
        if any(not value.index.equals(index) for value in obs):
            raise ValueError('misaligned observed channels')
        for value in obs[:3]:
            if not value.columns.equals(obs[0].columns):
                raise ValueError('misaligned terminal labels')
        days = [str(value) for value in index.normalize()]
        h = hashlib.sha256()
        for value in obs:
            h.update(pd.util.hash_pandas_object(value, index=True).values.tobytes())
        h.update(json.dumps(list(obs[0].columns), default=str).encode())
        h.update(json.dumps(days).encode())
        return obs, h.hexdigest()

    def score(self, fit, truth, root_label, observations=None):
        import numpy as np
        labels = list(fit['terminal_labels'])
        if observations is not None:self.selection_replay(fit, observations)
        result = {}
        for name in METHODS[:3]:
            point = fit['models'][name]
            result[name] = errors(prediction(point, labels), truth)
            uncut = point.get('selected_uncontracted')
            result[name]['before_contraction_errors'] = errors(prediction(uncut, labels), truth)['full_errors'] if uncut else None
            result[name]['selected_origin'] = point.get('selected_origin', point.get('origin'))
            if observations is not None:
                r, x = np.asarray(point['r_matrix']), np.asarray(point['x_matrix'])
                for role, frames in observations.items():
                    p, q, v, root = (value.to_numpy(dtype=float) for value in frames)
                    drop = root[:,None]**2-v**2
                    measured = float(np.mean(abs(p@r.T+q@x.T-drop)))
                    stored = point['train_mae' if role == 'training' else 'validation_mae']
                    difference = abs(measured-stored)
                    self.maxima['fit_loss'] = max(self.maxima['fit_loss'], difference)
                    self.check('same_observation_loss.'+name+'.'+role, difference <= 1e-10, difference)
        native = [point for point in fit['bank'] if point['config'] == 'raw_RX75_0.16']
        if len(native) != 1:
            raise ValueError('fixed RNJ native family missing')
        rnj = family([[labels[i] for i in s] for s in native[0]['native_family']])
        result[METHODS[3]] = errors(rnj, truth)
        # Existing graph functions only: no new sensitivity estimation or fit.
        from research_experiments.rnj.rooted_study_baselines import infer_classical_nj
        from rnj_wzzt.graph.rooted_hierarchy import rooted_clades
        cal = fit['calibrations']['raw']; front = fit['fronts']['raw']
        kernel = .75*np.asarray(front['R'])/cal['mu_r']+.25*np.asarray(front['X'])/cal['mu_x']
        tree = infer_classical_nj(kernel, kernel, labels, root_label, collapse_factor=.16)
        self.record_counts['stored_front_geometric_NJ_replays'] += 1
        nj = {(label,) for label in labels} | family(rooted_clades(tree.edges, root_label, labels))
        result[METHODS[4]] = errors(nj, truth)
        return result

    def selection_replay(self, fit, observations):
        """Recompute saved-point losses, 3% selection and output contraction."""
        import numpy as np
        n = len(fit['terminal_labels'])
        def matrices(point):
            r, x = np.zeros((n,n)), np.zeros((n,n))
            for support, rw, xw in zip(point['supports'],point['r'],point['x'],strict=True):
                r[np.ix_(support,support)] += rw
                x[np.ix_(support,support)] += xw
            return r,x
        def loss(point, role):
            r,x=matrices(point)
            p,q,v,root=(value.to_numpy(dtype=float) for value in observations[role])
            return float(np.mean(abs(p@r.T+q@x.T-(root[:,None]**2-v**2))))
        candidates=fit['selection_candidates'];bank=fit['bank']
        primary_count=len(fit['searches']['primary']['path'])
        self.check('selection.bank_prefix',candidates[:len(bank)]==bank and len(bank)==15)
        pools=dict(rnj_bank=bank,baseline25=candidates[:len(bank)+primary_count],multistart_original_rule=candidates)
        validation=[loss(point,'validation') for point in candidates]
        for i,(point,value) in enumerate(zip(candidates,validation,strict=True)):
            self.check('selection.candidate_validation.'+str(i),abs(value-point['validation_mae'])<=1e-10)
        cal=fit['calibrations']['raw'];threshold=.16*cal['median_root_depth']
        for name,points in pools.items():
            losses=validation[:len(points)]
            best=min(losses)
            eligible=[i for i,value in enumerate(losses) if value<=1.03*best+1e-12]
            selected=min(eligible,key=lambda i:(len(points[i]['supports']),losses[i],i))
            point,model=points[selected],fit['models'][name]
            uncut=model['selected_uncontracted']
            self.check('selection.'+name+'.index',model['selection_index']==selected==uncut['selection_index'])
            self.check('selection.'+name+'.uncut_support',uncut['supports']==point['supports'])
            self.same('selection.'+name+'.uncut_r',uncut['r'],point['r'])
            self.same('selection.'+name+'.uncut_x',uncut['x'],point['x'])
            keep=[i for i,s in enumerate(point['supports']) if len(s)==1 or cal['a']*point['r'][i]+cal['b']*point['x'][i]+1e-10>=threshold]
            expected=dict(supports=[point['supports'][i] for i in keep],r=[point['r'][i] for i in keep],x=[point['x'][i] for i in keep])
            self.check('selection.'+name+'.final_support',model['support_indices']==expected['supports'])
            self.same('selection.'+name+'.final_r',model['r_values'],expected['r'])
            self.same('selection.'+name+'.final_x',model['x_values'],expected['x'])
            r,x=matrices(expected)
            self.same('selection.'+name+'.final_matrix_R',model['r_matrix'],r)
            self.same('selection.'+name+'.final_matrix_X',model['x_matrix'],x)
            self.check('selection.'+name+'.uncut_validation',abs(model['selection_validation_mae']-losses[selected])<=1e-10)
            self.record_counts['saved_candidate_selection_replays'] += 1

    def compare_models(self, label, expected, saved):
        self.check(label+'.five_methods', set(saved) == set(METHODS))
        for method in METHODS:
            for key in ('full_errors', 'full_recovery', 'predicted_blocks', 'truth_blocks'):
                self.check(label+'.'+method+'.'+key, expected[method][key] == saved[method][key])
            self.check(label+'.'+method+'.missing', set(expected[method]['missing']) == family(saved[method]['missing']))
            self.check(label+'.'+method+'.extra', set(expected[method]['extra']) == family(saved[method]['extra']))
            if method in METHODS[:3]:
                self.check(label+'.'+method+'.uncut', expected[method]['before_contraction_errors'] == saved[method]['before_contraction_errors'])
                self.check(label+'.'+method+'.origin', expected[method]['selected_origin'] == saved[method]['selected_origin'])

    def receipt_metadata(self, case_key, phase, row, folder, oracle):
        """Retain fees and failure denominators even when identification failed."""
        label = case_key+'.'+phase+'.'+row['record_id']
        candidate = Path(row.get('physical_receipt_path', folder/'records'/row['record_id']/'physical'/'events_receipt.json'))
        if not candidate.is_file():
            self.record_counts['future_mappings_without_physical_receipt'] += 1
            self.check(label+'.missing_receipt_not_scored', row['status'] not in ('evaluated','evaluated_operation_failed'))
            return None, None
        digest = sha(candidate)
        if row.get('physical_receipt_sha256') is not None:
            self.check(label+'.receipt_hash', digest == row['physical_receipt_sha256'])
        self.files[str(candidate)] = digest
        value = read(candidate)
        self.record_counts['physical_receipts'] += 1
        self.record_counts['physical_event_records_metadata'] += len(value.get('event_records', []))
        self.record_counts['completed_physical_events_metadata'] += sum(e.get('status') == 'complete' for e in value.get('event_records', []))
        self.check(label+'.receipt_fee_retained', row['actual_payment'] == value.get('actual_payment') and row['committed_payment'] == value.get('committed_payment'))
        events = value.get('event_records', [])
        if not events:
            self.record_counts['physical_receipts_without_event_ledger'] += 1
            self.check(label+'.unresolved_worker_retained', value['status'] != 'complete' and row['status'] not in ('evaluated','evaluated_operation_failed'))
            if phase == 'test':
                self.check(label+'.test_generated_after_selection', utc(value['started_utc']) >= utc(oracle['frozen_utc']))
            return candidate, value
        committed = sum(float(e.get('committed_payment', 0.)) for e in events)
        known = sum(float(e['actual_payment']) for e in events if e.get('actual_payment') is not None)
        unknown = sum(e.get('actual_payment') is None for e in events)
        self.check(label+'.receipt_commitment_total', abs(committed-value['committed_payment']) <= 1e-12)
        self.check(label+'.receipt_known_fee_total', abs(known-value['known_actual_payment']) <= 1e-12)
        self.check(label+'.receipt_unknown_mass', unknown == value['unknown_payment_events'])
        self.check(label+'.receipt_settled_total', value['actual_payment'] is None if unknown else value['actual_payment'] is not None and abs(value['actual_payment']-known) <= 1e-12)
        self.check(label+'.receipt_hard_budget', committed <= 1.6+1e-12 and value['worst_payment'] <= 1.6+1e-12)
        if phase == 'test':
            self.check(label+'.test_generated_after_selection', utc(value['started_utc']) >= utc(oracle['frozen_utc']))
        return candidate, value

    def physical(self, case_key, case, phase, row, lock, receipt, packet, truth_packet):
        import numpy as np
        policy, path = row['policy'], int(row['path'])
        labels, base = lock['labels'], float(lock['base_kw'])
        plan = np.asarray(lock['plans'][policy], int)
        directions = np.asarray(lock['directions'][policy], int)
        orientations = np.asarray(lock['orientations_by_event'], int)
        n = len(labels)
        label = case_key+'.'+phase+'.'+row['record_id']
        self.check(label+'.prices_budget', plan.shape == (2,n) and np.isin(plan,[0,1,2]).all() and .4*plan.sum() <= 1.6+1e-12)
        self.check(label+'.worst_fee', abs(receipt['worst_payment']-.4*plan.sum()) <= 1e-12)
        self.check(label+'.receipt_identity', receipt['policy']==policy and receipt['path']==path and receipt['case']==case['case'])
        self.check(label+'.no_old_test_truth', receipt['old_test_or_truth_files_read'] is False)
        self.check(label+'.normal_noise', receipt['voltage_sigma']==.0002 and self.cfg['pq_sigma']==.005)
        seed, replicate_base = case['participation_seed'], case['replicate_base']
        self.check(label+'.rng_binding', receipt['seeds']['replicate_base']==replicate_base and receipt['seeds']['participation_seed']==seed)
        event_records = {int(v['event']):v for v in receipt['event_records']}
        physical_events = {int(v['event']):v for v in truth_packet['events']}
        observed = {}
        for role in ('training','validation'):
            events = packet['metadata'][role+'_event_ids']
            self.check(label+'.'+role+'.events', len(events)==len(packet[role]))
            observed.update({int(e):s for e,s in zip(events,packet[role],strict=True)})
        self.check(label+'.observed_only', packet['metadata']['observed_only'] is True and 'truth' not in packet and 'physical_network' not in packet)
        self.check(label+'.four_event_records', sorted(event_records)==list(range(4)))
        if receipt['status']=='complete':
            self.check(label+'.four_complete_events', sorted(physical_events)==sorted(observed)==list(range(4)))
        committed=known_fee=0.; unknown=0
        path_h=hashlib.sha256()
        for event in range(4):
            if event not in event_records:
                continue
            record=event_records[event]; price=plan[event%2]
            z=np.asarray(record['accepted_participants'],bool)
            rng=np.random.default_rng(np.random.SeedSequence([seed,path,event]))
            u=rng.random(n)
            if self.cfg['dependence']=='common':u[:]=u[0]
            self.check(label+f'.e{event}.mask',np.array_equal(z,u<np.array([0.,.5,.75])[price]))
            self.check(label+f'.e{event}.not_redrawn',np.array_equal(z,np.asarray(record['participants'],bool)))
            self.check(label+f'.e{event}.replicate',record['replicate']==replicate_base+10*path+event)
            fee=.2*float(price@z);committed+=fee
            self.check(label+f'.e{event}.committed',abs(fee-record['committed_payment'])<=1e-12)
            if record['actual_payment'] is None:unknown+=1
            else:known_fee+=record['actual_payment']
            self.record_counts['physical_event_records']+=1
            if event not in physical_events or event not in observed:
                continue
            t,scene=physical_events[event],observed[event]
            p,q,v,root=(np.asarray(t[k],float) for k in ('P_pu','Q_pu','V_pu','root_pu'))
            natural_p,natural_q=(np.asarray(t[k],float) for k in ('natural_P_pu','natural_Q_pu'))
            self.check(label+f'.e{event}.shape',p.shape==q.shape==v.shape==natural_p.shape==natural_q.shape==(96,n) and root.shape==(96,))
            direction=directions[event%2]*orientations[event]
            self.same(label+f'.e{event}.direction',record['directions'],direction,0.)
            delta=np.zeros((96,n));delta[76:84]=np.r_[np.full(4,-.2),np.full(4,.2)][:,None]*direction[None,:]*z[None,:]
            if lock['templates'][policy]=='alternating_opposed':
                raise ValueError('audit protocol does not declare alternating task shapes')
            ep=self.same(label+f'.e{event}.P_delta',(p-natural_p)*base,delta)
            eq=self.same(label+f'.e{event}.Q_delta',(q-natural_q)*base,.35*delta)
            self.maxima['physical_kw']=max(self.maxima['physical_kw'],ep)
            self.maxima['physical_kvar']=max(self.maxima['physical_kvar'],eq)
            energy=delta.sum(axis=0)*.25
            self.maxima['energy_kWh']=max(self.maxima['energy_kWh'],self.same(label+f'.e{event}.energy',energy,np.zeros(n)))
            for name,delivered in (('reduction',np.maximum(-delta,0).sum(axis=0)*.25),('rebound',np.maximum(delta,0).sum(axis=0)*.25)):
                self.same(label+f'.e{event}.'+name,delivered,.2*z)
                self.same(label+f'.e{event}.receipt_'+name,record['delivered_'+name+'_kWh'],delivered)
            self.check(label+f'.e{event}.settled_fee',record['actual_payment'] is not None and abs(record['actual_payment']-fee)<=1e-12)
            power=.2+delta[76:84];stock=np.cumsum(power,axis=0)*.25
            self.check(label+f'.e{event}.device',power.min()>=-1e-12 and power.max()<=.4+1e-12 and stock.min()>=-1e-12 and stock.max()<=.4+1e-12 and np.max(abs(stock[-1]-.4))<=1e-12)
            checks=record['checks']
            self.check(label+f'.e{event}.AC_receipt',checks['electrical_equations_pass'] and checks['ac_converged_samples']==96)
            self.check(label+f'.e{event}.operation',checks['energy_return_pass'] and checks['capacity_pass'] and checks['voltage_limits_pass'])
            self.check(label+f'.e{event}.saved_AC_convergence',bool(np.asarray(t['ac']['converged']).all()))
            measured=[scene[k].to_numpy(dtype=float) for k in ('P_terminal','Q_terminal','V_terminal')]
            root_m=scene['V_root_observed'].to_numpy(dtype=float)
            noises=[(a-b)/(sigma*np.maximum(abs(b),1e-12)) for a,b,sigma in zip(measured,(p,q,v),(.005,.005,.0002),strict=True)]
            noise_rng=np.random.default_rng(int(record['profile_seed'])+2000)
            for name,noise in zip(('P','Q','V'),noises,strict=True):
                self.same(label+f'.e{event}.nominal_noise_'+name,noise,noise_rng.standard_normal((96,n)),5e-10)
            root_seed=70_000_019+100_003*record['replicate']+record['profile_index']
            self.same(label+f'.e{event}.root_noise',(root_m-root)/(.0002*np.maximum(abs(root),1e-12)),np.random.default_rng(root_seed).standard_normal(96),5e-10)
            crn=(case_key,phase,path,event)
            if crn not in self.natural_reference:
                self.natural_reference[crn]=(natural_p.copy(),natural_q.copy(),root.copy())
                self.noise_reference[crn]=[x.copy() for x in noises]
                self.root_reference[crn]=root_m.copy()
            else:
                for name,a,b in zip(('P','Q','root'),(natural_p,natural_q,root),self.natural_reference[crn],strict=True):
                    self.same(label+f'.e{event}.natural_CRN_'+name,a,b,0.)
                for name,a,b in zip(('P','Q','V'),noises,self.noise_reference[crn],strict=True):
                    diff=self.same(label+f'.e{event}.noise_CRN_'+name,a,b,5e-10)
                    self.maxima['standardized_noise']=max(self.maxima['standardized_noise'],diff)
                self.same(label+f'.e{event}.root_observation_CRN',root_m,self.root_reference[crn],0.)
            for value in (natural_p,natural_q,root):path_h.update(value.tobytes())
            self.record_counts['physical_events_audited']+=1
            self.record_counts['AC_timepoints_audited']+=96
        self.check(label+'.committed_total',abs(receipt['committed_payment']-committed)<=1e-12)
        self.check(label+'.known_fee_total',abs(receipt['known_actual_payment']-known_fee)<=1e-12)
        self.check(label+'.unknown_fee_retained',receipt['unknown_payment_events']==unknown and ((unknown>0 and receipt['actual_payment'] is None) or (unknown==0 and abs(receipt['actual_payment']-known_fee)<=1e-12)))
        self.check(label+'.budget',committed<=1.6+1e-12 and receipt['worst_payment']<=1.6+1e-12)
        self.check(label+'.row_fee',row['actual_payment']==receipt['actual_payment'] and row['committed_payment']==receipt['committed_payment'])
        if policy=='zero':self.natural_path_hashes[(case_key,phase,path)]=path_h.hexdigest()

    def audit_rows(self):
        import numpy as np
        repo=Path(self.cfg['repository']);sys.path.insert(0,str(repo));sys.path.insert(0,str(repo/'rnj_wzzt_core'))
        for source, expected in self.cfg['source_sha256'].items():
            candidate = Path(source)
            if not candidate.is_absolute():candidate = repo/candidate
            self.check('source_closure.'+candidate.name, candidate.is_file() and sha(candidate) == expected)
        oracle=read(self.root/'oracle_selected.json')
        self.check('oracle.sha',sha(self.root/'oracle_selected.json')==read(self.root/'oracle_selected.sha256.json')['sha256'])
        self.check('oracle.protocol',oracle['protocol_sha256']==sha(self.root/'protocol.json'))
        self.check('oracle.test_not_used',oracle['selection_uses_test_paths'] is False and oracle['expectation_estimated_before_menu_selection'] is True)
        self.check('path_disjointness',not(set(self.cfg['planning_paths'])&set(self.cfg['test_paths'])))
        self.check('case_participant_streams_distinct',len({c['participation_seed'] for c in self.cfg['cases'].values()})==len(self.cfg['cases']))
        self.check('case_natural_streams_distinct',len({c['replicate_base'] for c in self.cfg['cases'].values()})==len(self.cfg['cases']))
        for case_key,case in self.cfg['cases'].items():
            lock=read(case['plan_lock']['path'])
            self.check(case_key+'.lock',sha(case['plan_lock']['path'])==case['plan_lock']['sha256']==self.p['price_lock_sha256'][case_key])
            history_packet=self.load(case['history_packet']['path'],case['history_packet']['sha256'])
            history=history_packet.get('history',history_packet)
            hobs={role:self.input_frames(history[role]) for role in ('training','validation')}
            self.check(case_key+'.complete_history',all(len(obs[0][0])==288 for obs in hobs.values()))
            first_truth=None
            for phase in ('planning','test'):
                folder=self.root/case_key/phase
                ended,frozen=read(folder/'phase_ended.json'),read(folder/'fits_frozen.json')
                self.phase_data[(case_key,phase)]=ended
                self.check(case_key+'.'+phase+'.protocol',ended['protocol_sha256']==frozen['protocol_sha256']==sha(self.root/'protocol.json'))
                paths=self.cfg['planning_paths' if phase=='planning' else 'test_paths']
                phase_truth=None
                expected={(int(path),policy) for path in paths for policy in self.cfg['policies']}
                rows=ended['rows'];by={(int(r['path']),r['policy']):r for r in rows}
                self.check(case_key+'.'+phase+'.all_records',len(rows)==len(by)==len(expected) and set(by)==expected)
                fb={row['record_id']:row for row in frozen['rows']}
                self.check(case_key+'.'+phase+'.freeze_records',set(fb)=={r['record_id'] for r in rows})
                if phase=='test':
                    self.check(case_key+'.test.oracle_hash',ended['oracle_selection_sha256']==sha(self.root/'oracle_selected.json'))
                    self.check(case_key+'.test.reference',ended['frozen_reference_policy']==oracle['cases'][case_key]['selected_policy'])
                for row in rows:
                    label=case_key+'.'+phase+'.'+row['record_id'];self.record_counts['future_mappings']+=1
                    frozen_row=fb[row['record_id']]
                    receipt_path, receipt = self.receipt_metadata(case_key,phase,row,folder,oracle)
                    # Physical delivery is auditable after freezing even if the
                    # downstream fit failed. Fit failure never hides paid data.
                    packet=truth=None
                    if receipt is not None:
                        physical_dir=receipt_path.parent;private=receipt.get('private_files',{})
                        if 'truth_evaluation.json' in private:
                            truth_path=physical_dir/'truth_evaluation.json';self.guard()
                            self.check(label+'.truth_hash',sha(truth_path)==private['truth_evaluation.json']['sha256'])
                            truth_record=read(truth_path);truth=family(truth_record['true_supports'])
                            self.check(label+'.truth_definition',truth_record['arbitrarily_weak_positive_lines_retained'] is True and truth_record['zero_impedance_merged'] is True)
                            if first_truth is None:first_truth=truth
                            if phase_truth is None:phase_truth=truth
                            self.check(label+'.static_truth',truth==first_truth==phase_truth)
                            self.record_counts['physical_truth_support_files_audited']+=1
                        else:self.record_counts['physical_truth_support_files_unavailable']+=1
                        if 'packet.pkl' in private:
                            packet=self.load(physical_dir/'packet.pkl',private['packet.pkl']['sha256'])
                        if packet is not None and 'physical_truth.pkl' in private:
                            physical_truth=self.load(physical_dir/'physical_truth.pkl',private['physical_truth.pkl']['sha256'])
                            self.physical(case_key,case,phase,row,lock,receipt,packet,physical_truth)
                        else:self.record_counts['physical_packets_unavailable_for_full_replay']+=1
                    if row['status'] not in ('evaluated','evaluated_operation_failed'):
                        self.record_counts['unresolved_future_mappings']+=1
                        self.check(label+'.failure_preserved',row['status'] not in ('started','fit_complete'))
                        continue
                    self.check(label+'.freeze_fit_identity',frozen_row['status']=='fit_complete' and frozen_row['fit_result_sha256']==row['fit_result_sha256'])
                    self.check(label+'.evaluation_after_freeze',utc(row['evaluation_utc'])>=utc(frozen['frozen_utc']))
                    self.check(label+'.truth_not_input',row['truth_read_before_fit'] is False)
                    fit_path=Path(row['fit_result_path']);self.check(label+'.fit_hash',sha(fit_path)==row['fit_result_sha256'])
                    fit=read(fit_path)
                    self.check(label+'.fit_status',fit['status']=='complete' and fit['final_strategy']=='multistart_original_rule')
                    self.check(label+'.algorithm_source',fit['source_sha256']==self.cfg['algorithm_source_sha256'])
                    self.check(label+'.settings',fit['settings']['per_path_seconds']==25. and fit['settings']['final_validation_fraction']==.03 and fit['settings']['resolution_factor']==.16)
                    if receipt is None:raise ValueError('scored mapping lacks physical receipt')
                    if packet is None or truth is None:raise ValueError('scored mapping lacks hash-bound observed packet or evaluator truth')
                    joined={role:self.input_frames(history[role]+packet[role]) for role in ('training','validation')}
                    self.check(label+'.full_rows',all(len(value[0][0])==480 for value in joined.values()) and fit['training_rows']==fit['validation_rows']==480)
                    self.check(label+'.exact_observed_hash',all(value[1]==fit['input_sha256'][role] for role,value in joined.items()))
                    self.check(label+'.cache_key',fit['matrix_cache']['cache_key']==row['observed_cache_key'])
                    independent=self.score(fit,truth,case['root_label'],{role:v[0] for role,v in joined.items()})
                    self.compare_models(label,independent,row['models'])
                    if row['status']=='evaluated':self.record_counts['evaluated_future_mappings']+=1
                    else:self.record_counts['operation_failed_future_mappings']+=1
                    self.full_score_rows.append(dict(case_key=case_key,phase=phase,path=row['path'],policy=row['policy'],status=row['status'],actual_payment=row['actual_payment'],models={name:{k:v for k,v in values.items() if k not in ('missing','extra')} for name,values in independent.items()}))
                self.check(case_key+'.'+phase+'.denominators',ended['planned']==len(expected) and ended['recorded']==len(rows) and ended['evaluated']==sum(r['status']=='evaluated' for r in rows) and ended['operation_failed_complete']==sum(r['status']=='evaluated_operation_failed' for r in rows))
                scored=sum(r['status'] in ('evaluated','evaluated_operation_failed') for r in rows)
                if ended['truth_signature'] is None:
                    self.check(case_key+'.'+phase+'.truth_not_scored_by_original_driver',scored==0 and ended['historical_stop'] is None)
                    self.record_counts['phases_with_original_truth_and_baseline_unscored']+=1
                elif phase_truth is not None:
                    signature=hashlib.sha256(json.dumps(sorted(phase_truth),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
                    self.check(case_key+'.'+phase+'.original_truth_signature',ended['truth_signature']==signature)
                if phase=='planning':self.check(case_key+'.planning.selection_binding',oracle['planning_phase_sha256'][case_key]==sha(folder/'phase_ended.json'))
            baseline=read(self.root/case_key/'baseline/result.json')
            self.record_counts['historical_baseline_mappings']+=1
            original_baseline_path=Path(case['baseline_frozen_result']['path'])
            self.check(case_key+'.baseline.frozen_hash',sha(original_baseline_path)==case['baseline_frozen_result']['sha256'])
            original_baseline=read(original_baseline_path)
            self.check(case_key+'.baseline.no_new_numerical_fit',{k:v for k,v in baseline.items() if k!='matrix_cache'}=={k:v for k,v in original_baseline.items() if k!='matrix_cache'})
            self.check(case_key+'.baseline.full_history',baseline['training_rows']==baseline['validation_rows']==288)
            self.check(case_key+'.baseline.input_hash',all(baseline['input_sha256'][role]==hobs[role][1] for role in hobs))
            if first_truth is not None:
                base_score=self.score(baseline,first_truth,case['root_label'],{role:pair[0] for role,pair in hobs.items()})
                self.independent_historical_stop[case_key]={name:{k:v for k,v in values.items() if k not in ('missing','extra')} for name,values in base_score.items()}
                for phase in ('planning','test'):
                    original_score=self.phase_data[(case_key,phase)]['historical_stop']
                    if original_score is not None:self.compare_models(case_key+'.'+phase+'.historical_stop',base_score,original_score)
                    else:self.record_counts['original_historical_stop_missing_retained']+=1
            cache=self.root/case_key/'fit_cache';ledger={};failed_cache=0;unknown_cache=[];cache_entries=0
            for item in cache.iterdir():
                if not item.is_dir():continue
                cache_entries+=1
                result=item/'result.json';checkpoint=item/'checkpoint.json'
                value=read(result if result.exists() else checkpoint) if result.exists() or checkpoint.exists() else {}
                if 'LP_solves' in value:ledger[item.name]=len(value['LP_solves'])
                else:unknown_cache.append(item.name)
                if value.get('status')!='complete':failed_cache+=1
                for lp in value.get('LP_solves',[]):
                    if lp.get('accepted'):
                        self.check(case_key+'.cached_LP.numerical_receipt',lp.get('status')==2 and lp.get('max_constraint_violation',0)<=1e-7 and lp.get('max_bound_violation',0)<=1e-7)
            final_ledger=self.phase_data[(case_key,'test')]['LP_ledger']
            self.check(case_key+'.cumulative_LP_ledger',final_ledger['by_key']==ledger and final_ledger['total']==sum(ledger.values()) and set(final_ledger['unknown_cache_keys'])==set(unknown_cache))
            self.check(case_key+'.fit_request_cap',cache_entries<=self.cfg['max_new_fits_per_case'])
            self.record_counts['unique_fit_cache_entries']+=cache_entries
            self.record_counts['unknown_fit_cache_entries']+=len(unknown_cache)
            self.record_counts['new_fixed_family_LP_receipts']+=sum(ledger.values())
            self.record_counts['failed_or_incomplete_fit_cache_entries']+=failed_cache
            zero_paths=[value for (key,phase,path),value in self.natural_path_hashes.items() if key==case_key]
            self.check(case_key+'.independent_path_hashes',len(set(zero_paths))==len(zero_paths))
        self.check('planned_future_ledger',self.record_counts['future_mappings']==self.cfg['planned_physical_fit_requests']==64)
        self.check('planned_baseline_ledger',self.record_counts['historical_baseline_mappings']==self.cfg['planned_historical_baselines']==2)
        self.check('fit_upper_66',self.record_counts['unique_fit_cache_entries']<=66)

    def audit_selection_and_aggregate(self):
        oracle=read(self.root/'oracle_selected.json');aggregate=read(self.root/'aggregate.json')
        independent_groups=[];independent_pairs=[]
        for case_key,case in self.cfg['cases'].items():
            lock=read(case['plan_lock']['path']);planning=self.phase_data[(case_key,'planning')]
            den=len(self.cfg['planning_paths']);cap=4*len(lock['labels']);summaries=[]
            for policy in self.cfg['policies']:
                rows=[r for r in planning['rows'] if r['policy']==policy]
                known=[r['models']['multistart_original_rule']['full_errors'] for r in rows if r['status']=='evaluated']
                fee=.4*sum(c*(0.,.5,.75)[c] for row in lock['plans'][policy] for c in row)
                summaries.append(dict(policy=policy,denominator=den,known=len(known),unresolved=den-len(known),mean_error_lower=sum(known)/den,mean_error_upper=(sum(known)+(den-len(known))*cap)/den,expected_fee=fee))
            actual=oracle['cases'][case_key]
            self.check(case_key+'.selection.risk_table',summaries==actual['policies'])
            resolved=all(r['unresolved']==0 for r in summaries)
            selected=None;ties=[]
            if resolved:
                best=min(r['mean_error_lower'] for r in summaries)
                risk_ties=[r for r in summaries if abs(r['mean_error_lower']-best)<=1e-12]
                best_fee=min(r['expected_fee'] for r in risk_ties)
                ties=[r['policy'] for r in risk_ties if abs(r['expected_fee']-best_fee)<=1e-12]
                selected=Random(str(self.cfg['oracle_tie_seed'])+':'+case_key).choice(ties)
            self.check(case_key+'.selection.expectation_before_choice',actual['selection_resolved']==resolved and actual['selected_policy']==selected and actual['all_error_and_fee_ties']==ties)
            test=self.phase_data[(case_key,'test')];lookup={(r['path'],r['policy']):r for r in test['rows']}
            for policy in self.cfg['policies']:
                rows=[r for r in test['rows'] if r['policy']==policy];evaluated=[r for r in rows if r['status']=='evaluated']
                group=dict(case_key=case_key,policy=policy,planned=len(self.cfg['test_paths']),evaluated=len(evaluated),unresolved=len(self.cfg['test_paths'])-len(evaluated),known_fee_sum=sum(r['actual_payment'] for r in rows if r['actual_payment'] is not None),unknown_fee_records=sum(r['actual_payment'] is None for r in rows),models={name:dict(evaluated=len(evaluated),full_error_sum=sum(r['models'][name]['full_errors'] for r in evaluated),full_recovery=sum(r['models'][name]['full_recovery'] for r in evaluated)) for name in METHODS},selected_by_planning_reference=selected==policy)
                independent_groups.append(group)
            if selected is not None:
                for baseline in ('zero','single','rotation'):
                    pairs=[]
                    for path in self.cfg['test_paths']:
                        a,b=lookup.get((path,selected)),lookup.get((path,baseline))
                        if a and b and a['status']==b['status']=='evaluated':pairs.append(dict(path=path,errors_delta=a['models']['multistart_original_rule']['full_errors']-b['models']['multistart_original_rule']['full_errors'],fee_delta=a['actual_payment']-b['actual_payment']))
                    independent_pairs.append(dict(case_key=case_key,comparison='frozen_reference_minus_'+baseline,planned=len(self.cfg['test_paths']),paired=len(pairs),rows=pairs))
        self.check('aggregate.frozen_reference',aggregate['oracle_selection_sha256']==sha(self.root/'oracle_selected.json'))
        self.check('aggregate.groups',aggregate['groups']==independent_groups)
        self.check('aggregate.paired_controls',aggregate['paired_frozen_reference']==independent_pairs)
        return dict(groups=independent_groups,paired_reference=independent_pairs)


def run(root,out):
    out=Path(out).resolve()
    if (out/'receipt.json').exists():raise FileExistsError('audit result retained; do not overwrite')
    a=Audit(root,out)
    result=None
    try:
        a.audit_rows();aggregation=a.audit_selection_and_aggregate();a.guard()
        failed=[row for row in a.checks if not row['passed']]
        result=dict(status='passed' if not failed else 'failed',checks_total=len(a.checks),checks_passed=len(a.checks)-len(failed),failed=failed,checks=a.checks,counts=dict(a.record_counts),maxima=a.maxima,aggregate=aggregation,score_rows=a.full_score_rows,independent_historical_stop=a.independent_historical_stop,original_unscored_fields_preserved=True,file_sha256=a.files,marker_sha256=a.markers,new_identifier_calls=0,new_LP_calls=0,new_AC_calls=0,new_future_generated=False,full_support_scoring_without_weight_threshold=True,oracle_chosen_from_planning_mean_not_per_realization=True,test_outcomes_used_to_choose_reference=False,limitations=['four simulated heldout paths per fixed case; not a stable population advantage claim','geometric NJ replay reuses established graph function on stored front; no new frontend fit','saved AC equation receipts checked but equations not independently re-solved','synthetic fixed device contract; no empirical user-response calibration'])
    except Exception as error:
        result=dict(status='audit_error',exception_type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),checks=a.checks,counts=dict(a.record_counts),maxima=a.maxima,new_identifier_calls=0,new_LP_calls=0,new_AC_calls=0)
    write(out/'receipt.json',result)
    summary={key:value for key,value in result.items() if key not in ('checks','score_rows','file_sha256','marker_sha256','traceback')}
    summary['receipt_sha256']=sha(out/'receipt.json');summary['audit_source_sha256']=sha(__file__)
    write(out/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k not in ('aggregate','failed')},ensure_ascii=True))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--root',required=True,type=Path);parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--expected-protocol-sha',default=EXPECTED_PROTOCOL)
    args=parser.parse_args()
    if args.mode=='prepare':prepare(args.root,args.out,args.expected_protocol_sha)
    else:raise SystemExit(run(args.root,args.out))
