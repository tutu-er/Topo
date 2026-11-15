"""Independent saved-array audit for node/hour P-Q bands and SDP seeds.

The unchanged single-target auditor supplies complete-native scoring, Gaussian
gates, paired metering, fees and electrical balance. This module independently
reconstructs the new training-only band. No optimization or AC is run here.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import csv
import hashlib
import json
import shutil
import traceback

import numpy as np

from ..single_target_20261115 import audit as single

base, panel, prob, previous, ROOT = single.base, single.panel, single.prob, single.previous, single.ROOT


def local_hours(timestamps, zone='Europe/Berlin'):
    """Require explicit timezone; row zero is not assumed to be midnight."""
    answer = []
    for stamp in timestamps:
        value = datetime.fromisoformat(str(stamp).replace('Z', '+00:00'))
        if value.tzinfo is None:
            raise ValueError('A timezone-aware source timestamp is required')
        answer.append(value.astimezone(ZoneInfo(zone)).hour)
    return np.asarray(answer, dtype=int)


def reconstruct_band(p, q, train_timestamps, decision_timestamps, zone='Europe/Berlin'):
    """Full-training affine slope; same-clock-hour residual max minus min."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.shape != q.shape or p.ndim != 2 or len(p) != len(train_timestamps):
        raise ValueError('Unaligned training powers/timestamps')
    if not np.isfinite(p).all() or not np.isfinite(q).all():
        raise ValueError('Nonfinite training powers')
    centered_p, centered_q = p-p.mean(0), q-q.mean(0)
    energy = np.square(centered_p).sum(0)
    if np.any(energy <= np.finfo(float).tiny):
        raise ValueError('Constant training P has no identifiable affine slope')
    slope = (centered_p*centered_q).sum(0)/energy
    intercept = q.mean(0)-slope*p.mean(0)
    residual = q-p*slope-intercept
    hours, decision_hours = local_hours(train_timestamps, zone), local_hours(decision_timestamps, zone)
    widths, counts = [], []
    for hour in range(24):
        values = residual[hours == hour]
        if len(values) < 2:
            raise ValueError('Every local hour needs at least two training observations')
        widths.append(values.max(0)-values.min(0)); counts.append(len(values))
    widths = np.asarray(widths)
    return dict(slope=slope, intercept_pu=intercept, residual_pu=residual,
        width_by_clock_hour_pu=widths, width_pu=widths[decision_hours],
        training_clock_hour=hours, decision_clock_hour=decision_hours,
        training_count_by_hour=np.asarray(counts, int))


def direct_band(delta_p, delta_q, slope, width_pu, tolerance=None):
    """Planning increments only; positive means the original band is exceeded."""
    p, q, k, b = map(lambda x: np.asarray(x, float), (delta_p, delta_q, slope, width_pu))
    if p.shape != q.shape or p.shape != b.shape or p.shape[1:] != k.shape:
        raise ValueError('Band/increment dimensions differ')
    if np.any(b < 0) or not all(np.isfinite(x).all() for x in (p, q, k, b)):
        raise ValueError('Nonfinite or negative band')
    deviation = q-p*k
    excess = np.abs(deviation)-b
    roundoff = 64*np.finfo(float).eps*(np.abs(q)+np.abs(p*k)+b)
    allowed = roundoff if tolerance is None else np.full_like(b,float(tolerance))
    encoded_excess = np.abs(deviation)-b*(1.-1e-7)
    return dict(passed=bool(np.all(excess <= allowed)),
        encoded_passed=bool(np.all(encoded_excess <= allowed)),
        encoded_maximum_violation_pu=float(np.maximum(encoded_excess,0.).max(initial=0.)),
        maximum_violation_pu=float(np.maximum(excess,0.).max(initial=0.)),
        minimum_slack_pu=float((-excess).min(initial=np.inf)),
        violated_cells=int(np.sum(excess > allowed)),
        cells=int(excess.size), maximum_roundoff_tolerance_pu=float(allowed.max(initial=0)),
        maximum_absolute_departure_pu=float(np.max(np.abs(deviation), initial=0)),
        zero_band_cells=int(np.sum(b == 0)),
        meaning='Incremental training-relation band; delivered violations are diagnostics, not physical device violations')


def no_sdp_certificate(receipt):
    """Approximate SDP output must not silently become a valid QCQP bound."""
    if (receipt.get('lower_bound_certified', False) or receipt.get('used_as_qcqp_lower_bound', False)
            or receipt.get('bound_validated', False) or receipt.get('effective_qcqp_lower_bound') is not None):
        raise ValueError('This experiment has no certified SDP lower-bound implementation')
    return dict(certified_lower_bound=False,
        scope='SDP objective/residuals are numerical diagnostics and seed generation only')


def validate_band_reference(protocol, baseline_protocol):
    if baseline_protocol.get('variant')!='band':
        raise ValueError('SDP comparison requires the no-SDP node/hour experiment')
    if not protocol.get('smoke',False) and baseline_protocol.get('smoke',False):
        raise ValueError('Formal SDP cannot use a smoke baseline')
    if baseline_protocol['repeats']<protocol['repeats']:
        raise ValueError('Band baseline has fewer paired paths than the SDP comparison')


def independent_mgf_polynomials(model, forecast, resource_map, mode):
    """Joint-Gaussian integral expansion, independent of production evidence."""
    if mode not in ('mgf_fixed','mgf_forecast'):
        raise ValueError('Only the two exponential constraints are lifted')
    t,n=forecast.shape[0],model['n'];h=t//4;r=resource_map.shape[1]
    score=model['score'];sigma2=model['sigma2']
    tie=max(1e-7,(score['cells']+t*n)*score['reference_mse']*score['validation_tie']/(2*sigma2))
    target=model['targets'][0];i=model['representatives'][target]
    rivals=[j for j,f in enumerate(model['families']) if f!=model['families'][i]]
    sd=model.get('forecast_relative_sd',.05)*abs(forecast) if mode=='mgf_forecast' else np.zeros_like(forecast)
    def expand(matrix):
        q=np.einsum('ci,tcd,dj->tij',resource_map,matrix,resource_map).reshape(h,4,r,r).sum(1)
        a=np.einsum('ci,tcd,td->ti',resource_map,matrix,forecast).reshape(h,4,r).sum(1)
        c=float(np.einsum('tc,tcd,td->',forecast,matrix,forecast))
        return q,a,c
    rows=[]
    for j in rivals:
        hist=(model['history'][j]-model['history'][i])/(2*sigma2)
        integral=prob.PairIntegral(model['H'][i],model['H'][j],model['means'][target],sigma2,model['covariance'],sd,hist)
        lam=integral.selected_lambda;eigen=1+2*lam*integral.eigenvalues
        effective=lam*integral.Q[None]-2*lam*lam*np.einsum('tki,tk,tkj->tij',integral.projected_G,1/eigen,integral.projected_G)
        q,a,c=expand(effective)
        c+=np.log(.1/len(rivals))+lam*(hist-tie)+.5*np.log(eigen).sum()
        rows.append((q,a,c))
    guards=[]
    for rep,j in enumerate(model['representatives']):
        if rep==target:
            continue
        di=model['means'][target]-model['H'][i];dj=model['means'][target]-model['H'][j]
        matrix=(dj.T@dj-di.T@di)/(2*sigma2)
        q,a,_=expand(np.broadcast_to(matrix,(t,2*n,2*n)))
        guards.append((q,a))
    return dict(gate_q=np.asarray([x[0] for x in rows]),gate_a=np.asarray([x[1] for x in rows]),gate_c=np.asarray([x[2] for x in rows]),
        guard_q=np.asarray([x[0] for x in guards]),guard_a=np.asarray([x[1] for x in guards]))


def independent_moments(problem, mean, moments):
    """Numerical diagnostics, deliberately not a primal/dual certificate."""
    z=np.asarray(mean);w=np.asarray(moments)
    if w.shape != (len(z),z.shape[1],z.shape[1]):
        raise ValueError('SDP moment dimensions differ')
    gates=np.einsum('khij,hij->k',problem['gate_q'],w)+2*np.einsum('khi,hi->k',problem['gate_a'],z)+problem['gate_c']
    guards=np.einsum('khij,hij->k',problem['guard_q'],w)+2*np.einsum('khi,hi->k',problem['guard_a'],z)
    mins=[]
    for hour in range(len(z)):
        active=np.flatnonzero((problem['lower'][hour]!=0)|(problem['upper'][hour]!=0))
        a=z[hour,active];b=w[hour][np.ix_(active,active)]
        block=np.block([[np.ones((1,1)),a[None]],[a[:,None],b]])
        mins.append(float(np.linalg.eigvalsh((block+block.T)/2).min()))
    return dict(minimum_eigenvalues=mins,gate_margins=gates.tolist(),scalar_guard_gains=guards.tolist(),
        certified_lower_bound=False,scope='Independently recomputed approximate SDP point diagnostics only')


class Audit(single.Audit):
    def __init__(self):
        super().__init__()
        self.band_context = {}
        self.band_records = []
        self.single_baseline_records = []
        self.band_baseline_records = []
        self.sdp_records = []

    def action_band(self, entry, action):
        c = self.band_context[entry['id']]
        power = prob.shift(c['config'], action, c['labels'], entry['base_kw'])
        self.require(power.shape[0] == len(action)*4, 'Unexpected action/measurement interval ratio')
        hourly = power[::4]
        self.same('constant_hourly_increment', power, np.repeat(hourly, 4, axis=0), atol=0, rtol=0)
        return direct_band(hourly[:, :c['n']], hourly[:, c['n']:], c['band']['slope'], c['band']['width_pu'])

    def single_quote(self, entry, name, quote, action, direct, bill, seeds, seed_check):
        if quote.get('solver_result_reused'):
            self.require(quote.get('optimizer_invoked') is False and quote.get('native_status') is None,
                         'Reused quote presented as a newly invoked native optimization')
            self.require(quote.get('seconds')==0 and quote.get('lower_bound_yuan')==0,
                         'Reused quote revives native timing or an unaudited bound')
            prior=base.read(self.checked(quote['reused_quote_ref']))
            prior_action=base.native.arrays(self.checked(quote['reused_action_ref']))['action']
            self.same('reused_band_action',action,prior_action,atol=0,rtol=0)
            self.require(bool(quote['new_gate_feasible'])==bool(prior['new_gate_feasible']) and
                         bool(quote['zero_fallback'])==bool(prior['zero_fallback']), 'Reused feasibility flag changed')
            if quote['new_gate_feasible']:
                self.same('reused_band_objective',quote['cost_yuan'],prior['cost_yuan'],atol=0,rtol=0)
        check = self.action_band(entry, action)
        self.require(bool(quote['planned_band_feasible']) == check['passed'], 'Reported P/Q band flag differs')
        self.require(check['passed'], 'Accepted or zero fallback request violates P/Q band')
        saved = quote['band_direct']
        self.require(bool(saved['passed']) == check['passed'], 'Saved direct band audit differs')
        if 'max_violation_kvar' in saved:
            self.same('band_maximum_violation', saved['max_violation_kvar'],
                      max(0.,check['maximum_violation_pu'])*entry['base_kw'], atol=1e-8)
        def checked_seed(seed, mode):
            gate, fee = seed_check(seed, mode)
            band = self.action_band(entry, seeds[seed])
            return dict(gate, feasible=gate['feasible'] and band['passed']), fee
        super().single_quote(entry, name, quote, action, direct, bill, seeds, checked_seed)
        row = self.quote_records[-1]
        encoded = (check['encoded_passed'] and
            all(r['margin'] >= 1e-8-1e-9 for r in direct['rows']) and
            all(g['gain'] >= -1e-9 for g in direct['guards']))
        cert = row['certificate']
        # Solver bounds concern the slightly smaller numerical encoding only.
        # Re-evaluate history conflicts only against a point in that domain.
        diagnostics = Path(self.where)/(name+'_diagnostics')
        log, trace = diagnostics/'native.log', diagnostics/'progress.jsonl'
        history = previous.native_bound_history(self.file(log) if log.exists() else None,
            self.file(trace) if trace.exists() else None,
            bill['activation_yuan'] if quote['new_gate_feasible'] else None,
            bool(quote['new_gate_feasible'] and encoded))
        cert.update(native_bound_history=history, encoded_domain_feasible=encoded,
            original_domain_feasible=bool(direct['feasible'] and check['passed']),
            lower_bound_valid_for_unbuffered_band=False,
            bound_scope='Activation objective on the encoded probability and node-hour-band QCQP; positive bands shrunk by1e-7; not an unbuffered-domain lower bound',
            audit_withdrawal_due_to_bound_trajectory=history['recorded_bound_conflict_with_verified_buffered_point'])
        if history['recorded_bound_conflict_with_verified_buffered_point'] or not encoded:
            cert['raw_encoded_lower_bound_yuan']=quote.get('lower_bound_yuan')
            cert.update(lower_bound_yuan=0.,relative_gap=1. if bill['activation_yuan']>0 else 0.,
                        lower_bound_withdrawn=True)
        row.update(planning_band=check, sdp_receipts=quote.get('sdp_receipts'),
            solver_result_reused=quote.get('solver_result_reused',False),
            reused_quote_ref=quote.get('reused_quote_ref'),reused_action_ref=quote.get('reused_action_ref'),
            original_initial_points=quote.get('original_initial_points'))

    def case(self, folder, entry, protocol, metadata, branches, buses):
        self.where = str(folder)
        seal = base.read(self.file(folder/'seal.json')); self.references(seal)
        self.require(seal['train'] == entry['files']['train.npz'], 'Band training source changed')
        self.require(seal['sampling_source'] == entry['files']['observations_sealed.json'], 'Training timestamp source changed')
        train = base.native.arrays(self.checked(seal['train']))
        sampling = base.read(self.checked(seal['sampling_source']))
        self.require(sampling['observations']['train']['ref'] == seal['train'], 'Train/timestamp ledger mismatch')
        timestamps = sampling['sampling']['train']['timestamps']
        indices = sampling['sampling']['train']['indices']
        self.require(indices == list(range(0,192,int(entry['stride']))), 'Historical sampled rows changed')
        decision_timestamps = sampling['sampling']['new_day']['timestamps'][::4]
        self.require(len(decision_timestamps) == 24, 'Expected original 24 decision hours')
        band = reconstruct_band(train['P'], train['Q'], timestamps, decision_timestamps, protocol['bandwidth_timezone'])
        meta = base.read(self.checked(seal['bandwidth_json'])); arrays = base.native.arrays(self.checked(seal['bandwidth_npz']))
        original_meta = base.read(self.checked(entry['node_time_bandwidth']))
        self.require(meta == original_meta, 'Band metadata changed between preparation and quote')
        self.require(meta['training_source'] == seal['train'] and meta['sampling_source'] == seal['sampling_source'],
                     'Band sources not limited to original training')
        self.require(meta['multiplier'] == 1 and meta['bandwidth_floor'] is None, 'Unannounced band scale/floor')
        self.same('band_base', meta['common_power_base_kva'], entry['base_kw'], atol=0, rtol=0)
        expected = dict(labels=train['labels'], kappa=band['slope'],
            intercept_kvar=band['intercept_pu']*entry['base_kw'], band_kvar=band['width_pu']*entry['base_kw'],
            counts=np.repeat(band['training_count_by_hour'][band['decision_clock_hour'],None], len(train['labels']), axis=1),
            decision_hours=band['decision_clock_hour'])
        self.require(set(arrays) == set(expected), 'Band arrays have missing/extra fields')
        for key, value in expected.items():
            self.same('training_only_band_'+key, arrays[key], value, atol=1e-10, rtol=1e-10)
        self.same('band_same_clock_as_actual_day', arrays['decision_hours'],
                  (protocol['bandwidth_start_hour']+np.arange(24))%24, atol=0, rtol=0)
        copied = base.native.arrays(self.checked(meta['arrays']))
        for key in arrays:
            self.same('copied_prepared_band_'+key, arrays[key], copied[key], atol=0, rtol=0)
        labels = list(map(int, train['labels'])); config = base.read(self.checked(seal['resource_case']))
        frozen_band=dict(band,slope=arrays['kappa'],width_pu=arrays['band_kvar']/entry['base_kw'])
        self.band_context[entry['id']] = dict(band=frozen_band, config=config, labels=labels, n=len(labels))
        original = base.native.arrays(self.checked(entry['initial_points']))
        saved_original = base.native.arrays(self.checked(seal['original_initial_points']))
        final = base.native.arrays(self.checked(seal['shared_initial_points']))
        self.require(set(original) == set(saved_original) and set(original) <= set(final), 'Original seed pool changed')
        for key in original:
            self.same('original_pool_copy_'+key, saved_original[key], original[key], atol=0, rtol=0)
            self.same('original_pool_preserved_'+key, final[key], original[key], atol=0, rtol=0)
        origins = base.read(self.checked(seal['sdp_seed_origins']))
        self.require(set(origins) == set(final)-set(original), 'Extra seed provenance incomplete')
        if not protocol['use_sdp']:
            self.require(not origins and not seal['sdp_receipts'], 'SDP data entered the no-SDP comparison')
        self.band_records.append(dict(id=entry['id'], training_rows=len(train['P']),
            source=seal['train'], sampling_source=seal['sampling_source'],
            minimum_width_kvar=float(arrays['band_kvar'].min()), maximum_width_kvar=float(arrays['band_kvar'].max()),
            zero_width_cells=int(np.sum(arrays['band_kvar']==0)), counts_min=int(arrays['counts'].min()),
            decision_hours=arrays['decision_hours'].tolist(), original_seed_count=len(original), extra_seed_count=len(origins)))
        # The inherited complete-AC audit reads exactly the final pool supplied to
        # the engine. Original-pool preservation/provenance is checked above.
        audited_entry = dict(entry, initial_points=seal['shared_initial_points'])
        before = len(self.records)
        super().case(folder, audited_entry, protocol, metadata, branches, buses)
        for record in self.records[before:]:
            if record['audit_status'] != 'passed':
                continue
            obs = base.native.arrays(self.file(Path(record['source'])/'observations.npz'))
            record['planning_band'] = self.action_band(entry, obs['requested_action'])
            record['delivered_band'] = self.action_band(entry, obs['delivered_action'])
            self.require(record['planning_band']['passed'], 'Sealed request exceeds its training band')
            self.require(all(np.isfinite(v) and v<=1e-7 for v in record['requested_violations'].values()),
                         'Accepted or fallback request violates original physical resources')
            self.require(all(np.isfinite(v) and v<=1e-7 for v in record['delivered_violations'].values()),
                         'Actual delivery violates original physical resources')
        for mode, reference in seal['sdp_receipts'].items():
            receipt = base.read(self.checked(reference)); self.references(receipt)
            no_sdp_certificate(receipt)
            self.sdp_records.append(dict(id=entry['id'], mode=mode, source=reference,
                status=receipt['status'], wall_seconds=receipt.get('wall_seconds'),
                added_seed_names=receipt.get('added_seed_names',[]), certified_lower_bound=False))
        # Every added seed is rechecked even when it is only a non-feasible warm
        # point. Its own mode probability flag does not imply gate feasibility in
        # the other mode sharing the union pool.
        if origins:
            ctx = self.context(entry); forecast = base.native.arrays(self.checked(seal['planner']))['forecast']
            noise = panel.noise_parameters(entry, protocol)
            model = single.historical_target(prob.build_model(ctx['score'],
                base.native.arrays(self.checked(entry['files']['validation.npz'])), ctx['native'],
                meter_relative_sd=noise['meter_relative_sd'],forecast_relative_sd=noise['forecast_relative_sd']))
            for name, origin in origins.items():
                self.require(origin['mode'] in protocol['modes'] and origin['receipt'] is not None, 'Missing SDP seed source')
                source=base.read(self.checked(origin['receipt']))
                candidate=next((r for r in source.get('candidates',[]) if r.get('pool_name')==origin['label']),None)
                self.require(candidate is not None and candidate['admitted_to_pool'],'SDP seed has no admitted extraction record')
                source_actions=base.native.arrays(self.checked(candidate['action_ref']))
                self.same('SDP_seed_from_saved_extraction',final[name],source_actions[candidate['array_key']],atol=0,rtol=0)
                _, violations, dp, dq = panel.resource_values(config, final[name], labels, forecast[:,:len(labels)], entry['base_kw'])
                self.require(all(np.isfinite(v) and v<=1e-7 for v in violations.values()), 'SDP seed violates physical resource constraints')
                self.require(self.action_band(entry, final[name])['passed'], 'SDP seed violates node/hour band')
                direct = self.gates(model, forecast, forecast+np.c_[dp,dq], origin['mode'], origin['direct_evidence'])
                self.require(bool(candidate['probability_feasible'])==direct['feasible'],'SDP candidate probability flag differs')
                self.sdp_records.append(dict(id=entry['id'], seed=name, mode=origin['mode'],
                    probability_feasible=direct['feasible'], band_feasible=True, physical_feasible=True))
        if protocol['use_sdp']:
            ctx=self.context(entry);forecast=base.native.arrays(self.checked(seal['planner']))['forecast']
            noise=panel.noise_parameters(entry,protocol)
            model=single.historical_target(prob.build_model(ctx['score'],
                base.native.arrays(self.checked(entry['files']['validation.npz'])),ctx['native'],
                meter_relative_sd=noise['meter_relative_sd'],forecast_relative_sd=noise['forecast_relative_sd']))
            resource_map=np.zeros((2*len(labels),len(config['resources'])))
            for col,resource in enumerate(config['resources']):
                node=labels.index(int(resource['node_label']))
                if config['action_units'][col]=='kW':
                    resource_map[node,col]=1/entry['base_kw']
                    resource_map[len(labels)+node,col]=resource.get('q_over_p',0.)/entry['base_kw']
                else:resource_map[len(labels)+node,col]=1/entry['base_kw']
            for mode in protocol['modes']:
                directory=folder/(mode+'_sdp')
                if not (directory/'problem.npz').exists():continue
                problem=base.native.arrays(self.file(directory/'problem.npz'))
                polynomials=independent_mgf_polynomials(model,forecast,resource_map,mode)
                for key,value in polynomials.items():
                    self.same('independent_SDP_integral_'+key,problem[key],value,atol=1e-8,rtol=1e-9)
                if (directory/'moments.npz').exists():
                    moments=base.native.arrays(self.file(directory/'moments.npz'))
                    self.sdp_records.append(dict(id=entry['id'],mode=mode,
                        moment_diagnostics=independent_moments(problem,moments['mean'],moments['moments'])))


def planning_boundary(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    runner = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run_case')
    calls = [n for n in ast.walk(runner) if isinstance(n,ast.Call)]
    solves = [n for n in calls if ast.unparse(n.func)=='engine.plan']
    seals = [n for n in calls if ast.unparse(n.func)=='io.write' and n.args and isinstance(n.args[0],ast.BinOp)
        and isinstance(n.args[0].right,ast.Constant) and n.args[0].right.value=='seal.json']
    executes = [n for n in calls if ast.unparse(n.func)=='paired.execute']
    forbidden = [n for n in calls if ast.unparse(n.func) in ('old.context','paired.old.context','physical_response')]
    expected = ['case',"planning['score']","planning['val']",'forecast',"planning['base']"]
    passed = (len(solves)==len(seals)==len(executes)==1 and not forbidden
        and solves[0].lineno<seals[0].lineno<executes[0].lineno
        and [ast.unparse(a) for a in solves[0].args]==expected
        and [ast.unparse(a) for a in executes[0].args[-2:]]==['{}','dict(guards=[])'])
    return dict(passed=passed,source=base.ref(path),scope='Training P/Q bandwidth and history/forecast-only solver; original physical execution after sealing')


def run(source, out):
    source, out = Path(source).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError(out)
    if not (source/'complete.json').exists():
        raise ValueError('Complete source run required')
    out.mkdir(parents=True); shutil.copyfile(__file__, out/'audit.py')
    audit=Audit(); p=base.read(audit.file(source/'protocol.json'))
    entries=base.read(audit.checked(p['inputs'])); complete=base.read(audit.file(source/'complete.json'))
    audit.require(complete['source_unchanged'] and p['oracle'] is False, 'Source changed or oracle scope')
    policies=single.policy_names(p)
    audit.require(p['policies']==policies, 'Policy list differs')
    for reference in p['sources']:
        original=Path(reference['path']).resolve(); snapshot=source/'source_snapshot'/original.relative_to(ROOT)
        audit.file(snapshot,reference['sha256'])
        audit.source_snapshots[(str(original),reference['sha256'])]=snapshot
        current=base.ref(original)
        if current['sha256']!=reference['sha256']:
            audit.post_execution_source_changes.append(dict(executed_source=reference,snapshot=base.ref(snapshot),current_source=current))
    boundary=planning_boundary(source/'source_snapshot/rnj_wzzt_dr/experiments/node_time_pq_20261115/run.py')
    audit.require(boundary['passed'],'Planning information boundary differs')
    original=base.read(audit.checked(p['source_protocol']))
    audit.require(p['seed']==original['seed'] and p['forecasts']==1, 'Original random stream changed')
    original_entries={e['id']:e for e in base.read(audit.checked(original['inputs']))}
    for entry in entries:
        for key in ('score','native_copy','files','physical_cache','p_resources','q_resources','index','base_kw','meter'):
            audit.require(entry[key]==original_entries[entry['id']][key], 'Frozen input changed: '+key)
    ids={e['id'] for e in entries}
    for field, destination, allowed in (
            ('baseline_audit',audit.baseline_records,set(panel.POLICIES)),
            ('single_baseline_audit',audit.single_baseline_records,{'single_'+m for m in panel.MODES}),
            ('band_baseline_audit',audit.band_baseline_records,{'single_'+m for m in p['modes']})):
        if not p.get(field):
            continue
        prior=base.read(audit.checked(p[field]));audit.require(prior['status']=='passed','Unverified baseline '+field)
        if field=='band_baseline_audit':
            validate_band_reference(p,base.read(audit.checked(prior['source_protocol'])))
        for row in (json.loads(line) for line in audit.checked(prior['records']).read_text(encoding='utf8').splitlines()):
            if row['id'] in ids and row['repeat']<p['repeats'] and row['strategy'] in allowed:
                destination.append(row)
        expected=len(entries)*p['repeats']*len(allowed)
        audit.require(len(destination)==expected and all(r['audit_status']=='passed' for r in destination), 'Baseline denominator/status differs: '+field)
    expected_rows=len(entries)*len(policies)*p['repeats']
    audit.require(p['expected_rows']==expected_rows and p['expected_quotes']==len(entries)*len(policies),'Protocol denominator differs')
    succeeded={r['id'] for r in complete.get('receipts',[])}; failures={r['id'] for r in complete.get('failed_tasks',[])}
    audit.require(not(succeeded&failures) and succeeded|failures==ids,'Missing/duplicate task')
    base.write(out/'protocol.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source=base.ref(__file__),
        source_protocol=base.ref(source/'protocol.json'),planning_boundary=boundary,expected_rows=expected_rows,
        scope='Independent training-only bands, original single-target formulas, native scores, paired saved AC, prices and SDP warm-point provenance; no new solve/AC'))
    for entry in entries:
        if entry['id'] in failures:
            for repeat in range(p['repeats']):
                for policy in policies:
                    audit.records.append(dict(**panel.identity(entry),forecast=0,repeat=repeat,strategy=policy,
                        audit_status='recorded_failure',original_status='task_failed',evaluation={'native_correct':False}))
            continue
        try:
            casepath=source/'source_snapshot/topology_identification_standalone/cases'/entry['case']
            metadata=base.read(audit.file(casepath/'metadata.json'))
            with audit.file(casepath/'branches.csv').open(encoding='utf-8-sig',newline='') as stream:branches=list(csv.DictReader(stream))
            with audit.file(casepath/'buses.csv').open(encoding='utf-8-sig',newline='') as stream:buses=list(csv.DictReader(stream))
            audit.case(source/'cases'/entry['id'],entry,p,metadata,branches,buses)
        except Exception as exc:
            audit.task_errors.append(dict(id=entry['id'],error=repr(exc),traceback=traceback.format_exc()))
        base.write(out/'progress.json',dict(rows=len(audit.records),expected=expected_rows,task_errors=len(audit.task_errors)))
    for path,sha in audit.hashes.items():
        if base.ref(path)['sha256']!=sha:
            audit.task_errors.append(dict(path=path,error='Input changed during audit'))
    passed=(len(audit.records)==expected_rows and not audit.task_errors
        and all(r['audit_status'] in ('passed','recorded_failure') for r in audit.records))
    files={}
    for name,rows in (('records',audit.records),('quotes',audit.quote_records),('baseline_records',audit.baseline_records),
                      ('single_baseline_records',audit.single_baseline_records),('band_baseline_records',audit.band_baseline_records)):
        with (out/(name+'.jsonl')).open('w',encoding='utf8') as stream:
            for row in rows:panel.json_line(stream,row)
        files[name]=base.ref(out/(name+'.jsonl'))
    result=dict(status='passed' if passed else 'failed',protocol=base.ref(out/'protocol.json'),
        source_protocol=base.ref(source/'protocol.json'),source_inputs=p['inputs'],**files,
        expected_rows=expected_rows,audited_rows=len(audit.records),baseline_rows=len(audit.baseline_records),
        single_baseline_rows=len(audit.single_baseline_records),band_baseline_rows=len(audit.band_baseline_records),
        input_coverage=audit.input_records,task_errors=audit.task_errors,
        audit_status_counts=dict(Counter(r['audit_status'] for r in audit.records)),
        original_failed_tasks=complete.get('failed_tasks',[]),original_quote_exceptions=complete.get('quote_exceptions',0),
        original_execution_failed_rows=complete.get('failed_rows',0),post_execution_source_changes=audit.post_execution_source_changes,
        maximum_numeric_errors=dict(audit.maxima),maximum_electrical_errors=dict(audit.operation.maxima),
        counts=dict(audit.counts),band_checks=audit.band_records,sdp_checks=audit.sdp_records)
    base.write(out/'report.json',result);base.write(out/'complete.json',dict(status=result['status'],report=base.ref(out/'report.json')))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();r=run(args.run,args.out)
    print({k:r[k] for k in ('status','expected_rows','audited_rows','task_errors')})
    if r['status']!='passed':
        raise SystemExit(1)
