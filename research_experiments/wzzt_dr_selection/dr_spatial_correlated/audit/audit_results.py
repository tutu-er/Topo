"""Independent arithmetic/provenance replay after the spatial AC batch ends.

Reads completed private receipts only. It does not import experiment modules,
run a probability evaluator, construct a network, refit, optimize, or solve AC.
Truth labels are opened only after observations and predictions are frozen.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np


OUTPUT = Path(__file__).resolve().parent
STAGE = OUTPUT.parent
DEFAULT_ROOT = STAGE/'run_v2'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(reference):
    path = Path(reference['path'])
    if sha(path) != reference['sha256']:
        raise ValueError('frozen-reference hash mismatch: '+str(path))
    return path


def load(path):
    with Path(path).open('rb') as stream:
        return pickle.load(stream)


def array(x):
    return np.asarray(x, dtype=float)


def fingerprint(x):
    a = np.ascontiguousarray(array(x))
    return hashlib.sha256(str(a.shape).encode()+a.tobytes()).hexdigest()


class Audit:
    def __init__(self):
        self.counts = Counter()
        self.errors = []
        self.maxima = defaultdict(float)

    def check(self, name, condition, detail=None):
        self.counts[name] += 1
        if not bool(condition):
            self.errors.append(dict(check=name, detail=detail))

    def close(self, name, x, y, tolerance=1.e-11, detail=None):
        x, y = array(x), array(y)
        if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
            self.check(name, False, dict(context=detail, shapes=[list(x.shape), list(y.shape)]))
            return
        maximum = float(np.max(np.abs(x-y), initial=0.))
        self.maxima[name] = max(self.maxima[name], maximum)
        self.check(name, maximum <= tolerance, dict(context=detail, maximum=maximum, tolerance=tolerance))


def independent_z(day_seed, law, active_positions, n=15, T=96):
    """Closed AR coefficient matrix, independent of response_law's recursion."""
    xi = np.random.default_rng(day_seed).standard_normal(1+n+T+n*T)
    phi = float(law['phi'])
    ar_matrix = np.zeros((T, T))
    ar_matrix[:, 0] = phi**np.arange(T)
    for t in range(1, T):
        ar_matrix[t, 1:t+1] = np.sqrt(1-phi*phi)*phi**np.arange(t-1, -1, -1)
    day = xi[0]
    node = xi[1:1+n]
    common = ar_matrix @ xi[1+n:1+n+T]
    individual = xi[1+n+T:].reshape(n, T) @ ar_matrix.T
    wg, wi, wc, we = array(law['weights'])
    z = np.sqrt(wg)*day + np.sqrt(wi)*node[:, None] + np.sqrt(wc)*common[None, :] + np.sqrt(we)*individual
    return z[np.asarray(active_positions, int)]


def errors(supports, truth):
    proposed = {tuple(sorted(map(int, s))) for s in supports}
    return dict(missing=len(truth-proposed), extra=len(proposed-truth), errors=len(proposed ^ truth))


def selection_replay(pool, historical_abs, scene):
    """Independent original common3%, new weight2, original complexity ties."""
    order = pool['selection_order']
    models = {m['candidate_id']: m for m in pool['models']}
    p, q = array(scene['P_terminal']), array(scene['Q_terminal'])
    y = array(scene['drop_target'])
    new_abs = {k:float(np.abs(y-(p @ array(models[k]['R']).T + q @ array(models[k]['X']).T)).sum()) for k in order}
    nh, nd = int(pool['historical_validation_cells']), y.size
    historical = {k:historical_abs[k]/nh for k in order}
    hmin = min(historical.values())
    scope = [k for k in order if historical[k] <= 1.03*hmin+1.e-12]
    means = {k:(historical_abs[k]+2*new_abs[k])/(nh+2*nd) for k in order}
    reference = min(scope, key=lambda k:(means[k], order.index(k)))
    threshold = 1.03*means[reference]+1.e-12
    eligible = [k for k in scope if means[k] <= threshold]
    selected = min(eligible, key=lambda k:(len(models[k]['point']['supports']), means[k], order.index(k)))
    return dict(selected=selected, eligible=eligible, historical_eligible=scope,
        new_losses={k:new_abs[k]/nd for k in order}, weighted_losses=means,
        margin_micro=(means['coarse']-threshold)*1.e6,
        D_micro=(new_abs['coarse']/nd-1.03*new_abs['refined']/nd)*1.e6)


def electrical_replay(private):
    """KCL/system balances from saved powers; no feeder construction/AC call."""
    ac = private['AC']
    p, q = private['physical_P_pu'], private['physical_Q_pu']
    buses = list(ac['V_bus_mag'].columns)
    incoming_p = {int(b):np.zeros(len(p)) for b in buses}
    incoming_q = {int(b):np.zeros(len(p)) for b in buses}
    outgoing_p = {int(b):np.zeros(len(p)) for b in buses}
    outgoing_q = {int(b):np.zeros(len(p)) for b in buses}
    branch_loss_error = 0.
    for edge in ac['metadata']['branch_edges']:
        label, a, b = edge['label'], edge['from_bus'], edge['to_bus']
        fp, fq = array(ac['branch_p_from_pu'][label]), array(ac['branch_q_from_pu'][label])
        tp, tq = array(ac['branch_p_to_pu'][label]), array(ac['branch_q_to_pu'][label])
        outgoing_p[a] += fp; outgoing_q[a] += fq
        incoming_p[b] += tp; incoming_q[b] += tq
        branch_loss_error = max(branch_loss_error,
            float(np.max(np.abs(fp-tp-array(ac['branch_loss_p_pu'][label])))),
            float(np.max(np.abs(fq-tq-array(ac['branch_loss_q_pu'][label])))))
    root = int(ac['metadata']['root_bus'])
    nodal_p = nodal_q = 0.
    for bus in buses:
        if bus == root:
            continue
        lp = array(p[bus]) if bus in p else np.zeros(len(p))
        lq = array(q[bus]) if bus in q else np.zeros(len(q))
        nodal_p = max(nodal_p, float(np.max(np.abs(incoming_p[bus]-outgoing_p[bus]-lp))))
        nodal_q = max(nodal_q, float(np.max(np.abs(incoming_q[bus]-outgoing_q[bus]-lq))))
    bp = outgoing_p[root]-array(p).sum(1)-array(ac['branch_loss_p_pu']).sum(1)
    bq = outgoing_q[root]-array(q).sum(1)-array(ac['branch_loss_q_pu']).sum(1)
    return dict(nodal_p=nodal_p, nodal_q=nodal_q, branch_loss=branch_loss_error,
                system_p=float(np.max(np.abs(bp))), system_q=float(np.max(np.abs(bq))))


def run(root):
    required = ['protocol.json', 'price_freeze.json', 'problem_frozen.json', 'observations_frozen.json',
                'predictions_frozen.json', 'summary.json', 'scored_private.json', 'truth_private.json', 'ended.json']
    missing = [name for name in required if not (root/name).is_file()]
    if missing:
        return dict(status='not_ready', missing=missing, actual_audit_started=False)
    audit = Audit()
    protocol = read(root/'protocol.json'); freeze = read(root/'price_freeze.json')
    ledger = read(root/'observations_frozen.json'); predictions = read(root/'predictions_frozen.json')
    primary = read(root/'summary.json'); scored = read(root/'scored_private.json')['rows']
    audit.check('protocol_and_prices_frozen', protocol['status']==freeze['status']=='frozen')
    audit.check('predictions_precede_truth_use', predictions['status']=='frozen' and predictions['truth_read'] is False)
    checked(freeze['protocol']); checked(freeze['problem']); checked(ledger['price_freeze'])
    audit.check('price_problem_identity',freeze['problem']==read(root/'problem_frozen.json')['input'])
    audit.check('price_protocol_identity', freeze['protocol']['sha256']==sha(root/'protocol.json'))
    audit.check('observation_price_identity', ledger['price_freeze']['sha256']==sha(root/'price_freeze.json'))
    audit.check('prediction_before_truth_file', (root/'predictions_frozen.json').stat().st_mtime_ns <= (root/'truth_private.json').stat().st_mtime_ns)
    checked(read(root/'ended.json')['summary'])
    for path, digest in protocol['source_sha256'].items():
        audit.check('current_source_hash', sha(path)==digest, path)
    with zipfile.ZipFile(root/'source_snapshot.zip') as archive:
        for i,(path,digest) in enumerate(protocol['source_sha256'].items()):
            content = archive.read(f'{i:03d}_{Path(path).name}')
            audit.check('frozen_source_archive_hash', hashlib.sha256(content).hexdigest()==digest, path)
    for reference in protocol['inputs'].values():
        checked(reference)
    for key in ('parent_protocol','parent_prices'):
        if key in protocol:
            checked(protocol[key])
    checked(protocol['source_metadata']); checked(protocol['legacy_price'])
    problem = load(checked(read(root/'problem_frozen.json')['input']))
    pool = read(checked(protocol['inputs']['pool']))
    history = load(checked(protocol['inputs']['history']))['history']
    labels, active = pool['labels'], problem['active_positions']
    controls = [labels[i] for i in active]
    audit.check('frozen_resource_order', controls==problem['control_labels'])
    audit.check('historical_day_denominators', [sum(len(x['P_terminal']) for x in history[s]) for s in ('training','validation')]==[288,288])
    natural_p = np.concatenate([array(s['P_terminal']) for s in history['training']+history['validation']])*200
    natural_q = np.concatenate([array(s['Q_terminal']) for s in history['training']+history['validation']])*200
    caps = np.maximum(0., np.minimum.reduce([np.ones(len(labels)), natural_p.min(0), natural_q.min(0)/.35]))
    audit.close('historical_capacity_rule', problem['capacity_caps'], caps[active])
    audit.check('historical_zero_capacity_exclusion', active==np.flatnonzero(caps>0).tolist())
    models = {m['candidate_id']:m for m in pool['models']}
    packet_models = {m['candidate_id']:m for m in problem['pool']['models']}
    historical_abs = {}
    for identity, model in models.items():
        audit.close('frozen_predictor_R',packet_models[identity]['R'],model['R'],0.,identity)
        audit.close('frozen_predictor_X',packet_models[identity]['X'],model['X'],0.,identity)
        audit.check('frozen_predictor_point',packet_models[identity]['point']==model['point'],identity)
        historical_abs[identity] = sum(float(np.abs(array(s['drop_target'])-(array(s['P_terminal'])@array(model['R']).T+array(s['Q_terminal'])@array(model['X']).T)).sum()) for s in history['validation'])
        audit.close('historical_validation_replay', historical_abs[identity], model['historical_validation_abs_sum'], 1.e-12, identity)
        point = model['point']; calibration = pool['calibration']
        kept = [s for s,r,x in zip(point['supports'],point['r'],point['x'])
                if len(s)==1 or calibration['a']*r+calibration['b']*x+1.e-10>=.16*calibration['median_root_depth']]
        audit.check('original_frozen_contract', kept==model['frozen_output_supports'], identity)
    policies = freeze['policies']
    policy_stats = {}
    for name, policy in policies.items():
        u, a = array(policy['mean_u']), array(policy['capacity_kw'])
        audit.check('control_shape_finite', u.shape==(len(active),96) and a.shape==(len(active),) and np.isfinite(u).all() and np.isfinite(a).all(), name)
        audit.check('allocation_domain', np.all(a>=0) and np.all(a<=caps[active]+1.e-7) and a.sum()<=2+1.e-7, name)
        audit.check('mean_power_domain', np.all(np.abs(u)<=a[:,None]+1.e-7), name)
        audit.check('zero_capacity_zero_command', np.all(u[a==0]==0), name)
        quote = np.zeros_like(u)
        np.divide(2*np.abs(u),a[:,None],out=quote,where=np.broadcast_to(a[:,None]>0,u.shape))
        audit.check('maximum_quote_two', np.all(quote<=2+1.e-7), name)
        fee = .25*float(np.sum(quote*np.abs(u)))
        audit.close('perspective_fee', fee, policy['cost'], 1.e-9, name)
        audit.check('planning_target_label',bool(policy['planning_probability']>=.95-1.e-7)==bool(policy['planning_target_met']),name)
        policy_stats[name] = dict(capacity_total=float(a.sum()), active_resources=int(np.sum(a>0)),
            maximum_quote=float(quote.max(initial=0.)), planned_fee=fee,
            capacity_by_node_kw={str(node):float(value) for node,value in zip(controls,a)})
    for name in ('rotation_uniform','rotation_selected_capacity'):
        audit.close('matched_rotation_fee', policies[name]['cost'], policies['open_correlated']['cost'], 1.e-8, name)
    expected = {(d['day'],law,name) for d in protocol['days'] for law in protocol['observed_laws'] for name in policies}
    actual = {(r['day'],r['law'],r['policy']) for r in ledger['rows']}
    audit.check('144_physical_record_denominator', len(ledger['rows'])==len(actual)==len(expected)==144 and actual==expected)
    predicted = {r['record_id']:r for r in predictions['rows']}
    final_rows = {r['record_id']:r for r in scored}
    audit.check('prediction_failure_denominator', len(predicted)==len(final_rows)==144)
    # Opening the saved evaluator labels occurs only after frozen predictions.
    truth = {tuple(sorted(map(int,s))) for s in read(root/'truth_private.json')['missed_blocks']}
    model_errors = {}
    for name, model in models.items():
        raw = [[labels[i] for i in s] for s in model['point']['supports']]
        final = model['frozen_output_support_labels']
        model_errors[name] = dict(raw=errors(raw,truth), final=errors(final,truth))
        audit.check('model_error_replay', model_errors[name]==primary['models'][name], name)
    baselines = {}; zcache = {}; rows = []
    for item in ledger['rows']:
        key, identity = (item['day'],item['law'],item['policy']), item['record_id']
        row = dict(day=item['day'], law=item['law'], policy=item['policy'], record_id=identity,
                   physical_status=item['status'], fee=policies[item['policy']]['cost'], selected=None)
        if item['status']!='observed' or 'packet' not in item:
            audit.check('failed_observation_remains_unresolved', final_rows[identity]['status']=='unresolved', identity)
            rows.append(row); continue
        packet = load(checked(item['packet'])); private, physical = packet['private'], packet['summary']
        audit.check('prices_precede_record', (root/'price_freeze.json').stat().st_mtime_ns <= checked(item['packet']).stat().st_mtime_ns,identity)
        audit.check('physical_receipt_replay', physical==item['physical'], identity)
        day = next(d for d in protocol['days'] if d['day']==item['day'])
        a = array(policies[item['policy']]['capacity_kw']); u=array(policies[item['policy']]['mean_u'])
        law = protocol['response_main' if item['law']=='main' else 'response_misspec']
        zkey = (item['day'],item['law'])
        if zkey not in zcache:
            zcache[zkey] = independent_z(day['response_seed'],law,active)
        zz = zcache[zkey]
        audit.close('prelocked_shared_response_Z', private['Z'],zz,2.e-12,identity)
        delivered = u*(law['mean']+law['relative_sd']*zz)
        audit.close('signed_uncapped_delivery',private['delivered_kw'],delivered,2.e-12,identity)
        audit.close('frozen_capacity_used',private['capacity_kw'],a,1.e-12,identity)
        audit.close('frozen_command_used',private['mean_u'],u,1.e-12,identity)
        p0,q0,root0 = array(private['natural_P_pu']),array(private['natural_Q_pu']),array(private['root_true'])
        bg_hash = (fingerprint(p0),fingerprint(q0),fingerprint(root0))
        if item['day'] in baselines:
            audit.check('same_background_all_policies_laws',bg_hash==baselines[item['day']],identity)
        else:
            baselines[item['day']] = bg_hash
        dp=np.zeros_like(p0); dq=np.zeros_like(q0)
        for i,j in enumerate(active):
            dp[:,j] += delivered[i]; dq[:,j] += .35*delivered[i]
        pp,qq = array(private['physical_P_pu']),array(private['physical_Q_pu'])
        audit.close('increment_added_once_P',pp,p0+dp/200.,2.e-14,identity)
        audit.close('increment_added_once_Q',qq,q0+dq/200.,2.e-14,identity)
        component=a[:,None]+delivered
        bad=(component < -1.e-10)|(component>2*a[:,None]+1.e-10)
        audit.close('component_capacity_scaling',private['resource_component_kw'],component,2.e-12,identity)
        audit.check('component_violation_retained',int(bad.sum())==physical['resource_violation_cells'] and bool(bad.any())==physical['resource_violation'],identity)
        audit.check('no_clipping_or_redraw',physical['no_clipping'] and physical['no_redraw'] and physical['resource_limit_does_not_skip_AC'],identity)
        audit.close('physical_prepaid_fee',physical['planned_cost'],row['fee'],1.e-9,identity)
        shifted=[42,7,21][day['profile_index']]+100003*day['replicate']
        audit.check('original_seed_identity',physical['profile_seed']==shifted and physical['root_seed']==shifted+1000 and physical['meter_seed']==shifted+2000 and physical['root_meter_seed']==70000019+100003*day['replicate']+day['profile_index'],identity)
        row.update(resource_violation=bool(bad.any()),resource_violation_cells=int(bad.sum()),
            actual_abs_modulation_kwh=.25*float(np.abs(delivered).sum()),
            actual_reduction_kwh=.25*float(np.maximum(-delivered,0).sum()),
            actual_increase_kwh=.25*float(np.maximum(delivered,0).sum()),
            actual_net_modulation_kwh=.25*float(delivered.sum()),
            peak_sum_abs_modulation_kw=float(np.abs(delivered).sum(0).max()),
            peak_abs_net_modulation_kw=float(np.abs(delivered.sum(0)).max()),
            network_positive_baseline_kwh=.25*float(np.maximum(p0*200,0).sum()),
            network_absolute_net_baseline_kwh=.25*float(np.abs(p0*200).sum()),
            enrolled_node_positive_baseline_kwh=.25*float(np.maximum(p0[:,np.asarray(active)[a>0]]*200,0).sum()),
            ac_converged_samples=int(physical.get('ac_converged_samples',0)),
            ac_convergence_known='ac_converged_samples' in physical,
            ac_calls=int(physical.get('ac_calls',0)),
            voltage_violation=(not bool(physical['voltage_limits_pass'])) if 'voltage_limits_pass' in physical else None,
            voltage_violation_cells=int(physical.get('voltage_violation_cells',0)),
            terminal_negative_cells=int((pp<0).sum()))
        if 'AC' in private:
            ac = private['AC']
            audit.check('convergence_count',int(array(ac['converged']).sum())==row['ac_converged_samples'],identity)
            if np.isfinite(array(ac['V_bus_mag'])).all():
                residuals=electrical_replay(private)
                for kind,value in residuals.items():
                    audit.maxima['electrical_'+kind]=max(audit.maxima['electrical_'+kind],value)
                    audit.check('saved_electrical_balance_'+kind,value<=1.e-8,dict(record=identity,residual=value))
        scenes=packet['scenes']
        if scenes.get('both') is None:
            audit.check('unusable_physics_unresolved',final_rows[identity]['status']=='unresolved',identity)
            rows.append(row);continue
        vv=array(private['voltage_true'])
        rng=np.random.default_rng(shifted+2000)
        zp,zq,zv = (rng.standard_normal(pp.shape) for _ in range(3))
        zr=np.random.default_rng(70000019+100003*day['replicate']+day['profile_index']).standard_normal(96)
        measured_p=pp+.005*np.maximum(np.abs(pp),1.e-12)*zp
        measured_q=qq+.005*np.maximum(np.abs(qq),1.e-12)*zq
        measured_v=vv+.0005*np.maximum(np.abs(vv),1.e-12)*zv
        measured_root=root0+.0005*np.maximum(np.abs(root0),1.e-12)*zr
        for noise,scene in scenes.items():
            power_noise=noise in ('pq','both');voltage_noise=noise in ('v','both')
            audit.close('meter_P_replay',scene['P_terminal'],measured_p if power_noise else pp,2.e-14,(identity,noise))
            audit.close('meter_Q_replay',scene['Q_terminal'],measured_q if power_noise else qq,2.e-14,(identity,noise))
            audit.close('meter_V_replay',scene['V_terminal'],measured_v if voltage_noise else vv,2.e-14,(identity,noise))
            audit.close('meter_root_replay',scene['V_root_observed'],measured_root if voltage_noise else root0,2.e-14,(identity,noise))
            audit.close('observed_drop_replay',scene['drop_target'],array(scene['V_root_observed'])[:,None]**2-array(scene['V_terminal'])**2,2.e-14,(identity,noise))
            audit.check('complete_observation_day',array(scene['P_terminal']).shape==(96,15),(identity,noise))
        manual=selection_replay(pool,historical_abs,scenes['both'])
        saved=read(checked(predicted[identity]['score']))
        primary_rule=next(q for q in saved['rules'] if q['new_day_weight']==2 and q['new_day_fraction']==.03)
        audit.check('original_w2_common3_selection',manual['selected']==primary_rule['selected_candidate_id']==predicted[identity]['selected'],identity)
        audit.check('original_eligible_set',manual['eligible']==primary_rule['eligible_candidate_ids'],identity)
        audit.close('new_day_L1_replay',[manual['new_losses'][x] for x in pool['selection_order']],
                    [next(r for r in saved['loss_rows'] if r['candidate_id']==x)['new_validation_mae'] for x in pool['selection_order']],1.e-13,identity)
        audit.close('raw_decision_margin_micro',manual['margin_micro'],predicted[identity]['margin_micro'],1.e-7,identity)
        audit.close('raw_D_micro',manual['D_micro'],predicted[identity]['D_micro'],1.e-7,identity)
        selected=manual['selected']; outcome=model_errors[selected]
        audit.check('complete_error_counts',all(final_rows[identity][k]==v for k,v in outcome['final'].items()) and final_rows[identity]['raw_errors']==outcome['raw']['errors'],identity)
        row.update(selected=selected,errors=outcome['final']['errors'],raw_errors=outcome['raw']['errors'],D_micro=manual['D_micro'])
        rows.append(row)
    tables=[]
    for law in protocol['observed_laws']:
        for name in policies:
            group=[r for r in rows if r['law']==law and r['policy']==name]
            valid=[r for r in group if r['selected'] is not None]
            arrays_available=[r for r in group if 'actual_abs_modulation_kwh' in r]
            table=dict(law=law,policy=name,planned=12,complete=len(valid),unresolved=12-len(valid),
                total_errors=sum(r['errors'] for r in valid),raw_errors=sum(r['raw_errors'] for r in valid),
                recoveries=sum(r['errors']==0 for r in valid),array_metric_records=len(arrays_available),
                resource_violation_paths=sum(r['resource_violation'] for r in arrays_available),
                resource_violation_cells=sum(r['resource_violation_cells'] for r in arrays_available),
                ac_calls=sum(r['ac_calls'] for r in arrays_available),
                ac_converged_samples=sum(r['ac_converged_samples'] for r in arrays_available),
                records_with_known_ac_convergence=sum(r['ac_convergence_known'] for r in arrays_available),
                records_with_unknown_ac_convergence=12-sum(r['ac_convergence_known'] for r in arrays_available),
                planned_ac_samples=12*96,voltage_violation_paths=sum(r['voltage_violation'] is True for r in arrays_available),
                records_with_unknown_voltage_limits=12-sum(r['voltage_violation'] is not None for r in arrays_available),
                voltage_violation_cells=sum(r['voltage_violation_cells'] for r in arrays_available),
                negative_terminal_paths=sum(r['terminal_negative_cells']>0 for r in arrays_available),
                total_prepaid_fee=12*policies[name]['cost'],**policy_stats[name])
            for metric in ('actual_abs_modulation_kwh','actual_reduction_kwh','actual_increase_kwh','actual_net_modulation_kwh',
                           'network_positive_baseline_kwh','network_absolute_net_baseline_kwh','enrolled_node_positive_baseline_kwh'):
                table[metric+'_total']=sum(r[metric] for r in arrays_available)
                table[metric+'_per_observed_day']=table[metric+'_total']/len(arrays_available) if arrays_available else None
            dr=table['actual_abs_modulation_kwh_total']
            for denominator in ('network_positive_baseline_kwh','network_absolute_net_baseline_kwh','enrolled_node_positive_baseline_kwh'):
                value=table[denominator+'_total']
                table['abs_DR_percent_of_'+denominator]=100*dr/value if value>0 else None
            table['maximum_peak_sum_abs_DR_kw']=max((r['peak_sum_abs_modulation_kw'] for r in arrays_available),default=None)
            table['maximum_peak_abs_net_DR_kw']=max((r['peak_abs_net_modulation_kw'] for r in arrays_available),default=None)
            supplied=next(t for t in primary['tables'] if t['law']==law and t['policy']==name)
            for key in ('planned','complete','unresolved','total_errors','raw_errors','recoveries'):
                audit.check('summary_'+key,table[key]==supplied[key],(law,name))
            audit.close('summary_fee',policy_stats[name]['planned_fee'],supplied['fee_per_day'],1.e-9,(law,name))
            audit.check('failure_denominator_preserved',table['complete']+table['unresolved']==12,(law,name))
            tables.append(table)
    audit.check('independent_background_denominator',len(baselines)==12 and primary['independent_backgrounds']==12 and primary['paired_laws']==2)
    audit.check('complete_selection_total',sum(t['complete'] for t in tables)==primary['complete_selections'])
    report=dict(status='passed' if not audit.errors else 'failed',created_utc=datetime.now(timezone.utc).isoformat(),
        run_root=str(root.resolve()),
        auditor_source_sha256=sha(__file__),protocol_sha256=sha(root/'protocol.json'),price_freeze_sha256=sha(root/'price_freeze.json'),
        check_counts=dict(audit.counts),checks=sum(audit.counts.values()),failures=audit.errors,max_errors=dict(audit.maxima),
        physical_record_denominator=144,independent_backgrounds=12,paired_response_laws=2,policies=len(policies),
        source_scope='current source and frozen source ZIP both checked; audit source is post-run independent replay',
        new_AC_calls=0,new_fits=0,new_optimization_calls=0,truth_used_only_after_predictions_frozen=True)
    metrics=dict(status=report['status'],tables=tables,records=rows,
        denominators=dict(independent_backgrounds=12,paired_laws=2,policies=len(policies),planned_physical_records=144),
        definitions=dict(absolute_modulation='dt*sum_resource,slot abs(realized U)',
            network_positive_baseline='dt*sum_node,slot max(natural P_kw,0)',
            enrolled_node_positive_baseline='same positive load, only nodes with allocated a_i>0',
            prepaid_fee='planned perspective fee, no execution clawback; sums across counterfactual policy-law records are not one actual invoice'),
        no_arrays_or_raw_load_trajectories_in_this_summary=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    (OUTPUT/'audit_receipt.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    (OUTPUT/'policy_actual_metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return {k:report[k] for k in ('status','checks','physical_record_denominator','independent_backgrounds','new_AC_calls')}|dict(failures=len(audit.errors))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path)
    parser.add_argument('--run',help='Stage-relative run name, e.g. run_v2, or an absolute run path')
    args=parser.parse_args()
    if args.root is not None and args.run is not None:
        parser.error('use one of --root and --run')
    root=args.root if args.root is not None else (Path(args.run) if args.run and Path(args.run).is_absolute() else STAGE/(args.run or DEFAULT_ROOT.name))
    print(json.dumps(run(root),ensure_ascii=False))
