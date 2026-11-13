"""Independent support-level scores, physical AC, fee and source checks."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

from ... import io
from ...observations import load
from ..customer_flexibility_20261113.model import CustomerCase
from ..customer_flexibility_20261113.placement import CandidateEvidence
from .config import ROOT
from .fitting import calibrated_means
from .simulation import runtime, forecast, response
from .study import verify_sources


def require(value, message):
    if not value:
        raise AssertionError(message)


def support_prediction(point, observation):
    answer = np.zeros_like(observation.y, dtype=np.longdouble)
    for C, r, x in zip(point['supports'], point['r'], point['x'], strict=True):
        answer[:, C] += (np.longdouble(r)*observation.p[:, C].sum(axis=1)+
                         np.longdouble(x)*observation.q[:, C].sum(axis=1))[:, None]
    return answer


def run(out):
    protocol = verify_sources(out)
    module = runtime(out)
    counts = dict(history_scores=0, policies_sealed=0, device_actions=0, AC_days=0, posterior_vectors=0)
    errors = dict(maximum_probability_error=0., maximum_ac_voltage_error=0., maximum_power_error=0., maximum_evidence_error=0.)
    for sealfile in sorted(out.glob('histories/*/fit_complete.json')):
        receipt = io.read(sealfile)
        if receipt['status'] != 'fitted':
            continue
        s = io.read(receipt['score']['path'])
        require(io.ref(receipt['score']['path'])['sha256'] == receipt['score']['sha256'], 'History score changed')
        job = next(j for f in out.glob('history_jobs_s*.json') for j in io.read(f) if j['id'] == receipt['id'])
        obs = load(job['observed_refs']['validation']['path'])
        losses = np.array([np.square(np.asarray(obs.y, np.longdouble)-support_prediction(p['point'], obs)).sum() for p in s['parameters']], dtype=np.longdouble)
        likelihood = -(losses-losses.min())/(2*np.longdouble(s['sigma2']))
        likelihood -= logsumexp(likelihood)
        discrepancy = float(np.max(np.abs(np.exp(likelihood)-np.array([m['probability'] for m in s['models']]))))
        require(discrepancy < 1e-8, 'Independent historical support scores differ')
        counts['history_scores'] += 1
    for phase in ('development', 'confirmation'):
        selectionfile = out/('development_selection.json' if phase == 'development' else 'confirmation_selection.json')
        if not selectionfile.exists():
            continue
        selection = io.read(selectionfile)
        if isinstance(selection, dict):
            selection = selection['entries']
        for entry in selection:
            s = io.read(entry['score']['path'])
            require(io.ref(entry['score']['path'])['sha256'] == entry['score']['sha256'], 'Execution used changed candidates')
            case = CustomerCase(io.read(out/'resource_configs'/f"{entry['adoption']}.json"))
            H = np.asarray([p['H'] for p in s['parameters']])
            target = s['map_index']
            competitors = [j for j in range(len(H)) if j != target]
            means = calibrated_means(s, load(entry['observed_refs']['validation']['path']))
            for daydir in sorted((out/phase/entry['id']).glob('day*')):
                if not (daydir/'policy_seal.json').exists():
                    continue
                seal, plans = io.read(daydir/'policy_seal.json'), io.read(daydir/'plans.json')
                for key in ('planner', 'plans'):
                    require(io.ref(seal[key]['path'])['sha256'] == seal[key]['sha256'], 'Sealed policy changed')
                require(not seal['planner_receives_truth'] and not seal['planner_receives_true_lines_or_voltage'], 'Planner input rule violated')
                counts['policies_sealed'] += 1
                day = int(daydir.name[3:])
                replicate = (100000 if phase == 'development' else 200000)+entry['index']*100+day
                with np.load(daydir/'physical_background.npz') as z:
                    b = {k: np.array(z[k]) for k in z.files}
                eta = response(case.R, replicate)
                require(np.array_equal(eta, b['response_multiplier']), 'Response stream differs')
                with np.load(daydir/'planner.npz') as z:
                    policies = {k: np.array(z[k]) for k in z.files}
                prediction = forecast(dict(p=b['P'], q=b['Q']), replicate)
                require(np.array_equal(prediction, policies.pop('forecast')), 'Forecast stream differs')
                # Reload the frozen physical network only in this evaluator.
                from .simulation import background
                actual = background(module, case, entry, replicate)
                require(np.array_equal(actual['p'], b['P']) and np.array_equal(actual['q'], b['Q']), 'Original physical generator differs')
                evidence = CandidateEvidence(case, H, target, competitors, prediction, s['sigma2'], actual['network']['base_kw'], means) if competitors else None
                for name, u in policies.items():
                    require(case.audit(u)['passed'] and case.audit(u*eta[None])['passed'], 'Device action is infeasible')
                    counts['device_actions'] += 1
                    if name.startswith('optimized'):
                        quote = plans[name.split('_')[1]]
                        if quote['feasible']:
                            require(abs(case.fee(u)-quote['cost_yuan']) < 1e-7, 'Quote and actual fee formula differ')
                            lower = quote.get('lower_bound_yuan')
                            require(lower is not None and 0 <= lower <= case.fee(u)+1e-7, 'Original fee bound contradicts a feasible point')
                        else:
                            require(np.all(u == 0), 'Unqualified quote must use the explicit zero fallback')
                    if name.startswith('optimized') and plans[name.split('_')[1]]['feasible'] and competitors:
                        gamma = plans[name.split('_')[1]]['gamma']
                        direct, quadratic = evidence.direct(u), evidence.values(u)
                        error = float(np.max(abs(direct-quadratic)))
                        errors['maximum_evidence_error'] = max(errors['maximum_evidence_error'], error)
                        require(np.allclose(direct, quadratic, rtol=1e-7, atol=1e-8), 'QCQP evidence differs from original formula')
                        require(np.min(direct-evidence.required(gamma)) >= -1e-7, 'A qualified quote misses the original evidence requirement')
                    path = daydir/(name+'.npz')
                    if not path.exists():
                        continue  # The failed row stays in the aggregate report.
                    obs = load(path)
                    with np.load(path) as z:
                        truep, trueq, truev, saved = [np.array(z[k]) for k in ('true_P', 'true_Q', 'true_V', 'log_probability')]
                    dp, dq = case.aggregate(case.delivery(u*eta[None]))
                    discrepancy = max(np.max(abs(truep-(b['P']+dp/actual['network']['base_kw']))),
                                      np.max(abs(trueq-(b['Q']+dq/actual['network']['base_kw']))))
                    errors['maximum_power_error'] = max(errors['maximum_power_error'], float(discrepancy))
                    require(discrepancy < 1e-13, 'Actual delivered P/Q differs')
                    v, ac = module._ac_voltages(actual['network'], actual['labels'], truep, trueq, b['root'])
                    discrepancy = float(np.max(abs(v-truev)))
                    errors['maximum_ac_voltage_error'] = max(errors['maximum_ac_voltage_error'], discrepancy)
                    require(discrepancy < 1e-12, 'Independent original AC differs')
                    require(np.allclose(obs.p, truep*(1+.005*b['meter_p']), atol=1e-14), 'P meter stream differs')
                    require(np.allclose(obs.q, trueq*(1+.005*b['meter_q']), atol=1e-14), 'Q meter stream differs')
                    require(np.allclose(obs.voltage, v*(1+entry['voltage_sigma']*b['meter_v']), atol=1e-14), 'V meter stream differs')
                    counts['AC_days'] += 1
                    loss = np.array([np.longdouble(p['history_sse'])+
                        np.square(np.asarray(obs.y, np.longdouble)-support_prediction(p['point'], obs)).sum() for p in s['parameters']], dtype=np.longdouble)
                    ll = -(loss-loss.min())/(2*np.longdouble(s['sigma2']))
                    ll -= logsumexp(ll)
                    discrepancy = float(np.max(abs(np.exp(ll)-np.exp(saved))))
                    errors['maximum_probability_error'] = max(errors['maximum_probability_error'], discrepancy)
                    require(discrepancy < 1e-8, 'Independent support-level posterior differs')
                    counts['posterior_vectors'] += 1
    result = dict(status='passed', counts=counts, errors=errors, source_protocol=io.ref(out/'protocol.json'),
        audit_source=io.ref(__file__), failed_jobs_preserved=True)
    io.write(out/'audit.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(run(args.out.resolve()))
