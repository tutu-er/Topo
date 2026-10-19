"""Independent arithmetic/data-boundary audit of saved response comparisons.

The auditor does not call a fitting or scoring function.  It rebuilds response
matrices, latent-input profiles, QP geometry and saved dual bounds directly.
AC sweeps are used only to check recorded physical explanations.  Case
regeneration verifies the existing observations; no truth enters a score.
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np

from .. import likelihood as lk
from ..local_rx_identifiability.cases import candidate_bank, make_case
from ..local_rx_identifiability.experiment import read, write
from .remainder import exact_remainder, remainder_bound


ROOT = Path(__file__).resolve().parents[3]
METHODS = ('AC', 'RESP', 'QP_N', 'QP_U')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _key(spec):
    return f"{spec['network']}_{spec['seed']}_N{spec['N']}_{spec['boundary']}"


def _close(actual, expected, label, *, atol=1e-8, rtol=1e-7):
    actual, expected = np.asarray(actual, float), np.asarray(expected, float)
    if actual.shape != expected.shape:
        raise AssertionError(label+': shape mismatch')
    if not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise AssertionError(label+': nonfinite values')
    if not np.allclose(actual, expected, atol=atol, rtol=rtol):
        error = float(np.max(np.abs(actual-expected)))
        raise AssertionError(label+f': maximum discrepancy {error:.6g}')


def _data(obs, sigma, v_ref=1.):
    p, q, v = (np.asarray(obs[key], float) for key in ('p', 'q', 'v'))
    if p.ndim != 2 or q.shape != p.shape or v.shape != p.shape:
        raise AssertionError('P/Q/V shapes do not agree')
    n = p.shape[1]
    d = np.zeros((n-1, n))
    for j in range(1, n):
        d[j-1, :j] = 1./np.sqrt(j*(j+1))
        d[j-1, j] = -j/np.sqrt(j*(j+1))
    sig = {key:np.broadcast_to(np.asarray(sigma.get(key, sigma.get('sigma_'+key)), float), p.shape)
           for key in ('p', 'q', 'v')}
    if any(np.any(a <= 0) or not np.isfinite(a).all() for a in sig.values()):
        raise AssertionError('invalid known meter scale')
    variance = 4*float(v_ref)**2*sig['v']**2+2*sig['v']**4
    cy = np.asarray([d @ np.diag(row) @ d.T for row in variance])
    return dict(p=p, q=q, v=v, sig=sig, d=d, x=np.c_[p, q],
                y=(v*v-sig['v']**2) @ d.T, cov_y=cy,
                sx=np.c_[sig['p'], sig['q']])


def _h(candidate, r, x, d):
    r, x = np.asarray(r, float), np.asarray(x, float)
    resistance = candidate.M @ np.diag(r) @ candidate.M.T
    reactance = candidate.M @ np.diag(x) @ candidate.M.T
    return np.c_[-2*d @ resistance, -2*d @ reactance]


def _profile(candidate, data, r, x):
    h = _h(candidate, r, x, data['d'])
    residual = data['y']-data['x'] @ h.T
    omega = np.asarray([cy+h @ np.diag(sx*sx) @ h.T
                        for cy, sx in zip(data['cov_y'], data['sx'])])
    w = np.linalg.solve(omega, residual[..., None])[..., 0]
    latent = data['x']+data['sx']**2*(w @ h)
    return dict(loss=float(np.sum(residual*w)), residual=residual,
                covariance=omega, H=h, latent=latent)


def _fixed_covariance(candidate, data, domain, weight):
    n, m, d = candidate.n, candidate.m, data['d']
    philo, phihi = np.arctan(domain['ratio_min']), np.arctan(domain['ratio_max'])
    if weight == 'nominal':
        mag = np.sqrt(domain['z_min']*domain['z_max'])
        phi = (philo+phihi)/2.
        h = _h(candidate, np.full(m, mag*np.cos(phi)), np.full(m, mag*np.sin(phi)), d)
        return np.asarray([cy+h @ np.diag(sx*sx) @ h.T
                           for cy, sx in zip(data['cov_y'], data['sx'])])
    if weight != 'upper':
        raise AssertionError('unknown fixed covariance weight')
    rmax, xmax = domain['z_max']*np.cos(philo), domain['z_max']*np.sin(phihi)
    # Entrywise upper bound on the topology response, then trace domination.
    absolute_r = 2*np.abs(d @ candidate.M) @ (candidate.M*rmax).T
    absolute_x = 2*np.abs(d @ candidate.M) @ (candidate.M*xmax).T
    absolute_h = np.c_[absolute_r, absolute_x]
    result = []
    for sv, sx in zip(data['sig']['v'], data['sx']):
        cy = d @ np.diag(4*domain['v_max']**2*sv*sv+2*sv**4) @ d.T
        result.append(cy+float(np.sum(absolute_h**2*sx[None, :]**2))*np.eye(n-1))
    return np.asarray(result)


def _geometry(candidate, domain, d):
    active = np.flatnonzero(np.linalg.norm(d @ candidate.M, axis=0)>1e-12)
    width = len(active)
    phi_lo, phi_hi = np.arctan(domain['ratio_min']), np.arctan(domain['ratio_max'])
    zlo, zhi = domain['z_min'], domain['z_max']
    lower = np.r_[np.full(width, zlo*np.cos(phi_hi)), np.full(width, zlo*np.sin(phi_lo))]
    upper = np.r_[np.full(width, zhi*np.cos(phi_lo)), np.full(width, zhi*np.sin(phi_hi))]
    rows, rhs = [], []
    for j in range(width):
        for rr, xx, target in [(domain['ratio_min'], -1., 0.),
                               (-domain['ratio_max'], 1., 0.),
                               (-np.cos((phi_lo+phi_hi)/2.), -np.sin((phi_lo+phi_hi)/2.),
                                -zlo*np.cos((phi_hi-phi_lo)/2.))]:
            row = np.zeros(2*width); row[j] = rr; row[width+j] = xx
            rows.append(row); rhs.append(target)
        for angle in np.linspace(phi_lo, phi_hi, 8):
            row = np.zeros(2*width); row[j] = np.cos(angle); row[width+j] = np.sin(angle)
            rows.append(row); rhs.append(zhi)
    return active, lower, upper, np.asarray(rows), np.asarray(rhs)


def audit_qp(candidate, obs, sigma, domain, fitted, *, v_ref=1.):
    """Reconstruct primal, original QP constraints, and its saved dual bound."""
    data = _data(obs, sigma, v_ref)
    active, lower, upper, a, rhs = _geometry(candidate, domain, data['d'])
    np.testing.assert_array_equal(fitted['active_edges'], active)
    r, x = np.asarray(fitted['r']), np.asarray(fitted['x'])
    point = np.r_[r[active], x[active]]
    covariance = _fixed_covariance(candidate, data, domain, fitted['weight'])
    _close(fitted['covariance'], covariance, 'QP fixed covariance', atol=1e-14)
    h = _h(candidate, r, x, data['d'])
    _close(fitted['H'], h, 'QP response matrix', atol=1e-13)
    residual = data['y']-data['x'] @ h.T
    weighted = np.linalg.solve(covariance, residual[..., None])[..., 0]
    primal = float(np.sum(residual*weighted))
    _close(fitted['loss'], primal, 'QP primal')
    _close(fitted['J'], .5*primal, 'QP J')
    design = []
    for edge in active:
        col = -2*(data['d'] @ candidate.M[:, edge])
        design.append((data['p'] @ candidate.M[:, edge])[:, None]*col[None, :])
    for edge in active:
        col = -2*(data['d'] @ candidate.M[:, edge])
        design.append((data['q'] @ candidate.M[:, edge])[:, None]*col[None, :])
    design = np.stack(design, axis=2)
    gradient = -2*np.einsum('tae,ta->e', design, weighted)
    _close(fitted['gradient_rx_active'], gradient, 'QP gradient')
    solved_design = np.linalg.solve(covariance, design)
    hessian = 2*np.einsum('tae,taf->ef', design, solved_design)
    linear = -2*np.einsum('tae,ta->e', design,
                        np.linalg.solve(covariance, data['y'][..., None])[..., 0])
    constant = float(np.sum(data['y']*np.linalg.solve(covariance, data['y'][..., None])[..., 0]))
    _close(fitted['quadratic_hessian'], hessian, 'QP Hessian', atol=1e-5)
    _close(fitted['quadratic_linear'], linear, 'QP linear term', atol=1e-5)
    _close(fitted['quadratic_constant'], constant, 'QP constant')
    violation = max(0., float(np.max(a @ point-rhs)), float(np.max(lower-point)), float(np.max(point-upper)))
    _close(fitted['constraint_violation'], violation, 'QP constraint violation', atol=1e-10)
    if fitted.get('qualified') and violation > 1e-9:
        raise AssertionError('qualified QP violates its original convex outer domain')
    cert = fitted['certificate']
    multiplier = np.asarray(cert['multiplier'], float)
    if multiplier.shape != rhs.shape or np.min(multiplier) < 0:
        raise AssertionError('QP dual multipliers must be nonnegative')
    adjusted = gradient+a.T @ multiplier
    corner = np.where(adjusted >= 0, lower, upper)
    dual_linear = float(adjusted @ corner-multiplier @ rhs)
    raw = primal-gradient @ point+dual_linear
    _close(cert['adjusted_gradient'], adjusted, 'QP dual adjusted gradient')
    _close(cert['minimizing_corner'], corner, 'QP dual corner')
    _close(cert['dual_linear'], dual_linear, 'QP dual linear value')
    _close(cert['raw_lower_bound'], raw, 'QP tangent lower value')
    guarded = max(0., raw-float(cert['numerical_guard']))
    _close(cert['lower_bound'], guarded, 'QP guarded bound')
    _close(fitted['lower_bound'], guarded, 'QP saved bound')
    _close(fitted['absolute_gap'], max(0., primal-guarded), 'QP gap')
    if guarded > primal+1e-6:
        raise AssertionError('QP lower bound exceeds its feasible objective')
    return dict(primal=primal, lower_bound=guarded, absolute_gap=max(0., primal-guarded),
                constraint_violation=violation)


def _audit_profile(candidate, obs, sigma, fitted, r, x, *, v_ref=1.):
    data = _data(obs, sigma, v_ref)
    value = _profile(candidate, data, r, x)
    _close(fitted['loss'], value['loss'], 'analytic P/Q profile loss')
    _close(fitted['J'], .5*value['loss'], 'analytic profile J')
    for key in ('H', 'residual', 'covariance'):
        if key in fitted:
            _close(fitted[key], value[key], 'analytic profile '+key,
                   atol=1e-13 if key == 'covariance' else 1e-8)
    if 'p' in fitted:
        _close(fitted['p'], value['latent'][:, :candidate.n], 'eliminated latent P', atol=1e-12)
        _close(fitted['q'], value['latent'][:, candidate.n:], 'eliminated latent Q', atol=1e-12)
    return value


def _line_legal(r, x, domain):
    r, x = np.asarray(r), np.asarray(x)
    mag, phi = np.hypot(r, x), np.arctan2(x, r)
    return bool(np.all(mag >= domain['z_min']-1e-10) and np.all(mag <= domain['z_max']+1e-10)
                and np.all(phi >= np.arctan(domain['ratio_min'])-1e-10)
                and np.all(phi <= np.arctan(domain['ratio_max'])+1e-10))


def _audit_ac(candidate, obs, sigma, domain, fitted, *, expected_lines=None):
    data = _data(obs, sigma)
    r, x, p, q, b = (np.asarray(fitted[k], float) for k in ('r', 'x', 'p', 'q', 'root_v'))
    if expected_lines is not None:
        _close(r, expected_lines[0], 'holdout fixed R', atol=0., rtol=0.)
        _close(x, expected_lines[1], 'holdout fixed X', atol=0., rtol=0.)
        if fitted.get('lines_updated') is not False:
            raise AssertionError('holdout AC score does not declare fixed training lines')
    physical = lk.ac_predict(candidate, r, x, p, q, root_v=b)
    residual = np.c_[(p-data['p'])/data['sig']['p'], (q-data['q'])/data['sig']['q'],
                     (physical['v']-data['v'])/data['sig']['v']]
    loss = float(np.sum(residual**2))
    _close(fitted['loss'], loss, 'full AC raw P/Q/V SSE', atol=1e-6)
    _close(fitted['J'], .5*loss, 'full AC J', atol=1e-6)
    _close(fitted['residual'], residual, 'full AC raw residual', atol=1e-7)
    _close(fitted['v'], physical['v'], 'full AC terminal voltage', atol=1e-10)
    voltage, current = physical['voltage'], physical['branch_current']
    ohm = []
    for e, parent in enumerate(candidate.parents):
        upstream = b if parent < 0 else voltage[:, parent]
        ohm.append(np.max(np.abs(upstream-voltage[:, e]-(r[e]+1j*x[e])*current[:, e])))
    max_ohm = float(max(ohm))
    if max_ohm > 1e-10 or physical['power_balance_residual'] > 1e-10:
        raise AssertionError('AC branch equations or complex-power balance failed')
    legal = bool(_line_legal(r,x,domain)
                 and np.all(np.abs(voltage)>=domain['v_min']-1e-10)
                 and np.all(np.abs(voltage)<=domain['v_max']+1e-10)
                 and np.all(b>=domain['root_min']-1e-10) and np.all(b<=domain['root_max']+1e-10)
                 and np.all(np.hypot(p,q)<=np.asarray(domain['power_caps'])+1e-10))
    if fitted.get('qualified') and not legal:
        raise AssertionError('qualified AC state violates the original physical domain')
    remainder = exact_remainder(candidate, r, x, p, q, physical=physical)
    actual = remainder['contrast_remainder']
    public = remainder_bound(candidate,domain)
    public_covered = bool(np.all(actual>=public['contrast_lower']) and np.all(actual<=public['contrast_upper']))
    if legal and not public_covered:
        raise AssertionError('public AC remainder envelope misses a legal recorded state')
    observed = remainder_bound(candidate,domain,obs,sigma,alpha_meter=.01)
    z = observed['terminal_envelope']['simultaneous_z']
    member = bool(legal and np.all(np.abs(p-data['p'])<=z*data['sig']['p'])
                  and np.all(np.abs(q-data['q'])<=z*data['sig']['q'])
                  and np.all(np.abs(physical['v']-data['v'])<=z*data['sig']['v']))
    observed_covered = bool(np.all(actual>=observed['contrast_lower']) and np.all(actual<=observed['contrast_upper']))
    if member and not observed_covered:
        raise AssertionError('meter-conditioned remainder box misses a member of its event')
    fixed = remainder_bound(candidate,domain,obs,sigma,alpha_meter=.01,fixed_r=r,fixed_x=x)
    fixed_covered = bool(np.all(actual>=fixed['contrast_lower']) and np.all(actual<=fixed['contrast_upper']))
    if member and not fixed_covered:
        raise AssertionError('fixed-line remainder box misses a member of its event')
    return dict(loss=loss, original_domain_legal=legal, max_ohm_residual=max_ohm,
                power_balance_residual=physical['power_balance_residual'],
                remainder_identity_error=remainder['identity_max_abs'],
                public_remainder_covered=public_covered,
                meter_rectangle_member=member,
                meter_remainder_coverage_claim_eligible=member,
                meter_remainder_covered=observed_covered,
                fixed_line_remainder_covered=fixed_covered)


def audit(directory):
    """Audit every saved condition; keep failures and unqualified fits counted."""
    start = perf_counter(); directory = Path(directory)
    config = read(directory/'configuration.json')
    errors, warnings, records = [], [], []
    source_mismatch = []
    for name, digest in config.get('source_manifest', {}).items():
        path = ROOT/name
        if not path.exists() or _sha(path) != digest:
            source_mismatch.append(name)
    if source_mismatch:
        errors.append(dict(phase='frozen_source',files=source_mismatch))
    bank = {c.name:c for c in candidate_bank()}
    baseline = Path(config['baseline'])
    for spec in config['tasks']:
        name = _key(spec); details = dict(key=name,methods={},errors=[])
        try:
            record_path=directory/'records'/f'{name}.json.gz'
            payload_path=directory/'inputs'/f'{name}.json.gz'
            saved, payload=read(record_path),read(payload_path)
            if saved.get('status') != 'completed':
                raise AssertionError('record did not complete; retained as failed condition')
            if saved.get('truth_used') is not False or payload.get('truth_used') is not False:
                raise AssertionError('observed-only data boundary is missing')
            if saved['input_sha256'] != _sha(payload_path):
                raise AssertionError('sealed input hash mismatch')
            if payload.get('train_indices') != [0,spec['N']] or payload.get('holdout_indices') != [672,768]:
                raise AssertionError('training or holdout interval changed')
            if spec['N'] > 672 or payload.get('disjoint_training_holdout') is not True:
                raise AssertionError('training and holdout are not disjoint')
            if payload.get('prediction_uses_observed_holdout_PQ') is not True:
                raise AssertionError('holdout P/Q conditioning rule is not declared')
            old_path=baseline/'inference'/f'{name}.json.gz';old=read(old_path)
            if payload['old_inference_sha256'] != _sha(old_path):
                raise AssertionError('old AC inference hash mismatch')
            generated=make_case(spec['network'],'ordinary',spec['seed'],total=768)
            for channel in ('p','q','v'):
                train=np.asarray(payload['train'][channel])
                hold=np.asarray(payload['holdout'][channel])
                np.testing.assert_array_equal(train,np.asarray(old['observed'][channel]))
                np.testing.assert_array_equal(train,np.asarray(generated['observed'][channel])[:spec['N']])
                np.testing.assert_array_equal(hold,np.asarray(generated['observed'][channel])[672:768])
            if old.get('root_known') is not None or old.get('root_meter') is not None:
                raise AssertionError('comparison includes extra boundary information')
            if set(saved['methods']) != set(METHODS):
                raise AssertionError('missing or unexpected score method')
            for method in METHODS:
                details['methods'][method]={}
                if set(saved['methods'][method]) != set(bank):
                    raise AssertionError('missing candidate retained in comparison')
                for candidate_name, row in saved['methods'][method].items():
                    candidate=bank[candidate_name];fit=row['training'];hold=row['holdout']
                    item=dict(training_qualified=bool(fit.get('qualified')))
                    details['methods'][method][candidate_name]=item
                    try:
                        if fit.get('truth_used') is True:
                            raise AssertionError('training declares truth use')
                        if 'r' not in fit:
                            warnings.append(dict(key=name,method=method,candidate=candidate_name,phase='training_failed'))
                            continue
                        r,x=np.asarray(fit['r']),np.asarray(fit['x'])
                        if method == 'AC':
                            _close(fit['r'],old['fits'][candidate_name]['r'],'reused AC R',atol=0.,rtol=0.)
                            _close(fit['x'],old['fits'][candidate_name]['x'],'reused AC X',atol=0.,rtol=0.)
                            item['training_ac']=_audit_ac(candidate,payload['train'],payload['sigma'],payload['domain'],fit)
                        elif method == 'RESP':
                            _audit_profile(candidate,payload['train'],payload['sigma'],fit,r,x,v_ref=config.get('v_ref',1.))
                            if not _line_legal(r,x,payload['domain']):
                                raise AssertionError('response fit left original line domain')
                        else:
                            item['training_qp']=audit_qp(candidate,payload['train'],payload['sigma'],payload['domain'],fit,v_ref=config.get('v_ref',1.))
                        profile=_audit_profile(candidate,payload['holdout'],payload['sigma'],hold['profile'],r,x,v_ref=config.get('v_ref',1.))
                        if hold['profile'].get('fixed_training_lines') is not True:
                            raise AssertionError('holdout profile fixed-line declaration missing')
                        legal_lines=_line_legal(r,x,payload['domain'])
                        if bool(hold['original_line_domain_feasible']) != legal_lines:
                            raise AssertionError('holdout line-domain membership changed')
                        if method.startswith('QP'):
                            data=_data(payload['holdout'],payload['sigma'],config.get('v_ref',1.))
                            covariance=_fixed_covariance(candidate,data,payload['domain'],'nominal' if method=='QP_N' else 'upper')
                            expected=float(np.sum(profile['residual']*np.linalg.solve(covariance,profile['residual'][...,None])[...,0]))
                            _close(hold['own']['loss'],expected,'holdout own fixed-weight loss')
                            _close(hold['own']['J'],.5*expected,'holdout own fixed-weight J')
                        elif method == 'RESP':
                            _close(hold['own']['loss'],profile['loss'],'holdout response own loss')
                        if legal_lines and 'p' in hold.get('ac',{}):
                            item['holdout_ac']=_audit_ac(candidate,payload['holdout'],payload['sigma'],payload['domain'],hold['ac'],expected_lines=(r,x))
                            if method == 'AC':
                                _close(hold['own']['loss'],hold['ac']['loss'],'holdout AC own loss')
                        elif legal_lines:
                            warnings.append(dict(key=name,method=method,candidate=candidate_name,phase='holdout_AC_failed'))
                        elif hold.get('ac',{}).get('status') != 'outer_parameter_not_in_original_domain':
                            raise AssertionError('nonphysical outer-QP coefficient backsub was not explicitly skipped')
                        if not fit.get('qualified'):
                            warnings.append(dict(key=name,method=method,candidate=candidate_name,phase='training_unqualified'))
                    except Exception as error:
                        details['errors'].append(dict(method=method,candidate=candidate_name,error=repr(error)))
        except Exception as error:
            details['errors'].append(dict(error=repr(error)))
        if details['errors']:
            errors.append(dict(key=name,errors=details['errors']))
        records.append(details)
    ac_states=[item[phase] for record in records for method in record['methods'].values()
               for item in method.values() for phase in ('training_ac','holdout_ac') if phase in item]
    return dict(audit_pass=not errors,errors=errors,warnings=warnings,records=records,
                expected_conditions=len(config['tasks']),audited_conditions=len(records),
                audited_ac_states=len(ac_states),
                meter_rectangle_member_count=sum(row['meter_rectangle_member'] for row in ac_states),
                meter_rectangle_nonmember_count=sum(not row['meter_rectangle_member'] for row in ac_states),
                max_ac_ohm_residual=max((row['max_ohm_residual'] for row in ac_states),default=None),
                max_remainder_identity_error=max((row['remainder_identity_error'] for row in ac_states),default=None),
                source_files_checked=len(config.get('source_manifest',{})),
                independent_math='response_profiles_QP_geometry_dual_raw_AC_residuals',
                holdout_data='regenerated_old_independent_noisy_observations_not_true_PQ_or_true_V',
                failed_and_unqualified_records_retained=True,seconds=perf_counter()-start)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--directory',required=True)
    args=parser.parse_args();directory=Path(args.directory)
    result=audit(directory);write(directory/'audit.json',result)
    print({key:result[key] for key in ('audit_pass','audited_conditions','audited_ac_states','seconds')},flush=True)
    if not result['audit_pass']:
        print(result['errors'],flush=True)
        raise SystemExit(1)


if __name__=='__main__':main()
