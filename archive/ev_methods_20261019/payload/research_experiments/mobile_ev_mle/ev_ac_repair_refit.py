"""Fixed-plan conditional full-AC fits; no truth or action selection enters.

Measured terminal P/Q are bias-corrected, then held fixed.  This is a distinct
conditional-output comparison, not the latent-P/Q Bayesian likelihood.  All
retained SSEs are attained local values, never global profile certificates.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from time import perf_counter
from collections.abc import Mapping

import numpy as np

from . import likelihood as lk
from . import ev_systematic_refit as legacy
from .ev_rx_fit import fit_rx_inputs


def _options(config):
    values = {} if config is None else dict(config)
    if 'fit_options' in values:
        if set(values) != {'fit_options'}:
            raise ValueError('refit config accepts only fit_options or the direct fit options')
        values = dict(values['fit_options'])
    return legacy._options(dict({'multistart': 2, 'max_nfev': 400}, **values))


def _serial(value):
    if isinstance(value, np.ndarray):
        return _serial(value.tolist())
    if isinstance(value, np.generic):
        return _serial(value.item())
    if isinstance(value, Mapping):
        return {str(k): _serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serial(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _fit_record(fit):
    result = legacy._summary(fit)
    if fit is not None:
        diagnostics = getattr(fit, 'diagnostics', {})
        for key in ('min_voltage', 'max_voltage', 'physical_voltage_feasible',
                    'ratio_constraints', 'restart_loss_spread', 'P_Q_likelihood_included',
                    'P_Q_measurement_error_propagated', 'parameterization'):
            if key in diagnostics:
                result['diagnostics'][key] = diagnostics[key]
    return _serial(result)


def _state(item):
    return legacy._state(item)


def _score(candidate, observations, meter, state, options, lengths, names, counter):
    return legacy._score(candidate, observations, meter, state, options,
                         lengths, names, counter)


def _history_fingerprint(bank, D, dm, options):
    digests = [legacy._fingerprint(c, (D,), (dm,), options) for c in bank]
    return sha256(json.dumps(digests).encode()).hexdigest()


@dataclass
class HistoryFits:
    """Action-independent cache; use record() for durable JSON evidence."""
    candidate_names: tuple
    fingerprint: str
    options: dict
    fits: dict
    candidates: dict
    failures: dict
    counters: dict
    seconds: float

    @property
    def complete(self):
        return (not self.failures and len(self.candidates) == len(self.candidate_names)
                and all(r['success'] for r in self.candidates.values()))

    def record(self):
        return _serial(dict(complete=self.complete, candidate_names=self.candidate_names,
            fingerprint=self.fingerprint, fit_options=self.options,
            candidates=self.candidates, failures=self.failures,
            counters=self.counters, seconds=self.seconds,
            scope='conditional output given measured P/Q; P/Q input noise ignored',
            observed_history_only=True, source_interface=1,
            source_components=['P', 'Q'], global_optimality_certified=False))


def fit_history(bank, D, dm, config=None, seed=0):
    """Fit every topology to history only, retaining all local failures."""
    began = perf_counter(); bank = tuple(bank); options = _options(config)
    names = tuple(c.name for c in bank)
    if not names or len(set(names)) != len(names):
        raise ValueError('nonempty bank with unique candidate names required')
    fits = {}; rows = {}; failures = {}; counter = {}
    fingerprint = _history_fingerprint(bank, D, dm, options)
    for j, candidate in enumerate(bank):
        start = perf_counter(); local_counter = {}
        try:
            fitted = fit_rx_inputs(candidate, D, dm, seed=int(seed)+j,
                call_counter=local_counter, **options)
            fits[candidate.name] = fitted
            score = _score(candidate, D, dm, _state(fitted), options,
                           (D.t,), ('history',), local_counter)
            success = bool(fitted.success and score['usable'])
            rows[candidate.name] = _serial(dict(**score, success=success,
                optimizer=_fit_record(fitted), state=_state(fitted),
                counters=local_counter, seconds=perf_counter()-start,
                global_optimality_certified=False))
            if not success:
                failures[candidate.name] = 'history_fit_not_converged_or_not_physical'
        except Exception as exc:
            failures[candidate.name] = type(exc).__name__+': '+str(exc)
            rows[candidate.name] = dict(success=False, error=failures[candidate.name],
                counters=dict(local_counter), seconds=perf_counter()-start)
        for key, value in local_counter.items():
            counter[key] = counter.get(key, 0)+value
    return HistoryFits(names, fingerprint, options, fits, rows, failures,
                       counter, perf_counter()-began)


def _source_witness(candidate, D, dm, E, em, r, x, options, seed, counter):
    """Both blocks are source-profiled at exactly the supplied line vector."""
    fixed = dict(options, multistart=1)
    dh = fit_rx_inputs(candidate, D, dm, fixed_r=r, fixed_x=x,
        seed=seed, call_counter=counter, **fixed)
    ef = fit_rx_inputs(candidate, E, em, fixed_r=r, fixed_x=x,
        seed=seed+1, call_counter=counter, **fixed)
    state = legacy._join_state(r, x, (dh, ef))
    return state, dict(history=_fit_record(dh), future=_fit_record(ef),
                      success=bool(dh.success and ef.success))


def _center(item):
    get = item.get if isinstance(item, Mapping) else lambda k: getattr(item, k)
    r = np.asarray(get('r'), float); x = np.asarray(get('x'), float)
    if r.ndim != 1 or x.shape != r.shape or not np.isfinite(r).all() or not np.isfinite(x).all():
        raise ValueError('Bayesian center requires finite one-dimensional r/x')
    return r.copy(), x.copy()


def evaluate_refits(bank, D, dm, E, em, historyfits, config=None, seed=0,
                    bayes_centers=None):
    """Compare fixed plans/data; on any necessary failure retain all candidates.

    Named starts are explicitly optimized separately.  A legal fixed-line
    witness always remains in the joint attained-value pool, including when
    every optimizer fails.  Such retention does not turn failure into success.
    """
    began = perf_counter(); bank = tuple(bank); options = _options(config)
    names = tuple(c.name for c in bank)
    if not isinstance(historyfits, HistoryFits):
        raise TypeError('fit_history HistoryFits cache required')
    if (names != historyfits.candidate_names
            or _history_fingerprint(bank, D, dm, options) != historyfits.fingerprint):
        raise ValueError('history cache differs in candidate/data/meter/options')
    centers = {} if bayes_centers is None else dict(bayes_centers)
    if set(centers)-set(names):
        raise ValueError('Bayesian center names must belong to the candidate bank')
    observations = lk.concat_observations(D, E)
    meter = legacy.concat_models((D, E), (dm, em))
    rows = {}; failures = {}; counter = {}
    for j, candidate in enumerate(bank):
        name = candidate.name; cb = perf_counter(); cc = {}; starts = []
        try:
            if name not in historyfits.fits:
                raise RuntimeError('history fit unavailable')
            history_fit = historyfits.fits[name]
            line_centers = [('history', (np.asarray(history_fit.r), np.asarray(history_fit.x)))]
            if name in centers:
                line_centers.append(('bayes', _center(centers[name])))
            attained = []; fixed_score = None; converged = []
            for i, (label, (r, x)) in enumerate(line_centers):
                sr = dict(label=label, success=False); sb = perf_counter()
                try:
                    warm, sources = _source_witness(candidate, D, dm, E, em,
                        r, x, options, int(seed)+1000*j+20*i, cc)
                    witness = _score(candidate, observations, meter, warm, options,
                                     (D.t, E.t), ('history', 'future'), cc)
                    sr.update(source_fits=sources, witness=witness,
                              witness_state=warm)
                    if not witness['usable']:
                        raise RuntimeError('source-profiled start is outside the physical domain')
                    attained.append((witness['loss'], 'witness:'+label, warm, witness))
                    if label == 'history':
                        fixed_score = dict(**witness, state=warm,
                            success=bool(historyfits.candidates[name]['success'] and sources['success']),
                            selected_origin='fixed_history_lines')
                    fitted = fit_rx_inputs(candidate, observations, meter,
                        initial=warm, seed=int(seed)+1000*j+20*i+2,
                        call_counter=cc, **dict(options, multistart=1))
                    score = _score(candidate, observations, meter, _state(fitted), options,
                                   (D.t, E.t), ('history', 'future'), cc)
                    sr.update(optimizer=_fit_record(fitted), score=score,
                              attained_state=_state(fitted))
                    if score['usable']:
                        attained.append((score['loss'], 'optimizer:'+label,
                                         _state(fitted), score))
                    tolerance = 1e-8*(1+abs(witness['loss']))
                    sr['success'] = bool(sources['success'] and fitted.success
                        and score['usable'] and score['loss'] <= witness['loss']+tolerance)
                    if sr['success']:
                        converged.append(label)
                except Exception as exc:
                    sr['error'] = type(exc).__name__+': '+str(exc)
                sr['seconds'] = perf_counter()-sb
                starts.append(_serial(sr))
            if fixed_score is None or not attained:
                raise RuntimeError('no usable fixed-history-line witness')
            _, origin, retained_state, retained_score = min(attained, key=lambda row: row[0])
            # A worse converged restart cannot certify a better failed restart.
            best_converged_loss = min((s['score']['loss'] for s in starts
                if s.get('success')), default=np.inf)
            retained_certified = bool(best_converged_loss <= retained_score['loss']
                +1e-8*(1+abs(retained_score['loss'])))
            success = bool(fixed_score['success'] and retained_certified
                and historyfits.candidates[name]['success'])
            joint_score = dict(**retained_score, state=retained_state,
                success=success, selected_origin=origin,
                converged_named_starts=converged,
                optimization_failure=not retained_certified,
                global_optimality_certified=False, global_loss_lower_bound=None,
                loss_scope='retained feasible upper bound on profile minimum')
            rows[name] = _serial(dict(fixed=fixed_score, joint=joint_score,
                starts=starts, success=success,
                joint_no_worse_than_fixed=bool(joint_score['loss'] <= fixed_score['loss']
                    +1e-8*(1+abs(fixed_score['loss']))),
                same_line_vector_all_times=True, source_interface=1,
                source_components=['P', 'Q'], counters=cc,
                seconds=perf_counter()-cb))
            if not success:
                failures[name] = 'required_history_source_or_joint_fit_not_converged'
        except Exception as exc:
            failures[name] = type(exc).__name__+': '+str(exc)
            rows[name] = _serial(dict(success=False, error=failures[name], starts=starts,
                                      counters=cc, seconds=perf_counter()-cb))
        for key, value in cc.items():
            counter[key] = counter.get(key, 0)+value
    complete = len(rows) == len(names) and not failures and historyfits.complete
    losses = {route: {name: row[route]['loss'] for name, row in rows.items()
               if route in row and row[route].get('usable')}
              for route in ('fixed', 'joint')}
    return _serial(dict(complete=complete, comparisons=rows, losses=losses,
        failures=failures, counters=counter, seconds=perf_counter()-began,
        candidate_names=names, keep_all_on_failure=not complete,
        retained_on_failure=list(names) if not complete else None,
        scope='conditional output given measured P/Q; P/Q input noise ignored',
        history_fingerprint=historyfits.fingerprint,
        history_cost_included=False, history_preparation_seconds=historyfits.seconds,
        truth_used=False, future_used_only_after_plan_fixed=True,
        shared_lines_history_and_future=True, source_interface=1,
        global_optimality_certified=False))


__all__ = ['HistoryFits', 'fit_history', 'evaluate_refits']
