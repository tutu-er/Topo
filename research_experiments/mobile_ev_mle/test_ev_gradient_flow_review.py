"""Independent-review fixtures: no data generation, fitting or policy calls."""
from copy import deepcopy

import numpy as np
import pytest

from . import ev_gradient_flow_review as review
from .information_context import meter_to_record
from .likelihood import MeasurementModel, RadialCandidate
from .simulation import model


@pytest.fixture(scope='module')
def trial():
    bank = [RadialCandidate.from_supports(name, 3, supports) for name, supports in (
        ('star', ()), ('clade_01', ((0, 1),)), ('clade_02', ((0, 2),)), ('clade_12', ((1, 2),)))]
    T, n = 2, 3
    p = np.array([[.12, .13, .11], [.13, .12, .14]])
    q = p*.25
    source = np.tile([.08, .025], (T, 1))
    root = np.ones(T)
    u = np.zeros_like(p)
    u[:, 0] = 2.
    meter = model(3)
    sigma = np.column_stack((np.broadcast_to(meter.v_sigma, (T, n)),
        np.broadcast_to(meter.master_p_sigma, (T,)), np.broadcast_to(meter.master_q_sigma, (T,))))
    lines = {c.name:dict(r=np.full(c.m, .004), x=np.full(c.m, .0015)) for c in bank}

    def values(c, power):
        out = review.forward_physics(c, lines[c.name]['r'], lines[c.name]['x'],
            p+power/20., q, source[:, 0], source[:, 1], root)
        return np.column_stack((out['v'], out['master_p'], out['master_q']))

    rows = []
    for a in bank:
        for b in bank:
            if a.name == b.name:
                continue
            ref, comp = values(a, u), values(b, u)
            residual = (comp-ref)/sigma
            gradient = np.zeros_like(u)
            for t, j in ((0, 0), (T-1, n-1)):
                plus, minus = u.copy(), u.copy()
                plus[t, j] += 1e-3
                minus[t, j] -= 1e-3
                rp = (values(b, plus)-values(a, plus))/sigma
                rm = (values(b, minus)-values(a, minus))/sigma
                gradient[t, j] = np.sum((rp-rm)/.002*residual)
            rows.append(dict(a=a.name, b=b.name, name=a.name+'__to__'+b.name,
                success=True, source=source.copy(), half_SSE=float(np.sum(residual**2))/2,
                reference_values=ref, competitor_values=comp, residual=residual,
                gradient_per_kw=gradient, source_interface=1, line_parameters_fixed=True))
    profile = dict(method='ac', pairs=rows, power_kw=u, success=True,
        line_parameters_unchanged=True, expected_pair_count=12)
    selected_score = review._aggregate(profile, 'nearest')
    fits = [dict(name=c.name, candidate=c.name, loss=float(i+1), **lines[c.name]) for i,c in enumerate(bank)]
    shortlist = ['star', 'clade_01']
    return dict(seed=1, config=dict(source=dict(source_p_bounds=[-.5,.5], source_q_bounds=[-.5,.5])),
        forecast=dict(p=p,q=q,root=root,source_by_candidate={c.name:dict(
            source_p=source[:,0],source_q=source[:,1]) for c in bank}),
        training_result=dict(fits=fits), candidates=[dict(name=c.name,supports=c.supports) for c in bank],
        future_meter=meter_to_record(meter), true_name='star', training_top2=dict(names=shortlist,miss_true=False),
        selection=dict(training_top2_names=shortlist,configuration=dict(base_kw=20.),
            methods=dict(gradac_v2g_fixed=dict(selected_profile=profile,selected_score=selected_score,
                initial_candidates=[dict(profile=deepcopy(profile))],aggregation='nearest'))))


def test_saved_profiles_deduplicate_and_replay_independent_AC(trial):
    result = review.review_profile_trial(trial)
    assert result['success'], result['errors']
    assert result['counts']['unique_profiles'] == 1
    assert result['counts']['duplicate_profiles'] == 1
    assert result['counts']['AC_pairs_verified'] == 12
    assert result['counts']['price_coordinates_verified'] == 24
    assert result['no_new_fit'] and result['no_selector']


def test_duplicate_payload_cannot_hide_tampered_residual(trial):
    altered = deepcopy(trial)
    profile = altered['selection']['methods']['gradac_v2g_fixed']['initial_candidates'][0]['profile']
    profile['pairs'][0]['residual'][0, 0] += .1
    result = review.review_profile_trial(altered)
    assert not result['success']
    assert any('inconsistent payload' in row['message'] for row in result['errors'])


@pytest.mark.parametrize('kind', ['source', 'pair', 'shortlist', 'gradient'])
def test_corrupt_physics_or_information_payload_is_detected(trial, kind):
    altered = deepcopy(trial)
    record = altered['selection']['methods']['gradac_v2g_fixed']
    record.pop('initial_candidates')
    if kind == 'source':
        record['selected_profile']['pairs'][0]['source'][0,0] = .7
    elif kind == 'pair':
        record['selected_profile']['pairs'].pop()
    elif kind == 'shortlist':
        altered['selection']['training_top2_names'] = ['clade_12', 'clade_02']
    else:
        record['selected_profile']['pairs'][0]['gradient_per_kw'][0,0] += 1.
    result = review.review_profile_trial(altered)
    assert not result['success']


def test_I2_payload_check_does_not_claim_AC_agreement(trial):
    altered = deepcopy(trial)
    row = altered['selection']['methods']['gradac_v2g_fixed']
    row.pop('initial_candidates')
    row['selected_profile']['method'] = 'i2'
    result = review.review_profile_trial(altered)
    assert result['success'], result['errors']
    assert result['counts']['I2_payload_pairs_verified'] == 12
    assert result['counts']['AC_pairs_verified'] == 0
    assert result['counts']['independent_forward_batches'] == 0
    assert not result['I2_AC_agreement_claimed']


def test_accepted_step_must_increase_complete_AC_value(trial):
    altered = deepcopy(trial)
    row = altered['selection']['methods']['gradac_v2g_fixed']
    score = row['selected_score']
    row['iterations'] = [dict(start_index=0,before_score=score+1,after_score=score,
        accepted=True,candidate_profile=deepcopy(row['selected_profile']),candidate_score=score)]
    result = review.review_profile_trial(altered)
    assert not result['success']
    assert any('strictly improve' in e['message'] for e in result['errors'])
