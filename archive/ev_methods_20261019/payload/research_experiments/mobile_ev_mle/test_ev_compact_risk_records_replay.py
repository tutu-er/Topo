from copy import deepcopy
from .ev_compact_risk_records_replay import replay_reports
from .ev_compact_risk_independent_audit import (
    PRIMARY, canonical_hash, reconstruct_calibration, reconstruct_evaluation,
    projected_records,
)


def reports():
    methods = ['none', *PRIMARY]
    spec = dict(policy_id='p', config_sha256='c', source_manifest_sha256='s',
        reference_id='r', candidates=['a', 'b'], methods=methods, strata=['s'],
        alpha=.1, familywise=False, stratum_mode='worst', stratum_is_public=False)
    def row(i, true):
        return dict(history_id=str(i), independent_unit_id=str(i), true_name=true,
            stratum='s', candidate_names=['a', 'b'], candidate_miss=False,
            base_candidate_miss=False, method_candidate_miss={m: False for m in methods},
            methods={m: dict(status='complete', losses={'a': 0., 'b': 1.}) for m in methods})
    def make(er):
        bundle=dict(spec=spec, calibration=[row(1, 'a'), row(2, 'b')], evaluation=er,
            planned_calibration_ids=['1', '2'], planned_evaluation_ids=['3'],
            raw_directories={'calibration': '/not-present/raw', 'evaluation': '/not-present/raw'},
            raw_manifest={'calibration': [], 'evaluation': []})
        bundle['bundle_id'] = canonical_hash(bundle)
        out = dict(standard_records=bundle)
        for label, mm, family in (('individual_policies', methods, False), ('primary_family', list(PRIMARY), True)):
            local = dict(spec, methods=mm, familywise=family)
            cr = projected_records(bundle['calibration'], mm) if family else bundle['calibration']
            ee = projected_records(er, mm) if family else er
            cal = reconstruct_calibration(cr, local, ['1', '2'])
            out[label] = dict(calibration=cal, evaluation=reconstruct_evaluation(ee, cal))
        return out
    return make([]), make([row(3, 'a')])


def test_portable_replay_and_small_record_tamper():
    before, final = reports()
    assert replay_reports(before, final)['success']  # Deliberately absent raw paths.
    broken = deepcopy(final)
    broken['standard_records']['evaluation'][0]['methods']['none']['losses']['b'] = 2.
    result = replay_reports(before, broken)
    assert not result['success'] and 'final_bundle_hash' in result['errors']


def test_threshold_tamper_is_rejected():
    before, final = reports()
    final['individual_policies']['calibration']['thresholds']['s']['a']['none']['threshold'] = 42.
    result = replay_reports(before, final)
    assert not result['success']
    assert 'individual_policies_calibration_identity' in result['errors']
    assert 'individual_policies_whole_calibration_replay' in result['errors']
