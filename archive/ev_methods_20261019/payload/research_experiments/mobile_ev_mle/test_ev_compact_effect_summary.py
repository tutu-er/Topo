"""Pure saved-table checks: independence, failures, provenance and decisions."""
from copy import deepcopy
from dataclasses import replace
import csv
import json
import pytest
from . import ev_compact_effect_summary as e
from . import ev_policy_calibration as c
from . import ev_compact_analysis as a


def rows(seeds=(11, 12), phase='development'):
    answer = []
    for seed in seeds:
        for method in (*e.METHODS, 'v2g_fixed', 'charge_fixed'):
            primary = method in e.PRIMARY
            answer.append(dict(seed=seed, method=method, phase=phase, policy_id='policy',
                family='star' if seed == seeds[0] else 'cherry', domain='asset', prediction='ensemble',
                history_sha256='H'+str(seed), truth_sha256='T'+str(seed), noise_sha256='N'+str(seed),
                measurement_sha256='meter', device_sha256='device', data_sha256='D'+str(seed),
                status='complete', actual_M=(15. if seed == seeds[0] else 1.) if primary else 5.,
                noiseless_M=10. if primary else 4., noiseless_complete=True,
                R_relative=.1 if primary else .3, X_relative=.6 if primary else .7,
                candidate_miss=False, operation_feasible=True, selection_failures=0,
                selected_origin='MILP' if primary else '', selected_is_new_MILP_proposal=primary,
                proxy_completed=primary))
    return answer


def get(rows, method, seed=11):
    return next(r for r in rows if str(r['seed']) == str(seed) and r['method'] == method)


def test_history_is_unit_and_negative_paired_effect_retained():
    result = e.summarize_rows(rows())
    assert result['independent_histories'] == 2
    assert result['method_measurements'] == result['paired_measurements'] == 8
    contrast = result['contrasts']['refined_v2g_mobile_vs_constant']
    assert contrast['history_denominator'] == 2
    metric = contrast['metrics']['actual_M']
    assert metric['n'] == 2 and metric['mean'] == 3. and metric['median'] == 3.
    assert metric['positive'] == metric['negative'] == 1
    assert contrast['metrics']['R_relative']['mean'] == pytest.approx(-.2)
    assert len(result['family_contrasts']['star']) == 4


def test_failed_fit_and_unexecuted_history_have_different_denominators():
    data = rows()
    failed = get(data, e.PRIMARY[0], 12)
    failed.update(status='required_joint_fit_failure', actual_M=900., R_relative=.2,
                  noiseless_complete=False, noiseless_M=900.)
    result = e.summarize_rows(data, metadata=dict(planned_histories=3))
    rec = result['methods'][e.PRIMARY[0]]
    assert result['pending_histories'] == 1
    assert rec['history_denominator'] == 2 and rec['failed_or_missing'] == 1
    assert rec['metrics']['actual_M']['n'] == rec['metrics']['noiseless_M']['n'] == 1
    assert rec['metrics']['R_relative']['n'] == 2  # saved true-candidate estimate still available
    paired = result['contrasts']['refined_v2g_mobile_vs_constant']
    assert paired['complete_actual_pairs'] == 1 and paired['history_denominator'] == 2
    assert next(r for r in result['paired_rows'] if r['seed'] == '12')['delta_actual_M'] is None


def test_missing_comparator_row_is_preserved_as_failure():
    data = [r for r in rows() if not (r['method'] == 'random_mobile' and r['seed'] == 12)]
    result = e.summarize_rows(data)
    assert result['methods']['random_mobile']['status_counts']['missing_method'] == 1
    assert result['contrasts']['capped_v2g_mobile_vs_random_mobile']['metrics']['actual_M']['n'] == 1


@pytest.mark.parametrize('field,value', [('phase', 'evaluation'), ('policy_id', 'another'),
    ('history_sha256', 'another'), ('device_sha256', 'another')])
def test_different_inputs_cannot_be_paired(field, value):
    data = rows()
    data[0][field] = value
    with pytest.raises(ValueError):
        e.summarize_rows(data)


def test_duplicate_method_or_metadata_history_mismatch_rejected():
    with pytest.raises(ValueError, match='duplicate'):
        e.summarize_rows([*rows(), rows()[0]])
    with pytest.raises(ValueError, match='coverage'):
        e.summarize_rows(rows(), metadata=dict(seeds=[11, 99]))


def test_inherited_MILP_not_miscounted_as_non_MILP_or_direct_new():
    data = rows(seeds=(11,))
    get(data, e.PRIMARY[0]).update(selected_origin='v2g_fixed', selected_is_new_MILP_proposal=False)
    get(data, 'v2g_fixed').update(selected_origin='MILP', selected_is_new_MILP_proposal=True)
    result = e.summarize_rows(data)
    rec = result['methods'][e.PRIMARY[0]]
    assert rec['direct_new_MILP_count'] == rec['direct_MILP_origin_count'] == 0
    assert rec['ancestral_MILP_count'] == rec['from_MILP_count'] == rec['ancestral_new_MILP_count'] == 1
    row = next(r for r in result['method_rows'] if r['method'] == e.PRIMARY[0])
    assert row['source_lineage_path'] == [e.PRIMARY[0], 'v2g_fixed', 'MILP']
    direct = result['methods'][e.PRIMARY[1]]
    assert direct['direct_new_MILP_count'] == direct['from_MILP_count'] == 1
    assert direct['ancestral_MILP_count'] == 0


def test_quality_parent_and_missing_or_cyclic_origins_are_distinct():
    data = rows(seeds=(11,))
    get(data, e.PRIMARY[0]).update(selected_origin='v2g_fixed', selected_is_new_MILP_proposal=False)
    get(data, 'v2g_fixed').update(selected_origin='quality_v2g_fixed')
    rec = e.summarize_rows(data)['methods'][e.PRIMARY[0]]
    assert rec['from_MILP_count'] == rec['source_lineage_unknown_count'] == 0
    get(data, 'v2g_fixed').update(selected_origin='charge_fixed')
    get(data, 'charge_fixed').update(selected_origin='v2g_fixed')
    result = e.summarize_rows(data)
    assert result['methods'][e.PRIMARY[0]]['source_lineage_unknown_count'] == 1
    get(data, 'v2g_fixed').update(selected_origin='absent_domain')
    assert e.summarize_rows(data)['methods'][e.PRIMARY[0]]['source_lineage_unknown_count'] == 1


def test_random_path_is_flagged_for_primary_even_if_parent_is_inherited():
    data = rows(seeds=(11,))
    get(data, e.PRIMARY[0]).update(selected_origin='v2g_fixed', selected_is_new_MILP_proposal=False)
    get(data, 'v2g_fixed').update(selected_origin='random_mobile')
    rec = e.summarize_rows(data)['methods'][e.PRIMARY[0]]
    assert rec['source_lineage_random_path_count'] == 1 and rec['from_MILP_count'] == 0


@pytest.mark.parametrize('kind', ['direct', 'inherited', 'quality', 'constant', 'missing', 'cycle', 'random'])
def test_independent_lineage_recursions_agree_on_resolved_and_failed_paths(kind):
    data = rows(seeds=(11,))
    child, parent = get(data, e.PRIMARY[0]), get(data, 'v2g_fixed')
    child.update(selected_origin='v2g_fixed', selected_is_new_MILP_proposal=False)
    parent.update(selected_origin='MILP', selected_is_new_MILP_proposal='False')
    if kind == 'direct':
        child.update(selected_origin='MILP', selected_is_new_MILP_proposal=True)
    elif kind == 'quality':
        parent.update(selected_origin='quality_v2g_fixed')
    elif kind == 'constant':
        parent.update(selected_origin='constant')
    elif kind == 'missing':
        parent.update(selected_origin='missing_parent')
    elif kind == 'cycle':
        parent.update(selected_origin=e.PRIMARY[0])
    elif kind == 'random':
        parent.update(selected_origin='random_mobile')
    assert e.crosscheck_lineages(data, a.trace_source_lineage)['success']


def test_inconsistent_direct_new_flag_is_not_hidden_by_matching_lineages():
    data = rows(seeds=(11,))
    get(data, e.PRIMARY[0]).update(selected_origin='v2g_fixed', selected_is_new_MILP_proposal=True)
    get(data, 'v2g_fixed').update(selected_origin='MILP', selected_is_new_MILP_proposal=False)
    checked = e.crosscheck_lineages(data, a.trace_source_lineage)
    assert not checked['success']
    assert checked['differences'][0]['fields'] == ['new_curve_flag_without_direct_MILP_origin']


def risk_fixture(candidate_miss=False):
    spec = c.CalibrationSpec(policy_id='policy', config_sha256='config', source_manifest_sha256='source',
        reference_id='synthetic', candidates=('star', 'cherry'), methods=e.METHODS, alpha=.5)
    def record(seed, true_cost=1., failure=False):
        return dict(policy_id='policy', config_sha256='config', source_manifest_sha256='source',
            reference_id='synthetic', history_id=str(seed), independent_unit_id=str(seed),
            stratum='declared_asset_reference', true_name='star', candidate_names=['star', 'cherry'],
            candidate_miss=False, methods={m:dict(status='fit_failure' if failure and m == e.PRIMARY[0] else 'complete',
                losses=dict(star=true_cost, cherry=0.)) for m in e.METHODS})
    cr, er = [record(1), record(2)], [record(11, 3., failure=True)]
    er[0]['candidate_miss'] = candidate_miss
    bundle = dict(schema='ev_compact_standard_records_v1', spec=spec.summary(),
        calibration=cr, evaluation=er, raw_manifest=dict(calibration=[], evaluation=[]))
    bundle['bundle_id'] = e._hash(bundle)
    result = dict(standard_records=bundle, alpha=.5, planned_calibration=2, actual_calibration=2,
                  planned_evaluation=1, actual_evaluation=1)
    for label, methods, familywise in [('individual_policies', e.METHODS, False),
                                      ('primary_family', e.PRIMARY, True)]:
        local = replace(spec, methods=methods, familywise=familywise)
        def subset(rs):
            return [dict(r, methods={m:r['methods'][m] for m in methods}) for r in rs]
        calibrated = c.calibrate(subset(cr), local)
        result[label] = dict(calibration=calibrated, evaluation=c.evaluate(subset(er), calibrated))
    return result


def test_saved_calibrated_decisions_not_guessed_from_margin_sign():
    result = e.summarize_rows(rows(seeds=(11,), phase='evaluation'), risk=risk_fixture())
    risk = result['risk']
    # CSV has positive M for this primary, but the saved calibration says fit failure/keep-all.
    assert risk['methods'][e.PRIMARY[0]]['computational_failures'] == 1
    assert risk['methods'][e.PRIMARY[0]]['unresolved'] == 1
    assert risk['methods'][e.PRIMARY[0]]['false_rejections'] == 0
    assert risk['methods'][e.PRIMARY[1]]['false_rejections'] == 1
    assert risk['primary_family']['any_primary_false_rejection'] == 1
    assert risk['primary_family']['history_denominator'] == 1
    assert len(risk['decisions']) == 6  # four individual + two joint, one independent H


def test_risk_calibration_phase_only_exports_thresholds():
    result = e.summarize_rows(rows(seeds=(1, 2), phase='calibration'), risk=risk_fixture())
    assert result['risk']['decisions'] == [] and result['risk']['methods'] == {}
    assert result['risk']['thresholds']['primary_family']
    assert '仅保存' in e.markdown(result)


def test_candidate_miss_is_not_silently_added_as_a_successful_false_rejection_test():
    result = e.summarize_rows(rows(seeds=(11,), phase='evaluation'), risk=risk_fixture(candidate_miss=True))
    rec = result['risk']['methods'][e.PRIMARY[0]]
    assert rec['history_denominator'] == rec['candidate_misses'] == 1
    assert rec['known_covered_histories'] == rec['false_rejections'] == 0
    assert result['risk']['decisions'][0]['false_rejection'] is None
    assert '候选漏真' in e.markdown(result)


def test_risk_version_and_independent_history_bindings_rejected():
    risk = risk_fixture()
    with pytest.raises(ValueError, match='development'):
        e.summarize_rows(rows(seeds=(11,)), risk=risk)
    with pytest.raises(ValueError, match='coverage'):
        e.summarize_rows(rows(seeds=(12,), phase='evaluation'), risk=risk)
    risk['standard_records']['spec']['policy_id'] = 'other'
    with pytest.raises(ValueError, match='hash'):
        e.summarize_rows(rows(seeds=(11,), phase='evaluation'), risk=risk)


def test_raw_trial_hash_binding_and_calibration_hash_checked():
    data = rows(seeds=(11,), phase='evaluation')
    with pytest.raises(ValueError, match='raw-trial'):
        e.summarize_rows(data, metadata=dict(trial_hashes=[dict(name='trial_11.json.gz', sha256='wrong')]), risk=risk_fixture())
    risk = risk_fixture()
    risk['individual_policies']['calibration']['thresholds']['declared_asset_reference']['star'][e.PRIMARY[0]]['threshold'] = 99.
    with pytest.raises(ValueError, match='calibration content'):
        e.summarize_rows(data, risk=risk)


def test_csv_json_delivery_has_input_and_helper_hashes(tmp_path):
    directory, output = tmp_path/'analysis', tmp_path/'effects'
    directory.mkdir()
    data = rows()
    with (directory/'method_rows.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    (directory/'summary.json').write_text(json.dumps(dict(phase='development', policy_id='policy',
        seeds=[11, 12], planned_histories=3)), encoding='utf-8')
    result = e.write_summary(directory, output)
    assert set(result['input_sha256']) == {'method_rows.csv', 'summary.json'}
    assert len(result['assistant_sha256']) == 64
    assert json.loads((output/'summary.json').read_text(encoding='utf-8'))['pending_histories'] == 1
    assert len(list(csv.DictReader((output/'paired_primary.csv').open(encoding='utf-8-sig')))) == 8
    assert '不从M正负' in (output/'summary.md').read_text(encoding='utf-8')
