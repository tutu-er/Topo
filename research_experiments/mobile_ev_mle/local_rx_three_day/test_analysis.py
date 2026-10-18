import gzip
import hashlib
import json

from .analysis import _attempt_summary, _ranking, summarize, write_outputs


def test_ranking_requires_all_qualified_and_retains_absolute_ties():
    fits = {name:dict(qualified=True, loss=10.+i) for i, name in enumerate(('star', 'pair01', 'pair02', 'pair12'))}
    fits['pair01']['loss'] = 10.+.5e-6
    ranking = _ranking(fits)
    assert ranking['available']
    assert ranking['tied_groups'][0] == ['star', 'pair01']
    assert not ranking['unique_first']
    fits['pair12']['qualified'] = False
    assert not _ranking(fits)['available']


def test_attempt_objective_agreement_does_not_claim_parameter_agreement():
    result = _attempt_summary(dict(attempts=[dict(loss=3., projected_gradient=.00001, qualified=True,
        physical=True, reason='projected_gradient')]*4))
    assert result['count'] == 4
    assert result['loss_range'] == 0.
    assert not result['parameter_spread_available']


def test_saved_data_summary_keeps_missing_denominator_and_pairing(tmp_path):
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = json.dumps(data).encode()
        path.write_bytes(gzip.compress(raw) if path.suffix == '.gz' else raw)
    specs = [dict(network='paper15', seed=1, N=n, boundary='U') for n in (288, 672)]
    write(tmp_path/'configuration.json', dict(tasks=specs, pilot=False, workers=4,
        independent_random_families=1, source_manifest={}))
    fits = {name:dict(qualified=True, status='qualified', loss=10.+i, J=5.+i/2,
        information=dict(rank=6, width=8, active_line_coordinate_count=0),
        attempts=[dict(loss=10.+i, physical=True, qualified=True, projected_gradient=0., reason='projected_gradient')]*4,
        seconds=1., ac_calls=4) for i, name in enumerate(('star', 'pair01', 'pair02', 'pair12'))}
    inf = tmp_path/'inference/paper15_1_N288_U.json.gz'
    write(inf, dict(fits=fits, diagnostics={}))
    write(tmp_path/'results/paper15_1_N288_U.json.gz', dict(spec=specs[0], status='completed',
        qualified=4, seconds=5., inference_sha256=hashlib.sha256(inf.read_bytes()).hexdigest(),
        evaluation=dict(truth_read_after_inference_seal=True, candidates={'star':dict(r_leaf_relative=.2,
            x_leaf_relative=.4, r_total_relative=.9, x_total_relative=.8, beta_leaf_relative=.1)})))
    result = summarize(tmp_path)
    assert result['expected_conditions'] == 2
    assert result['completed_conditions'] == 1
    assert result['candidate_denominator'] == 8
    assert result['qualified_candidates'] == 4
    assert result['independent_random_families'] == 1
    assert result['audit']['status'] == 'pending_or_failed'
    assert result['paired_length_groups'][0]['available'] == 0
    assert result['paired_length_groups'][0]['denominator'] == 1
    assert result['groups'][1]['qualified_true_star'] == 0
    write_outputs(tmp_path, result)
    assert (tmp_path/'candidate_details.csv').exists()
    assert '0/1' in (tmp_path/'results_generated.md').read_text(encoding='utf-8')
