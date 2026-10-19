from copy import deepcopy
import pytest
from research_experiments.mobile_ev_mle.ac_history_refresh.test_io_replay import _raw
from research_experiments.mobile_ev_mle.ac_history_refresh.evidence.pack_replay_inputs import thin_record,verify_inputs,pack_file,pack_batch
from research_experiments.mobile_ev_mle.ac_history_refresh import io


def test_thin_inputs_keep_exact_history_future_and_physical_fingerprints():
    raw=_raw();raw['plans']['SQMD'].update(operation_x=[0.,1.],decoded={'u_kw':[[1.]]},physicalcheck={'feasible':True})
    raw['operation_config']={'battery_kwh':50.};raw['training_result']['fits'][0]['diagnostics']={'large_matrix':'omit'}
    raw['refit_history']['candidates']['star']['state']['unused_future_joint']='omit'
    thin=thin_record(raw);proof=verify_inputs(raw,thin)
    assert proof['passed'],proof
    for key in ('prediction','posterior_centers','actual_white_noise','true_lines_evaluation_only'):assert key not in thin
    assert 'diagnostics' not in thin['training_result']['fits'][0]
    assert 'unused_future_joint' not in thin['refit_history']['candidates']['star']['state']
    assert 'truth_inputs_evaluation_only' not in thin['actions']['a']['regimes']['regular']
    assert thin['plans']['SQMD']['operation_x']==[0.,1.]
    assert proof['new_optimizations']==proof['new_scientific_samples']==proof['new_noise']==0


def test_original_process_failure_is_preserved_with_identical_missing_input_reason():
    raw=dict(status='process_failure',spec={'index':0},failures={'worker':'timeout'})
    proof=verify_inputs(raw,thin_record(raw))
    assert proof['passed'] and not proof['preparation']['available']
    assert proof['preparation']['errors']['original']==proof['preparation']['errors']['packed']


def test_packed_bytes_have_separate_hash_and_are_never_overwritten(tmp_path):
    source=tmp_path/'original.json.gz';destination=tmp_path/'thin.json.gz';io.save_record(source,_raw())
    record=pack_file(source,destination)
    assert record['passed']
    assert record['original_raw_sha256']!=record['packed_raw_sha256']
    assert record['verification']['evaluation']['sha256']['original']==record['verification']['evaluation']['sha256']['packed']
    with pytest.raises(FileExistsError):pack_file(source,destination)


def test_batch_guard_does_not_open_main_raw_before_A_completion(tmp_path):
    io.save_record(tmp_path/'controller.json',{'status':'running'})
    with pytest.raises(RuntimeError,match='no scientific raw opened'):pack_batch(tmp_path,tmp_path/'packed')
