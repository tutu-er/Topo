"""Paired-noise, fixed-history, checkpoint and conditional-denominator tests."""
from types import SimpleNamespace

import numpy as np
import pytest

from . import ev_mobile_v4_gap_noise as n


def fixture():
    clean=dict(p=np.ones((2,3)),q=np.full((2,3),.4),v=np.ones((2,3)),root_v=np.ones(2),
        master_p=np.ones(2)*3,master_q=np.ones(2),source_p=None,source_q=None)
    meter=SimpleNamespace(**{k+'_sigma':.03*(i+1) for i,k in enumerate(n.CHANNELS)})
    return n.core.io.observations_record(n.core._observation(clean)),meter


def test_noise_is_reproducible_independent_channels_and_shared_between_actions():
    clean,meter=fixture();shapes={k:np.asarray(clean[k]).shape for k in n.CHANNELS}
    white=n.white_noise(n.SEEDS[0],shapes)
    assert n.core.fingerprint(white)==n.core.fingerprint(n.white_noise(n.SEEDS[0],shapes))
    assert n.core.fingerprint(white)!=n.core.fingerprint(n.white_noise(n.SEEDS[1],shapes))
    assert not np.array_equal(white['p'],white['q'])
    other={**clean,'p':np.asarray(clean['p'])+5.,'v':np.asarray(clean['v'])+.02}
    a,b=n.add_white(clean,meter,white),n.add_white(other,meter,white)
    for k in n.CHANNELS:
        assert np.asarray(a[k])-clean[k]==pytest.approx(np.asarray(b[k])-other[k])
    assert np.array_equal(a['root_v'],clean['root_v']) and a['source_p'] is None
    # Zero white preserves already included EV and calibration offsets exactly.
    zero={k:np.zeros(shape) for k,shape in shapes.items()}
    assert n.core.fingerprint(n.add_white(clean,meter,zero))==n.core.fingerprint(clean)


def test_slot_keeps_history_fit_seed_and_checkpoint(tmp_path):
    clean,meter=fixture();D=n.core._observation(clean);before=n.core.fingerprint(clean)
    meta=dict(observed=clean,observed_meter={},future_meter={},trained={},fit_options={'max_nfev':7},
        fit_seed=100,candidate_bank=[],true_name='a',candidate_names=list('abcd'))
    context=dict(D=D,dm=object(),fm=meter,bank=list('abcd'),fits={})
    seed=n.SEEDS[0];white=n.white_noise(seed,{k:np.asarray(clean[k]).shape for k in n.CHANNELS})
    n.core._write(tmp_path/'white_noise'/(str(seed)+'.json.gz'),dict(seed=seed,white=white,white_sha256=n.core.fingerprint(white)))
    calls=[]
    def evaluate(bank,history,dm,future,fm,fits,options,fit_seed):
        calls.append(1)
        assert history is D and fit_seed==100 and options=={'max_nfev':7}
        assert n.core.fingerprint(n.core.io.observations_record(history))==before
        return dict(complete=True,comparisons={k:dict(joint=dict(success=True,loss=float(i+1))) for i,k in enumerate(bank)},failures={})
    row=n.evaluate_slot(tmp_path,meta,context,clean,seed,n.METHODS[0],evaluate=evaluate)
    assert row['status']=='complete' and row['halfM']==.5
    assert row['candidate_slots']==dict(expected=4,attempted=4,success=4,failed=0,not_run=0)
    assert n.evaluate_slot(tmp_path,meta,context,clean,seed,n.METHODS[0],evaluate=evaluate)==row
    assert len(calls)==1 and n.core.fingerprint(n.core.io.observations_record(D))==before
    cache=n.core._read(row['fit_source']['path'])
    assert cache['fit_seed']==100 and set(cache['outcome']['comparisons'])==set('abcd')


def test_missing_pairs_keep_denominator_and_are_not_zero_filled():
    slots={}
    for seed,value in zip(n.SEEDS,(1.,-3.,0.)):
        slots[seed,n.METHODS[0]]=dict(status='complete',halfM=10.)
        slots[seed,n.METHODS[1]]=dict(status='complete',halfM=10.+value)
    slots[n.SEEDS[3],n.METHODS[0]]=dict(status='complete',halfM=10.)
    slots[n.SEEDS[3],n.METHODS[1]]=dict(status='fit_failure',halfM=None)
    result=n.paired_statistics(slots,dict(noiseless111_delta=2.,original_observed_delta=-3.))
    assert result['history_denominator']==1 and result['conditional_repeat_denominator']==32
    assert result['complete_pairs']==3 and result['incomplete_pairs']==29
    assert result['mean']==pytest.approx(-2/3)
    assert result['sample_sd']==pytest.approx(np.std([1.,-3.,0.],ddof=1))
    assert result['Monte_Carlo_SE']==pytest.approx(result['sample_sd']/np.sqrt(3))
    assert (result['win'],result['tie'],result['loss'])==(1,1,1)
    assert result['fraction_at_or_below_original']==pytest.approx(1/3)
    assert result['pairs'][3]['delta_halfM'] is None


def test_empty_summary_retains_all_declared_slots_and_no_new_history(tmp_path):
    result=n.summarize(tmp_path)
    assert result['history_denominator']==1 and result['target_slot_denominator']==64
    assert result['candidate_slots']==dict(expected=256,attempted=0,success=0,failed=0,not_run=256)
    assert result['statistics']['complete_pairs']==0 and 'mean' not in result['statistics']
    assert len(n.SEEDS)==32 and n.SEEDS[0]==2026102200 and n.SEEDS[-1]==2026102231
    assert not result['independent_confirmation'] and not result['used_to_change_policy']
    assert n.core.METHODS==('guarded__fixed_bidir','guarded__mobile_bidir')
