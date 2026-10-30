from .actual_runner import summarize


def test_failed_cost_is_not_lost():
    result=summarize([dict(policy='zero',status='failed',actual_payment=.2)])
    group=next(g for g in result['policy_groups'] if g['policy']=='zero')
    assert group['known_actual_payment']==.2 and group['complete']==0
    assert group['structure_denominator']==0 and group['wzzt_total_errors'] is None


def test_unknown_cost_is_not_zero():
    result=summarize([dict(policy='rotation',status='timeout')])
    group=next(g for g in result['policy_groups'] if g['policy']=='rotation')
    assert group['payment_unknown_records']==1 and group['payment_known_records']==0
    assert len(result['policy_groups'])==8


def test_incomplete_search_keeps_its_denominator():
    row=dict(policy='wzzt_response_table',status='complete',actual_payment=.4,
        fit_path='one',all_local_scans_complete=False,
        evaluation=dict(multistart_original_rule=dict(full_block_errors=1,full_recovery=False),
                        rnj_bank=dict(full_block_errors=2,full_recovery=False)))
    result=summarize([row]);group=next(g for g in result['policy_groups'] if g['policy']==row['policy'])
    assert group['incomplete_search_records']==1 and group['structure_denominator']==1
    assert group['wzzt_total_errors']==1 and result['planned']==64
