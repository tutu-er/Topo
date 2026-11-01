"""Counterexamples for structural deletion versus coefficient refitting."""
from time import perf_counter
import numpy as np
from rnj_wzzt.estimation import laminar_l1_milp as core
from research_experiments.local_rebuild.model import canonical
from research_experiments.flexible_exit.model import MaskBank, original_step, select_block


def example(true_block, old_block):
    n = 7; z = np.zeros(n); z[list(true_block)] = 1.
    y = np.eye(n)+np.outer(z, z)
    data = core._prepare_scenarios([dict(P_terminal=np.eye(n), Q_terminal=np.zeros((n,n)), drop_target=y)])
    basis = canonical([*((i,) for i in range(n)), tuple(range(n))])
    cal = dict(a=1., b=1., median_root_depth=1.)
    return data, basis, cal


def test_delete_refit_does_not_repair_a_blocked_true_connection():
    correct = tuple(range(1,7)); wrong = tuple(range(6)); data, basis, cal = example(correct, wrong)
    bank = MaskBank(data, [*basis, wrong, correct], cal, deadline=perf_counter()+60.)
    try:
        before = bank.fit([*basis, wrong], preserve=True)
        assert select_block(before,7) == wrong  # zero response may still block.
        dropped = bank.fit(basis, preserve=True)
        assert dropped['objective'] >= before['objective']-1e-8 and dropped['objective'] > .01
        repaired, event = original_step(bank, dropped, penalty=.0001, deadline=perf_counter()+30., per_mip=20.)
        assert event['accepted'] and repaired['objective'] < 1e-8
        assert correct in canonical(repaired['supports'])
    finally: bank.close()


def test_mask_prevents_rediscovering_a_correct_removed_block():
    correct = tuple(range(1,7)); data, basis, cal = example(correct, correct)
    free = MaskBank(data, [*basis, correct], cal, deadline=perf_counter()+60.)
    masked = MaskBank(data, basis, cal, forbidden=[correct], deadline=perf_counter()+60.)
    try:
        start = free.fit(basis, preserve=True)
        recovered, _ = original_step(free, start, penalty=.0001, deadline=perf_counter()+25., per_mip=20.)
        alternative, event = original_step(masked, masked.fit(basis,preserve=True), penalty=.0001, deadline=perf_counter()+25., per_mip=20.)
        assert recovered['objective'] < 1e-8 and correct in canonical(recovered['supports'])
        assert correct not in canonical(alternative['supports']) and alternative['objective'] > 1e-4
        assert event['audit']['valid']
    finally: free.close(); masked.close()
