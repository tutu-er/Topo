"""Independent checks of relative loss and conservative simulated tails."""
from .relative_audit import recompute_relative, recompute_tail


def test_relative_loss_uses_minimum_over_all_candidates():
    fits = {name:dict(qualified=True, loss=loss) for name, loss in
            (("star", 17.), ("pair01", 12.), ("pair02", 15.), ("pair12", 9.))}
    assert recompute_relative(fits, "star")["statistic"] == 8.
    assert recompute_relative(fits, "pair12")["statistic"] == 0.
    fits["pair01"]["qualified"] = False
    result = recompute_relative(fits, "star")
    assert result["valid"] is False
    assert result["invalid"] == ["pair01"]
    assert result["statistic"] is None


def test_failed_statistic_counts_in_tail_instead_of_being_dropped():
    values = [dict(valid=True, statistic=1.), dict(valid=True, statistic=5.),
              dict(valid=True, statistic=10.), dict(valid=False, statistic=None)]
    result = recompute_tail(5., values)
    assert result == dict(pvalue=4/5, replicates=4, failed=1, valid=3, upper_tail=3)
