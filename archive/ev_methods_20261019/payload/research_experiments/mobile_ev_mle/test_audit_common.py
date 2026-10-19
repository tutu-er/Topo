import numpy as np
from .audit_common import build_bank,vectorize,unvectorize,changed_prior,atom_mean


def test_finite_bank_is_proper_external_and_all_actions_are_feasible():
    bank=build_bank()
    assert len(bank.atoms)==48
    np.testing.assert_array_equal(np.bincount(bank.labels),[24,24])
    assert bank.prior.min()>0 and np.isclose(bank.prior.sum(),1.)
    assert bank.diagnostics["source_point"]=="node1"
    assert bank.diagnostics["ac_equation_residual"]<2e-10
    assert bank.diagnostics["power_balance_residual"]<2e-10
    assert all(row["feasible"] for row in bank.diagnostics["action_checks"].values())
    for i in range(2):
        pairs={(a["rscale"],a["xscale"]) for a in bank.atoms if a["topology"]==i}
        assert len(pairs)==4  # X scale is not tied to the R scale.


def test_meter_roundtrip_and_ev_is_counted_once_in_all_atoms():
    bank=build_bank()
    for g in range(len(bank.atoms)):
        none=unvectorize(bank.future_means["none"][g],bank.meter,8)
        np.testing.assert_allclose(vectorize(none,bank.meter),bank.future_means["none"][g])
        for name,action in bank.actions.items():
            obs=unvectorize(bank.future_means[name][g],bank.meter,8)
            np.testing.assert_allclose(obs.p-none.p,action.p,atol=2e-16)
            np.testing.assert_allclose(obs.q-none.q,action.q,atol=2e-16)


def test_lines_are_shared_and_common_source_is_time_varying():
    bank=build_bank()
    for g,a in enumerate(bank.atoms):
        assert np.ptp(a["source_p"])>.02
        mean,physical=atom_mean(bank.candidates[bank.labels[g]],a,bank.meter,slice(0,12))
        np.testing.assert_allclose(mean,bank.history_means[g])
        # AC total master equals terminal consumption, same-point source, losses.
        losses=np.sum(abs(physical["branch_current"])**2*(a["r"]+1j*a["x"]),axis=1)
        master=physical["master_p"]+1j*physical["master_q"]
        total=a["p"][:12].sum(axis=1)+1j*a["q"][:12].sum(axis=1)+a["source_p"][:12]+1j*a["source_q"][:12]
        np.testing.assert_allclose(master,total+losses,atol=2e-10)


def test_prior_sensitivity_changes_mass_without_changing_support():
    bank=build_bank()
    prior=changed_prior(bank,.2,.7)
    assert prior.min()>0 and np.isclose(prior.sum(),1.)
    assert np.isclose(prior[bank.labels==0].sum(),.2)
    weak=np.array([a["strength"]==0 for a in bank.atoms])
    assert np.isclose(prior[weak].sum(),.7)
