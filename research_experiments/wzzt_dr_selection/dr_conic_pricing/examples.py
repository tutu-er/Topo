"""Small explicit synthetic input, not behavior calibration or a grid result."""
from __future__ import annotations

import numpy as np

from .model import PriceProblem


def synthetic_problem():
    """Two independent devices, two exclusive energy-neutral types per device.

    Four one-hour slots, two terminal nodes. Flatten P first and Q second;
    each channel is time-major. The benefit rows are signed temporal contrasts,
    not actual topology losses. No minimum spatial dispersion is imposed.
    """
    slots, nodes, contracts, factors = 4, 2, 4, 3
    p = np.zeros((slots, nodes, contracts))
    p[0, 0, 0], p[1, 0, 0] = -1., 1.
    p[2, 0, 1], p[3, 0, 1] = -1., 1.
    p[0, 1, 2], p[1, 1, 2] = -1., 1.
    p[2, 1, 3], p[3, 1, 3] = -1., 1.
    hp = p.reshape(slots*nodes, contracts)
    hq = hp*np.array([.2, .2, .4, .4])
    h = np.vstack((hp, hq))
    mu = np.r_[np.ones(slots*nodes), np.tile([.2, .4], slots)]
    # Two explicitly assumed background load/PV factors and a third independent
    # response factor. Both load and DR use the same factors, preserving covariance.
    f = np.zeros((len(mu), factors))
    f[:slots*nodes, 0] = np.tile([.025, .015], slots)
    f[:slots*nodes, 1] = [-.01, -.01, .01, .01, -.01, -.01, .01, .01]
    f[slots*nodes:, :] = .3*f[:slots*nodes, :]
    d = np.array([[.03, .04, .06], [.02, -.06, .08],
                  [.04, .02, -.05], [.01, .03, -.07]])
    g = np.zeros((2, len(mu)))
    g[0, 0], g[0, 2] = -.5, .5
    g[1, 5], g[1, 7] = -.5, .5
    # P in [.5,1.5] and all prefix incremental energies in [-.45,.45].
    a = np.vstack((hp, -hp, np.cumsum(p, axis=0).reshape(slots*nodes, contracts),
                   -np.cumsum(p, axis=0).reshape(slots*nodes, contracts)))
    b = np.r_[np.full(2*slots*nodes, .5), np.full(2*slots*nodes, .45)]
    return PriceProblem(
        mu0=mu, background_factors=f, profiles=h, response_factors=d,
        elasticities=np.full(contracts, .2), reduction_hours=np.ones(contracts),
        price_upper=np.full(contracts, 2.), actual_amplitude_upper=np.full(contracts, .4),
        benefit_directions=g, benefit_offsets=np.zeros(2), risk_allocations=[.05, .05],
        budget=1.6, device_matrix=a, device_rhs=b,
        contract_ids=["node1_early", "node1_late", "node2_early", "node2_late"],
        exclusive_groups=[[0, 1], [2, 3]],
        metadata=dict(source="explicit synthetic teaching/solver qualification case",
                      calibrated=False, node_labels=[201, 204], slots=slots,
                      interval_hours=1., stacking="P_then_Q_time_major",
                      benefit_units="kW temporal contrast", fixed_socp_allowed=[0, 3],
                      resource_semantics="one independently metered device at each terminal; one contract type each",
                      background_semantics="assumed net-load factors already combine natural load and renewables"))


def resource_day_problem():
    """96-slot DR plus EV, using original resource declarations and profiles.

    Inputs are explicitly assumed. This qualifies a full-day data path; it is
    not a household calibration or an AC/topology recovery experiment.
    """
    from research_experiments.wzzt_dr_selection.dr_day_modules.day_schedule import (
        Battery, ResourceDay, ev_resource, shift_module, ev_reallocation_module,
    )
    from .resources import resource_contract_data
    slots = 96
    shift = ResourceDay("shift", 201, .2, .4, -.4, .4,
                        reactive_ratio=.2, declaration_source="synthetic full-day qualification")
    baseline = np.zeros(slots)
    baseline[40:80] = .2
    available = np.zeros(slots, dtype=bool)
    available[40:80] = True
    ev = ev_resource("ev", 204, baseline, .4, available,
                     Battery(10., 2., 4.), service_deadline_slot=80,
                     declaration_source="synthetic EV; fixed declared connection and task")
    modules = [shift_module("shift_18_to_20", "shift", range(72, 80), range(80, 88), .2),
               shift_module("shift_16_to_18", "shift", range(64, 72), range(72, 80), .2),
               ev_reallocation_module("ev_early", ev, range(44, 52), range(56, 64), .2),
               ev_reallocation_module("ev_late", ev, range(60, 68), range(72, 80), .2)]
    d = [[.03, .04, .06], [.02, -.06, .08], [.04, .02, -.05], [.01, .03, -.07]]
    partial = resource_contract_data([shift, ev], modules, node_labels=[201, 204],
                                    elasticities=[.1]*4, price_upper=[2.]*4,
                                    response_factors=d)
    h = np.asarray(partial["profiles"])
    # Explicit P-only temporal contrasts associated with requested module types.
    g = np.zeros((2, 4*slots))
    for k, j in enumerate((0, 3)):
        g[k, :2*slots] = h[:2*slots, j]/np.sum(h[:2*slots, j]**2)
    mu_p = np.column_stack((np.ones(slots), .8+baseline))
    mu = np.r_[mu_p.ravel(), (mu_p*np.array([.2, .4])).ravel()]
    f = np.zeros((4*slots, 3))
    f[:2*slots, 0] = .025
    f[:2*slots, 1] = np.repeat(.01*np.sin(np.arange(slots)*2*np.pi/slots), 2)
    f[2*slots:, :] = .3*f[:2*slots, :]
    partial["metadata"].update(source="explicit synthetic 24h DR+EV qualification",
                                calibrated=False, fixed_socp_allowed=[0, 3],
                                benefit_units="kW signed temporal contrast",
                                background_semantics="net background includes natural load and renewables; resource baselines already included")
    return PriceProblem(mu0=mu, background_factors=f, benefit_directions=g,
                        benefit_offsets=[0., 0.], risk_allocations=[.05, .05],
                        budget=1.6, **partial)
