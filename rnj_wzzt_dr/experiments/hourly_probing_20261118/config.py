"""Settings of the hourly probing development runs (PROTOCOL v3 draft defaults).

Every value in `Settings` is the v3 draft default unless marked as a development choice; the defaults are
kept so that every development run reproduces. The block selector and its settings were locked on
2026-11-23 (PROTOCOL section 11, user decision): use `locked_settings()` and `LOCKED_ARM` for them. The rest
of the protocol is still a draft.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math


@dataclass(frozen=True)
class Settings:
    # Timing
    slots_per_hour: int = 4
    hours: int = 24
    gate_lead_hours: int = 1          # block for h+1 sealed at start of h, data to end of h-1
    horizon_hours: int = 2            # confirmation target H_c
    # Scoring and stopping
    alpha: float = .05                # confirmation level 1-alpha (posterior 0.95)
    delta: float = .1                 # miss probability of the planner condition
    omega: float = .5                 # Chernoff order
    confirm_hours: int = 2            # consecutive hourly checks
    equivalence_pu: float = 5e-4      # meter resolution eps_V in (19), 0.05 % of nominal
    no_progress_hours: int = 6
    no_progress_gain: float = 1.
    max_active: int = 3
    tail_order: float = .5            # Chernoff order of rivals outside the active set (1.0: variant)
    # Refit
    r_min_ohm: float = .003           # R_min; r_min = 2 R_min / Z_base
    upper_bound: float = 2.           # QPFitter box bound
    first_gate_fixed_point: int = 3
    use_loss_term: bool = True
    refit: bool = True                # False only for frozen scoring
    # Planner
    kappa_grid: int = 24
    bisection_steps: int = 30
    voltage_band: tuple = (.96, 1.035)
    z_voltage: float = 2.33
    plausible_weight: float = .01
    unbalance_factor: float = 6.      # kappa_U = kappa_V / 6 for single-phase delivery
    hourly_cap_cny: float = 5.
    daily_cap_cny: float = 50.
    # Procurement cost model. "contract": price x u^2/headroom at 0.30-0.60 CNY/kWh (contract (3), v3 draft).
    # "linear": demand-response settlement, CNY per kWh moved: reductions (discharge, paused charging) at
    # price_down_cny, increases (extra charging) at price_up_cny. Chinese peak-shaving DR pays about 3-4 CNY/kWh,
    # valley filling 0.3-1.2 CNY/kWh (development choice 2026-11-22, upper ends).
    cost_model: str = "contract"
    price_down_cny: float = 4.
    price_up_cny: float = 1.2
    forecast_sigma: float = .08       # relative 1-sigma load forecast error used by the planner
    # Broad check arm
    check_hours: int = 2
    check_rho: float = .5
    # Resources (scenario R1)
    household_kw: float = .8
    battery_share: float = .25
    ev_share: float = .20
    enrolled_share: float = .25
    battery_kw: float = 5.
    battery_kwh: float = 13.5
    ev_kw: float = 7.
    ev_hours: tuple = (18, 8)         # plugged in 18:00-08:00
    ev_down_fraction: float = .5      # share of enrolled EV power that is charging (can pause)
    soc_bounds: tuple = (.2, .8)
    battery_eta: float = .95
    # Measurement (development defaults)
    voltage_sigma: float = .0005      # random part of voltage meter error (relative)
    pq_sigma: float = .005
    root_sigma: float = .003          # physical root-voltage variation
    meter_bias_sigma: float = 0.      # fixed per-meter voltage gain bias (relative); typical .0005, stress .001
    voltage_resolution_v: float = 0.  # quantisation step in volts on a 230 V base, stress test
    meter_offsets: bool = False       # revised step 1: a free constant per meter in every refit
    root_whitening: bool = False      # GLS for the root-meter error shared by every terminal's y
    tail_guarantee: bool = True       # False: best-effort planner on the top rivals (development variant)
    cert_value_cny: float = 50.       # certified planner: price of certainty V (CNY per unit miss bound)
    cert_rivals: int = 20             # certified planner: nearest rivals with q_j in the bound
    bayes_members: int = 10           # Bayes-risk greedy: most probable candidates whose class pairs enter U
    trust_gate: bool = False          # probe only if the closure completed and the leader is adequate
    adequacy_max: float = 1.2         # trust gate: leader residual variance / meter-class variance
    prequential_check: bool = False   # evaluation only: log one-hour-ahead prediction errors (DEVELOPMENT_RUNS section 18)
    delivery_gain: tuple = (.9, 1.)
    # Physics of the evaluation (single phase; the three-phase option is out of scope since 2026-11-20)
    unbalanced: bool = False
    z0_over_z1: float = 3.
    transformer_kva: dict = field(default_factory=lambda: {"soumalas11": 125., "pengwah18": 200.,
                                                                      "flynn16": 200., "paper15": 200.})

    @property
    def ln19(self):
        return math.log((1-self.alpha)/self.alpha)

    def with_(self, **changes):
        return replace(self, **changes)


# Block selector lock (PROTOCOL section 11, user decision 2026-11-23): the strong greedy (one-step Bayes-risk
# greedy, class weights) at V = 2,000 CNY, settled at linear DR prices, on the refined pipeline: revised
# step 1 (bank.build(step1="v2"), at least one day of history), per-meter offsets, GLS for the shared root-meter
# error and the trust gate. These are exactly the settings confirmed on fresh seeds in DEVELOPMENT_RUNS
# section 18 (L1, L2). T1 (truth removed) failed there; the lock was made with that failure disclosed
# (sections 18-19), so Stage M keeps reporting truth-removed and natural truth-missing runs.
LOCKED_ARM = "bgreedy_P"
LOCKED = dict(meter_offsets=True, root_whitening=True, trust_gate=True, adequacy_max=1.2,
              cost_model="linear", price_down_cny=4., price_up_cny=1.2, hourly_cap_cny=20., daily_cap_cny=100.,
              cert_value_cny=2000., bayes_members=10,
              # protocol v4 section 11 (2026-11-23): the rules used in every development run
              alpha=.05, confirm_hours=2, equivalence_pu=5e-4, gate_lead_hours=1, no_progress_hours=6,
              no_progress_gain=1., horizon_hours=2, cert_rivals=20, voltage_band=(.96, 1.035), z_voltage=2.33,
              plausible_weight=.01, unbalance_factor=6.)


def locked_settings(**measurement):
    """Settings of the locked block selector. `measurement` may set the measurement condition (meter bias,
    noise, resources) but none of the locked values."""
    clash = sorted(set(measurement) & set(LOCKED))
    if clash:
        raise ValueError(f"locked settings cannot be changed: {clash}")
    return Settings().with_(**LOCKED).with_(**measurement)


ARMS = ("passive_bic", "t_opt_P", "frozen_replay", "check_then_t_opt", "random_matched_P")
# certified planner and its comparison arms (PLANNER_REVISION.md, DEVELOPMENT_RUNS section 13)
CERT_ARMS = ("passive_bic", "cert_P", "random_matched_P", "qbc_matched_P", "qbc_P", "greedy_P")
DEV_ARMS = ("passive_bic", "cert_P", "bgreedy_P", "random_matched_P", "qbc_matched_P", "qbc_P", "greedy_P")
# hard-case test (DEVELOPMENT_RUNS section 16): realistic DR prices, ambiguous inputs only
HARD_ARMS = ("passive_bic", "cert_P", "random_matched_P", "qbc_P", "bgreedy_P", "bgreedy_lowV_P")
