"""Predeclared development settings; not fitted on new-day outcomes."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Config:
    dt_hours: float = .25
    steps: int = 96
    beta: float = .9
    eta: float = .9
    lambda_start: float = .5
    initial_delivery_gain: float = .95
    minimum_delivery_gain: float = .9
    confirmation_hours: float = 2.
    recovery_hours: float = 2.
    diagnostic_standard_deviations: float = 2.
    caution_fraction: float = .25
    day_solver_seconds: float = 1200.
    round_solver_seconds: float = 30.
    kappa_p: float = .5
    kappa_q: float = .5
    progress: bool = True
    modes: tuple = ('normal_fixed', 'mgf_fixed', 'mgf_forecast')

    def __post_init__(self):
        if self.steps != 96 or self.dt_hours != .25:
            raise ValueError('This protocol uses exactly 96 quarter-hour decisions')
        values = (self.beta, self.eta, self.lambda_start, self.day_solver_seconds,
                  self.round_solver_seconds, self.confirmation_hours, self.recovery_hours)
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Finite positive configuration values required')
        if not 0 < self.beta < 1 or not 0 < self.eta < 1:
            raise ValueError('beta and eta must lie in (0,1)')
        if not self.minimum_delivery_gain <= self.initial_delivery_gain <= 1:
            raise ValueError('Initial gain outside the declared device range')

    def record(self):
        return asdict(self)
