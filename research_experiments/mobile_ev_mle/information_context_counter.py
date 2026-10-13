"""Serial instrumentation of context construction, including failed attempts.

The context module has two exclusive physical-AC entry paths: its imported
ac_predict alias for historical Jacobians and likelihood.ac_predict for fits
and direct loss checks. Inner fit accounting wraps the latter but delegates
to our wrapper once, so its self-reported count is not added a second time.
Global function bindings are restored in finally. Run this wrapper serially.
"""
from __future__ import annotations
from time import perf_counter
from . import information_context as context
from . import likelihood


class ContextBuildError(ValueError):
    """An exceptional required context build with complete attempted counts."""
    def __init__(self, message, *, ac_calls, fit_calls, seconds, cause_kind,
                 alias_ac_calls, module_ac_calls):
        self.ac_calls, self.fit_calls = int(ac_calls), int(fit_calls)
        self.seconds, self.cause_kind = float(seconds), str(cause_kind)
        self.alias_ac_calls, self.module_ac_calls = int(alias_ac_calls), int(module_ac_calls)
        self.call_count_available = True
        super().__init__(message+f" [context attempted AC={self.ac_calls}, fits={self.fit_calls}]")

    def record(self):
        return dict(kind=type(self).__name__, message=str(self),
            ac_calls=self.ac_calls, fit_calls=self.fit_calls, seconds=self.seconds,
            cause_kind=self.cause_kind, alias_ac_calls=self.alias_ac_calls,
            module_ac_calls=self.module_ac_calls, call_count_available=True,
            call_count_unavailable=False)


def build_centres(*args, **kwargs):
    """Call the existing builder without changing its numerical decisions.

    Count at entry, before the AC or nonlinear fit can fail. The successful
    record preserves the builder's reported counts and whether they match,
    then uses observed counts as ac_calls/fit_calls. Any exception carries the
    actual attempted denominator; no unknown count is replaced with zero.
    """
    original_alias = context.ac_predict
    original_module_ac = likelihood.ac_predict
    original_fit = likelihood.fit_candidate
    counts = dict(alias=0, module=0, fit=0)
    started = perf_counter()

    def alias_counted(*ac_args, **ac_kwargs):
        counts["alias"] += 1
        return original_alias(*ac_args, **ac_kwargs)

    def module_counted(*ac_args, **ac_kwargs):
        counts["module"] += 1
        return original_module_ac(*ac_args, **ac_kwargs)

    def fit_counted(*fit_args, **fit_kwargs):
        counts["fit"] += 1
        return original_fit(*fit_args, **fit_kwargs)

    context.ac_predict = alias_counted
    likelihood.ac_predict = module_counted
    likelihood.fit_candidate = fit_counted
    try:
        centers, original_record = context.build_centres(*args, **kwargs)
        record = dict(original_record)
        ac_calls = counts["alias"]+counts["module"]
        reported_ac = record.get("ac_calls")
        reported_fit = record.get("fit_calls")
        verification = dict(reported_ac_calls=reported_ac,
            reported_fit_calls=reported_fit, instrumented_ac_calls=ac_calls,
            instrumented_fit_calls=counts["fit"],
            alias_ac_calls=counts["alias"], module_ac_calls=counts["module"],
            ac_count_matches=reported_ac == ac_calls,
            fit_count_matches=reported_fit == counts["fit"])
        record.update(ac_calls=ac_calls, fit_calls=counts["fit"],
            instrumented_ac_calls=ac_calls, instrumented_fit_calls=counts["fit"],
            instrumented_build_seconds=perf_counter()-started,
            call_count_available=True, call_count_unavailable=False,
            call_counter_verification=verification,
            call_counter_verification_passed=bool(verification["ac_count_matches"] and verification["fit_count_matches"]))
        return centers, record
    except Exception as error:
        raise ContextBuildError(str(error), ac_calls=counts["alias"]+counts["module"],
            fit_calls=counts["fit"], seconds=perf_counter()-started,
            cause_kind=type(error).__name__, alias_ac_calls=counts["alias"],
            module_ac_calls=counts["module"]) from error
    finally:
        context.ac_predict = original_alias
        likelihood.ac_predict = original_module_ac
        likelihood.fit_candidate = original_fit


build_centers = build_centres