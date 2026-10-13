"""Research-only search adapter with copied globals and an unchanged search body.

The adapter is opt-in. It does not edit the original policy or install a global
solver wrapper. The archived pilots used the same construction with common QP
instrumentation for both upper backends.
"""
from __future__ import annotations

import copy
import hashlib
import marshal
import time
import types

import numpy as np

from . import ev_information_milp as original
from .shared_power_square_upper import upper_master_shared


def _function_identity(function):
    return {
        "module": function.__module__,
        "qualname": function.__qualname__,
        "code_sha256": hashlib.sha256(marshal.dumps(function.__code__)).hexdigest(),
    }


def make_search_adapter(upper_backend=upper_master_shared, *, qp_trace=None):
    """Return an isolated adaptive_search clone and an inspectable adapter audit.

    Only upper_master is replaced as a policy component. Supplying a trace list
    also installs a pass-through solve_pair logger in the clone's private globals.
    Exceptions remain exceptions and are included in the trace.
    """
    source = original.adaptive_search
    source_upper = source.__globals__["upper_master"]
    source_qp = source.__globals__["solve_pair"]
    globals_copy = source.__globals__.copy()
    globals_copy["upper_master"] = upper_backend

    if qp_trace is not None:
        def trace_qp(pair, u, *args, **kwargs):
            started = time.perf_counter()
            try:
                result = source_qp(pair, u, *args, **kwargs)
            except Exception as error:
                qp_trace.append({
                    "pair": pair.name, "u_kw": np.asarray(u).copy(),
                    "seconds": time.perf_counter() - started,
                    "exception": type(error).__name__, "message": str(error),
                    "success": False,
                })
                raise
            fields = ("success", "status", "message", "z", "q_lower", "q_upper",
                      "numeric_gap", "optimality", "roundoff_guard", "acceleration")
            qp_trace.append({
                "pair": pair.name, "u_kw": np.asarray(u).copy(),
                "seconds": time.perf_counter() - started,
                **{key: result[key] for key in fields},
            })
            return result

        trace_qp.__wrapped__ = source_qp
        globals_copy["solve_pair"] = trace_qp

    clone = types.FunctionType(
        source.__code__, globals_copy, source.__name__ + "_upper_adapter",
        source.__defaults__, source.__closure__,
    )
    clone.__kwdefaults__ = copy.deepcopy(source.__kwdefaults__)
    clone.__annotations__ = source.__annotations__.copy()
    changed = [key for key in source.__globals__
               if globals_copy[key] is not source.__globals__[key]]
    expected = set()
    if upper_backend is not source_upper:
        expected.add("upper_master")
    if qp_trace is not None:
        expected.add("solve_pair")
    if set(changed) != expected:
        raise AssertionError("unexpected adapter policy change")
    if source.__globals__["upper_master"] is not source_upper:
        raise AssertionError("original upper backend changed")
    return clone, {
        "same_code_object": clone.__code__ is source.__code__,
        "keyword_defaults_copied": clone.__kwdefaults__ == source.__kwdefaults__,
        "changed_globals": changed,
        "common_pass_through_instrumentation": [] if qp_trace is None else ["solve_pair"],
        "unwrapped_qp_target": _function_identity(source_qp),
        "policy_backend_target": _function_identity(upper_backend),
        "original_global_upper_unmodified": True,
    }


def adaptive_search_shared(config, pairs, **kwargs):
    """Opt in to the shared-square upper while retaining the original search."""
    search, _ = make_search_adapter()
    return search(config, pairs, **kwargs)
