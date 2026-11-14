"""Original-form continuation control matched to the sealed reduced trial.

Reuse trial preparation and billing. Only the trial module's runtime dispatch
is replaced by the original solver; no frozen source file is modified.
"""
import argparse
from pathlib import Path
from types import SimpleNamespace

from ... import io
from ..absolute_identification_20261114 import solver as original_solver
from . import reduced_trial


def original_engine(*args, **kwargs):
    receipt, action = original_solver.plan(*args, **kwargs)
    receipt.update(engine='original_s_auxiliary_form', reformulation='none',
        engine_source=io.ref(original_solver.__file__),
        dispatch_source=io.ref(__file__),
        comparison_role='Same original180 incumbent and shared seeds, same additional180 budget as reduced trial')
    return receipt, action


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('outputs/flexibility_reaudit_long180_20261114_v1'))
    parser.add_argument('--out', type=Path, default=Path('outputs/flexibility_reaudit_original_control_20261114_v1'))
    parser.add_argument('--reduced', type=Path, default=Path('outputs/flexibility_reaudit_reduced_trial_20261114_v1'))
    parser.add_argument('--seconds', type=float, default=180.)
    parser.add_argument('--forecast', type=int, default=0)
    args = parser.parse_args()
    protocol = reduced_trial.prepare(args.run, args.out, seconds=args.seconds, forecast=args.forecast)
    counterpart = io.read(args.reduced/'protocol.json')
    for name in ('planner', 'plans', 'previous_shared_points'):
        if protocol[name]['sha256'] != counterpart[name]['sha256']:
            raise ValueError('Original/reduced initial inputs differ: '+name)
    if protocol['solver_seconds'] != counterpart['solver_seconds']:
        raise ValueError('Original/reduced additional budgets differ')
    protocol.update(phase='Single original-form additional-budget control matched to reduced trial',
        engine='original_s_auxiliary_form', engine_source=io.ref(original_solver.__file__),
        runtime_dispatch='Assign a SimpleNamespace(plan=original_engine) only to reduced_trial.reduced_solver; all source files unchanged',
        dispatch_source=io.ref(__file__), reduced_counterpart=io.ref(args.reduced/'protocol.json'),
        matched_initial_inputs_verified=True,
        proof_scope='Original buffered QCQP, same representatives, resource domain, forecast, risk, guards and original180 incumbent as reduced trial')
    io.write(args.out/'protocol.json', protocol)
    reduced_trial.reduced_solver = SimpleNamespace(plan=original_engine)
    receipt = reduced_trial.solve(args.out)
    print({key:receipt.get(key) for key in ('engine', 'status', 'source', 'cost_yuan',
        'lower_bound_yuan', 'relative_gap', 'charged_total_seconds',
        'fee_improvement_over_original_180_yuan')}, flush=True)


if __name__ == '__main__':
    main()
