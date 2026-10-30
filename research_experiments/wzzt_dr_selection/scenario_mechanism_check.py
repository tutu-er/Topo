"""Exact arithmetic for scenario reasoning; no identification experiment."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
from fractions import Fraction as F
from pathlib import Path
import platform
import sys


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def matvec(a, x):
    return [sum(v*w for v, w in zip(row, x)) for row in a]


def atom(z):
    return [[i*j for j in z] for i in z]


def kernel(diagonal, weight, z):
    zz = atom(z)
    return [[(diagonal[i] if i == j else F(0)) + weight*zz[i][j]
             for j in range(3)] for i in range(3)]


def event_probability(success):
    """Enumerate all four independent events using rational probabilities."""
    total = F(0)
    for bits in itertools.product((0, 1), repeat=4):
        mass = F(1)
        for bit in bits:
            mass *= success if bit else 1-success
        if any(bits):
            total += mass
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError('Existing arithmetic record retained; do not overwrite')
    cfg = json.loads(args.protocol.read_text(encoding='utf-8'))
    for path, digest in cfg['source_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('Frozen source hash mismatch: ' + path)
    cases = []
    d, delta = F(1, 10), F(1, 1000)
    for c, t, s in itertools.product((F(0), F(1, 100), F(3, 50)),
                                     (F(0), d/4, d/2), (F(1, 50), F(2, 50))):
        a = kernel([d]*3, c, [1, 1, 0])
        bdiag = [d+2*c, d+2*c-2*t, d-2*t]
        assert min(bdiag) >= 0
        b = kernel(bdiag, t, [0, 1, 1])
        p = [s]*3
        assert matvec(a, p) == matvec(b, p) == [s*(d+2*c), s*(d+2*c), s*d]
        minus, plus = [s, s-delta, s], [s, s+delta, s]
        ya, yb = [matvec(m, minus) for m in (a, b)], [matvec(m, plus) for m in (a, b)]
        assert yb[0][0]-ya[0][0] == 2*c*delta
        assert yb[1][0]-ya[1][0] == 0
        # The midpoint achieves the universal two-point constant-fit lower bound.
        midpoint = (ya[0][0]+yb[0][0])/2
        residual = abs(ya[0][0]-midpoint)+abs(yb[0][0]-midpoint)
        assert residual == 2*c*delta
        cases.append(dict(c=str(c), t=str(t), s=str(s), history_equal=True,
                          zero_weight=(c == 0 or t == 0 or min(bdiag) == 0),
                          two_point_row1_lower_bound=str(residual)))
    single, sync = event_probability(F(3, 4)), event_probability(F(1, 2))
    assert single == F(255, 256) and sync == F(15, 16)
    p = F(1, 2)
    assert 2*p*(1-p) == F(1, 2)
    single_worst = sync_worst = 4*F(1, 5)*2
    single_expected = single_worst*F(3, 4)
    sync_expected = sync_worst*F(1, 2)
    assert single_worst == F(8, 5)
    result = dict(status='complete', scope=cfg['scope'], protocol_sha256=sha(args.protocol),
                  python=sys.version, platform=platform.platform(), parameter_examples=cases,
                  parameter_example_count=len(cases),
                  probability_checks=dict(single_relative_at_least_once=str(single),
                      independent_sync_relative_at_least_once=str(sync),
                      common_sync_relative_at_least_once='0',
                      equal_mean_relative_response='0', probability_is_topology_recovery=False),
                  costs=dict(single_worst=str(single_worst), sync_worst=str(sync_worst),
                             single_expected=str(single_expected), sync_expected=str(sync_expected)),
                  solver_calls=0, AC_calls=0, topology_algorithm_calls=0,
                  future_or_household_files_read=0, new_recovery_result=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ('status', 'parameter_example_count', 'probability_checks', 'costs', 'solver_calls', 'new_recovery_result')}))


if __name__ == '__main__':
    main()
