"""Exact four-terminal B checks; no identifier, solver, data or simulations.

This verifies a declared seven-coefficient family, not the full laminar domain.
Usage: python exact_algebra_audit.py DIRECTORY
"""
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import hashlib
import json
import platform
import sys


def serial(value):
    if isinstance(value, F):
        return str(value)
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def atom(support):
    return [[F(i in support and j in support) for j in range(4)] for i in range(4)]


def weighted(weights, bases):
    return [[sum(w * b[i][j] for w, b in zip(weights, bases))
             for j in range(4)] for i in range(4)]


def mv(matrix, vector):
    return [sum(a * b for a, b in zip(row, vector)) for row in matrix]


def rank(rows):
    a = [[F(v) for v in row] for row in rows]
    r = 0
    for c in range(len(a[0])):
        pivot = next((k for k in range(r, len(a)) if a[k][c]), None)
        if pivot is None:
            continue
        a[r], a[pivot] = a[pivot], a[r]
        den = a[r][c]
        a[r] = [v / den for v in a[r]]
        for k in range(len(a)):
            if k != r:
                factor = a[k][c]
                a[k] = [v - factor * w for v, w in zip(a[k], a[r])]
        r += 1
        if r == len(a):
            break
    return r


def features(vector, bases):
    columns = [mv(b, vector) for b in bases]
    return [[column[i] for column in columns] for i in range(4)]


def main():
    root = Path(sys.argv[1]).resolve()
    protocol_path = root / 'protocol.json'
    protocol = json.loads(protocol_path.read_text(encoding='utf-8-sig'))
    sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    assert sha == protocol['audit_source_sha256'], 'source drift'
    output = root / 'result.json'
    if output.exists():
        raise FileExistsError('completed exact audit must not be overwritten')
    one = [F(1)] * 4
    plus = [F(1), F(0), F(1), F(0)]
    minus = [F(1), F(0), F(-1), F(0)]
    next_direction = [F(0), F(1), F(-1), F(0)]
    # coefficient order: b, a1, a2, a3, a4, c12, d34
    bases = [atom(range(4))] + [atom([i]) for i in range(4)] + [atom([0, 1]), atom([2, 3])]
    checks = []
    def check(name, ok):
        checks.append({'name': name, 'passed': bool(ok)})
        if not ok:
            raise AssertionError(name)
    direction = weighted([F(1), 0, 0, 0, 0, -2, -2], bases)
    check('ancestor_child_compensation_cancels_common_history', mv(direction, one) == [0] * 4)
    check('same_phase_preserves_compensation', mv(direction, plus) == [0] * 4)
    check('opposed_phase_breaks_compensation', mv(direction, minus) == [-2, -2, 2, 2])
    positive = [F(3, 10), *([F(2, 5)] * 4), F(1, 2), F(1, 2)]
    alternative = [v + F(1, 4) * d for v, d in zip(positive, [1, 0, 0, 0, 0, -2, -2])]
    check('positive_to_zero_children_alternative_is_nonnegative', min(alternative) >= 0 and alternative[-2:] == [0, 0])
    k = weighted(positive, bases)
    k_alt = weighted(alternative, bases)
    check('positive_and_zero_children_share_history', mv(k, one) == mv(k_alt, one))
    check('positive_and_zero_children_share_same_phase', mv(k, plus) == mv(k_alt, plus))
    check('positive_children_remote_outputs', mv(k, minus)[1::2] == [positive[-2], -positive[-1]])
    check('history_rank_four', rank(features(one, bases)) == 4)
    check('history_and_same_rank_six', rank(features(one, bases) + features(plus, bases)) == 6)
    check('history_and_opposed_rank_seven', rank(features(one, bases) + features(minus, bases)) == 7)
    check('single_point_does_not_identify_both_blocks', rank(features(one, bases) + features([1, 0, 0, 0], bases)) == 6)
    # A different, legal laminar family defeats a global conclusion at c12=0.
    # Add E={1,3,4}; reduce root by 3e/4; increase leaf2 by 3e.
    e = F(1, 25)
    full_zero_direction = weighted([e, -F(3, 4) * e, 3 * e],
                                  [atom([0, 2, 3]), atom(range(4)), atom([1])])
    check('zero_boundary_alternative_preserves_history', mv(full_zero_direction, one) == [0] * 4)
    check('zero_boundary_alternative_preserves_opposed_probe', mv(full_zero_direction, minus) == [0] * 4)
    check('second_probe_breaks_this_alternative', mv(full_zero_direction, next_direction) == [-e, 3 * e, -e, -e])
    check('zero_boundary_reweighted_root_stays_nonnegative', F(3, 10) - F(3, 4) * e >= 0)
    masks = list(product((0, 1), repeat=2))
    patterns = {}
    for name, relative_sign in [('same', 1), ('opposed', -1)]:
        records = []
        for first, second in product(masks, repeat=2):
            rows = features(one, bases)
            for z in (first, second):
                rows += features([z[0], 0, relative_sign * z[1], 0], bases)
            records.append({'states': [first, second], 'rank': rank(rows)})
        count = sum(row['rank'] == 7 for row in records)
        check(name + '_exact_full_rank_count', count == {'same': 6, 'opposed': 9}[name])
        patterns[name] = {'records': records, 'sufficient_event_probability': F(count, 16)}
    common = {}
    for name in ('same', 'opposed'):
        records = patterns[name]['records']
        subset = [v for v in records if all(a == b for a, b in v['states'])]
        probability = F(sum(v['rank'] == 7 for v in subset), len(subset))
        check(name + '_common_participation_probability', probability == {'same': F(0), 'opposed': F(3, 4)}[name])
        common[name] = probability
    wave = [F(-1, 5)] * 4 + [F(1, 5)] * 4
    check('same_peak_and_energy', max(map(abs, wave)) == F(1, 5)
          and sum(max(-v, 0) for v in wave) / 4 == F(1, 5) and sum(wave) == 0)
    for orientation in (1, -1):
        power = [F(1, 5) + orientation * v for v in wave]
        energy = [sum(power[:i + 1]) / 4 for i in range(8)]
        check('device_power_energy_orientation_' + str(orientation),
              min(power) >= 0 and max(power) <= F(2, 5)
              and min(energy) >= 0 and max(energy) <= F(2, 5) and energy[-1] == F(2, 5))
    prices = {'low_pair': [[1, 0, 1, 0]] * 2,
              'high_pair_once': [[2, 0, 2, 0], [0, 0, 0, 0]],
              'high_rotation': [[2, 0, 0, 0], [0, 0, 2, 0]]}
    payment = {}
    for name, rows in prices.items():
        worst = F(2, 5) * sum(map(sum, rows))
        expected = F(2, 5) * sum(F(c) * {0: F(0), 1: F(1, 2), 2: F(3, 4)}[c] for row in rows for c in row)
        check(name + '_budget', worst == F(8, 5))
        check(name + '_expected_payment', expected == (F(4, 5) if name == 'low_pair' else F(6, 5)))
        payment[name] = {'four_event_worst': worst, 'four_event_expected': expected}
    # Independent readiness counts for two high-price single calls / one pair.
    check('high_price_two_required_acceptances', F(3, 4) ** 2 == F(9, 16))
    result = {'status': 'complete', 'scope': 'exact conditional algebra only; no identifier or DR performance',
              'checks': checks, 'check_count': len(checks), 'patterns': patterns,
              'common_participation_sufficient_events': common, 'payments': payment,
              'LP_calls': 0, 'identifier_calls': 0, 'physical_simulations': 0,
              'history_or_future_data_loaded': False,
              'environment': {'python': sys.version, 'platform': platform.platform()},
              'audit_source_sha256': sha,
              'protocol_sha256': hashlib.sha256(protocol_path.read_bytes()).hexdigest()}
    output.write_text(json.dumps(serial(result), indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': result['status'], 'checks': len(checks), 'LP_calls': 0,
                      'same': '6/16', 'opposed': '9/16', 'output': str(output)}))


if __name__ == '__main__':
    main()
