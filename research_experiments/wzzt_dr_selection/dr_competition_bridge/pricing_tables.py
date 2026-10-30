"""Exact price-to-participation tables; no fitting, solver, or future-data reads.

Events are ordered train row0, train row1, validation row0, validation row1.
The two validation events repeat prices, but use independent participation.
Node position j is bit 1<<j in a pattern key. Prices 0/1/2 have probabilities
0, 1/2, 3/4. Given participation, power is independent of the price amount.
"""
from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
import hashlib
import itertools
import math
from typing import Mapping, Callable

import numpy as np


PROBABILITIES = (Fraction(0), Fraction(1, 2), Fraction(3, 4))
DEFAULT_ENERGY_KWH = Fraction(1, 5)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{name} must be an integer')
    value = int(value)
    if value < minimum:
        raise ValueError(f'{name} must be at least {minimum}')
    return value


def _fraction(value):
    return value if isinstance(value, Fraction) else Fraction(str(value))


def _menu(menu):
    a = np.asarray(menu)
    if a.ndim != 2 or a.shape[0] != 2 or a.shape[1] == 0:
        raise ValueError('menu must have shape (2,n_nodes), n_nodes>0')
    if not np.isin(a, (0, 1, 2)).all():
        raise ValueError('prices must be 0,1,2')
    return a.astype(np.uint8, copy=False)


def _law(dependence):
    if dependence == 'independent':
        return 'local'
    if dependence not in ('local', 'common'):
        raise ValueError('dependence must be local or common')
    return dependence


def enumerate_menus(n_nodes, max_units=4):
    """All menus with prices 0/1/2 and sum<=max_units, including zero.

    The bound counts price units over two training rows; their prices are reused
    for validation. Default max_units=4 is worst payment <=1.6 currency units.
    """
    n = _integer(n_nodes, 'n_nodes', 1)
    units = min(_integer(max_units, 'max_units'), 4*n)
    width = 2*n
    flat = [0]*width
    rows = []

    def visit(position, left):
        if position == width:
            rows.append(tuple(flat))
            return
        for price in range(min(2, left)+1):
            flat[position] = price
            visit(position+1, left-price)

    visit(0, units)
    return np.asarray(rows, dtype=np.uint8).reshape(-1, 2, n)


def single_node_menus(n_nodes, max_units=4):
    """All feasible menus supported on at most one node, zero included once."""
    n = _integer(n_nodes, 'n_nodes', 1)
    local = enumerate_menus(1, max_units)
    rows = [np.zeros((2, n), dtype=np.uint8)]
    for j in range(n):
        for small in local:
            if not small.any():
                continue
            menu = np.zeros((2, n), dtype=np.uint8)
            menu[:, j] = small[:, 0]
            rows.append(menu)
    return np.stack(rows)


@lru_cache(maxsize=None)
def _event_keys(prices, dependence):
    """Tuple of (one-event node-bit key, exact probability)."""
    probs = [PROBABILITIES[p] for p in prices]
    if dependence == 'common':
        cut = sorted(set((Fraction(0), Fraction(1), *probs)))
        mass_by_key = {}
        for left, right in zip(cut, cut[1:]):
            if right == left:
                continue
            midpoint = (left+right)/2
            key = sum(1 << j for j, p in enumerate(probs) if midpoint < p)
            mass_by_key[key] = mass_by_key.get(key, Fraction(0))+(right-left)
        return tuple(sorted(mass_by_key.items()))
    active = [j for j, p in enumerate(probs) if p]
    rows = []
    for bits in itertools.product((0, 1), repeat=len(active)):
        key, mass = 0, Fraction(1)
        for j, bit in zip(active, bits):
            mass *= probs[j] if bit else 1-probs[j]
            if bit:
                key |= 1 << j
        rows.append((key, mass))
    return tuple(sorted(rows))


def enumerate_training_outcomes(menu, dependence='local'):
    """Return ((first-event bits,second-event bits), exact probability)."""
    a, law = _menu(menu), _law(dependence)
    one = _event_keys(tuple(int(v) for v in a[0]), law)
    two = _event_keys(tuple(int(v) for v in a[1]), law)
    return [(tuple((ka, kb)), pa*pb) for (ka, pa), (kb, pb) in itertools.product(one, two)]


def enumerate_outcome_keys(menu, dependence='local'):
    """Return four-event bit keys and exact joint probabilities.

    Train and validation are independent given prices, even under common law.
    Common law only couples nodes *within* each event through a fresh U.
    """
    pairs = enumerate_training_outcomes(menu, dependence)
    return [(train+validation, pt*pv)
            for (train, pt), (validation, pv) in itertools.product(pairs, pairs)]


def masks_from_key(key, n_nodes):
    n = _integer(n_nodes, 'n_nodes', 1)
    if len(key) not in (2, 4):
        raise ValueError('key must contain 2 training events or all 4 events')
    out = np.zeros((len(key), n), dtype=np.uint8)
    for i, value in enumerate(key):
        value = _integer(value, 'event bit key')
        if value >= 1 << n:
            raise ValueError('key contains a bit outside n_nodes')
        for j in range(n):
            out[i, j] = (value >> j) & 1
    return out


def outcome_key(mask):
    a = np.asarray(mask)
    if a.ndim != 2 or a.shape[0] not in (2, 4) or a.shape[1] == 0:
        raise ValueError('mask must have shape (2,n) or (4,n)')
    if not np.isin(a, (0, 1)).all():
        raise ValueError('participation values must be 0 or 1')
    return tuple(sum(int(value) << j for j, value in enumerate(row)) for row in a)


def enumerate_event_outcomes(menu, dependence='local'):
    """Return [(mask shape[4,n_nodes], Fraction probability), ...]."""
    a = _menu(menu)
    return [(masks_from_key(key, a.shape[1]), mass)
            for key, mass in enumerate_outcome_keys(a, dependence)]


def unique_training_masks(menus, dependence='local'):
    """Reachable two-event keys, deduplicated before fitting LPs."""
    a = np.asarray(menus)
    if a.ndim != 3 or a.shape[1] != 2:
        raise ValueError('menus must have shape (M,2,n_nodes)')
    keys = set()
    for menu in a:
        keys.update(key for key, _ in enumerate_training_outcomes(menu, dependence))
    return sorted(keys)


def payment(menu, energy_kwh=DEFAULT_ENERGY_KWH):
    """Exact worst/expected payment for all four events, not per-event budget."""
    a = _menu(menu)
    energy = _fraction(energy_kwh)
    if energy < 0:
        raise ValueError('energy_kwh must be nonnegative')
    worst = 2*energy*sum(int(p) for p in a.flat)
    expected = 2*energy*sum((int(p)*PROBABILITIES[int(p)] for p in a.flat), Fraction(0))
    return {'worst': worst, 'expected': expected}


def actual_payment(menu, key, energy_kwh=DEFAULT_ENERGY_KWH):
    a = _menu(menu)
    mask = masks_from_key(key, a.shape[1])
    if mask.shape[0] != 4:
        raise ValueError('actual payment requires all four events')
    energy = _fraction(energy_kwh)
    if energy < 0:
        raise ValueError('energy_kwh must be nonnegative')
    return energy*sum(int(a[i % 2, j])*int(mask[i, j])
                      for i in range(4) for j in range(a.shape[1]))


def expected_loss(loss_table: Mapping | Callable, menu, dependence='local'):
    """Probability integration only. Missing/unresolved table entries raise."""
    weighted = []
    for key, mass in enumerate_outcome_keys(menu, dependence):
        value = loss_table(key) if callable(loss_table) else loss_table[key]
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f'nonfinite conditional loss for {key}')
        weighted.append(float(mass)*value)
    return math.fsum(weighted)


def menu_tie_key(menu, tie_seed=0):
    a = _menu(menu)
    prefix = f'{tie_seed}|{a.shape[1]}|'.encode('utf-8')
    return hashlib.sha256(prefix+a.tobytes(order='C')).hexdigest()


def score_menus(menus, loss_by_four_mask, dependence='local', *, tie_seed=0,
                loss_tol=1e-12, energy_kwh=DEFAULT_ENERGY_KWH):
    """Select by loss tolerance, exact expected fee, seeded hash, then index.

    This function does not enforce a payment budget beyond the passed menu set.
    Use enumerate_menus(max_units=4) for the frozen <=1.6 worst-payment domain.
    """
    a = np.asarray(menus)
    if a.ndim != 3 or a.shape[1] != 2 or len(a) == 0:
        raise ValueError('menus must be a nonempty (M,2,n_nodes) array')
    if not math.isfinite(loss_tol) or loss_tol < 0:
        raise ValueError('loss_tol must be finite and nonnegative')
    law = _law(dependence)
    records, fees = [], []
    for index, menu in enumerate(a):
        cost = payment(menu, energy_kwh)
        fees.append(cost['expected'])
        records.append(dict(menu_id=index, expected_loss=expected_loss(loss_by_four_mask, menu, law),
                            expected_fee=float(cost['expected']), worst_fee=float(cost['worst']),
                            tie_key=menu_tie_key(menu, tie_seed)))
    best = min(r['expected_loss'] for r in records)
    eligible = [r['menu_id'] for r in records if r['expected_loss'] <= best+loss_tol]
    selected = min(eligible, key=lambda i:(fees[i], records[i]['tie_key'], i))
    return dict(selected_index=selected, selected_menu=a[selected].copy(),
                best_expected_loss=best, loss_tied_indices=eligible, records=records,
                dependence=law, loss_tol=loss_tol, tie_seed=tie_seed)
