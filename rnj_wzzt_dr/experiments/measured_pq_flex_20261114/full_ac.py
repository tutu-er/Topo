"""Full-state cache of the existing balanced radial PQ AC equations.

This extends the original backward/forward sweep to retain all buses and
branches. The cache independently compares terminal voltages to the original
solver and the already sealed physical inputs. Outputs are evaluator-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np

from topology_identification_standalone import data as backend
from ... import io
from ..measured_pq_20261114.physical import _network_hash


def solve(network, labels, p, q, root, *, tolerance=1e-10, max_iterations=200):
    p, q, root = np.asarray(p, float), np.asarray(q, float), np.asarray(root, float)
    labels = tuple(map(int, labels))
    if p.shape != q.shape or p.shape != (root.size, len(labels)) or root.ndim != 1:
        raise ValueError("P/Q/root dimensions differ")
    if not all(np.isfinite(x).all() for x in (p, q, root)) or np.any(root <= 0):
        raise ValueError("Finite P/Q and positive finite root voltage required")
    order, parent, incoming = backend._radial_order(network)
    position = {bus: i for i, bus in enumerate(order)}
    terminal = np.array([position[label] for label in labels])
    source = np.array([position[parent[c]] for c in order[1:]])
    sink = np.arange(1, len(order))
    load = np.zeros((root.size, len(order)), dtype=complex)
    load[:, terminal] = p+1j*q
    zbase = network['base_kv']**2/network['base_mva']
    z = np.array([complex(incoming[c]['r_ohm'], incoming[c]['x_ohm'])/zbase for c in order[1:]])
    if np.any(np.abs(z) == 0):
        raise ValueError("Full-state branch reconstruction requires nonzero impedances")
    voltage = np.repeat(root[:, None], len(order), axis=1).astype(complex)
    for iteration in range(max_iterations):
        if np.any(np.abs(voltage) < .2):
            raise RuntimeError("AC iteration reached collapsed voltage")
        current = np.conj(load/voltage)
        for child in reversed(order[1:]):
            current[:, position[parent[child]]] += current[:, position[child]]
        updated = voltage.copy()
        updated[:, 0] = root
        for k, child in enumerate(order[1:]):
            updated[:, k+1] = updated[:, source[k]]-z[k]*current[:, k+1]
        error = float(np.max(np.abs(updated-voltage)))
        voltage = updated
        if error <= tolerance:
            break
    else:
        raise RuntimeError("AC did not converge")
    # Reconstruct branch flow from Ohm's law, then check nodal current balance.
    branch_i = (voltage[:, source]-voltage[:, sink])/z
    sending = voltage[:, source]*branch_i.conj()
    receiving = voltage[:, sink]*branch_i.conj()
    losses = z*np.abs(branch_i)**2
    out_i = np.zeros_like(voltage)
    for k, a in enumerate(source):
        out_i[:, a] += branch_i[:, k]
    mismatch = branch_i-out_i[:, 1:]-np.conj(load[:, 1:]/voltage[:, 1:])
    root_power = voltage[:, 0]*out_i[:, 0].conj()+load[:, 0]
    base = network['base_kw']
    balance = root_power-load.sum(axis=1)-losses.sum(axis=1)
    state = dict(bus_ids=np.array(order), terminal_labels=np.array(labels), terminal_positions=terminal,
        branch_from=np.array([parent[c] for c in order[1:]]), branch_to=np.array(order[1:]),
        voltage_complex_pu=voltage, voltage_magnitude_pu=np.abs(voltage), voltage_angle_rad=np.angle(voltage),
        terminal_voltage_pu=np.abs(voltage[:, terminal]), root_voltage_pu=root,
        specified_load_kw=load.real*base, specified_load_kvar=load.imag*base,
        branch_current_complex_pu=branch_i,
        branch_current_A=np.abs(branch_i)*base/(np.sqrt(3)*network['base_kv']),
        branch_sending_kw=sending.real*base, branch_sending_kvar=sending.imag*base,
        branch_receiving_kw=receiving.real*base, branch_receiving_kvar=receiving.imag*base,
        branch_loss_kw=losses.real*base, branch_loss_kvar=losses.imag*base,
        root_supply_kw=root_power.real*base, root_supply_kvar=root_power.imag*base,
        power_balance_residual_kw=balance.real*base, power_balance_residual_kvar=balance.imag*base)
    checks = dict(iterations=iteration+1, maximum_voltage_update=error,
        max_KCL_residual_pu=float(np.max(np.abs(mismatch))),
        max_power_balance_residual_kVA=float(np.max(np.abs(balance))*base),
        max_branch_loss_identity_residual_kVA=float(np.max(np.abs(sending-receiving-losses))*base),
        min_terminal_voltage_pu=float(state['terminal_voltage_pu'].min()),
        max_terminal_voltage_pu=float(state['terminal_voltage_pu'].max()),
        thermal_rating_check=False, branch_orientation='away from root; negative flow means reverse')
    if checks['max_KCL_residual_pu'] > 1e-7 or checks['max_power_balance_residual_kVA'] > 1e-5:
        raise ArithmeticError("Full AC state balance check failed")
    return state, checks


def cache(index, output):
    index, output = Path(index).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve previous AC cache")
    catalog = io.read(index)
    inputs = catalog['inputs']
    context_ref = catalog['resource_context']
    if io.ref(context_ref['path'])['sha256'] != context_ref['sha256']:
        raise ValueError('Source resource-context receipt changed')
    physical_refs = {(r['case'], r['start_date']): r['existing_physical']
                     for r in io.read(context_ref['path'])['rows']}
    output.mkdir(parents=True)
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(), input_index=io.ref(index),
        source=io.ref(__file__), original_AC_source=io.ref(backend.__file__), count=len(inputs),
        purpose='Evaluator-only baseline AC cache; no new loads, noise, fitting or procurement',
        units='Three-phase kW/kvar; per-unit balanced voltages; line RMS A; no shunts in original cases',
        environment=io.environment())
    io.write(output/'protocol.json', protocol)
    rows = []
    for item in inputs:
        began = perf_counter()
        directory = Path(item['files']['train.npz']['path']).parent
        original = directory/'private_physical.npz'
        receipt = io.read(directory/'physical_receipt.json')
        network = backend.load_case(item['case'])
        if io.ref(original)['sha256'] != physical_refs[item['case'], item['start_date']]['sha256']:
            raise ValueError('Sealed physical input changed')
        if _network_hash(network) != receipt['original_network_sha256']:
            raise ValueError('Physical network changed')
        with np.load(original, allow_pickle=False) as z:
            p, q, root, labels = z['P'], z['Q'], z['root'], z['labels']
            state, checks = solve(network, labels, p, q, root)
            native, _ = backend._ac_voltages(network, labels, p, q, root)
            checks['max_original_solver_difference_pu'] = float(np.max(np.abs(native-state['terminal_voltage_pu'])))
            checks['max_sealed_voltage_difference_pu'] = float(np.max(np.abs(z['V']-state['terminal_voltage_pu'])))
            if max(checks['max_original_solver_difference_pu'], checks['max_sealed_voltage_difference_pu']) > 1e-12:
                raise ArithmeticError('Cache differs from original physical solver or sealed input')
            state['timestamps_utc'] = z['timestamps']
        dest = output/(item['case']+'__'+item['start_date']+'.npz')
        np.savez_compressed(dest, **state)
        rows.append(dict(case=item['case'], start_date=item['start_date'], points=len(root),
            original=io.ref(original), network_hash=receipt['original_network_sha256'],
            cache=io.ref(dest), checks=checks, seconds=perf_counter()-began))
    result = dict(status='complete', protocol=io.ref(output/'protocol.json'), inputs=len(rows),
                  total_time_points=sum(r['points'] for r in rows), rows=rows)
    io.write(output/'index.json', result)
    io.write(output/'complete.json', dict(index=io.ref(output/'index.json')))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = cache(args.index, args.out)
    print({k:v for k,v in result.items() if k != 'rows'})
