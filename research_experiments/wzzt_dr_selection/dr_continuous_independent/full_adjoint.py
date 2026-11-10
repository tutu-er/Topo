"""Coefficient adjoints of the existing independent-voltage probability.

The frequency nodes, conditional distribution, and stable Fourier segments are
the same as gauss_frequency.py.  This function differentiates with respect to
each coarse/fine residual mean without allocating cells x frequencies x prices.
Only a single frequency chunk and two coefficient vectors are needed.
"""
from __future__ import annotations

import math
import time

import numpy as np

try:
    from .continuous import _truncated_fourier
    from .gauss_frequency import _dyadic_nodes
except ImportError:
    from continuous import _truncated_fourier
    from gauss_frequency import _dyadic_nodes


def _integrate_adjoint(model, threshold, nodes, weights, normalization, chunk):
    center, scale = normalization
    normalized_threshold = (model.cells * threshold - center) / scale
    probability_integral = 0.
    gradient_coarse = np.zeros(model.cells)
    gradient_fine = np.zeros(model.cells)
    peak_known_bytes = 0
    max_cf_modulus = 0.

    for start in range(0, nodes.size, chunk):
        frequencies = nodes[start:start + chunk]
        integration_weights = weights[start:start + chunk] / (math.pi * frequencies)
        raw_frequency = frequencies[None, :] / scale
        shape = (model.indices.size, frequencies.size)
        values = np.zeros(shape, complex)
        partial_coarse = np.zeros(shape, complex)
        partial_fine = np.zeros(shape, complex)

        for segment in range(3):
            mask = model.nonempty[:, segment]
            if not np.any(mask):
                continue
            interval_frequency = model.slope[mask, segment, None] * raw_frequency
            integral = _truncated_fourier(model.left[mask, segment],
                                         model.right[mask, segment], interval_frequency)
            centered_intercept = model.intercept[mask, segment] - model.means[mask]
            contribution = np.exp(1j * centered_intercept[:, None] * raw_frequency) * integral
            values[mask] += contribution
            partial_coarse[mask] += (1j * raw_frequency * model.sign_coarse[mask, segment, None]
                                    * contribution)
            partial_fine[mask] += (-model.kappa * 1j * raw_frequency
                                  * model.sign_fine[mask, segment, None] * contribution)

        # Prefix/suffix products handle genuine zeros of a local CF. Dividing
        # the total product by a local CF would produce invalid derivatives.
        before = np.vstack((np.ones((1, frequencies.size), complex),
                            np.cumprod(values[:-1], axis=0)))
        after = np.vstack((np.cumprod(values[:0:-1], axis=0)[::-1],
                           np.ones((1, frequencies.size), complex)))
        excluding_cell = before * after
        product = np.prod(values, axis=0)
        normalization_phase = np.exp(1j * raw_frequency.ravel() * (model.sum_mean - center))
        characteristic = normalization_phase * product
        max_cf_modulus = max(max_cf_modulus, float(np.abs(characteristic).max()))
        event_phase = np.exp(-1j * frequencies * normalized_threshold)
        combined_phase = event_phase * normalization_phase
        probability_integral += float(integration_weights @ np.imag(event_phase * characteristic))

        derivative = combined_phase[None, :] * excluding_cell * partial_coarse
        gradient_coarse[model.indices] += np.imag(derivative) @ integration_weights
        derivative = combined_phase[None, :] * excluding_cell * partial_fine
        gradient_fine[model.indices] += np.imag(derivative) @ integration_weights

        fixed = ~model.random
        if np.any(fixed):
            constant_derivative = float(integration_weights @ np.imag(
                event_phase * characteristic * 1j * raw_frequency.ravel()))
            gradient_coarse[fixed] += np.sign(model.coarse[fixed]) * constant_derivative
            gradient_fine[fixed] -= model.kappa * np.sign(model.fine[fixed]) * constant_derivative

        # A lower accounting of the live NumPy work arrays, not process RSS or
        # memory inside SciPy. Its shape remains O(cells x chunk), independent
        # of the number of price variables.
        arrays = (values, partial_coarse, partial_fine, before, after,
                  excluding_cell, derivative)
        peak_known_bytes = max(peak_known_bytes, sum(array.nbytes for array in arrays))

    probability = .5 + probability_integral
    if (not np.isfinite(probability) or not np.isfinite(gradient_coarse).all()
            or not np.isfinite(gradient_fine).all()):
        raise FloatingPointError('nonfinite adjoint probability or coefficient derivative')
    return probability, gradient_coarse, gradient_fine, max_cf_modulus, peak_known_bytes


def probability_with_adjoint(model, threshold, *, cutoff=512., order=32,
                             normalization=None, refinements=0, tolerance=1e-7,
                             gradient_tolerance=1e-7, chunk=64):
    """Return a finite-quadrature probability and its two residual-mean gradients.

    Inputs use the existing IndependentResiduals object. threshold has the same
    mean-loss units as its contributions. normalization is (sum_center,
    sum_scale), fixed across prices when differentiating the finite function.

    gradient_coarse[i] and gradient_fine[i] differentiate with respect to the
    i-th candidate residual mean, with sigma held fixed. Refinement is a
    diagnostic. The analytic tail bound excludes quadrature and FP errors.
    """
    began = time.perf_counter()
    if (not np.isfinite(threshold) or not np.isfinite(cutoff) or cutoff <= 0
            or order < 2 or int(order) != order or refinements < 0
            or int(refinements) != refinements or tolerance <= 0
            or gradient_tolerance <= 0 or chunk <= 0 or int(chunk) != chunk):
        raise ValueError('finite threshold and valid positive integration controls required')
    if model.sum_sd == 0:
        margin = model.sum_mean / model.cells - threshold
        return dict(probability=float(margin > 0),
                    gradient_coarse=np.zeros(model.cells), gradient_fine=np.zeros(model.cells),
                    gradient_valid=bool(margin != 0), numerical_pass=True,
                    seconds=time.perf_counter() - began, certified_error=0.,
                    method='deterministic_strict_event', trace=[], known_workspace_peak_bytes=0)

    center, scale = ((model.sum_mean, model.sum_sd) if normalization is None
                     else tuple(map(float, normalization)))
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('finite center and positive scale required')
    trace = []
    previous_coarse = previous_fine = None
    probability_difference = gradient_difference = None
    tail = model.tail_bound(cutoff, scale)
    peak_bytes = 0
    for level in range(refinements + 1):
        current_order = int(order) * 2**level
        nodes, weights, endpoints = _dyadic_nodes(cutoff, current_order)
        probability, coarse, fine, modulus, workspace = _integrate_adjoint(
            model, threshold, nodes, weights, (center, scale), int(chunk))
        peak_bytes = max(peak_bytes, workspace)
        if trace:
            probability_difference = abs(probability - trace[-1]['probability'])
            gradient_difference = float(max(np.max(np.abs(coarse - previous_coarse)),
                                            np.max(np.abs(fine - previous_fine))))
        trace.append(dict(order=current_order, nodes=int(nodes.size), probability=probability,
                          refinement_difference=probability_difference,
                          gradient_refinement_difference=gradient_difference,
                          max_cf_modulus=modulus))
        previous_coarse, previous_fine = coarse, fine
        if (probability_difference is not None and probability_difference < tolerance
                and gradient_difference < gradient_tolerance):
            break

    fixed_kink = np.any((~model.random) & ((model.coarse == 0) | (model.fine == 0)))
    converged = (probability_difference is not None and probability_difference < tolerance
                 and gradient_difference < gradient_tolerance)
    return dict(probability=probability, gradient_coarse=coarse, gradient_fine=fine,
                gradient_valid=bool(not fixed_kink),
                gradient_scope='fixed-node conditional probability; sigma and normalization fixed',
                moments=model.moments(), normalization=[center, scale],
                cutoff=float(cutoff), order=trace[-1]['order'], nodes=trace[-1]['nodes'],
                intervals=endpoints, trace=trace, analytic_tail_bound=tail,
                refinement_difference=probability_difference,
                gradient_refinement_difference=gradient_difference,
                numerical_pass=bool(converged and tail < tolerance and 0 <= probability <= 1),
                certified_error=None, seconds=time.perf_counter() - began,
                method='deterministic_fourier_coefficient_adjoint',
                known_workspace_peak_bytes=peak_bytes,
                largest_cell_frequency_shape=[int(model.indices.size), min(int(chunk), int(nodes.size))],
                error_scope='IBP tail is bounded; quadrature and FP error uncertified')
