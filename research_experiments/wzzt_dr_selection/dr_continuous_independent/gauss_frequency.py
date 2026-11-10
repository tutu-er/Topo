"""Dyadic Gauss--Legendre inversion of the unchanged residual-cell model.

This uses the same characteristic function and the same frequency cutoff as
the uniform Simpson implementation. It changes quadrature nodes only. Order
refinement is a numerical diagnostic, not a rigorous quadrature error bound.
"""
from __future__ import annotations

import math
import time

import numpy as np
from numpy.polynomial.legendre import leggauss


def _dyadic_nodes(cutoff, order):
    endpoints = [0., min(1., cutoff)]
    while endpoints[-1] < cutoff:
        endpoints.append(min(cutoff, 2 * endpoints[-1]))
    standard_nodes, standard_weights = leggauss(order)
    nodes, weights = [], []
    for left, right in zip(endpoints[:-1], endpoints[1:]):
        radius = (right - left) / 2
        nodes.append((left + right) / 2 + radius * standard_nodes)
        weights.append(radius * standard_weights)
    return np.concatenate(nodes), np.concatenate(weights), endpoints


def gauss_probability(model, threshold, *, cutoff=512., order=16,
                      refinements=2, tolerance=1e-7, normalization=None,
                      daC=None, daF=None, chunk=128, gradient_tolerance=1e-5):
    """Integrate the existing characteristic function without a uniform grid.

    With order=16, refinements=2, check orders 16 and 32, using 64 only if the
    previous difference exceeds tolerance. For fixed-node direction tests use
    the same normalization and order, refinements=0, at every parameter point.
    The optional coefficient directions use the existing analytic CF gradient.
    """
    began = time.perf_counter()
    if (not np.isfinite(threshold) or not np.isfinite(cutoff) or cutoff <= 0
            or order < 2 or int(order) != order or refinements < 0
            or int(refinements) != refinements or tolerance <= 0):
        raise ValueError('finite threshold, positive cutoff/tolerance and valid quadrature orders required')
    if model.sum_sd == 0:
        return dict(probability=float(model.sum_mean / model.cells > threshold), gradient=None,
                    seconds=time.perf_counter()-began, method='deterministic',
                    numerical_pass=True, certified_error=0., trace=[])
    center, scale = ((model.sum_mean, model.sum_sd) if normalization is None
                     else tuple(map(float, normalization)))
    if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
        raise ValueError('finite center and positive normalization scale required')
    directions = None
    if daC is not None or daF is not None:
        dc, df = np.asarray(daC, float), np.asarray(daF, float)
        if dc.ndim == 1:
            dc = dc[:, None]
        if df.ndim == 1:
            df = df[:, None]
        if (dc.shape != df.shape or dc.shape[0] != model.cells
                or not np.isfinite(dc).all() or not np.isfinite(df).all()):
            raise ValueError('finite coefficient directions (cells,directions) required')
        directions = dc, df
    normalized_threshold = (model.cells * threshold - center) / scale
    tail = model.tail_bound(cutoff, scale)
    trace = []
    result_gradient = None
    difference = None
    for level in range(refinements + 1):
        current_order = int(order) * 2**level
        nodes, weights, endpoints = _dyadic_nodes(cutoff, current_order)
        cf, cf_gradient = model._characteristic_and_gradient(nodes, chunk, (center, scale), directions)
        phase = np.exp(-1j * nodes * normalized_threshold)
        integrand = np.imag(phase * cf) / nodes
        probability = float(.5 + weights @ integrand / math.pi)
        if directions is not None:
            integrand_gradient = np.imag(phase[:, None] * cf_gradient) / nodes[:, None]
            result_gradient = weights @ integrand_gradient / math.pi
        difference = None if not trace else abs(probability - trace[-1]['probability'])
        gradient_difference = None
        if result_gradient is not None and trace:
            gradient_difference = float(np.max(np.abs(result_gradient - np.asarray(trace[-1]['gradient']))))
        trace.append(dict(level=level, order=current_order, nodes=int(nodes.size),
                          probability=probability, refinement_difference=difference,
                          gradient=None if result_gradient is None else result_gradient.tolist(),
                          gradient_refinement_difference=gradient_difference,
                          max_cf_modulus=float(np.abs(cf).max())))
        gradient_converged = (directions is None or (gradient_difference is not None
                              and gradient_difference < gradient_tolerance))
        if difference is not None and difference < tolerance and gradient_converged:
            break
    in_range = bool(0 <= probability <= 1)
    converged = difference is not None and difference < tolerance and gradient_converged
    return dict(probability=probability,
                gradient=None if result_gradient is None else result_gradient.tolist(),
                moments=model.moments(), normalization=[center, scale],
                normalized_threshold=float(normalized_threshold), cutoff=float(cutoff),
                order=trace[-1]['order'], nodes=trace[-1]['nodes'], intervals=endpoints,
                trace=trace, analytic_tail_bound=tail, refinement_difference=difference,
                gradient_refinement_difference=gradient_difference,
                probability_in_unit_interval=in_range,
                numerical_pass=bool(converged and tail < tolerance and in_range),
                certified_error=None, seconds=time.perf_counter()-began,
                method='deterministic_fourier_dyadic_gauss_legendre',
                gradient_scope='finite quadrature with normalization fixed; probability gradient uncertified',
                error_scope='same analytic IBP tail; order refinement is empirical; quadrature/FP uncertified')
