"""Probability and derivatives for a declared pair with shared Gaussian errors.

This is the verified three-segment Fourier kernel, collected in one file.
The two candidates share each cell's noise; cells themselves are independent.
Quadrature refinement is an empirical diagnostic, not an interval certificate.
"""
from __future__ import annotations
import math
import time
import numpy as np
from scipy.special import erfcx, ndtr
from numpy.polynomial.legendre import leggauss

NORMAL_AT_ZERO = 1.0 / math.sqrt(2.0 * math.pi)


def _inputs(coarse, fine, sigma, kappa):
    coarse, fine, sigma = [np.asarray(a, dtype=float).reshape(-1) for a in (coarse, fine, sigma)]
    if not coarse.size or coarse.shape != fine.shape or coarse.shape != sigma.shape:
        raise ValueError('three nonempty vectors of the same shape are required')
    if not all(np.isfinite(a).all() for a in (coarse, fine, sigma)):
        raise ValueError('finite coefficients are required')
    if np.any(sigma < 0) or not np.isfinite(kappa) or kappa <= 1:
        raise ValueError('nonnegative sigma and kappa > 1 are required')
    return coarse, fine, sigma


def _normal_density(x):
    return np.exp(-0.5 * x * x) * NORMAL_AT_ZERO


def _normal_mass(left, right):
    # Choosing the tail avoids subtraction of CDFs both rounded to one.
    return np.where(left >= 0, ndtr(-left) - ndtr(-right), ndtr(right) - ndtr(left))


def _tail_fourier(x, frequency):
    """Integral_x^infinity phi(z) exp(i frequency z) dz for x >= 0."""
    return (0.5 * np.exp(-0.5 * x * x + 1j * frequency * x)
            * erfcx((x - 1j * frequency) / math.sqrt(2.0)))


def _truncated_fourier(left, right, frequency):
    """Stable real-interval normal Fourier integral, broadcasting (cell,freq)."""
    result = np.zeros_like(frequency, dtype=complex)
    positive = left >= 0
    negative = right <= 0
    crossing = ~(positive | negative)
    for mask, sign in ((positive, 1.0), (negative, -1.0)):
        if not np.any(mask):
            continue
        lo = left[mask] if sign > 0 else -right[mask]
        hi = right[mask] if sign > 0 else -left[mask]
        freq = sign * frequency[mask]
        part = _tail_fourier(lo[:, None], freq)
        finite = np.isfinite(hi)
        if np.any(finite):
            part[finite] -= _tail_fourier(hi[finite, None], freq[finite])
        result[mask] = part
    if np.any(crossing):
        freq = frequency[crossing]
        part = np.exp(-0.5 * freq * freq).astype(complex)
        lo, hi = left[crossing], right[crossing]
        finite = np.isfinite(lo)
        if np.any(finite):
            part[finite] -= _tail_fourier(-lo[finite, None], -freq[finite])
        finite = np.isfinite(hi)
        if np.any(finite):
            part[finite] -= _tail_fourier(hi[finite, None], freq[finite])
        result[crossing] = part
    return result


class IndependentResiduals:
    """D = mean(abs(coarse+sigma*Z) - kappa*abs(fine+sigma*Z))."""

    def __init__(self, coarse, fine, sigma, kappa=1.03):
        self.coarse, self.fine, self.sigma = _inputs(coarse, fine, sigma, kappa)
        self.kappa = float(kappa)
        self.cells = self.coarse.size
        self.random = self.sigma > 0
        self.indices = np.flatnonzero(self.random)
        self.constant_sum = float(np.sum(np.abs(self.coarse[~self.random])
                                        - self.kappa * np.abs(self.fine[~self.random])))
        c, f, sd = self.coarse[self.random], self.fine[self.random], self.sigma[self.random]
        knots = np.sort(np.stack((-c / sd, -f / sd), axis=1), axis=1)
        self.left = np.column_stack((np.full(sd.size, -np.inf), knots))
        self.right = np.column_stack((knots, np.full(sd.size, np.inf)))
        # A representative in each of the three intervals determines signs.
        middle = np.column_stack((knots[:, 0] - 1., knots.mean(axis=1), knots[:, 1] + 1.))
        sign_c = np.sign(c[:, None] + sd[:, None] * middle)
        sign_f = np.sign(f[:, None] + sd[:, None] * middle)
        self.sign_coarse, self.sign_fine = sign_c, sign_f
        self.intercept = sign_c * c[:, None] - self.kappa * sign_f * f[:, None]
        self.slope = sd[:, None] * (sign_c - self.kappa * sign_f)
        self.nonempty = self.right > self.left
        # At coincident roots a zero-width middle interval is never evaluated.
        self.slope[~self.nonempty] = 1.
        folded = lambda a: (sd * math.sqrt(2. / math.pi) * np.exp(-0.5 * (a / sd)**2)
                             + a * (2 * ndtr(a / sd) - 1.))
        self.means = folded(c) - self.kappa * folded(f)
        variance = np.zeros(sd.size)
        c_constant = np.zeros(sd.size)
        for j in range(3):
            lo, hi = self.left[:, j], self.right[:, j]
            mass = _normal_mass(lo, hi)
            density_lo, density_hi = _normal_density(lo), _normal_density(hi)
            first = density_lo - density_hi
            lo_phi = np.zeros(sd.size)
            hi_phi = np.zeros(sd.size)
            np.multiply(lo, density_lo, out=lo_phi, where=np.isfinite(lo))
            np.multiply(hi, density_hi, out=hi_phi, where=np.isfinite(hi))
            second = mass + lo_phi - hi_phi
            centered = self.intercept[:, j] - self.means
            slope = self.slope[:, j]
            terms = centered * centered * mass + 2 * centered * slope * first + slope * slope * second
            variance += np.where(self.nonempty[:, j], terms, 0.)
            nearest_zero = np.minimum(np.maximum(0., lo), hi)
            c_constant += np.where(self.nonempty[:, j],
                                   2 * _normal_density(nearest_zero) / np.abs(slope), 0.)
        if np.any(variance < -1e-8 * np.maximum(1., sd * sd)):
            raise ArithmeticError('negative computed local variance')
        self.variances = np.maximum(0., variance)
        self.tail_constants = c_constant
        self.sum_mean = self.constant_sum + float(self.means.sum())
        self.sum_sd = float(np.sqrt(self.variances.sum()))
        if self.indices.size and (self.sum_sd == 0 or not np.isfinite(self.sum_sd)):
            raise FloatingPointError('positive-noise model has unresolved numerical variance')

    def moments(self):
        return dict(mean=self.sum_mean / self.cells,
                    sd=self.sum_sd / self.cells,
                    independent_cells=int(self.random.sum()), cells=self.cells)

    def tail_bound(self, cutoff, scale=None):
        """IBP bound on probability integral beyond normalized frequency cutoff.

        Minimize over product bounds using the smallest 1..N constants. This
        bounds analytic truncation only, excluding quadrature and FP errors.
        """
        if cutoff <= 0:
            raise ValueError('positive cutoff required')
        if self.sum_sd == 0:
            return 0.
        constants = np.sort(self.tail_constants * (self.sum_sd if scale is None else scale))
        ranks = np.arange(1, constants.size + 1)
        logs = np.cumsum(np.log(constants)) - ranks * math.log(cutoff) - np.log(math.pi * ranks)
        log_bound = float(np.min(logs))
        return float(math.exp(max(log_bound, math.log(np.finfo(float).tiny))))


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


def probability_and_gradient(coarse, fine, sigma, threshold, *, kappa=1.03,
                             normalization=None, cutoff=512., order=32,
                             refinements=0, tolerance=1e-7, chunk=64):
    """Return crossing probability and derivatives with respect to both means.

    Inputs are equally shaped finite arrays (flattened in row-major order).
    Sigma is nonnegative, held fixed while differentiating. The same Gaussian
    error is used for coarse/fine in each cell. No error or load is sampled.
    """
    if not np.isfinite(threshold):
        raise ValueError("Finite loss threshold required")
    model = IndependentResiduals(coarse, fine, sigma, kappa)
    # The reverse triangle inequality gives d_i <= abs(coarse_i-fine_i).
    # A strict gap leaves a neighborhood with exactly zero event probability,
    # so its mean-coefficient derivatives are zero as well. Equality is left
    # to the ordinary kernel: no derivative is inferred at that boundary.
    upper_bound = float(np.mean(np.abs(model.coarse - model.fine)))
    if upper_bound < threshold:
        return dict(probability=0., gradient_coarse=np.zeros(model.cells),
                    gradient_fine=np.zeros(model.cells), gradient_valid=True,
                    moments=model.moments(), numerical_pass=True,
                    analytic_upper_bound=upper_bound, analytic_tail_bound=0.,
                    certified_error=0., method="analytic_unreachable_event",
                    trace=[], seconds=0.)
    result = probability_with_adjoint(model, threshold, normalization=normalization,
        cutoff=cutoff, order=order, refinements=refinements,
        tolerance=tolerance, chunk=chunk)
    # The deterministic kernel has no Fourier tail, but exposes the same fields
    # as stochastic evaluations so callers do not need a second code path.
    result.setdefault("moments", model.moments())
    result.setdefault("analytic_tail_bound", 0.)
    return result
