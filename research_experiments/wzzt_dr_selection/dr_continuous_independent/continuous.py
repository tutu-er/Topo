"""Deterministic signed-L1 probability for independent *residual cells*.

The two candidates share each cell's Gaussian innovation. Independence of
physical P/Q or device errors alone does not establish cell independence.
No physical simulator, topology truth, or historical archive is read here.
"""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.integrate import simpson
from scipy.special import erfcx, ndtr


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
    """D = mean(|coarse+sigma*Z| - kappa*|fine+sigma*Z|)."""

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

    def characteristic(self, frequency, chunk=128, normalization=None):
        """CF of (sum contributions - normalization[0]) / normalization[1]."""
        return self._characteristic_and_gradient(frequency, chunk, normalization, None)[0]

    def _characteristic_and_gradient(self, frequency, chunk, normalization, directions):
        frequency = np.asarray(frequency, float).reshape(-1)
        center, scale = (self.sum_mean, self.sum_sd) if normalization is None else normalization
        if self.sum_sd == 0:
            cf = np.exp(1j * frequency * (self.sum_mean - center) / scale)
            return cf, None
        result = np.empty(frequency.size, complex)
        gradient = None
        if directions is not None:
            dc, df = directions
            dimensions = dc.shape[1]
            gradient = np.empty((frequency.size, dimensions), complex)
            fixed_derivative = (np.sign(self.coarse[~self.random]) @ dc[~self.random]
                                - self.kappa * np.sign(self.fine[~self.random]) @ df[~self.random])
        for start in range(0, frequency.size, chunk):
            t = frequency[start:start+chunk][None, :] / scale
            value = np.zeros((self.indices.size, t.shape[1]), complex)
            derivatives = None
            if directions is not None:
                derivatives = np.zeros((self.indices.size, t.shape[1], dimensions), complex)
            for j in range(3):
                mask = self.nonempty[:, j]
                if not np.any(mask):
                    continue
                freq = self.slope[mask, j, None] * t
                integral = _truncated_fourier(self.left[mask, j], self.right[mask, j], freq)
                centered = self.intercept[mask, j] - self.means[mask]
                contribution = np.exp(1j * centered[:, None] * t) * integral
                value[mask] += contribution
                if directions is not None:
                    indices = self.indices[mask]
                    da = (self.sign_coarse[mask, j, None] * dc[indices]
                          - self.kappa * self.sign_fine[mask, j, None] * df[indices])
                    derivatives[mask] += contribution[:, :, None] * (1j * t[:, :, None]) * da[:, None, :]
            product = np.prod(value, axis=0)
            phase = np.exp(1j * t.ravel() * (self.sum_mean - center))
            result[start:start+t.shape[1]] = phase * product
            if directions is not None:
                before = np.vstack((np.ones((1, t.shape[1]), complex), np.cumprod(value[:-1], axis=0)))
                after = np.vstack((np.cumprod(value[:0:-1], axis=0)[::-1], np.ones((1, t.shape[1]), complex)))
                dg = np.sum((before * after)[:, :, None] * derivatives, axis=0)
                dg += product[:, None] * (1j * t.ravel()[:, None]) * fixed_derivative
                gradient[start:start+t.shape[1]] = phase[:, None] * dg
        if not np.isfinite(result).all():
            raise FloatingPointError('nonfinite characteristic function')
        return result, gradient

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

    def probability(self, threshold, tolerance=1e-5, cutoff=None, step=.05,
                    refinements=2, max_cutoff=2048., chunk=128,
                    normalization=None, daC=None, daF=None):
        """Deterministic Simpson inversion with explicit, limited diagnostics.

        Success checks refinement and the analytic tail. Refinement differences
        are not a rigorous quadrature error bound. No probability is clipped.
        """
        began = time.perf_counter()
        if not np.isfinite(threshold) or tolerance <= 0 or step <= 0:
            raise ValueError('finite threshold and positive tolerance/step required')
        if self.sum_sd == 0:
            return dict(probability=float(self.sum_mean / self.cells > threshold),
                        moments=self.moments(), seconds=time.perf_counter()-began,
                        method='deterministic', numerical_pass=True, certified_error=0.)
        center, scale = (self.sum_mean, self.sum_sd) if normalization is None else map(float, normalization)
        if not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
            raise ValueError('finite normalization center and positive scale required')
        directions = None
        if daC is not None or daF is not None:
            dc, df = np.asarray(daC, float), np.asarray(daF, float)
            if dc.ndim == 1:
                dc = dc[:, None]
            if df.ndim == 1:
                df = df[:, None]
            if dc.shape != df.shape or dc.shape[0] != self.cells or not np.isfinite(dc).all() or not np.isfinite(df).all():
                raise ValueError('finite coefficient directions (cells, directions) required')
            directions = dc, df
        selected = 8. if cutoff is None else float(cutoff)
        if cutoff is None:
            while self.tail_bound(selected, scale) > tolerance / 4 and selected < max_cutoff:
                selected *= 2
            selected = min(selected, max_cutoff)
        normalized_threshold = (self.cells * threshold - center) / scale
        trace = []
        for level in range(refinements + 1):
            intervals = int(math.ceil(selected / (step / 2**level)))
            intervals += intervals % 2
            frequencies = np.linspace(0., selected, intervals + 1)
            cf, cf_gradient = self._characteristic_and_gradient(frequencies, chunk, (center, scale), directions)
            integrand = np.empty_like(frequencies)
            integrand[0] = (self.sum_mean - self.cells * threshold) / scale
            integrand[1:] = np.imag(np.exp(-1j * frequencies[1:] * normalized_threshold) * cf[1:]) / frequencies[1:]
            probability = float(.5 + simpson(integrand, x=frequencies) / math.pi)
            probability_gradient = None
            if directions is not None:
                gradient_integrand = np.empty_like(cf_gradient.real)
                # The t=0 limit is the derivative of E[sum d]/scale.
                expected_sign_c, expected_sign_f = np.sign(self.coarse), np.sign(self.fine)
                expected_sign_c[self.random] = 2 * ndtr(self.coarse[self.random] / self.sigma[self.random]) - 1
                expected_sign_f[self.random] = 2 * ndtr(self.fine[self.random] / self.sigma[self.random]) - 1
                mean_direction = expected_sign_c @ dc - self.kappa * expected_sign_f @ df
                gradient_integrand[0] = mean_direction / scale
                gradient_integrand[1:] = np.imag(np.exp(-1j * frequencies[1:, None] * normalized_threshold) * cf_gradient[1:]) / frequencies[1:, None]
                probability_gradient = simpson(gradient_integrand, x=frequencies, axis=0) / math.pi
            difference = None if not trace else abs(probability - trace[-1]['probability'])
            trace.append(dict(level=level, nodes=frequencies.size, probability=probability,
                              max_cf_modulus=float(np.abs(cf).max()), refinement_difference=difference))
            if difference is not None and difference < tolerance / 4:
                break
        tail = self.tail_bound(selected, scale)
        in_range = 0 <= probability <= 1
        converged = difference is not None and difference < tolerance / 4
        return dict(probability=probability, normalized_threshold=float(normalized_threshold),
                    gradient=None if probability_gradient is None else probability_gradient.tolist(),
                    gradient_scope='finite quadrature with normalization fixed; infinite-integral gradient not certified',
                    normalization=[center, scale],
                    moments=self.moments(), cutoff=selected, analytic_tail_bound=tail,
                    trace=trace, refinement_difference=difference, seconds=time.perf_counter()-began,
                    method='deterministic_fourier_simpson', numerical_pass=bool(converged and tail < tolerance / 4 and in_range),
                    certified_error=None,
                    error_scope='IBP tail bound; refinement is empirical, quadrature/FP uncertified',
                    probability_in_unit_interval=in_range)


def exact_single_probability(coarse, fine, sigma, threshold, kappa=1.03):
    """Independent elementary reference: integrate three affine event intervals."""
    model = IndependentResiduals([coarse], [fine], [sigma], kappa)
    if sigma == 0:
        return float(abs(coarse) - kappa * abs(fine) > threshold)
    total = 0.
    for j in range(3):
        lo, hi = model.left[0, j], model.right[0, j]
        if not lo < hi:
            continue
        a, b = model.intercept[0, j], model.slope[0, j]
        root = (threshold - a) / b
        left, right = (max(lo, root), hi) if b > 0 else (lo, min(hi, root))
        if left < right:
            total += float(_normal_mass(np.asarray(left), np.asarray(right)))
    return total
