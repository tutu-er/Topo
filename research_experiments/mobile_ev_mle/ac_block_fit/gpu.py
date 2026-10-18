"""Optional FP64 GPU arithmetic for the unchanged radial AC equations.

This module is a kernel comparison, not a replacement optimizer.  CuPy is
imported only when GPUWorkspace is constructed; no dependency is installed by
the module.  Exact zero-impedance edges remain zero in the Zbus formulation.
"""
from __future__ import annotations


def configure_dependency_path(path, *, work_dir=None):
    """Discover a pip --target CUDA install in this process only.

    cuda-pathfinder searches site.getsitepackages(), not arbitrary sys.path
    entries. Register the isolated target for that search without editing an
    installed package, conda environment, user site, or system environment.
    Import original CPU NumPy/SciPy first so their BLAS does not change.
    """
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import site
    import sys
    import os
    import tempfile
    from pathlib import Path
    root = str(Path(path).resolve())
    scratch = Path(work_dir).resolve() if work_dir is not None else Path(root).parent
    temporary, cache = scratch/'gpu_tmp', scratch/'cupy_cache'
    temporary.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    # NVRTC uses tempfile in addition to CuPy's persistent compilation cache.
    # These assignments affect this child process only.
    os.environ['TMP'] = os.environ['TEMP'] = str(temporary)
    os.environ['CUPY_CACHE_DIR'] = str(cache)
    tempfile.tempdir = str(temporary)
    if root not in sys.path:
        sys.path.insert(0, root)
    original = getattr(site, '_topo_original_getsitepackages', site.getsitepackages)
    site._topo_original_getsitepackages = original
    site.getsitepackages = lambda prefixes=None: [root]+list(original(prefixes))
    return root


class GPUWorkspace:
    """Keep one candidate's topology and repeated input arrays on one GPU."""

    def __init__(self, candidate):
        import cupy as cp
        self.cp = cp
        self.candidate = candidate
        self.B = cp.asarray(candidate.B, dtype=cp.float64)
        self.ids = list(candidate.terminal_edges)

    def inputs(self, r, x, p, q, source_p=0., source_q=0., root_v=1.):
        """Upload or reuse the same finite, broadcastable physical inputs."""
        cp = self.cp
        p = cp.asarray(p, dtype=cp.float64)
        if p.ndim == 1:
            p = p[None, :]
        if p.ndim != 2 or p.shape[1] != self.candidate.n or not len(p):
            raise ValueError('p must have shape (time, terminal)')
        t, m = len(p), self.candidate.m
        def array(value, shape):
            out = cp.broadcast_to(cp.asarray(value, dtype=cp.float64), shape).copy()
            if not bool(cp.isfinite(out).all().item()):
                raise ValueError('nonfinite physical input')
            return out
        data = dict(p=array(p, p.shape), q=array(q, p.shape),
                    r=array(r, (m,)), x=array(x, (m,)),
                    source_p=array(source_p, (t,)), source_q=array(source_q, (t,)),
                    root_v=array(root_v, (t,)))
        if bool(((data['r'] < 0).any() | (data['x'] < 0).any()).item()):
            raise ValueError('nonnegative R/X required')
        if bool((data['root_v'] <= 0).any().item()):
            raise ValueError('positive root voltage required')
        return data

    def predict(self, data, *, tol=2e-13, max_iter=400):
        """Original constant-power fixed point, same stopping/physical checks.

        Results stay on device.  Scalar checks synchronize as in a normal
        Python-controlled GPU loop; this cost belongs to the measured kernel.
        """
        cp = self.cp
        p, q, r, x, sp, sq, root = (data[k] for k in
            ('p', 'q', 'r', 'x', 'source_p', 'source_q', 'root_v'))
        t, m = len(p), self.candidate.m
        loads = cp.zeros((t, m), dtype=cp.complex128)
        loads[:, 0] = sp+1j*sq
        loads[:, self.ids] += p+1j*q
        z = r+1j*x
        zbus = (self.B*z)@self.B.T
        voltage = cp.broadcast_to(root[:, None], loads.shape).astype(cp.complex128).copy()
        for iteration in range(1, max_iter+1):
            if float(cp.min(abs(voltage)).item()) < .1:
                raise RuntimeError('AC low-voltage iterate; restart is invalid')
            new = root[:, None]-cp.conj(loads/voltage)@zbus.T
            if not bool(cp.isfinite(new).all().item()):
                raise RuntimeError('nonfinite AC iterate; restart is invalid')
            change = float(cp.max(abs(new-voltage)).item())
            voltage = new
            if change <= tol:
                break
        else:
            raise RuntimeError('AC sweep failed to converge; restart is invalid')
        currents = cp.conj(loads/voltage)
        branch_current = currents@self.B
        master = root*cp.conj(currents.sum(axis=1))
        losses = (abs(branch_current)**2*z).sum(axis=1)
        balance = master-loads.sum(axis=1)-losses
        equation = voltage-root[:, None]+currents@zbus.T
        eq = float(cp.max(abs(equation)).item())
        bal = float(cp.max(abs(balance)).item())
        if eq > 2e-10 or bal > 2e-10:
            raise RuntimeError('AC equation or power balance failed')
        return dict(p=p, q=q, v=abs(voltage[:, self.ids]), root_v=root,
            master_p=master.real, master_q=master.imag, source_p=sp, source_q=sq,
            voltage=voltage, loads=loads, branch_current=branch_current, zbus=zbus,
            ac_equation_residual=eq, power_balance_residual=bal,
            p_balance_max_abs=float(cp.max(abs(balance.real)).item()),
            q_balance_max_abs=float(cp.max(abs(balance.imag)).item()), iterations=iteration)

    def derivatives(self, physical, *, include_lines=True):
        """Exact selected-RHS implicit derivatives, same as CPU fitter."""
        import cupyx
        cp = self.cp
        voltage, loads, zbus = (physical[k] for k in ('voltage', 'loads', 'zbus'))
        t, m = voltage.shape
        width = 2*m+2 if include_lines else 2
        coefficient = cp.conj(loads)/cp.conj(voltage)**2
        k = zbus[None, :, :]*coefficient[:, None, :]
        eye = cp.eye(m, dtype=cp.float64)[None, :, :]
        matrix = cp.concatenate((cp.concatenate((eye-k.real, -k.imag), axis=2),
            cp.concatenate((-k.imag, eye+k.real), axis=2)), axis=1)
        rhs = cp.zeros((t, m, width), dtype=cp.complex128)
        if include_lines:
            edge = -self.B[None, :, :]*physical['branch_current'][:, None, :]
            rhs[:, :, :m], rhs[:, :, m:2*m] = edge, 1j*edge
        ds = cp.zeros((m, 2), dtype=cp.complex128)
        ds[0, 0], ds[0, 1] = 1., 1j
        direct = cp.conj(ds)[None, :, :]/cp.conj(voltage)[:, :, None]
        rhs[:, :, -2:] = -cp.einsum('ij,tjk->tik', zbus, direct)
        # Raise for invalid/singular solves rather than retaining silent NaNs.
        with cupyx.errstate(linalg='raise'):
            solution = cp.linalg.solve(matrix, cp.concatenate((rhs.real, rhs.imag), axis=1))
        dv = solution[:, :m]+1j*solution[:, m:]
        di = -coefficient[:, :, None]*cp.conj(dv)
        di[:, :, -2:] += direct
        master = physical['root_v'][:, None]*cp.conj(di.sum(axis=1))
        magnitude = (cp.conj(voltage[:, self.ids, None])*dv[:, self.ids]).real/abs(voltage[:, self.ids, None])
        return dict(v=magnitude, master_p=master.real, master_q=master.imag)

    def host(self, result):
        """Explicit device-to-host boundary for benchmark or optimizer use."""
        cp = self.cp
        return {key: cp.asnumpy(value) if isinstance(value, cp.ndarray) else value
                for key, value in result.items()}

    def synchronize(self):
        self.cp.cuda.get_current_stream().synchronize()


def gpu_problem_context(problem):
    """Context manager for a complete *hybrid* fit with CPU QR/BVLS.

    The base residual evaluation and exact block derivative callback use GPU
    FP64 kernels. Residuals, G/S blocks, and physical cache are copied to host
    on every call. Thus all transfers are included in an external fit timer;
    this deliberately simple first version does not promise device residency.
    The context restores the original bound evaluate method on exit. Its
    ``blocks(problem, vector)`` method matches blocks.problem_blocks.
    """
    return _GPUProblemContext(problem)


class _GPUProblemContext:
    def __init__(self, problem):
        self.problem = problem
        self.base = problem.base
        self.work = GPUWorkspace(self.base.candidate)
        self.cp = self.work.cp
        self.original_evaluate = None
        self.physical = None
        self.fixed = {}

    def __enter__(self):
        cp, base = self.cp, self.base
        self.fixed = dict(p=cp.asarray(base.p), q=cp.asarray(base.q),
                          root_v=cp.asarray(base.observations.root_v))
        self.bias_v = cp.asarray(base.bias['v'])
        self.observed = {key: cp.asarray(getattr(base.observations, key))
                         for key in ('v', 'master_p', 'master_q')}
        self.sigmas = {key: cp.asarray(value) for key, value in base.sigmas.items()}
        self.original_evaluate = base.evaluate
        base.evaluate = self.evaluate
        base.cache = {}
        return self

    def __exit__(self, kind, value, trace):
        self.base.evaluate = self.original_evaluate
        self.work.synchronize()
        return False

    def evaluate(self, vector):
        import numpy as np
        base, cp = self.base, self.cp
        if 'vector' in base.cache and np.array_equal(vector, base.cache['vector']):
            return base.cache['residual']
        r, x, source = base.unpack(vector)
        if not np.isfinite(np.r_[r, x, source.ravel()]).all() or (r < 0).any() or (x < 0).any():
            raise ValueError('nonfinite input or negative R/X')
        base.ac_calls += 1
        if base.counter is not None:
            base.counter['ac_calls'] = base.counter.get('ac_calls', 0)+1
        data = dict(self.fixed, r=cp.asarray(r), x=cp.asarray(x),
                    source_p=cp.asarray(source[:, 0]), source_q=cp.asarray(source[:, 1]))
        phy = self.work.predict(data)
        residual = cp.column_stack(((phy['v']+self.bias_v-self.observed['v'])/self.sigmas['v'],
            (phy['master_p']-self.observed['master_p'])/self.sigmas['master_p'],
            (phy['master_q']-self.observed['master_q'])/self.sigmas['master_q'])).ravel()
        residual = cp.asnumpy(residual)
        self.physical = phy
        base.cache = dict(vector=np.asarray(vector).copy(), residual=residual,
                          physical=self.work.host(phy))
        return residual

    def blocks(self, problem, vector):
        """Return host G[T,n+2,d] and S[T,n+2,2], exact same coordinates."""
        import numpy as np
        if problem is not self.problem:
            raise ValueError('GPU block callback belongs to another problem')
        problem.evaluate(vector)
        base, cp = self.base, self.cp
        derivatives = self.work.derivatives(self.physical, include_lines=not base.line_fixed)
        rows = []
        for name in ('v', 'master_p', 'master_q'):
            raw = derivatives[name]
            sigma = self.sigmas[name]
            if name != 'v':
                raw = raw[:, None, :]; sigma = sigma[:, None]
            rows.append(raw/sigma[:, :, None])
        physical = cp.concatenate(rows, axis=1)
        source = physical[:, :, -2:]
        if base.line_fixed:
            return np.zeros((*source.shape[:2], 0)), cp.asnumpy(source)
        full = problem.expand(vector)
        _, _, dx_dr, dx_da = problem.mapping.forward(full[:problem.m], full[problem.m:problem.shared])
        jr, jx = physical[:, :, :problem.m], physical[:, :, problem.m:problem.shared]
        shared = cp.concatenate((jr+jx*cp.asarray(dx_dr)[None, None, :],
                                 jx*cp.asarray(dx_da)[None, None, :]), axis=2)
        shared = shared[:, :, cp.asarray(problem.free[:problem.shared])]
        return cp.asnumpy(shared), cp.asnumpy(source)


__all__ = ['GPUWorkspace', 'gpu_problem_context', 'configure_dependency_path']
