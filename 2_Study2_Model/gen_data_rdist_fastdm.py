from __future__ import annotations

import importlib
import math
from dataclasses import dataclass
from typing import Any, Sequence


# This file mirrors the "fastdm" path used by rtdists::rdiffusion.
#
# Big picture for non-technical readers:
# 1) We first clean and standardize the user parameters.
# 2) We build a time-evolving probability table for how likely each response is.
# 3) We draw random values and map them through that table to get RTs.
# 4) We return a table with one row per simulated trial:
#      - rt: response time in seconds
#      - response: lower or upper boundary
#
# Function relationship map:
# - rdiffusion_fastdm -> prepare_diffusion_parameters -> validate_prepared_params
# - rdiffusion_fastdm -> FastDMSampler.sample
# - FastDMSampler.sample -> _sample_group (per unique parameter row)
# - _sample_group -> _build_fc_from_row (build CDF engine stack)
# - CDF engine stack: _FPlain (+ optional _FSz, _FSv, _FSt0 wrappers)
# - _sample_group then performs inverse-CDF sampling to generate trial data.


def _np() -> Any:
    return importlib.import_module("numpy")


def _pd() -> Any:
    return importlib.import_module("pandas")


@dataclass(frozen=True)
class PreparedParams:
    # Parameters in backend-ready form, all as 1D arrays of equal length.
    # Each index i represents one trial's parameter set after recycling/scaling.
    a: Any
    v: Any
    t0: Any
    d: Any
    szr: Any
    sv: Any
    st0: Any
    zr: Any

    @property
    def n(self) -> int:
        return int(self.a.shape[0])

    def as_matrix(self) -> Any:
        np = _np()
        return np.column_stack(
            [
                self.a,
                self.v,
                self.t0,
                self.d,
                self.szr,
                self.sv,
                self.st0,
                self.zr,
            ]
        )


@dataclass(frozen=True)
class TuneParams:
    pde_dt_min: float
    pde_dt_max: float
    pde_dt_scale: float
    dz: float
    dv: float
    dt0: float
    int_t0: float
    int_z: float
    sv_epsilon: float
    sz_epsilon: float
    st0_epsilon: float


def _set_precision(precision: float) -> TuneParams:
    # These constants control numerical step sizes in the PDE solver.
    # Larger precision usually means smaller steps and more compute.
    p = float(precision)
    return TuneParams(
        pde_dt_min=10.0 ** (-0.400825 * p - 1.422813),
        pde_dt_max=10.0 ** (-0.627224 * p + 0.492689),
        pde_dt_scale=10.0 ** (-1.012677 * p + 2.261668),
        dz=10.0 ** (-0.5 * p - 0.033403),
        dv=10.0 ** (-1.0 * p + 1.4),
        dt0=10.0 ** (-0.5 * p - 0.323859),
        int_t0=0.089045 * math.exp(-1.037580 * p),
        int_z=0.508061 * math.exp(-1.022373 * p),
        sv_epsilon=10.0 ** (-(p + 2.0)),
        sz_epsilon=10.0 ** (-(p + 2.0)),
        st0_epsilon=10.0 ** (-(p + 2.0)),
    )


def recalc_t0(t0: Any, st0: Any) -> Any:
    # Same convention as rtdists fastdm wrapper.
    # If st0 > 0, t0 is treated as the lower edge of a uniform range,
    # so we shift to the center used by the backend equations.
    return t0 + st0 / 2.0


def _to_1d_array(x: float | Sequence[float] | Any) -> Any:
    np = _np()
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim != 1:
        raise ValueError("Expected scalar or 1D input.")
    return arr


def _recycle(arr: Any, n: int) -> Any:
    np = _np()
    if arr.size == n:
        return arr
    if arr.size == 1:
        return np.repeat(arr, n)
    reps = int(np.ceil(n / arr.size))
    return np.tile(arr, reps)[:n]


def _validate_inputs_for_prepare(
    a: Any,
    v: Any,
    t0: Any,
    z: Any,
    d: Any,
    sz: Any,
    sv: Any,
    st0: Any,
    s: Any,
) -> None:
    np = _np()
    all_params = [a, v, t0, z, d, sz, sv, st0, s]
    if not all(np.isfinite(x).all() for x in all_params):
        raise ValueError("All diffusion parameters must be finite.")


def prepare_diffusion_parameters(
    n: int,
    a: float | Sequence[float] | Any,
    v: float | Sequence[float] | Any,
    t0: float | Sequence[float] | Any,
    z: float | Sequence[float] | Any,
    d: float | Sequence[float] | Any,
    sz: float | Sequence[float] | Any,
    sv: float | Sequence[float] | Any,
    st0: float | Sequence[float] | Any,
    s: float | Sequence[float] | Any,
) -> PreparedParams:
    # Convert mixed scalar/vector inputs into trial-length vectors,
    # then apply the same transformations used by rtdists.
    if n <= 0:
        raise ValueError("n must be a positive integer.")

    a_arr = _recycle(_to_1d_array(a), n)
    v_arr = _recycle(_to_1d_array(v), n)
    t0_arr = _recycle(_to_1d_array(t0), n)
    z_arr = _recycle(_to_1d_array(z), n)
    d_arr = _recycle(_to_1d_array(d), n)
    sz_arr = _recycle(_to_1d_array(sz), n)
    sv_arr = _recycle(_to_1d_array(sv), n)
    st0_arr = _recycle(_to_1d_array(st0), n)
    s_arr = _recycle(_to_1d_array(s), n)

    _validate_inputs_for_prepare(
        a=a_arr,
        v=v_arr,
        t0=t0_arr,
        z=z_arr,
        d=d_arr,
        sz=sz_arr,
        sv=sv_arr,
        st0=st0_arr,
        s=s_arr,
    )

    z_rel = z_arr / a_arr
    sz_rel = sz_arr / a_arr
    t0_adj = recalc_t0(t0_arr, st0_arr)

    # fastdm works in scaled space:
    # - a, v, sv are divided by s
    # - z and sz are represented relative to a
    prepared = PreparedParams(
        a=a_arr / s_arr,
        v=v_arr / s_arr,
        t0=t0_adj,
        d=d_arr,
        szr=sz_rel,
        sv=sv_arr / s_arr,
        st0=st0_arr,
        zr=z_rel,
    )
    validate_prepared_params(prepared)
    return prepared


def validate_prepared_params(params: PreparedParams) -> None:
    # These checks match the same safety checks done in fastdm/rtdists.
    # They prevent invalid geometry (e.g., start point outside boundaries)
    # and impossible timing combinations.
    np = _np()
    if np.any(params.a <= 0):
        raise ValueError("Invalid parameter: a must be > 0.")
    if np.any((params.szr < 0) | (params.szr > 1)):
        raise ValueError("Invalid parameter: szr must be in [0, 1].")
    if np.any(params.st0 < 0):
        raise ValueError("Invalid parameter: st0 must be >= 0.")
    if np.any(params.sv < 0):
        raise ValueError("Invalid parameter: sv must be >= 0.")
    if np.any(params.t0 - np.abs(0.5 * params.d) - 0.5 * params.st0 < 0):
        raise ValueError("Invalid parameter combination: t0, d, st0.")
    if np.any(params.zr - 0.5 * params.szr <= 0):
        raise ValueError("Invalid parameter combination: zr - szr/2 must be > 0.")
    if np.any(params.zr + 0.5 * params.szr >= 1):
        raise ValueError("Invalid parameter combination: zr + szr/2 must be < 1.")


def _f_limit(a: float, z: float, v: float) -> float:
    # Closed-form long-run boundary probability used to initialize the PDE grid.
    if abs(v) < 1e-8:
        return 1.0 - z / a
    return (math.exp(-2.0 * v * z) - math.exp(-2.0 * v * a)) / (1.0 - math.exp(-2.0 * v * a))


def _phi(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _phi_inverse(y: float) -> float:
    if y <= 0.5:
        l = -1.0
        while _phi(l) >= y:
            l -= 1.0
        r = l + 1.0
    else:
        r = 0.0
        while _phi(r) < y:
            r += 1.0
        l = r - 1.0
    while (r - l) > 1e-8:
        m = 0.5 * (l + r)
        if _phi(m) < y:
            l = m
        else:
            r = m
    return 0.5 * (l + r)


class _FBase:
    # Minimal interface all CDF engines follow.
    # Wrappers (_FSz/_FSv/_FSt0) decorate another engine and modify outputs.
    def __init__(self) -> None:
        self.N = 0
        self.plus = -1

    def start(self, plus: int) -> None:
        raise NotImplementedError

    def get_F(self, t: float) -> Any:
        raise NotImplementedError

    def get_z(self, i: int) -> float:
        raise NotImplementedError

    def get_val(self, t: float, z: float) -> float:
        np = _np()
        F = self.get_F(t)
        N = self.N
        if N == 0:
            return float(F[0])
        z0 = self.get_z(0)
        z1 = self.get_z(N)
        i = int(N * (z - z0) / (z1 - z0))
        if i < N:
            zi0 = self.get_z(i)
            zi1 = self.get_z(i + 1)
            p = (zi1 - z) / (zi1 - zi0)
            return float(p * F[i] + (1.0 - p) * F[i + 1])
        return float(F[N])


class _FPlain(_FBase):
    # Base CDF engine with no inter-trial variability terms.
    # Uses a Crank-Nicolson PDE stepper over a z-grid.
    def __init__(self, a: float, v: float, t0: float, d: float, tune: TuneParams) -> None:
        super().__init__()
        np = _np()
        n_grid = 2 * int(a * 0.5 / tune.dz + 0.5)
        if n_grid < 4:
            n_grid = 4
        self.N = n_grid
        self.a = float(a)
        self.v = float(v)
        self.t0 = float(t0)
        self.d = float(d)
        self.dz = self.a / self.N
        self.t_offset = 0.0
        self.t = 0.0
        self.F = np.zeros(self.N + 1, dtype=float)
        self.tune = tune

    def start(self, plus: int) -> None:
        # plus=1 means upper boundary CDF mode, plus=0 means lower.
        self.plus = plus
        self.t_offset = self.t0 - self.d * (0.5 if plus == 1 else -0.5)
        self.t = 0.0
        self.F[0] = 1.0 if plus == 1 else 0.0
        for i in range(1, self.N):
            z = self.get_z(i)
            self.F[i] = _f_limit(self.a, z, self.v)
        self.F[self.N] = 1.0 if plus == 1 else 0.0

    def get_F(self, t: float) -> Any:
        t_adj = t - self.t_offset
        if t_adj > self.t:
            self._advance_to(self.t, t_adj)
            self.t = t_adj
        return self.F

    def get_z(self, i: int) -> float:
        return i * self.dz

    def _solve_tridiag(self, rhs: Any, left: float, mid: float, right: float) -> Any:
        # Thomas algorithm for tridiagonal linear systems.
        # This is the core linear solve used every PDE time step.
        np = _np()
        n = rhs.shape[0]
        tmp = np.empty(n - 1, dtype=float)
        res = np.empty(n, dtype=float)
        old_tmp = right / mid
        old_res = rhs[0] / mid
        tmp[0] = old_tmp
        res[0] = old_res
        for i in range(1, n - 1):
            p = 1.0 / (mid - left * old_tmp)
            old_res = (rhs[i] - left * old_res) * p
            old_tmp = right * p
            res[i] = old_res
            tmp[i] = old_tmp
        p = 1.0 / (mid - left * old_tmp)
        res[n - 1] = (rhs[n - 1] - left * old_res) * p
        for i in range(n - 1, 0, -1):
            res[i - 1] = res[i - 1] - tmp[i - 1] * res[i]
        return res

    def _make_step(self, dt: float) -> None:
        # One Crank-Nicolson update from t to t+dt.
        np = _np()
        N = self.N
        left = (1.0 - self.dz * self.v) / (2.0 * self.dz * self.dz)
        mid = -1.0 / (self.dz * self.dz)
        right = (1.0 + self.dz * self.v) / (2.0 * self.dz * self.dz)
        tmp = np.empty(N + 1, dtype=float)
        tmp[1] = dt * left * self.F[0] + (1.0 + 0.5 * dt * mid) * self.F[1] + 0.5 * dt * right * self.F[2]
        for i in range(2, N - 1):
            tmp[i] = 0.5 * dt * left * self.F[i - 1] + (1.0 + 0.5 * dt * mid) * self.F[i] + 0.5 * dt * right * self.F[i + 1]
        tmp[N - 1] = 0.5 * dt * left * self.F[N - 2] + (1.0 + 0.5 * dt * mid) * self.F[N - 1] + dt * right * self.F[N]
        solved = self._solve_tridiag(tmp[1:N], -0.5 * dt * left, 1.0 - 0.5 * dt * mid, -0.5 * dt * right)
        self.F[1:N] = solved

    def _advance_to(self, t0: float, t1: float) -> None:
        done = False
        while not done:
            dt = self.tune.pde_dt_min + self.tune.pde_dt_scale * t0
            if dt > self.tune.pde_dt_max:
                dt = self.tune.pde_dt_max
            if t0 + dt >= t1:
                dt = t1 - t0
                done = True
            self._make_step(dt)
            t0 = t0 + dt


class _FSz(_FBase):
    # Adds trial-to-trial start-point variability (sz).
    # Conceptually: average neighboring z-grid CDF values with trapezoidal weights.
    def __init__(self, base_fc: _FBase, sz_abs: float, tune: TuneParams) -> None:
        super().__init__()
        np = _np()
        self.base_fc = base_fc
        base_n = base_fc.N
        dz = self.base_fc.get_z(1) - self.base_fc.get_z(0)
        tmp = sz_abs / (2.0 * dz)
        self.k = int(math.ceil(tmp) + 0.5)
        if 2 * self.k > base_n:
            raise ValueError("Invalid sz setup for grid size.")
        self.N = base_n - 2 * self.k
        self.q = self.k - tmp
        self.f = dz / sz_abs
        self.avg = np.empty(self.N + 1, dtype=float)

    def start(self, plus: int) -> None:
        self.plus = plus
        self.base_fc.start(plus)

    def get_F(self, t: float) -> Any:
        F = self.base_fc.get_F(t)
        m = 2 * self.k
        q = self.q
        f = self.f
        if m >= 3:
            for i in range(0, self.N + 1):
                tmp = F[i] * 0.5 * (1.0 - q) * (1.0 - q)
                tmp = tmp + F[i + 1] * (1.0 - 0.5 * q * q)
                for j in range(i + 2, i + m - 1):
                    tmp = tmp + F[j]
                tmp = tmp + F[i + m - 1] * (1.0 - 0.5 * q * q)
                tmp = tmp + F[i + m] * 0.5 * (1.0 - q) * (1.0 - q)
                self.avg[i] = tmp * f
        else:
            for i in range(0, self.N + 1):
                tmp = F[i] * 0.5 * (1.0 - q) * (1.0 - q)
                tmp = tmp + F[i + 1] * (1.0 - q * q)
                tmp = tmp + F[i + 2] * 0.5 * (1.0 - q) * (1.0 - q)
                self.avg[i] = tmp * f
        return self.avg

    def get_z(self, i: int) -> float:
        return self.base_fc.get_z(i + self.k)


class _FSv(_FBase):
    # Adds drift variability (sv).
    # Conceptually: evaluate multiple drift values and average their CDFs.
    def __init__(self, base_fcs: list[_FBase]) -> None:
        super().__init__()
        np = _np()
        self.base_fcs = base_fcs
        self.nv = len(base_fcs)
        self.N = base_fcs[0].N
        self.avg = np.empty(self.N + 1, dtype=float)

    def start(self, plus: int) -> None:
        self.plus = plus
        for fc in self.base_fcs:
            fc.start(plus)

    def get_F(self, t: float) -> Any:
        np = _np()
        F = self.base_fcs[0].get_F(t)
        self.avg[:] = F
        for j in range(1, self.nv):
            self.avg[:] = self.avg + self.base_fcs[j].get_F(t)
        self.avg[:] = self.avg / self.nv
        return self.avg

    def get_z(self, i: int) -> float:
        return self.base_fcs[0].get_z(i)


class _FSt0(_FBase):
    # Adds non-decision-time variability (st0).
    # Conceptually: cache nearby time slices and average them.
    def __init__(self, base_fc: _FBase, st0: float, dt0_tune: float) -> None:
        super().__init__()
        np = _np()
        self.base_fc = base_fc
        self.N = base_fc.N
        n_cache = int(st0 / dt0_tune + 1.5)
        if n_cache < 3:
            n_cache = 3
        self.st0 = float(st0)
        self.M = n_cache
        self.dt = self.st0 / (self.M - 2)
        self.start_t = -float("inf")
        self.values = np.zeros((self.M, self.N + 1), dtype=float)
        self.valid = np.zeros(self.M, dtype=bool)
        self.base = 0
        self.avg = np.zeros(self.N + 1, dtype=float)

    def start(self, plus: int) -> None:
        self.plus = plus
        self.base_fc.start(plus)
        self.start_t = -float("inf")
        self.valid[:] = False
        self.base = 0

    def _get_row(self, j: int) -> Any:
        idx = (self.base + j) % self.M
        if not self.valid[idx]:
            t = self.start_t + j * self.dt
            self.values[idx, :] = self.base_fc.get_F(t)
            self.valid[idx] = True
        return self.values[idx, :]

    def get_F(self, t: float) -> Any:
        np = _np()
        a = t - 0.5 * self.st0
        b = t + 0.5 * self.st0
        if a - self.start_t >= self.M * self.dt:
            shift = self.M
        else:
            shift = int((a - self.start_t) / self.dt)
            if shift < 0:
                shift = 0
        for j in range(shift):
            self.valid[(self.base + j) % self.M] = False
        if shift < self.M:
            self.start_t = self.start_t + shift * self.dt
            self.base = (self.base + shift) % self.M
        else:
            self.start_t = a
        self.avg[:] = 0.0
        tmp = (b - self.start_t) / self.dt
        m = int(math.ceil(tmp) + 0.5)
        if m >= self.M:
            m = self.M - 1
        q = (a - self.start_t) / self.dt
        r = m - tmp
        if m >= 3:
            self.avg[:] = self.avg + 0.5 * (1.0 - q) * (1.0 - q) * self._get_row(0)
            self.avg[:] = self.avg + (1.0 - 0.5 * q * q) * self._get_row(1)
            for j in range(2, m - 1):
                self.avg[:] = self.avg + self._get_row(j)
            self.avg[:] = self.avg + (1.0 - 0.5 * r * r) * self._get_row(m - 1)
            self.avg[:] = self.avg + 0.5 * (1.0 - r) * (1.0 - r) * self._get_row(m)
        elif m == 2:
            self.avg[:] = self.avg + 0.5 * (1.0 - q) * (1.0 - q) * self._get_row(0)
            self.avg[:] = self.avg + (1.0 - 0.5 * (q * q + r * r)) * self._get_row(1)
            self.avg[:] = self.avg + 0.5 * (1.0 - r) * (1.0 - r) * self._get_row(2)
        elif m == 1:
            self.avg[:] = self.avg + 0.5 * ((1.0 - q) * (1.0 - q) - r * r) * self._get_row(0)
            self.avg[:] = self.avg + 0.5 * ((1.0 - r) * (1.0 - r) - q * q) * self._get_row(1)
        self.avg[:] = self.avg * (self.dt / (b - a))
        return self.avg

    def get_z(self, i: int) -> float:
        return self.base_fc.get_z(i)


def _build_fc_from_row(row: Any, precision: float) -> _FBase:
    # Build the same layered CDF engine structure as fastdm:
    # plain -> (optional sz) -> (optional sv) -> (optional st0)
    a, v, t0, d, szr, sv, st0, zr = [float(x) for x in row.tolist()]
    tune = _set_precision(precision)
    base: _FBase = _FPlain(a=a, v=v, t0=t0, d=d, tune=tune)
    sz_abs = szr * a
    if sz_abs >= tune.sz_epsilon:
        base = _FSz(base_fc=base, sz_abs=sz_abs, tune=tune)
    if sv >= tune.sv_epsilon:
        nv = int(sv / tune.dv + 0.5)
        if nv < 3:
            nv = 3
        base_list: list[_FBase] = []
        for j in range(nv):
            x = _phi_inverse((0.5 + j) / nv)
            vj = sv * x + v
            one = _FPlain(a=a, v=vj, t0=t0, d=d, tune=tune)
            if sz_abs >= tune.sz_epsilon:
                one = _FSz(base_fc=one, sz_abs=sz_abs, tune=tune)
            base_list.append(one)
        base = _FSv(base_fcs=base_list)
    if st0 > tune.dt0 * 1e-6:
        base = _FSt0(base_fc=base, st0=st0, dt0_tune=tune.dt0)
    _ = zr
    return base


class FastDMSampler:
    # Public sampler class. It groups identical parameter rows and samples each
    # group once, which keeps simulation fast for repeated conditions.
    def __init__(self, precision: float = 3.0, max_time: float = 20.0) -> None:
        self.precision = float(precision)
        self.max_time = float(max_time)

    def sample(self, params: PreparedParams, rng: Any) -> tuple[Any, Any]:
        np = _np()
        rt = np.empty(params.n, dtype=float)
        boundary = np.empty(params.n, dtype=np.int64)

        # If many trials share the same parameters, reuse one CDF build per group.
        unique_rows, inverse = np.unique(params.as_matrix(), axis=0, return_inverse=True)
        for group_idx, row in enumerate(unique_rows):
            group_mask = inverse == group_idx
            group_size = int(np.sum(group_mask))
            group_rt, group_boundary = self._sample_group(row=row, n=group_size, rng=rng)
            rt[group_mask] = group_rt
            boundary[group_mask] = group_boundary
        return rt, boundary

    def _sample_group(self, row: Any, n: int, rng: Any) -> tuple[Any, Any]:
        # Data generation logic for one homogeneous parameter group:
        # 1) Build CDF engine
        # 2) Draw uniform random numbers
        # 3) Convert uniforms to signed decision times via inverse-CDF lookup
        # 4) Signed time >= 0 -> upper boundary, else lower
        # 5) RT is absolute value of signed time
        np = _np()
        a, _v, _t0, _d, _szr, _sv, _st0, zr = [float(x) for x in row.tolist()]
        fc = _build_fc_from_row(row=row, precision=self.precision)
        fs = rng.uniform(0.0, 1.0, size=n)
        fs_min = float(np.min(fs))
        fs_max = float(np.max(fs))
        scaled_z = zr * a

        t_max = 0.5
        fc.start(1)
        guard = 0
        while fc.get_val(t_max, scaled_z) < fs_max:
            t_max = t_max + 0.1
            guard += 1
            if guard > 100000:
                raise RuntimeError("Failed to bracket upper sampling time.")

        t_min = -0.5
        fc.start(0)
        guard = 0
        while fc.get_val(-t_min, scaled_z) > fs_min:
            t_min = t_min - 0.1
            guard += 1
            if guard > 100000:
                raise RuntimeError("Failed to bracket lower sampling time.")

        N = int((t_max - t_min) / 0.001 + 0.5)
        dt = (t_max - t_min) / N
        cdf_lookup = np.zeros(N + 1, dtype=float)

        # Build one monotonic lookup table that spans both boundaries.
        # Positive signed times correspond to upper; negative to lower.

        fc.start(1)
        for i in range(0, N + 1):
            t = t_min + i * dt
            if t < 0:
                continue
            cdf_lookup[i] = fc.get_val(t, scaled_z)

        fc.start(0)
        for i in range(N, -1, -1):
            t = -(t_min + i * dt)
            if t < 0:
                continue
            cdf_lookup[i] = fc.get_val(t, scaled_z)

        cdf_lookup = np.clip(cdf_lookup, 0.0, 1.0)
        cdf_lookup.sort()
        if cdf_lookup[0] > fs_min:
            cdf_lookup[0] = fs_min
        if cdf_lookup[N] < fs_max:
            cdf_lookup[N] = fs_max

        idx = np.searchsorted(cdf_lookup, fs, side="right") - 1
        idx = np.clip(idx, 0, N - 1)
        f0 = cdf_lookup[idx]
        f1 = cdf_lookup[idx + 1]
        denom = np.where(f1 > f0, f1 - f0, 1.0)
        frac = np.where(f1 > f0, (fs - f0) / denom, 0.0)
        signed_t = t_min + (idx + frac) * dt

        boundary = (signed_t >= 0).astype(np.int64)
        rt = np.abs(signed_t)
        if self.max_time > 0:
            rt = np.minimum(rt, self.max_time)
        return rt, boundary


def rdiffusion_fastdm(
    n: int,
    a: float | Sequence[float] | Any,
    v: float | Sequence[float] | Any,
    t0: float | Sequence[float] | Any,
    z: float | Sequence[float] | Any | None = None,
    d: float | Sequence[float] | Any = 0.0,
    sz: float | Sequence[float] | Any = 0.0,
    sv: float | Sequence[float] | Any = 0.0,
    st0: float | Sequence[float] | Any = 0.0,
    s: float | Sequence[float] | Any = 1.0,
    precision: float = 3.0,
    max_time: float = 20.0,
    seed: int | None = None,
) -> Any:
    # Main user-facing function:
    # - preprocess parameters
    # - sample RTs/boundaries
    # - return a tidy DataFrame
    np = _np()
    pd = _pd()
    z_eff = 0.5 * np.asarray(a, dtype=float) if z is None else z
    prepared = prepare_diffusion_parameters(
        n=n,
        a=a,
        v=v,
        t0=t0,
        z=z_eff,
        d=d,
        sz=sz,
        sv=sv,
        st0=st0,
        s=s,
    )
    rng = np.random.default_rng(seed)
    sampler = FastDMSampler(precision=precision, max_time=max_time)
    rt, boundary = sampler.sample(prepared, rng=rng)
    response = pd.Categorical.from_codes(boundary, categories=["lower", "upper"])
    return pd.DataFrame({"rt": rt, "response": response})


__all__ = [
    "PreparedParams",
    "FastDMSampler",
    "prepare_diffusion_parameters",
    "recalc_t0",
    "rdiffusion_fastdm",
    "validate_prepared_params",
]
