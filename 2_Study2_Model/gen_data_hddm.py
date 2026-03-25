from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Mapping, cast

_DEFAULT_SHARE_NOISE = {"a", "v", "t", "st", "sz", "sv", "z"}
_BOUNDS = {
    "a": (0.0, float("inf")),
    "z": (0.0, 1.0),
    "t": (0.0, float("inf")),
    "st": (0.0, float("inf")),
    "sv": (0.0, float("inf")),
    "sz": (0.0, 1.0),
}
_BASE_KEYS = ("v", "a", "t", "z", "sv", "sz", "st")


def _np() -> Any:
    return importlib.import_module("numpy")


def _pd() -> Any:
    return importlib.import_module("pandas")


@dataclass(frozen=True)
class GenRandDataPlan:
    conditions: dict[str, dict[str, float]]
    size: int
    subjs: int
    subj_noise: float | dict[str, float]
    exclude_params: tuple[str, ...]
    share_noise: set[str]
    n_fast_outliers: int
    n_slow_outliers: int
    seed: int | None
    method: str


def gen_single_params_set(include: tuple[str, ...] = ()) -> dict[str, float]:
    np = _np()
    include_set = set(include)
    if "all" in include_set:
        include_set.update({"z", "sv", "sz", "st"})
    if "all_inter" in include_set:
        include_set.update({"sv", "sz", "st"})

    params = {
        "v": float((np.random.rand() - 0.5) * 8.0),
        "a": float(0.5 + np.random.rand() * 1.5),
        "t": float(0.2 + np.random.rand() * 0.3),
        "z": 0.5,
        "sv": 0.0,
        "sz": 0.0,
        "st": 0.0,
    }
    if "z" in include_set:
        params["z"] = float(0.4 + np.random.rand() * 0.2)
    if "sv" in include_set:
        params["sv"] = float(np.random.rand() * 2.5)
    if "sz" in include_set:
        params["sz"] = float(np.random.rand() * 0.4)
    if "st" in include_set:
        params["st"] = float(np.random.rand() * 0.35)
    return params


def gen_rand_params(
    include: tuple[str, ...] = (),
    cond_dict: Mapping[str, Mapping[str, float]] | None = None,
    seed: int | None = None,
) -> dict[str, float] | dict[str, dict[str, float]]:
    np = _np()
    if seed is not None:
        np.random.seed(seed)

    if cond_dict is None:
        return gen_single_params_set(include=include)

    out: dict[str, dict[str, float]] = {}
    for cond_name, fixed in cond_dict.items():
        generated = gen_single_params_set(include=include)
        generated.update({k: float(v) for k, v in fixed.items()})
        out[str(cond_name)] = generated
    return out


def build_generation_plan(
    params: Mapping[str, float] | Mapping[str, Mapping[str, float]] | None = None,
    *,
    n_fast_outliers: int = 0,
    n_slow_outliers: int = 0,
    size: int = 50,
    subjs: int = 1,
    subj_noise: float | dict[str, float] = 0.1,
    exclude_params: tuple[str, ...] = (),
    share_noise: set[str] | None = None,
    seed: int | None = None,
    method: str = "cdf",
) -> GenRandDataPlan:
    if params is None:
        params = gen_rand_params()

    normalized = _normalize_conditions(params)
    _validate_non_negative_int(n_fast_outliers, "n_fast_outliers")
    _validate_non_negative_int(n_slow_outliers, "n_slow_outliers")
    _validate_positive_int(size, "size")
    _validate_positive_int(subjs, "subjs")
    _validate_method(method)

    if isinstance(subj_noise, Mapping):
        subj_noise = {str(k): float(v) for k, v in subj_noise.items()}
    else:
        subj_noise = float(subj_noise)
        if subj_noise < 0:
            raise ValueError("subj_noise must be >= 0.")

    for cond_params in normalized.values():
        _validate_params(cond_params)

    share = set(_DEFAULT_SHARE_NOISE if share_noise is None else share_noise)
    return GenRandDataPlan(
        conditions=normalized,
        size=int(size),
        subjs=int(subjs),
        subj_noise=subj_noise,
        exclude_params=tuple(str(x) for x in exclude_params),
        share_noise=share,
        n_fast_outliers=int(n_fast_outliers),
        n_slow_outliers=int(n_slow_outliers),
        seed=None if seed is None else int(seed),
        method=method,
    )


def gen_rand_data(
    params: Mapping[str, float] | Mapping[str, Mapping[str, float]] | None = None,
    n_fast_outliers: int = 0,
    n_slow_outliers: int = 0,
    *,
    execute: bool = False,
    **kwargs: Any,
) -> tuple[Any, dict[str, dict[str, float]] | dict[str, float]]:
    plan = build_generation_plan(
        params=params,
        n_fast_outliers=n_fast_outliers,
        n_slow_outliers=n_slow_outliers,
        **kwargs,
    )

    if not execute:
        return _empty_result_frame(), _unbox_single_condition(plan.conditions)

    df = _simulate_dataset(plan)
    return df, _unbox_single_condition(plan.conditions)


def _simulate_dataset(plan: GenRandDataPlan) -> Any:
    np = _np()
    pd = _pd()
    rng = np.random.default_rng(plan.seed)

    subj_noise_map = _normalize_subj_noise(plan.subj_noise)
    rows: list[dict[str, Any]] = []
    condition_items = list(plan.conditions.items())
    shared_by_subj = {
        subj_idx: _sample_subject_offsets(
            rng,
            condition_items,
            subj_noise_map,
            plan.share_noise,
            plan.exclude_params,
        )
        for subj_idx in range(plan.subjs)
    }

    # We simulate subject-by-subject so each person can have stable noise offsets.
    for subj_idx in range(plan.subjs):
        for condition_name, base in condition_items:
            trial_params = _apply_subject_offsets(
                base,
                shared_by_subj[subj_idx].get(condition_name, {}),
            )
            _validate_params(trial_params)
            for _ in range(plan.size):
                rt, response = _simulate_ddm_trial(rng, trial_params, plan.method)
                rows.append(
                    {
                        "rt": float(rt),
                        "response": int(response),
                        "subj_idx": int(subj_idx),
                        "condition": condition_name,
                    }
                )

    data = pd.DataFrame.from_records(rows)
    if data.empty:
        return _empty_result_frame()

    _inject_outliers(data, rng, n_fast=plan.n_fast_outliers, n_slow=plan.n_slow_outliers)
    data["rt"] = data["rt"].astype(float)
    data["response"] = data["response"].astype(int)
    data["subj_idx"] = data["subj_idx"].astype(int)
    return data


def _normalize_subj_noise(subj_noise: float | dict[str, float]) -> dict[str, float]:
    if isinstance(subj_noise, Mapping):
        out = {str(k): max(0.0, float(v)) for k, v in subj_noise.items()}
    else:
        value = max(0.0, float(subj_noise))
        out = {k: value for k in _BASE_KEYS}
    out.setdefault("v", 0.0)
    out.setdefault("a", 0.0)
    out.setdefault("t", 0.0)
    out.setdefault("z", 0.0)
    out.setdefault("sv", 0.0)
    out.setdefault("sz", 0.0)
    out.setdefault("st", 0.0)
    return out


def _sample_subject_offsets(
    rng: Any,
    condition_items: list[tuple[str, dict[str, float]]],
    subj_noise_map: dict[str, float],
    share_noise: set[str],
    exclude_params: tuple[str, ...],
) -> dict[str, dict[str, float]]:
    excluded = set(exclude_params)
    allowed_shared = share_noise - excluded
    shared_values = {
        k: float(rng.normal(0.0, subj_noise_map.get(k, 0.0)))
        for k in allowed_shared
        if subj_noise_map.get(k, 0.0) > 0
    }

    out: dict[str, dict[str, float]] = {}
    for condition_name, _ in condition_items:
        offsets: dict[str, float] = {}
        for key, sigma in subj_noise_map.items():
            if key in excluded or sigma <= 0:
                continue
            if key in shared_values:
                offsets[key] = shared_values[key]
            else:
                offsets[key] = float(rng.normal(0.0, sigma))
        out[condition_name] = offsets
    return out


def _apply_subject_offsets(
    base: dict[str, float],
    offsets: dict[str, float],
) -> dict[str, float]:
    adjusted = dict(base)
    for key, delta in offsets.items():
        adjusted[key] = adjusted.get(key, 0.0) + float(delta)

    adjusted["a"] = max(1e-6, adjusted["a"])
    adjusted["t"] = max(1e-6, adjusted["t"])
    adjusted["z"] = min(1.0 - 1e-6, max(1e-6, adjusted["z"]))
    adjusted["sv"] = max(0.0, adjusted["sv"])
    adjusted["st"] = max(0.0, adjusted["st"])
    adjusted["sz"] = max(0.0, adjusted["sz"])

    max_sz = min(2.0 * adjusted["z"], 2.0 * (1.0 - adjusted["z"]))
    adjusted["sz"] = min(adjusted["sz"], max_sz)
    adjusted["st"] = min(adjusted["st"], 2.0 * adjusted["t"])
    return adjusted


def _simulate_ddm_trial(rng: Any, params: dict[str, float], method: str) -> tuple[float, int]:
    np = _np()
    a = params["a"]
    z = params["z"]
    t = params["t"]
    sv = params["sv"]
    sz = params["sz"]
    st = params["st"]

    v_trial = float(params["v"] + (rng.normal(0.0, sv) if sv > 0 else 0.0))
    z_low = max(1e-6, z - sz / 2.0)
    z_high = min(1.0 - 1e-6, z + sz / 2.0)
    z_trial = float(rng.uniform(z_low, z_high) if sz > 0 else z)
    t_low = max(1e-6, t - st / 2.0)
    t_high = max(t_low, t + st / 2.0)
    t_trial = float(rng.uniform(t_low, t_high) if st > 0 else t)

    # Smaller dt means finer integration (slower), larger dt means rougher (faster).
    dt_map = {"cdf": 0.001, "cdf_py": 0.0015, "drift": 0.002}
    dt = dt_map.get(method, 0.001)
    max_t = 20.0
    max_steps = int(max_t / dt)
    sqrt_dt = float(np.sqrt(dt))
    x = float(a * z_trial)

    # Euler walk: keep stepping until one boundary is hit.
    for step in range(1, max_steps + 1):
        x += v_trial * dt + float(rng.normal(0.0, sqrt_dt))
        if x >= a:
            return t_trial + step * dt, 1
        if x <= 0.0:
            return t_trial + step * dt, 0

    response = 1 if x >= a / 2.0 else 0
    return t_trial + max_t, response


def _inject_outliers(data: Any, rng: Any, *, n_fast: int, n_slow: int) -> None:
    np = _np()
    n_rows = int(len(data))
    if n_rows == 0:
        return

    fast_n = int(min(max(0, n_fast), n_rows))
    slow_n = int(min(max(0, n_slow), n_rows - fast_n))
    if fast_n == 0 and slow_n == 0:
        return

    indices = np.arange(n_rows)
    if fast_n > 0:
        fast_idx = rng.choice(indices, size=fast_n, replace=False)
        data.loc[fast_idx, "rt"] = rng.uniform(0.02, 0.15, size=fast_n)
        data.loc[fast_idx, "response"] = rng.integers(0, 2, size=fast_n)

        mask = np.ones(n_rows, dtype=bool)
        mask[fast_idx] = False
        indices = indices[mask]

    if slow_n > 0:
        slow_idx = rng.choice(indices, size=slow_n, replace=False)
        data.loc[slow_idx, "rt"] = data.loc[slow_idx, "rt"].to_numpy(dtype=float) + rng.uniform(
            2.0,
            5.0,
            size=slow_n,
        )
        data.loc[slow_idx, "response"] = rng.integers(0, 2, size=slow_n)


def _normalize_conditions(
    params: Mapping[str, float] | Mapping[str, Mapping[str, float]],
) -> dict[str, dict[str, float]]:
    if not params:
        raise ValueError("params must not be empty.")

    if all(isinstance(v, Mapping) for v in params.values()):
        out = {
            str(cond): _coerce_param_dict(cast(Mapping[str, Any], cond_params))
            for cond, cond_params in params.items()
        }
    else:
        out = {"none": _coerce_param_dict(cast(Mapping[str, Any], params))}

    return out


def _coerce_param_dict(values: Mapping[str, Any]) -> dict[str, float]:
    out = {k: float(values[k]) for k in _BASE_KEYS if k in values}
    missing = [k for k in ("v", "a", "t") if k not in out]
    if missing:
        raise ValueError(f"Missing required parameter(s): {', '.join(missing)}")
    out.setdefault("z", 0.5)
    out.setdefault("sv", 0.0)
    out.setdefault("sz", 0.0)
    out.setdefault("st", 0.0)
    return out


def _validate_params(params: Mapping[str, float]) -> None:
    np = _np()
    a = params["a"]
    z = params["z"]
    t = params["t"]
    sv = params["sv"]
    st = params["st"]
    sz = params["sz"]

    for key, value in params.items():
        if not np.isfinite(value):
            raise ValueError(f"Parameter {key} must be finite.")
        if key in _BOUNDS:
            lo, hi = _BOUNDS[key]
            if value < lo or value > hi:
                raise ValueError(f"Parameter {key} out of bounds: {value}")

    if a <= 0:
        raise ValueError("Parameter a must be > 0.")
    if z + sz / 2.0 > 1.0:
        raise ValueError("Invalid combination: z + sz/2 > 1.")
    if z - sz / 2.0 < 0.0:
        raise ValueError("Invalid combination: z - sz/2 < 0.")
    if t - st / 2.0 < 0.0:
        raise ValueError("Invalid combination: t - st/2 < 0.")
    if sv < 0 or st < 0 or sz < 0:
        raise ValueError("sv, st, sz must be >= 0.")


def _validate_positive_int(value: int, name: str) -> None:
    if int(value) != value or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")


def _validate_non_negative_int(value: int, name: str) -> None:
    if int(value) != value or value < 0:
        raise ValueError(f"{name} must be a non-negative integer.")


def _validate_method(method: str) -> None:
    if method not in {"cdf", "drift", "cdf_py"}:
        raise ValueError("method must be one of {'cdf', 'drift', 'cdf_py'}.")


def _empty_result_frame() -> Any:
    pd = _pd()
    return pd.DataFrame(
        {
            "rt": pd.Series(dtype=float),
            "response": pd.Series(dtype=int),
            "subj_idx": pd.Series(dtype=int),
            "condition": pd.Series(dtype=object),
        }
    )


def _unbox_single_condition(
    conditions: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]] | dict[str, float]:
    if len(conditions) == 1:
        return next(iter(conditions.values()))
    return conditions


if __name__ == "__main__":
    demo_data, demo_params = gen_rand_data(
        params={"v": 0.5, "a": 1.2, "t": 0.3},
        size=10,
        subjs=2,
        n_fast_outliers=1,
        n_slow_outliers=1,
        method="cdf",
        execute=True,
        seed=42,
    )
    print("gen_rand_data_hddm implementation ready")
    print(f"rows={len(demo_data)}")
    print(f"params={demo_params}")
