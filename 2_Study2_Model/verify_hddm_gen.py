from __future__ import annotations

"""
Comparison and validation workflow for HDDM reverse engineering.

Big picture for beginners:
1) We simulate data from our Python implementation (`gen_data_hddm.py`).
2) We optionally simulate reference data from HDDM (inside Docker).
3) We compare summary statistics and quantiles.
4) We can also export reusable HDDM reference CSVs and draw a panel figure.

This file now combines three older scripts into one CLI:
- parity checks
- HDDM reference export
- comparison panel plotting
"""

import argparse
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Study2.gen_data_hddm import gen_rand_data


@dataclass(frozen=True)
class Scenario:
    name: str
    params: dict[str, float]
    method: str
    n_fast_outliers: int = 0
    n_slow_outliers: int = 0


@dataclass(frozen=True)
class SummaryStats:
    mean_rt: float
    var_rt: float
    p_upper: float
    q10: float
    q30: float
    q50: float
    q70: float
    q90: float


METHODS: tuple[str, ...] = ("cdf", "cdf_py", "drift")
METHOD_STYLES: dict[str, dict[str, str]] = {
    "cdf": {"color": "#d62728", "marker": "o", "label": "Py-cdf"},
    "cdf_py": {"color": "#ff7f0e", "marker": "o", "label": "Py-cdf_py"},
    "drift": {"color": "#2ca02c", "marker": "o", "label": "Py-drift"},
}
HDDM_COLOR = "#1f77b4"


def _np() -> Any:
    return importlib.import_module("numpy")


def _pd() -> Any:
    return importlib.import_module("pandas")


def _plt() -> Any:
    return importlib.import_module("matplotlib.pyplot")


def _gaussian_kde() -> Any:
    return importlib.import_module("scipy.stats").gaussian_kde


def scenario_list() -> list[Scenario]:
    return [
        Scenario(
            name="baseline",
            params={"v": 0.8, "a": 1.2, "t": 0.3, "z": 0.5, "sv": 0.0, "sz": 0.0, "st": 0.0},
            method="cdf",
        ),
        Scenario(
            name="variability",
            params={"v": 0.5, "a": 1.4, "t": 0.35, "z": 0.48, "sv": 0.8, "sz": 0.2, "st": 0.15},
            method="cdf_py",
        ),
        Scenario(
            name="scale_invariance",
            params={"v": 1.2, "a": 1.8, "t": 0.42, "z": 0.5, "sv": 0.0, "sz": 0.0, "st": 0.0},
            method="drift",
            n_fast_outliers=10,
            n_slow_outliers=10,
        ),
    ]


def scenario_seed(base_seed: int, index: int) -> int:
    return int(base_seed + index * 1000)


def hddm_cache_filename(scenario_name: str, size: int, subjs: int, seed: int) -> str:
    return f"hddm_gen_rand_data_{scenario_name}_n{size}_subjs{subjs}_seed{seed}.csv"


def summarize_dataframe(df: Any) -> SummaryStats:
    np = _np()
    if df.empty:
        nan = float("nan")
        return SummaryStats(nan, nan, nan, nan, nan, nan, nan, nan)

    rt = df["rt"].to_numpy(dtype=float)
    response = df["response"].to_numpy()
    q = np.quantile(rt, [0.1, 0.3, 0.5, 0.7, 0.9])
    return SummaryStats(
        mean_rt=float(np.mean(rt)),
        var_rt=float(np.var(rt, ddof=1)) if rt.size > 1 else 0.0,
        p_upper=float(np.mean(response == 1)),
        q10=float(q[0]),
        q30=float(q[1]),
        q50=float(q[2]),
        q70=float(q[3]),
        q90=float(q[4]),
    )


def _try_hddm_reference(
    params: dict[str, float],
    size: int,
    subjs: int,
    seed: int,
    n_fast_outliers: int,
    n_slow_outliers: int,
) -> Any:
    pd = _pd()
    try:
        hddm = importlib.import_module("hddm")
    except Exception as exc:
        raise RuntimeError("HDDM is not installed in this environment.") from exc

    data, _ = hddm.generate.gen_rand_data(
        params=params,
        size=size,
        subjs=subjs,
        n_fast_outliers=n_fast_outliers,
        n_slow_outliers=n_slow_outliers,
        seed=seed,
    )
    return pd.DataFrame(data)


def _load_cached_hddm_reference(cache_dir: Path, scenario_name: str, size: int, subjs: int, seed: int) -> Any:
    pd = _pd()
    detailed = cache_dir / hddm_cache_filename(scenario_name, size, subjs, seed)
    latest = cache_dir / f"hddm_gen_rand_data_{scenario_name}_latest.csv"
    if detailed.exists():
        return pd.read_csv(detailed)
    if latest.exists():
        return pd.read_csv(latest)
    return None


def _save_cached_hddm_reference(
    cache_dir: Path,
    scenario_name: str,
    size: int,
    subjs: int,
    seed: int,
    df: Any,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    detailed = cache_dir / hddm_cache_filename(scenario_name, size, subjs, seed)
    latest = cache_dir / f"hddm_gen_rand_data_{scenario_name}_latest.csv"
    df.to_csv(detailed, index=False)
    df.to_csv(latest, index=False)


def _run_candidate(scenario: Scenario, size: int, subjs: int, seed: int) -> tuple[Any, SummaryStats]:
    df, _ = gen_rand_data(
        params=scenario.params,
        size=size,
        subjs=subjs,
        n_fast_outliers=scenario.n_fast_outliers,
        n_slow_outliers=scenario.n_slow_outliers,
        seed=seed,
        method=scenario.method,
        execute=True,
    )
    return df, summarize_dataframe(df)


def _format_delta(candidate: SummaryStats, reference: SummaryStats) -> str:
    fields = ("mean_rt", "var_rt", "p_upper", "q10", "q30", "q50", "q70", "q90")
    parts: list[str] = []
    for field in fields:
        cand_v = float(getattr(candidate, field))
        ref_v = float(getattr(reference, field))
        parts.append(f"{field}_abs_diff={abs(cand_v - ref_v):.6f}")
    return ",".join(parts)


def run_parity(args: argparse.Namespace) -> None:
    print("phase4_status,ok")
    print(f"phase4_size,{args.size}")
    print(f"phase4_subjs,{args.subjs}")
    print(f"phase4_seed,{args.seed}")
    print(f"phase4_with_hddm_ref,{int(args.with_hddm_ref)}")

    for idx, scenario in enumerate(scenario_list()):
        run_seed = scenario_seed(args.seed, idx)
        candidate_df, cand_stats = _run_candidate(scenario, args.size, args.subjs, run_seed)
        print(f"scenario,{scenario.name}")
        print(f"candidate_rows,{len(candidate_df)}")
        print(f"candidate_summary,{cand_stats}")

        if not args.with_hddm_ref:
            continue

        try:
            ref_df = _try_hddm_reference(
                params=scenario.params,
                size=args.size,
                subjs=args.subjs,
                seed=run_seed,
                n_fast_outliers=scenario.n_fast_outliers,
                n_slow_outliers=scenario.n_slow_outliers,
            )
            ref_stats = summarize_dataframe(ref_df)
            print(f"reference_rows,{len(ref_df)}")
            print(f"reference_summary,{ref_stats}")
            print(f"parity_deltas,{_format_delta(cand_stats, ref_stats)}")
        except RuntimeError as exc:
            print(f"reference_status,skipped,{exc}")


def run_export(args: argparse.Namespace) -> None:
    for idx, scenario in enumerate(scenario_list()):
        run_seed = scenario_seed(args.seed, idx)
        ref_df = _try_hddm_reference(
            params=scenario.params,
            size=args.size,
            subjs=args.subjs,
            seed=run_seed,
            n_fast_outliers=scenario.n_fast_outliers,
            n_slow_outliers=scenario.n_slow_outliers,
        )
        ref_df["scenario"] = scenario.name
        ref_df["source"] = "hddm_reference"
        _save_cached_hddm_reference(
            cache_dir=args.hddm_cache_dir,
            scenario_name=scenario.name,
            size=args.size,
            subjs=args.subjs,
            seed=run_seed,
            df=ref_df,
        )
        print(f"hddm_export_saved,{scenario.name},{len(ref_df)}")


def _generate_candidate_by_method(scenario: Scenario, size: int, subjs: int, seed: int) -> dict[str, Any]:
    pd = _pd()
    out: dict[str, Any] = {}
    for method_idx, method in enumerate(METHODS):
        method_seed = int(seed + method_idx * 100)
        candidate_df, _ = gen_rand_data(
            params=scenario.params,
            size=size,
            subjs=subjs,
            n_fast_outliers=scenario.n_fast_outliers,
            n_slow_outliers=scenario.n_slow_outliers,
            seed=method_seed,
            method=method,
            execute=True,
        )
        out[method] = pd.DataFrame(candidate_df)
    return out


def _plot_density_row(ax: Any, candidate_by_method: dict[str, Any], reference_df: Any) -> None:
    np = _np()
    gaussian_kde = _gaussian_kde()
    rt_sets: list[Any] = [candidate_by_method[m]["rt"].to_numpy(dtype=float) for m in METHODS]
    if reference_df is not None and not reference_df.empty:
        rt_ref = reference_df["rt"].to_numpy(dtype=float)
        rt_sets.append(rt_ref)
    else:
        rt_ref = None

    lo = float(min(arr.min() for arr in rt_sets))
    hi = float(max(arr.max() for arr in rt_sets))
    bins = 80
    x = np.linspace(lo, hi, 512)

    if reference_df is not None and not reference_df.empty:
        ax.hist(rt_ref, bins=bins, range=(lo, hi), density=True, alpha=0.28, color=HDDM_COLOR, label="HDDM ref hist")
        ref_kde = gaussian_kde(rt_ref)
        ax.plot(x, ref_kde(x), color=HDDM_COLOR, linewidth=2.6, linestyle="--", label="HDDM ref density")

    for method in METHODS:
        style = METHOD_STYLES[method]
        rt_candidate = candidate_by_method[method]["rt"].to_numpy(dtype=float)
        ax.hist(rt_candidate, bins=bins, range=(lo, hi), density=True, alpha=0.16, color=style["color"], label=f"{style['label']} hist")
        py_kde = gaussian_kde(rt_candidate)
        ax.plot(x, py_kde(x), color=style["color"], linewidth=2.0, label=f"{style['label']} density")

    ax.legend(frameon=False, fontsize=8, title="Source", title_fontsize=8)


def _plot_quantile_row(ax: Any, candidate_by_method: dict[str, Any], reference_df: Any, quantile_levels: Any) -> None:
    np = _np()
    q_labels = ["q10", "q30", "q50", "q70", "q90"]
    if reference_df is not None and not reference_df.empty:
        ref_q = np.quantile(reference_df["rt"].to_numpy(dtype=float), quantile_levels)
        ax.plot(quantile_levels, ref_q, marker="s", color=HDDM_COLOR, linewidth=2.6, linestyle="--", label="HDDM ref quantiles")

    for method in METHODS:
        style = METHOD_STYLES[method]
        cand_q = np.quantile(candidate_by_method[method]["rt"].to_numpy(dtype=float), quantile_levels)
        ax.plot(quantile_levels, cand_q, marker=style["marker"], color=style["color"], linewidth=2.2, label=f"{style['label']} quantiles")

    ax.set_xticks(quantile_levels, q_labels)
    ax.legend(frameon=False, fontsize=8, title="Source", title_fontsize=8)


def run_panel(args: argparse.Namespace) -> None:
    np = _np()
    plt = _plt()
    scenarios = scenario_list()
    quantile_levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(scenarios), figsize=(6 * len(scenarios), 8.2), dpi=130)
    if len(scenarios) == 1:
        axes = np.array([[axes[0]], [axes[1]]])

    has_reference = False
    for col, scenario in enumerate(scenarios):
        run_seed = scenario_seed(args.seed, col)
        candidate_by_method = _generate_candidate_by_method(scenario, args.size, args.subjs, run_seed)

        reference_df = _load_cached_hddm_reference(
            cache_dir=args.hddm_cache_dir,
            scenario_name=scenario.name,
            size=args.size,
            subjs=args.subjs,
            seed=run_seed,
        )
        if reference_df is None:
            try:
                reference_df = _try_hddm_reference(
                    params=scenario.params,
                    size=args.size,
                    subjs=args.subjs,
                    seed=run_seed,
                    n_fast_outliers=scenario.n_fast_outliers,
                    n_slow_outliers=scenario.n_slow_outliers,
                )
                _save_cached_hddm_reference(
                    cache_dir=args.hddm_cache_dir,
                    scenario_name=scenario.name,
                    size=args.size,
                    subjs=args.subjs,
                    seed=run_seed,
                    df=reference_df,
                )
            except RuntimeError:
                reference_df = None

        if reference_df is not None:
            has_reference = True

        ax_hist = axes[0, col]
        ax_q = axes[1, col]
        _plot_density_row(ax_hist, candidate_by_method, reference_df)
        _plot_quantile_row(ax_q, candidate_by_method, reference_df, quantile_levels)
        ax_hist.set_title(f"{scenario.name}: histogram + density")
        ax_hist.set_xlabel("RT (seconds)")
        ax_hist.set_ylabel("Density")
        ax_q.set_title(f"{scenario.name}: cumulative quantiles")
        ax_q.set_xlabel("Cumulative quantile")
        ax_q.set_ylabel("RT (seconds)")
        ax_q.grid(alpha=0.25, linewidth=0.6)

    if not has_reference and not args.allow_missing_hddm:
        raise RuntimeError(
            "HDDM reference unavailable. Run export/panel in hddm-shell once, or use --allow-missing-hddm intentionally."
        )

    title_suffix = "with HDDM reference" if has_reference else "(HDDM unavailable)"
    fig.suptitle(f"gen_rand_data comparison panel {title_suffix}", y=0.995, fontsize=11)
    fig.tight_layout()
    fig.savefig(args.output)
    print(f"phase5_panel_saved,{args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="HDDM reverse-engineering comparison toolkit.")
    parser.add_argument("--mode", choices=["parity", "export", "panel"], default="parity")
    parser.add_argument("--size", type=int, default=3000)
    parser.add_argument("--subjs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--with-hddm-ref", action="store_true", help="Used in parity mode.")
    parser.add_argument("--allow-missing-hddm", action="store_true", help="Used in panel mode.")
    parser.add_argument("--output", type=Path, default=Path("Study2/figures/gen_rand_data_hddm_compare_panel.png"))
    parser.add_argument("--hddm-cache-dir", type=Path, default=Path("Study2/data/hddm_synthetic"))
    args = parser.parse_args()

    if args.mode == "parity":
        run_parity(args)
    elif args.mode == "export":
        run_export(args)
    else:
        run_panel(args)


if __name__ == "__main__":
    main()
