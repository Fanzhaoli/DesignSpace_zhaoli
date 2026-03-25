from __future__ import annotations

import subprocess
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Study2.gen_data_rdist_fastdm import rdiffusion_fastdm


def _np() -> Any:
    return importlib.import_module("numpy")


def _plt() -> Any:
    return importlib.import_module("matplotlib.pyplot")


def _gaussian_kde() -> Any:
    return importlib.import_module("scipy.stats").gaussian_kde


np = _np()
plt = _plt()
gaussian_kde = _gaussian_kde()


@dataclass(frozen=True)
class ParamSet:
    name: str
    a: float
    v: float
    t0: float
    z: float | None = None
    d: float = 0.0
    sz: float = 0.0
    sv: float = 0.0
    st0: float = 0.0
    s: float = 1.0


def _r_stats(param: ParamSet, n: int, seed: int) -> Any:
    z_expr = "NULL" if param.z is None else f"{param.z}"
    expr = (
        "suppressPackageStartupMessages(library(rtdists));"
        f"set.seed({seed});"
        "x <- rtdists::rdiffusion("
        f"n={n},a={param.a},v={param.v},t0={param.t0},"
        + ("z=0.5*" + str(param.a) + "," if param.z is None else f"z={z_expr},")
        + f"d={param.d},sz={param.sz},sv={param.sv},st0={param.st0},s={param.s});"
        "q <- as.numeric(stats::quantile(x$rt, probs=c(0.1,0.5,0.9), names=FALSE));"
        "vals <- c(mean(x$rt), stats::var(x$rt), mean(x$response=='upper'), q);"
        "cat(paste(vals, collapse=','))"
    )
    out = subprocess.run(
        ["Rscript", "-e", expr],
        check=True,
        capture_output=True,
        text=True,
    )
    return np.array([float(v) for v in out.stdout.strip().split(",")], dtype=float)


def _r_rt_samples(param: ParamSet, n: int, seed: int) -> Any:
    z_expr = "NULL" if param.z is None else f"{param.z}"
    expr = (
        "suppressPackageStartupMessages(library(rtdists));"
        f"set.seed({seed});"
        "x <- rtdists::rdiffusion("
        f"n={n},a={param.a},v={param.v},t0={param.t0},"
        + ("z=0.5*" + str(param.a) + "," if param.z is None else f"z={z_expr},")
        + f"d={param.d},sz={param.sz},sv={param.sv},st0={param.st0},s={param.s});"
        "cat(paste(x$rt, collapse=','))"
    )
    out = subprocess.run(
        ["Rscript", "-e", expr],
        check=True,
        capture_output=True,
        text=True,
    )
    txt = out.stdout.strip()
    if not txt:
        return np.array([], dtype=float)
    return np.array([float(v) for v in txt.split(",")], dtype=float)


def _py_stats(param: ParamSet, n: int, seed: int) -> Any:
    df = rdiffusion_fastdm(
        n=n,
        a=param.a,
        v=param.v,
        t0=param.t0,
        z=param.z,
        d=param.d,
        sz=param.sz,
        sv=param.sv,
        st0=param.st0,
        s=param.s,
        seed=seed,
        precision=3,
        max_time=20.0,
    )
    q = np.quantile(df["rt"].to_numpy(), [0.1, 0.5, 0.9])
    vals = np.array(
        [
            float(df["rt"].mean()),
            float(df["rt"].var(ddof=1)),
            float((df["response"] == "upper").mean()),
            float(q[0]),
            float(q[1]),
            float(q[2]),
        ],
        dtype=float,
    )
    return vals


def _py_rt_samples(param: ParamSet, n: int, seed: int) -> Any:
    df = rdiffusion_fastdm(
        n=n,
        a=param.a,
        v=param.v,
        t0=param.t0,
        z=param.z,
        d=param.d,
        sz=param.sz,
        sv=param.sv,
        st0=param.st0,
        s=param.s,
        seed=seed,
        precision=3,
        max_time=20.0,
    )
    return df["rt"].to_numpy(dtype=float)


def _plot_distribution_panel(data: list[tuple[ParamSet, Any, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, len(data), figsize=(6 * len(data), 8.2), dpi=130)
    if len(data) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    q_levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=float)
    q_labels = ["q10", "q30", "q50", "q70", "q90"]

    for col, (param, r_rt, py_rt) in enumerate(data):
        ax = axes[0, col]
        bins = 80
        lo = float(min(r_rt.min(), py_rt.min()))
        hi = float(max(r_rt.max(), py_rt.max()))

        ax.hist(r_rt, bins=bins, range=(lo, hi), density=True, alpha=0.28, color="#1f77b4", label="R hist")
        ax.hist(py_rt, bins=bins, range=(lo, hi), density=True, alpha=0.28, color="#d62728", label="Py hist")

        x = np.linspace(lo, hi, 512)
        r_kde = gaussian_kde(r_rt)
        py_kde = gaussian_kde(py_rt)
        ax.plot(x, r_kde(x), color="#1f77b4", linewidth=2.0, label="R density")
        ax.plot(x, py_kde(x), color="#d62728", linewidth=2.0, label="Py density")

        ax.set_title(f"{param.name}: histogram + density")
        ax.set_xlabel("RT (seconds)")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)

        ax_q = axes[1, col]
        r_q = np.quantile(r_rt, q_levels)
        py_q = np.quantile(py_rt, q_levels)
        ax_q.plot(q_levels, r_q, marker="o", linewidth=2.0, color="#1f77b4", label="R quantiles")
        ax_q.plot(q_levels, py_q, marker="o", linewidth=2.0, color="#d62728", label="Py quantiles")
        ax_q.set_xticks(q_levels, q_labels)
        ax_q.set_xlabel("Cumulative quantile")
        ax_q.set_ylabel("RT (seconds)")
        ax_q.set_title(f"{param.name}: cumulative quantiles")
        ax_q.grid(alpha=0.25, linewidth=0.6)
        ax_q.legend(frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "rdiffusion_fastdm_compare_panel.png")
    plt.close(fig)


def main() -> None:
    n = 5000
    seed = 42
    params = [
        ParamSet(name="basic", a=1.0, v=2.0, t0=0.3),
        ParamSet(name="scaled_s", a=0.1, v=0.2, t0=0.3, s=0.1),
        ParamSet(name="with_var", a=1.2, v=1.3, t0=0.25, z=0.55, d=0.05, sz=0.1, sv=0.3, st0=0.08),
    ]
    metric_names = ["mean_rt", "var_rt", "p_upper", "q10", "q50", "q90"]
    out_dir = PROJECT_ROOT / "Study2" / "figures"
    plot_data: list[tuple[ParamSet, Any, Any]] = []

    print("name,metric,r_value,py_value,abs_diff,rel_diff")
    for p in params:
        r_vals = _r_stats(p, n=n, seed=seed)
        py_vals = _py_stats(p, n=n, seed=seed)
        for i, metric in enumerate(metric_names):
            r_v = r_vals[i]
            py_v = py_vals[i]
            abs_diff = abs(py_v - r_v)
            rel_diff = abs_diff / max(abs(r_v), 1e-12)
            print(f"{p.name},{metric},{r_v:.8f},{py_v:.8f},{abs_diff:.8f},{rel_diff:.8f}")
        r_rt = _r_rt_samples(p, n=n, seed=seed)
        py_rt = _py_rt_samples(p, n=n, seed=seed)
        plot_data.append((p, r_rt, py_rt))

    _plot_distribution_panel(plot_data, out_dir=out_dir)


if __name__ == "__main__":
    main()
