"""Five spec-§4 plots → PNG (+ optional Plotly HTML fragment).

All captions include the data-as-of date, key params, and the
"analysis tool, not investment advice" disclaimer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ilb.estimate import RegimeVols, ewma_sigma, rolling_drift, vol_table
from ilb.leverage import breakeven_map, breakeven_sigma
from ilb.scenarios import NamedPath
from ilb.simulate import ConditionalResult, simulate_leveraged_nav

matplotlib.use("Agg")
DISCLAIMER = "Analysis tool — not investment advice."


@dataclass(frozen=True)
class PlotContext:
    spot: float
    asof: date
    horizon_years: float
    regime: RegimeVols
    leverage: float = 2.0
    financing_rate: float = 0.0
    expense_ratio: float = 0.0
    theme: str = "dark"

    def caption(self, extra: str = "") -> str:
        base = (
            f"IREN | as of {self.asof} | spot=${self.spot:.2f} | "
            f"β={self.leverage} | c={self.financing_rate:.2%} φ={self.expense_ratio:.2%} | "
            f"regimes σ_low/base/high = {self.regime.low:.2f}/{self.regime.base:.2f}/"
            f"{self.regime.high:.2f}"
        )
        return f"{base}\n{extra}\n{DISCLAIMER}".strip()


def _apply_theme(theme: str) -> None:
    if theme == "dark":
        plt.style.use("dark_background")
    else:
        plt.style.use("default")


def _save(fig: plt.Figure, outdir: Path, name: str) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_vol_drift_timeseries(
    df: pd.DataFrame,
    ctx: PlotContext,
    windows: list[int],
    ewma_lambda: float,
    outdir: Path,
) -> Path:
    """1: rolling σ (W∈windows) + EWMA + regime bands; drift + price overlay below."""
    _apply_theme(ctx.theme)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    vt = vol_table(df["log_return"], windows)
    for col in vt.columns:
        ax1.plot(vt.index, vt[col], lw=1.1, label=col.replace("sigma_", "σ "))
    ewma = ewma_sigma(df["log_return"], lam=ewma_lambda)
    ax1.plot(ewma.index, ewma.values, lw=1.0, ls="--", label=f"EWMA λ={ewma_lambda}")
    for lvl, val, color in [
        ("σ_low", ctx.regime.low, "#5fb56f"),
        ("σ_base", ctx.regime.base, "#cccc55"),
        ("σ_high", ctx.regime.high, "#e07060"),
    ]:
        ax1.axhline(val, color=color, lw=1, ls=":", alpha=0.75, label=f"{lvl}={val:.2f}")
    ax1.set_ylabel("annualized σ")
    ax1.set_title("Rolling volatility + regime quantiles")
    ax1.legend(ncol=3, fontsize=8, loc="upper left")

    ax2.plot(df.index, df["close"], color="white" if ctx.theme == "dark" else "black",
             lw=0.9, label="close")
    ax2b = ax2.twinx()
    drift60 = rolling_drift(df["log_return"], 60)
    ax2b.plot(drift60.index, drift60.values, color="#7aa6ff", lw=0.7, alpha=0.7,
              label="60d annualized ν̂")
    ax2b.axhline(0, color="grey", lw=0.5, alpha=0.4)
    ax2.set_ylabel("close ($)")
    ax2b.set_ylabel("ν̂ (annual log)")
    ax2.set_xlabel("date")
    fig.suptitle("IREN volatility & drift over time", fontsize=12)
    fig.text(0.01, -0.02, ctx.caption("ν̂ shown for context only — not used as a forecast."),
             fontsize=7.5, alpha=0.7)
    return _save(fig, outdir, "vol_drift_timeseries.png")


def plot_breakeven_map(
    ctx: PlotContext,
    targets: np.ndarray,
    horizons: np.ndarray,
    outdir: Path,
) -> Path:
    """2: T × K grid coloured by L_T/L_0 ÷ S_T/S_0 at σ_base (>1 ⇒ 2x wins)."""
    _apply_theme(ctx.theme)
    grid = breakeven_map(
        targets, horizons, spot=ctx.spot, sigma_base=ctx.regime.base,
        leverage=ctx.leverage, financing_rate=ctx.financing_rate,
        expense_ratio=ctx.expense_ratio,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    extent = [targets.min(), targets.max(), horizons.min(), horizons.max()]
    im = ax.imshow(
        grid, origin="lower", extent=extent, aspect="auto",
        cmap="RdYlGn", vmin=0.5, vmax=2.0,
    )
    cs = ax.contour(targets, horizons, grid, levels=[1.0], colors="white", linewidths=1.5)
    ax.clabel(cs, fmt={1.0: "tie"}, fontsize=8)
    for k in [150, 450, 700]:
        if extent[0] <= k <= extent[1]:
            ax.axvline(k, color="white", lw=0.6, ls=":", alpha=0.7)
            ax.text(k, horizons.max(), f" K=${k}", va="top", fontsize=8, alpha=0.85)
    fig.colorbar(im, ax=ax, label="L_T/L_0  ÷  S_T/S_0   (>1 ⇒ 2x wins)")
    ax.set_xlabel("target price K ($)")
    ax.set_ylabel("holding period T (years)")
    ax.set_title("Breakeven map: holding period × target  (at σ_base)")
    fig.text(0.01, -0.02, ctx.caption(
        f"Coloured at σ_base={ctx.regime.base:.2f}. Horizon often dominates."
    ), fontsize=7.5, alpha=0.7)
    return _save(fig, outdir, "breakeven_map.png")


def plot_breakeven_curve(
    ctx: PlotContext,
    targets: np.ndarray,
    outdir: Path,
    sigma_max: float = 2.0,
    n: int = 200,
) -> Path:
    """3: ratio vs realized σ, one curve per K, with regime band overlay."""
    _apply_theme(ctx.theme)
    sigmas = np.linspace(1e-4, sigma_max, n)
    T = ctx.horizon_years
    beta = ctx.leverage
    fig, ax = plt.subplots(figsize=(10, 6))
    for K in targets:
        gross = K / ctx.spot
        qv = (sigmas**2) * T
        drag = (
            0.5 * (beta**2 - beta) * qv
            + (beta - 1.0) * ctx.financing_rate * T
            + ctx.expense_ratio * T
        )
        ratio = (gross**beta) * np.exp(-drag) / gross
        sigma_be = float(breakeven_sigma(
            K, ctx.spot, T,
            financing_rate=ctx.financing_rate, expense_ratio=ctx.expense_ratio,
            leverage=beta,
        ))
        line, = ax.plot(sigmas, ratio, lw=1.4, label=f"K=${int(K)}  σ_be={sigma_be:.2f}")
        ax.axvline(sigma_be, color=line.get_color(), lw=0.7, ls=":", alpha=0.6)
    ax.axhline(1.0, color="white" if ctx.theme == "dark" else "black", lw=0.6, alpha=0.6)
    # Regime band overlay (single shaded region from σ_low → σ_high, σ_base line)
    ax.axvspan(ctx.regime.low, ctx.regime.high, color="orange", alpha=0.10,
               label="regime σ low–high")
    ax.axvline(ctx.regime.base, color="orange", lw=1, ls="--", alpha=0.75,
               label=f"σ_base={ctx.regime.base:.2f}")
    ax.set_xlabel("realized annualized volatility (σ)")
    ax.set_ylabel("L_T/L_0  ÷  S_T/S_0")
    ax.set_title(f"Breakeven volatility curves (T={T:.2f}y)")
    ax.legend(fontsize=8, loc="upper right")
    fig.text(0.01, -0.02, ctx.caption("Above 1.0 → 2x wins; below → 1x wins."),
             fontsize=7.5, alpha=0.7)
    return _save(fig, outdir, "breakeven_curve.png")


def plot_named_paths(
    ctx: PlotContext,
    paths: list[NamedPath],
    outdir: Path,
) -> Path:
    """4: same-K named paths — left panel prices, right panel 1x vs 2x terminal bars."""
    _apply_theme(ctx.theme)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 5),
                                   gridspec_kw={"width_ratios": [2, 1]})
    names, term_1x, term_2x, qvs = [], [], [], []
    for p in paths:
        axL.plot(p.prices, lw=1.4, label=f"{p.name}  QV≈{p.realized_qv:.3f}")
        nav, _ = simulate_leveraged_nav(
            p.log_returns.reshape(1, -1), leverage=ctx.leverage,
            financing_rate=ctx.financing_rate, expense_ratio=ctx.expense_ratio,
        )
        names.append(p.name)
        term_1x.append(p.prices[-1] / p.prices[0])
        term_2x.append(float(nav[0, -1]))
        qvs.append(p.realized_qv)
    axL.set_xlabel("trading day")
    axL.set_ylabel("price ($)")
    axL.set_title("Same-K paths, different realized variance")
    axL.legend(fontsize=8)

    x = np.arange(len(names))
    axR.bar(x - 0.18, term_1x, width=0.36, label="1x (S_T/S_0)", alpha=0.85)
    axR.bar(x + 0.18, term_2x, width=0.36, label="2x (L_T/L_0)", alpha=0.85)
    for i, (a, b) in enumerate(zip(term_1x, term_2x, strict=True)):
        axR.text(i - 0.18, a, f"{a:.2f}", ha="center", va="bottom", fontsize=8)
        axR.text(i + 0.18, b, f"{b:.2f}", ha="center", va="bottom", fontsize=8)
    axR.set_xticks(x)
    axR.set_xticklabels(names, rotation=15)
    axR.axhline(1.0, color="grey", lw=0.5, alpha=0.5)
    axR.set_title("Terminal multiple: 1x vs 2x")
    axR.legend(fontsize=8)
    fig.text(0.01, -0.02, ctx.caption(
        "Continuous-time: 2x depends only on QV. Discrete daily rebalance breaks ties."
    ), fontsize=7.5, alpha=0.7)
    return _save(fig, outdir, "named_paths.png")


def plot_conditional_dispersion(
    ctx: PlotContext,
    results: list[ConditionalResult],
    outdir: Path,
) -> Path:
    """5: violin of conditional 2x terminal NAV per target + impairment annotations."""
    _apply_theme(ctx.theme)
    fig, ax = plt.subplots(figsize=(10, 6))
    labels, dists = [], []
    for r in results:
        if not np.isfinite(r.mean_2x_terminal) or r.n_paths_in_band == 0:
            continue
        # Re-derive distribution from impairment stats — we don't keep full dist
        # here; rely on summary annotations and a synthetic violin of percentiles.
        labels.append(
            f"K=${int(r.target_price)}\n"
            f"n_in={r.n_paths_in_band}/{r.n_paths_total}\n"
            f"win 2x={r.win_rate_2x:.0%}\n"
            f"impair={r.impairment.impair_fraction:.1%}"
        )
        # Build a 3-point pseudo-distribution from percentiles for visual cue
        dists.append([r.impairment.p01, r.impairment.p05,
                      r.median_2x_terminal, r.mean_2x_terminal])
    if not dists:
        ax.text(0.5, 0.5, "No paths fell within any target band — widen band.",
                ha="center", va="center", transform=ax.transAxes)
    else:
        positions = np.arange(len(dists))
        bp = ax.boxplot(dists, positions=positions, widths=0.55, patch_artist=True,
                        medianprops={"color": "white"})
        for patch in bp["boxes"]:
            patch.set_alpha(0.6)
        for i, r in enumerate(results):
            if np.isfinite(r.mean_1x_terminal):
                ax.scatter([i], [r.mean_1x_terminal], marker="x", color="white",
                           s=60, label="1x mean" if i == 0 else None, zorder=5)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=8)
        ax.axhline(1.0, color="grey", lw=0.5, alpha=0.6)
        ax.set_ylabel("terminal multiple")
        ax.set_title("Terminal-K conditioned dispersion: 2x vs 1x (+ impairment)")
        ax.legend(fontsize=8, loc="upper left")
    fig.text(0.01, -0.02, ctx.caption(
        "Box = (p01, p05, median, mean) of 2x terminal multiple among in-band paths."
    ), fontsize=7.5, alpha=0.7)
    return _save(fig, outdir, "conditional_dispersion.png")
