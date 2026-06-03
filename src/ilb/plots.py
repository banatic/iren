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

from ilb.btc_regime import DecouplingResult
from ilb.estimate import RegimeVols, ewma_sigma, rolling_drift, rolling_sigma, vol_table
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


def _shade_regimes(ax, split, x0, x1) -> None:
    """Shade miner-era (≤ split) red and cloud-era (> split) green on a time axis."""
    if split is None:
        return
    ax.axvspan(x0, split, color="#e07060", alpha=0.07, zorder=0)
    ax.axvspan(split, x1, color="#5fb56f", alpha=0.07, zorder=0)


def plot_btc_decoupling(
    ctx: PlotContext,
    result: DecouplingResult,
    outdir: Path,
) -> Path:
    """6: IREN↔BTC rolling β (top) and ρ (bottom) with detected breaks + regime shading.

    Diagnostic for the regime-contamination story: IREN's BTC β/ρ are regime-
    switching (not a clean monotone decouple), so a full-history σ blends eras.
    """
    _apply_theme(ctx.theme)
    bc = result.beta_corr
    x0, x1 = bc.index.min(), bc.index.max()
    fig, (axb, axr) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})

    beta_colors = {60: "#7aa6ff", 120: "#cccc55"}
    rho_colors = {60: "#5fb56f", 120: "#e07060"}
    for w in result.windows:
        if f"beta_{w}" in bc:
            axb.plot(bc.index, bc[f"beta_{w}"], lw=1.1, color=beta_colors.get(w),
                     label=f"β {w}d")
        if f"rho_{w}" in bc:
            axr.plot(bc.index, bc[f"rho_{w}"], lw=1.1, color=rho_colors.get(w),
                     label=f"ρ {w}d")

    axb.axhline(1.0, color="grey", lw=0.6, ls="-", alpha=0.5)
    axb.axhline(0.0, color="grey", lw=0.5, ls=":", alpha=0.4)
    axr.axhline(0.0, color="grey", lw=0.5, ls=":", alpha=0.4)

    # all detected breaks as faint vlines; the chosen split prominent + shading
    for d in result.break_dates:
        for ax in (axb, axr):
            ax.axvline(d, color="white", lw=0.5, ls=":", alpha=0.35, zorder=1)
    _shade_regimes(axb, result.split_date, x0, x1)
    _shade_regimes(axr, result.split_date, x0, x1)
    if result.split_date is not None:
        for ax in (axb, axr):
            ax.axvline(result.split_date, color="orange", lw=1.3, ls="--", alpha=0.9, zorder=2)
        axb.text(result.split_date, axb.get_ylim()[1], "  miner ↔ cloud split",
                 va="top", ha="left", fontsize=8, color="orange", alpha=0.9)
    for d in result.catalysts:
        for ax in (axb, axr):
            ax.axvline(d, color="#36d6e7", lw=1.2, ls="-.", alpha=0.85, zorder=2)

    axb.set_ylabel("β  (IREN on BTC)")
    axb.set_title("IREN ↔ BTC decoupling: rolling OLS β and correlation ρ")
    axb.legend(fontsize=8, loc="upper left", ncol=len(result.windows))
    axr.set_ylabel("ρ  (Pearson)")
    axr.set_xlabel("date")
    axr.legend(fontsize=8, loc="upper left", ncol=len(result.windows))
    axr.set_ylim(-0.2, 1.0)

    split_str = result.split_date.date().isoformat() if result.split_date is not None else "none"
    fig.suptitle("BTC coupling over time (miner → AI-cloud regime)", fontsize=12)
    fig.text(0.01, -0.02, ctx.caption(
        f"Breaks via {result.method.upper()} (pen={result.penalty:g}) on β_60. "
        f"Regime split {split_str}. β is regime-switching, not a clean decouple — the "
        f"material change is the σ level (see next figure)."
    ), fontsize=7.5, alpha=0.7)
    return _save(fig, outdir, "btc_decoupling.png")


def plot_regime_sigma_compare(
    ctx: PlotContext,
    df: pd.DataFrame,
    result: DecouplingResult,
    verdict_targets: list[float],
    outdir: Path,
    window: int = 60,
) -> Path:
    """7: full-history vs post-break σ distributions/bands + σ_be(K) verdict markers.

    Shows which regime a target's breakeven σ lands in — i.e. whether using the
    post-pivot (lower-σ) regime instead of the contaminated full-history regime
    flips the 2x-vs-1x verdict.
    """
    _apply_theme(ctx.theme)
    r = df["log_return"]
    full_roll = rolling_sigma(r, window).dropna()
    if result.split_date is not None:
        post_roll = rolling_sigma(r[r.index > result.split_date], window).dropna()
    else:
        post_roll = full_roll
    samples = [full_roll.to_numpy(), post_roll.to_numpy()]
    regimes = [result.full_regime, result.post_regime]
    pos = [1, 2]
    fill = ["#e07060", "#5fb56f"]  # miner-era / cloud-era
    split_str = result.split_date.date().isoformat() if result.split_date is not None else "none"
    labels = [f"Full history\n(miner+cloud, n={result.n_obs_full})",
              f"Post-break\n(>{split_str}, n={result.n_obs_post})"]

    fig, ax = plt.subplots(figsize=(10, 6))
    vp = ax.violinplot(samples, positions=pos, widths=0.7, showextrema=False)
    for body, c in zip(vp["bodies"], fill, strict=True):
        body.set_facecolor(c)
        body.set_alpha(0.30)
        body.set_edgecolor(c)

    # low / base / high markers per regime
    for p, reg in zip(pos, regimes, strict=True):
        for val, col, lw in [(reg.low, "#5fb56f", 1.0), (reg.base, "#cccc55", 2.0),
                             (reg.high, "#e07060", 1.0)]:
            ax.hlines(val, p - 0.34, p + 0.34, color=col, lw=lw, alpha=0.9)
        ax.text(p, reg.base, f" σ_base={reg.base:.3f}", va="center", ha="left",
                fontsize=8, color="#cccc55")

    # σ_be(K) horizontal verdict markers; emphasize the flipper(s)
    flips = []
    for K in verdict_targets:
        be = float(breakeven_sigma(K, ctx.spot, ctx.horizon_years))
        flipped = (be > result.post_regime.base) != (be > result.full_regime.base)
        if flipped:
            flips.append(int(K))
        ax.axhline(be, color=("#36d6e7" if flipped else "grey"),
                   lw=(1.8 if flipped else 1.0),
                   ls=("-" if flipped else "--"), alpha=(0.95 if flipped else 0.55))
        ax.text(2.55, be, f"σ_be({int(K)})={be:.3f}" + ("  ⟵ flips" if flipped else ""),
                va="center", ha="left", fontsize=8,
                color=("#36d6e7" if flipped else "grey"))

    ax.set_xticks(pos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_xlim(0.4, 3.4)
    ax.set_ylabel("annualized realized σ")
    ax.set_title("Regime σ: full history vs post-break — does the verdict flip?")
    flip_note = (f"K={', '.join(map(str, flips))} flips verdict between regimes. "
                 if flips else "No marked K flips between these regimes. ")
    fig.text(0.01, -0.02, ctx.caption(
        flip_note + "A σ_be line above a regime's σ_base ⇒ 2x wins in that regime."
    ), fontsize=7.5, alpha=0.7)
    return _save(fig, outdir, "regime_sigma_compare.png")
