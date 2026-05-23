"""Typer CLI: fetch / breakeven / paths / run / build-site (spec §7)."""

from __future__ import annotations

import contextlib
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import typer

from ilb import site as site_mod
from ilb.config import load_config
from ilb.data import latest_spot, load_prices, write_snapshot
from ilb.estimate import RegimeVols, regime_vols, rolling_sigma
from ilb.generators import bootstrap_paths, gbm_paths
from ilb.leverage import breakeven_sigma
from ilb.plots import (
    PlotContext,
    plot_breakeven_curve,
    plot_breakeven_map,
    plot_conditional_dispersion,
    plot_named_paths,
    plot_vol_drift_timeseries,
)
from ilb.scenarios import build_named_paths
from ilb.simulate import terminal_conditioning

app = typer.Typer(help="IREN 2x leverage breakeven analyzer", no_args_is_help=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")


def _force_utf8_streams() -> None:
    """Windows consoles (cp949/cp1252) can't render σ β φ ₀; coerce streams to UTF-8."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            with contextlib.suppress(Exception):
                stream.reconfigure(encoding="utf-8")


_force_utf8_streams()


def _parse_targets(targets: str | None, default: list[float]) -> list[float]:
    if not targets:
        return default
    return [float(t.strip()) for t in targets.split(",") if t.strip()]


def _resolve_spot(df, override: float | None) -> tuple[float, date]:
    if override is not None:
        return float(override), df.index[-1].date()
    return latest_spot(df)


def _regimes_from_returns(df, windows: list[int]) -> RegimeVols:
    return regime_vols(rolling_sigma(df["log_return"], window=max(windows[0], 60)))


@app.command()
def fetch(
    ticker: str = typer.Option(None, help="override config.ticker"),
    refresh: bool = typer.Option(False, "--refresh", help="force re-download"),
    snapshot: bool = typer.Option(False, "--snapshot",
                                  help="also write data_snapshot/<ticker>_daily.csv"),
    config: str = typer.Option("config.yaml"),
) -> None:
    """Populate the parquet cache (and optionally the committed CSV snapshot)."""
    cfg = load_config(config)
    t = ticker or cfg.ticker
    df = load_prices(t, refresh=refresh)
    spot, asof = latest_spot(df)
    typer.echo(f"{t}: {len(df)} rows, spot=${spot:.2f} as of {asof}")
    if snapshot:
        path = write_snapshot(df, t)
        typer.echo(f"snapshot written → {path}")


@app.command()
def breakeven(
    targets: str = typer.Option(None, help="comma-separated K list"),
    spot: float = typer.Option(None, help="override spot"),
    horizon: float = typer.Option(None, help="override horizon years"),
    financing: float = typer.Option(None, help="annual financing rate c"),
    expense: float = typer.Option(None, help="annual expense ratio φ"),
    leverage: float = typer.Option(None),
    config: str = typer.Option("config.yaml"),
) -> None:
    """Print σ_be(K, T) table (drift-free closed-form)."""
    cfg = load_config(config)
    K = _parse_targets(targets, cfg.targets)
    T = horizon if horizon is not None else cfg.horizon_years
    c = financing if financing is not None else cfg.financing_rate
    phi = expense if expense is not None else cfg.expense_ratio
    beta = leverage if leverage is not None else cfg.leverage
    s = spot
    if s is None:
        df = load_prices(cfg.ticker)
        s, _ = latest_spot(df)
    typer.echo(
        f"σ_be at spot=${s:.2f}, T={T:.2f}y, β={beta}, c={c:.2%}, φ={phi:.2%}:"
    )
    typer.echo(f"{'K':>10}  {'σ_be':>10}  {'log(K/S₀)':>12}")
    for k in K:
        sig = float(breakeven_sigma(k, s, T, financing_rate=c, expense_ratio=phi,
                                    leverage=beta))
        typer.echo(f"{k:>10.2f}  {sig:>10.4f}  {np.log(k / s):>12.4f}")


@app.command()
def paths(
    target: float = typer.Option(..., help="K to converge on"),
    horizon: float = typer.Option(None),
    config: str = typer.Option("config.yaml"),
    outdir: str = typer.Option("plots"),
    spot: float = typer.Option(None),
) -> None:
    """Build the named-path comparison plot for a single K."""
    cfg = load_config(config)
    T = horizon if horizon is not None else cfg.horizon_years
    df = load_prices(cfg.ticker)
    s, asof = _resolve_spot(df, spot)
    reg = _regimes_from_returns(df, cfg.windows)
    n_days = max(int(round(T * 252)), 5)
    nps = build_named_paths(s, target, n_days)
    ctx = PlotContext(spot=s, asof=asof, horizon_years=T, regime=reg,
                      leverage=cfg.leverage, financing_rate=cfg.financing_rate,
                      expense_ratio=cfg.expense_ratio, theme=cfg.theme)
    out = plot_named_paths(ctx, nps, Path(outdir))
    typer.echo(f"wrote {out}")


@app.command()
def run(
    targets: str = typer.Option(None),
    horizon: float = typer.Option(None),
    generator: str = typer.Option(None, help="bootstrap | gbm"),
    regime: str = typer.Option(None),
    drift_scenario: float = typer.Option(None),
    sims: int = typer.Option(None),
    band: float = typer.Option(None),
    block_size: int = typer.Option(None),
    seed: int = typer.Option(None),
    spot: float = typer.Option(None),
    outdir: str = typer.Option("plots"),
    config: str = typer.Option("config.yaml"),
) -> None:
    """Full pipeline → 5 plots + breakeven table."""
    cfg = load_config(config)
    K_list = _parse_targets(targets, cfg.targets)
    T = horizon if horizon is not None else cfg.horizon_years
    gen = generator or cfg.generator
    rg = regime or cfg.regime
    drift = drift_scenario if drift_scenario is not None else cfg.drift_scenario
    n_sims = sims or cfg.mc_sims
    bw = band if band is not None else cfg.band_pct
    bs = block_size or cfg.block_size
    sd = seed if seed is not None else cfg.seed
    n_days = max(int(round(T * 252)), 5)

    df = load_prices(cfg.ticker)
    s, asof = _resolve_spot(df, spot)
    reg = _regimes_from_returns(df, cfg.windows)
    ctx = PlotContext(spot=s, asof=asof, horizon_years=T, regime=reg,
                      leverage=cfg.leverage, financing_rate=cfg.financing_rate,
                      expense_ratio=cfg.expense_ratio, theme=cfg.theme)
    outp = Path(outdir)

    typer.echo("[1/5] vol+drift timeseries")
    plot_vol_drift_timeseries(df, ctx, cfg.windows, cfg.ewma_lambda, outp)

    typer.echo("[2/5] breakeven map (T × K)")
    horizons = np.linspace(0.25, 3.0, 28)
    target_grid = np.linspace(max(20.0, s * 0.5), max(K_list) * 1.1, 60)
    plot_breakeven_map(ctx, target_grid, horizons, outp)

    typer.echo("[3/5] breakeven curve (per-K)")
    plot_breakeven_curve(ctx, np.array(K_list), outp)

    typer.echo(f"[4/5] named paths (K=${K_list[0]:.0f})")
    nps = build_named_paths(s, K_list[0], n_days)
    plot_named_paths(ctx, nps, outp)

    typer.echo(f"[5/5] conditional dispersion (gen={gen}, sims={n_sims})")
    if gen == "gbm":
        sigma_used = reg.get(rg)
        prices, returns = gbm_paths(
            s, sigma_used, n_days, n_sims,
            drift_scenario=drift, seed=sd,
        )
    else:
        prices, returns = bootstrap_paths(
            df["log_return"].dropna().to_numpy(),
            spot=s, n_days=n_days, n_paths=n_sims, block_size=bs,
            drift_scenario=drift, seed=sd,
        )
    results = [
        terminal_conditioning(
            prices, returns, spot=s, target=k, band_pct=bw,
            leverage=cfg.leverage, financing_rate=cfg.financing_rate,
            expense_ratio=cfg.expense_ratio, impair_threshold=cfg.impair_threshold,
        )
        for k in K_list
    ]
    plot_conditional_dispersion(ctx, results, outp)

    typer.echo("\nσ_be (drift-free):")
    typer.echo(f"{'K':>10}  {'σ_be':>10}")
    for k in K_list:
        sig = float(breakeven_sigma(k, s, T,
                                    financing_rate=cfg.financing_rate,
                                    expense_ratio=cfg.expense_ratio,
                                    leverage=cfg.leverage))
        typer.echo(f"{k:>10.2f}  {sig:>10.4f}")
    typer.echo(f"\nDone. {len(K_list)} targets, plots in {outp.resolve()}")


@app.command("build-site")
def build_site(
    outdir: str = typer.Option("site"),
    config: str = typer.Option("config.yaml"),
    skip_plots: bool = typer.Option(False, help="reuse existing plots/*.png"),
) -> None:
    """Build the static site for GitHub Pages (spec §12)."""
    cfg = load_config(config)
    out = site_mod.build(cfg, Path(outdir), skip_plots=skip_plots)
    typer.echo(f"site built → {out.resolve()}")


if __name__ == "__main__":
    app()
