"""Static-site builder for GitHub Pages (spec §12).

Two layers:
  - **Pre-rendered (heavy)** — `plots/*.png` and a short HTML metadata block.
    These come from the discrete simulator and depend on randomness, so
    they're frozen at build time.
  - **Client-side interactive (light)** — `breakeven.js` recomputes the
    drift-free closed-form σ_be(K,T) live in the browser using regime σ
    sliders. Inputs are baked into `inputs.json` so the page works offline.

To keep deps minimal we *don't* require Jinja2: the index template uses
simple `{{name}}` placeholders that we substitute by string. Spec §12.5
parity test: `breakeven.js` must agree with `leverage.breakeven_sigma`
to ±1e-6.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np

from ilb import btc_regime
from ilb.config import Config
from ilb.data import latest_spot, load_prices
from ilb.estimate import regime_vols, rolling_sigma
from ilb.generators import bootstrap_paths
from ilb.leverage import breakeven_sigma
from ilb.plots import (
    PlotContext,
    plot_breakeven_curve,
    plot_breakeven_map,
    plot_btc_decoupling,
    plot_conditional_dispersion,
    plot_named_paths,
    plot_regime_sigma_compare,
    plot_vol_drift_timeseries,
)
from ilb.scenarios import build_named_paths
from ilb.simulate import terminal_conditioning

WEB_DIR = Path("web")
PLOTS_DIR = Path("plots")


def _render(template: str, ctx: dict) -> str:
    """Tiny `{{name}}` substituter (we don't need full Jinja for this template)."""
    out = template
    for k, v in ctx.items():
        out = out.replace("{{" + k + "}}", str(v))
    return out


def _sigma_now(returns, windows: list[int]) -> dict:
    """Latest rolling σ per window + its percentile in the full historical distribution.
    Frames the snapshot honestly: "today's vol vs everything we've seen", not a forecast."""
    out: dict[str, dict] = {}
    for w in windows:
        s = rolling_sigma(returns, w).dropna()
        if s.empty:
            continue
        latest = float(s.iloc[-1])
        pct = float((s <= latest).mean())
        out[f"d{w}"] = {"sigma": round(latest, 4), "pct": round(pct, 3), "window": w}
    return out


def _build_inputs(cfg: Config, df, spot: float, asof: date) -> dict:
    reg = regime_vols(rolling_sigma(df["log_return"], window=max(cfg.windows[0], 60)))
    horizons = list(np.linspace(0.25, 3.0, 28).round(4))
    K_grid = list(np.linspace(max(20.0, spot * 0.5), max(cfg.targets) * 1.1, 60).round(2))
    return {
        "meta": {
            "ticker": cfg.ticker,
            "asof": asof.isoformat(),
            "built_at": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
            "data_source": cfg.site.data_source,
            "base_path": cfg.site.base_path,
            "title": cfg.site.title,
        },
        "spot": float(spot),
        "leverage": cfg.leverage,
        "financing_rate": cfg.financing_rate,
        "expense_ratio": cfg.expense_ratio,
        "regimes": {"low": reg.low, "base": reg.base, "high": reg.high},
        "sigma_now": _sigma_now(df["log_return"], cfg.windows),
        "targets": list(map(float, cfg.targets)),
        "horizons": horizons,
        "target_grid": K_grid,
        "default_horizon_years": cfg.horizon_years,
    }


def _build_plots(cfg: Config, df, spot: float, asof: date, outdir: Path) -> list[Path]:
    reg = regime_vols(rolling_sigma(df["log_return"], window=max(cfg.windows[0], 60)))
    ctx = PlotContext(
        spot=spot, asof=asof, horizon_years=cfg.horizon_years, regime=reg,
        leverage=cfg.leverage, financing_rate=cfg.financing_rate,
        expense_ratio=cfg.expense_ratio, theme=cfg.theme,
    )
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.append(plot_vol_drift_timeseries(df, ctx, cfg.windows, cfg.ewma_lambda, PLOTS_DIR))
    horizons = np.linspace(0.25, 3.0, 28)
    target_grid = np.linspace(max(20.0, spot * 0.5), max(cfg.targets) * 1.1, 60)
    paths.append(plot_breakeven_map(ctx, target_grid, horizons, PLOTS_DIR))
    paths.append(plot_breakeven_curve(ctx, np.array(cfg.targets), PLOTS_DIR))
    n_days = max(int(round(cfg.horizon_years * 252)), 5)
    nps = build_named_paths(spot, cfg.targets[0], n_days)
    paths.append(plot_named_paths(ctx, nps, PLOTS_DIR))
    prices, returns = bootstrap_paths(
        df["log_return"].dropna().to_numpy(),
        spot=spot, n_days=n_days, n_paths=cfg.mc_sims,
        block_size=cfg.block_size, drift_scenario=cfg.drift_scenario, seed=cfg.seed,
    )
    results = [
        terminal_conditioning(
            prices, returns, spot=spot, target=k, band_pct=cfg.band_pct,
            leverage=cfg.leverage, financing_rate=cfg.financing_rate,
            expense_ratio=cfg.expense_ratio, impair_threshold=cfg.impair_threshold,
        )
        for k in cfg.targets
    ]
    paths.append(plot_conditional_dispersion(ctx, results, PLOTS_DIR))

    # BTC-decoupling regime diagnostic: is the full-history σ contaminated by the
    # miner→cloud pivot? Fetched via the same load_prices path (live → snapshot).
    btc = load_prices("BTC-USD")
    decoup = btc_regime.analyze(df, btc)
    paths.append(plot_btc_decoupling(ctx, decoup, PLOTS_DIR))
    verdict_targets = [165.0, *map(float, cfg.targets)]
    paths.append(plot_regime_sigma_compare(ctx, df, decoup, verdict_targets, PLOTS_DIR))
    return paths


def build(cfg: Config, outdir: Path, skip_plots: bool = False) -> Path:
    df = load_prices(cfg.ticker)
    spot, asof = latest_spot(df)
    inputs = _build_inputs(cfg, df, spot, asof)

    outdir.mkdir(parents=True, exist_ok=True)
    assets = outdir / "assets"
    assets.mkdir(exist_ok=True)

    # 1) heavy pre-rendered figures
    if not skip_plots:
        png_paths = _build_plots(cfg, df, spot, asof, outdir)
    else:
        png_paths = list(PLOTS_DIR.glob("*.png"))
    for p in png_paths:
        shutil.copy2(p, assets / p.name)

    # 2) interactive bundle
    inputs_path = outdir / "inputs.json"
    inputs_path.write_text(json.dumps(inputs, indent=2), encoding="utf-8")

    # ship the JS / CSS / template as-is
    for name in ("breakeven.js", "styles.css"):
        src = WEB_DIR / name
        if src.exists():
            shutil.copy2(src, assets / name)

    template = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    sample_be = {
        f"K={int(k)}": round(float(breakeven_sigma(k, spot, cfg.horizon_years,
                                                   financing_rate=cfg.financing_rate,
                                                   expense_ratio=cfg.expense_ratio,
                                                   leverage=cfg.leverage)), 4)
        for k in cfg.targets
    }
    rendered = _render(template, {
        "TITLE": cfg.site.title,
        "TICKER": cfg.ticker,
        "ASOF": asof.isoformat(),
        "SPOT": f"{spot:.2f}",
        "REGIME_LOW": f"{inputs['regimes']['low']:.3f}",
        "REGIME_BASE": f"{inputs['regimes']['base']:.3f}",
        "REGIME_HIGH": f"{inputs['regimes']['high']:.3f}",
        "BE_SAMPLE_JSON": json.dumps(sample_be),
        "BUILT_AT": inputs["meta"]["built_at"],
        "BASE_PATH": cfg.site.base_path.rstrip("/"),
    })
    (outdir / "index.html").write_text(rendered, encoding="utf-8")
    (outdir / ".nojekyll").write_text("", encoding="utf-8")
    (outdir / "robots.txt").write_text("User-agent: *\nAllow: /\n", encoding="utf-8")
    return outdir
