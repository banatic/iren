"""Discrete daily-rebalance simulator + terminal-K conditioning + impairment (spec §2.6-§2.7).

The discrete daily rebal is the **ground truth** for the leveraged NAV;
the closed-form QV identity in `leverage.py` is a continuous-time
approximation that misses gap risk and the wipe-out asymmetry from
single-day β·r ≤ −1.

NAV recursion:
    L_t = L_{t-1} · max(0, 1 + β·r_t − (c+φ)/252)
We clamp at 0 (NAV cannot go negative). The number of times the unclamped
multiplier was ≤ 0 per path is the **wipe-out count** — a critical impairment
signal that the QV approximation entirely hides.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


def simulate_leveraged_nav(
    log_returns: np.ndarray,
    leverage: float = 2.0,
    financing_rate: float = 0.0,
    expense_ratio: float = 0.0,
    initial_nav: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply daily-resettable leverage to a (paths × days) log-return matrix.

    Returns (nav_paths, wipeout_counts) where:
      - nav_paths has shape (paths, days+1) with nav_paths[:, 0] = initial_nav
      - wipeout_counts has shape (paths,) — number of days where the
        unclamped multiplier (1 + β·r_t − fee) was ≤ 0 (theoretical wipe-out)
    """
    r = np.asarray(log_returns, dtype=float)
    simple = np.expm1(r)  # convert log to simple returns for daily NAV update
    fee_daily = (financing_rate * (leverage - 1.0) + expense_ratio) / TRADING_DAYS
    multipliers = 1.0 + leverage * simple - fee_daily
    wipeouts = (multipliers <= 0.0).sum(axis=1)
    safe = np.clip(multipliers, 0.0, None)
    cum = np.cumprod(safe, axis=1)
    nav = np.empty((r.shape[0], r.shape[1] + 1), dtype=float)
    nav[:, 0] = initial_nav
    nav[:, 1:] = initial_nav * cum
    return nav, wipeouts


@dataclass(frozen=True)
class ImpairmentStats:
    """Path-level impairment signals not visible in the closed-form QV identity."""
    impair_fraction: float       # share of paths whose min NAV ≤ threshold·L_0
    wipeout_fraction: float      # share of paths that experienced any β·r ≤ −1 day
    p01: float                   # 1st-percentile terminal NAV (left tail)
    p05: float                   # 5th-percentile terminal NAV
    median_terminal: float
    n_paths: int


def impairment_stats(
    nav_paths: np.ndarray,
    wipeout_counts: np.ndarray,
    threshold: float = 0.05,
    initial_nav: float = 1.0,
) -> ImpairmentStats:
    """Spec §2.6 impairment indicators on a (paths, days+1) NAV matrix."""
    if nav_paths.size == 0:
        return ImpairmentStats(np.nan, np.nan, np.nan, np.nan, np.nan, 0)
    min_nav = nav_paths.min(axis=1)
    impair = (min_nav <= threshold * initial_nav).mean()
    wipeout = (wipeout_counts > 0).mean()
    terminal = nav_paths[:, -1]
    p01, p05, p50 = np.percentile(terminal, [1, 5, 50])
    return ImpairmentStats(
        impair_fraction=float(impair),
        wipeout_fraction=float(wipeout),
        p01=float(p01),
        p05=float(p05),
        median_terminal=float(p50),
        n_paths=int(nav_paths.shape[0]),
    )


@dataclass(frozen=True)
class ConditionalResult:
    """Output of terminal-K conditioned 2x vs 1x analysis."""
    n_paths_in_band: int
    n_paths_total: int
    actual_band_pct: float
    target_price: float
    win_rate_2x: float                # share where L_T/L_0 > S_T/S_0
    mean_2x_terminal: float           # L_T/L_0
    mean_1x_terminal: float           # S_T/S_0
    median_2x_terminal: float
    realized_vol_mean: float          # annualized realized vol across in-band paths
    impairment: ImpairmentStats


def terminal_conditioning(
    price_paths: np.ndarray,
    log_return_paths: np.ndarray,
    spot: float,
    target: float,
    band_pct: float = 0.05,
    leverage: float = 2.0,
    financing_rate: float = 0.0,
    expense_ratio: float = 0.0,
    impair_threshold: float = 0.05,
    auto_widen: bool = True,
    max_band_pct: float = 0.40,
) -> ConditionalResult:
    """Rejection-conditioning: keep paths whose terminal price is in K·(1±band).

    If the band is empty and `auto_widen=True`, doubles the band up to
    `max_band_pct` to avoid empty-sample crashes (spec §2.7).
    """
    terminal_prices = price_paths[:, -1]
    n_total = price_paths.shape[0]
    band = band_pct
    while True:
        lo, hi = target * (1.0 - band), target * (1.0 + band)
        mask = (terminal_prices >= lo) & (terminal_prices <= hi)
        if mask.sum() > 0 or not auto_widen or band >= max_band_pct:
            break
        band = min(band * 2.0, max_band_pct)

    n_in = int(mask.sum())
    if n_in == 0:
        empty_imp = ImpairmentStats(np.nan, np.nan, np.nan, np.nan, np.nan, 0)
        return ConditionalResult(
            n_paths_in_band=0, n_paths_total=n_total, actual_band_pct=band,
            target_price=target, win_rate_2x=float("nan"),
            mean_2x_terminal=float("nan"), mean_1x_terminal=float("nan"),
            median_2x_terminal=float("nan"), realized_vol_mean=float("nan"),
            impairment=empty_imp,
        )

    sel_returns = log_return_paths[mask]
    nav, wipeouts = simulate_leveraged_nav(
        sel_returns,
        leverage=leverage,
        financing_rate=financing_rate,
        expense_ratio=expense_ratio,
    )
    terminal_2x = nav[:, -1]
    terminal_1x = terminal_prices[mask] / spot
    # Realized annualized vol per path: std(log r) · √252, then mean
    per_path_vol = sel_returns.std(axis=1, ddof=1) * np.sqrt(TRADING_DAYS)
    imp = impairment_stats(nav, wipeouts, threshold=impair_threshold)
    return ConditionalResult(
        n_paths_in_band=n_in,
        n_paths_total=n_total,
        actual_band_pct=band,
        target_price=target,
        win_rate_2x=float((terminal_2x > terminal_1x).mean()),
        mean_2x_terminal=float(terminal_2x.mean()),
        mean_1x_terminal=float(terminal_1x.mean()),
        median_2x_terminal=float(np.median(terminal_2x)),
        realized_vol_mean=float(np.nanmean(per_path_vol)),
        impairment=imp,
    )
