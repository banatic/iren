"""Volatility / drift estimation (spec §2.2).

Three layers:
  1. Rolling window σ̂, ν̂ over W ∈ {20, 60, 120} business days.
  2. EWMA(λ=0.94) of squared log-returns (RiskMetrics-style).
  3. Regime σ_{low/base/high} from the {25, 50, 75} percentiles of the
     full *rolling* σ distribution (avoids a single point-estimate that
     would be lost in its own SE).

Annualization caveat: √252 scaling is the convenience baseline; for honest
horizon comparisons we also expose `horizon_realized_vol(returns, H)` which
computes std of overlapping H-day log-return sums and re-annualizes — this
captures vol clustering/autocorr that iid √252 ignores.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class RegimeVols:
    low: float
    base: float
    high: float

    def get(self, name: str) -> float:
        return {"low": self.low, "base": self.base, "high": self.high}[name]


def rolling_sigma(returns: pd.Series, window: int) -> pd.Series:
    """Annualized rolling std of log-returns (√252 scaling)."""
    return returns.rolling(window, min_periods=window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def rolling_drift(returns: pd.Series, window: int) -> pd.Series:
    """Annualized rolling mean log-return (×252). Reported but NEVER trusted as a forecast."""
    return returns.rolling(window, min_periods=window).mean() * TRADING_DAYS


def ewma_sigma(returns: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics EWMA: σ²_t = λ σ²_{t-1} + (1−λ) r²_{t-1}. Returned annualized."""
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1)")
    r2 = returns.fillna(0.0).to_numpy() ** 2
    var = np.zeros_like(r2)
    if len(r2) == 0:
        return pd.Series(dtype=float, index=returns.index)
    var[0] = r2[0]
    for t in range(1, len(r2)):
        var[t] = lam * var[t - 1] + (1.0 - lam) * r2[t]
    return pd.Series(np.sqrt(var * TRADING_DAYS), index=returns.index, name="ewma_sigma")


def regime_vols(rolling_series: pd.Series) -> RegimeVols:
    """Quantile-derived low/base/high vol regimes from a rolling σ series."""
    s = rolling_series.dropna()
    if s.empty:
        raise ValueError("rolling σ series is empty after dropna")
    q = s.quantile([0.25, 0.50, 0.75])
    return RegimeVols(low=float(q.loc[0.25]), base=float(q.loc[0.50]), high=float(q.loc[0.75]))


def horizon_realized_vol(returns: pd.Series, horizon_days: int) -> float:
    """Annualized realized vol from overlapping H-day log-return sums.

    For H > 1 this measures vol at the actual investment horizon (captures
    autocorrelation/vol clustering), avoiding the iid √H scaling that
    masks IREN's volatility persistence.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be ≥ 1")
    r = returns.dropna()
    if len(r) < horizon_days + 1:
        return float("nan")
    sums = r.rolling(horizon_days).sum().dropna()
    sd = float(sums.std(ddof=1))
    return sd * np.sqrt(TRADING_DAYS / horizon_days)


def vol_table(returns: pd.Series, windows: list[int]) -> pd.DataFrame:
    """One-shot table of rolling σ across windows (NaN-aligned), for plotting."""
    cols = {f"sigma_{w}": rolling_sigma(returns, w) for w in windows}
    return pd.DataFrame(cols)
