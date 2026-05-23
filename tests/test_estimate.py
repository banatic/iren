"""Vol estimation: rolling/EWMA/regime/horizon-scaled realized vol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ilb.estimate import (
    TRADING_DAYS,
    RegimeVols,
    ewma_sigma,
    horizon_realized_vol,
    regime_vols,
    rolling_sigma,
)


def _synthetic_returns(n: int = 1000, sigma_daily: float = 0.03, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = rng.normal(loc=0.0, scale=sigma_daily, size=n)
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(r, index=idx, name="log_return")


def test_rolling_sigma_matches_iid_truth():
    r = _synthetic_returns(n=2000, sigma_daily=0.03, seed=1)
    s = rolling_sigma(r, window=120).dropna()
    annual = s.median()
    # Expected ≈ 0.03 · √252 ≈ 0.476
    assert annual == pytest.approx(0.03 * np.sqrt(TRADING_DAYS), rel=0.05)


def test_ewma_sigma_stationary_level():
    r = _synthetic_returns(n=3000, sigma_daily=0.03, seed=2)
    e = ewma_sigma(r, lam=0.94)
    # Burn-in: take the last half
    tail = e.iloc[-1000:]
    assert tail.median() == pytest.approx(0.03 * np.sqrt(TRADING_DAYS), rel=0.10)


def test_regime_vols_ordering_and_quantiles():
    r = _synthetic_returns(n=5000, sigma_daily=0.03, seed=3)
    s = rolling_sigma(r, window=60)
    reg = regime_vols(s)
    assert isinstance(reg, RegimeVols)
    assert reg.low < reg.base < reg.high
    # base should match the median of the rolling series
    assert reg.base == pytest.approx(s.dropna().median(), rel=1e-9)


def test_horizon_realized_vol_returns_finite_and_reasonable():
    r = _synthetic_returns(n=1500, sigma_daily=0.03, seed=4)
    one_day = horizon_realized_vol(r, horizon_days=1)
    twenty_day = horizon_realized_vol(r, horizon_days=20)
    # For iid Gaussians these should both ≈ daily σ · √252
    target = 0.03 * np.sqrt(TRADING_DAYS)
    assert one_day == pytest.approx(target, rel=0.08)
    assert twenty_day == pytest.approx(target, rel=0.15)


def test_horizon_realized_vol_short_series_returns_nan():
    r = pd.Series([0.01, -0.02, 0.005], index=pd.bdate_range("2024-01-02", periods=3))
    assert np.isnan(horizon_realized_vol(r, horizon_days=10))
