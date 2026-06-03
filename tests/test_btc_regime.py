"""BTC-decoupling diagnostic: alignment, rolling β/ρ, break detection, regime split."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ilb import btc_regime as br


def _frame(index: pd.DatetimeIndex, close: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame({"close": close}, index=index)
    df["log_return"] = np.log(df["close"]).diff()
    return df


def test_align_drops_btc_weekends():
    bdays = pd.bdate_range("2022-01-03", periods=150)
    caldays = pd.date_range("2022-01-01", periods=220)  # includes weekends
    rng = np.random.default_rng(0)
    iren = _frame(bdays, 50 * np.exp(np.cumsum(rng.normal(0, 0.03, len(bdays)))))
    btc = _frame(caldays, 20000 * np.exp(np.cumsum(rng.normal(0, 0.03, len(caldays)))))
    ret = br.align_log_returns(iren, btc)
    assert {"iren", "btc"} == set(ret.columns)
    assert set(ret.index) <= set(bdays)          # only IREN trading days survive
    assert bool((ret.index.dayofweek < 5).all())  # no weekends


def test_rolling_beta_is_one_when_identical():
    idx = pd.bdate_range("2022-01-03", periods=300)
    rng = np.random.default_rng(1)
    df = _frame(idx, 100 * np.exp(np.cumsum(rng.normal(0, 0.04, len(idx)))))
    bc = br.rolling_beta_corr(br.align_log_returns(df, df), (60,))
    assert np.allclose(bc["beta_60"].dropna(), 1.0, atol=1e-9)
    assert np.allclose(bc["rho_60"].dropna(), 1.0, atol=1e-9)


def test_rolling_beta_recovers_known_slope():
    idx = pd.bdate_range("2022-01-03", periods=400)
    rng = np.random.default_rng(2)
    btc_r = rng.normal(0, 0.03, len(idx))
    iren_r = 1.8 * btc_r + rng.normal(0, 0.001, len(idx))  # β ≈ 1.8
    ret = pd.DataFrame({"iren": iren_r, "btc": btc_r}, index=idx)
    assert abs(br.rolling_beta_corr(ret, (120,))["beta_120"].dropna().mean() - 1.8) < 0.05


def test_cusum_detects_single_level_shift():
    idx = pd.bdate_range("2022-01-03", periods=300)
    s = pd.Series(np.r_[np.full(150, 0.5), np.full(150, 1.5)], index=idx)
    breaks = br._cusum_break(s)
    assert len(breaks) == 1
    assert abs((breaks[0] - idx[150]).days) < 25


def test_detect_breaks_finds_the_step():
    idx = pd.bdate_range("2022-01-03", periods=320)
    rng = np.random.default_rng(3)
    s = pd.Series(np.r_[np.full(160, 0.4), np.full(160, 1.4)]
                  + rng.normal(0, 0.05, 320), index=idx)
    breaks, method = br.detect_breaks(s)
    assert method in {"pelt", "cusum"}
    assert any(abs((b - idx[160]).days) < 35 for b in breaks)


def test_analyze_splits_on_vol_regime_change():
    idx = pd.bdate_range("2022-01-03", periods=600)
    rng = np.random.default_rng(4)
    btc_r = rng.normal(0.0, 0.03, 600)
    # IREN: high-vol + low-β era, then low-vol + high-β era (break at day 300)
    iren_r = np.r_[0.3 * btc_r[:300] + rng.normal(0, 0.06, 300),
                   1.5 * btc_r[300:] + rng.normal(0, 0.02, 300)]
    iren = _frame(idx, 50 * np.exp(np.cumsum(iren_r)))
    btc = _frame(idx, 20000 * np.exp(np.cumsum(btc_r)))
    res = br.analyze(iren, btc)
    assert res.break_dates                      # a structural break was found
    assert res.split_date is not None           # post sample clears the floor
    assert res.full_regime.base > 0 and res.post_regime.base > 0
    # post-split σ (low-vol era) is materially below the blended full-history σ
    assert res.post_regime.base < res.full_regime.base


def test_analyze_handles_no_break_gracefully():
    """Identical series → β≡1, no break, post regime falls back to full."""
    idx = pd.bdate_range("2022-01-03", periods=400)
    rng = np.random.default_rng(5)
    df = _frame(idx, 100 * np.exp(np.cumsum(rng.normal(0, 0.03, len(idx)))))
    res = br.analyze(df, df)
    assert res.split_date is None
    assert res.post_regime == res.full_regime
