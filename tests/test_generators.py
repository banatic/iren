"""Generators: shape contracts, fat-tail preservation, drift injection, bridge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ilb.generators import (
    bootstrap_paths,
    gbm_bridge_paths,
    gbm_paths,
    stationary_block_bootstrap,
)


def _fat_tail_returns(n: int = 2000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # Student-t with df=3 → kurtosis is infinite in theory; sample kurt >> 3
    return rng.standard_t(df=3, size=n) * 0.03


def test_block_bootstrap_shape_and_finiteness():
    rng = np.random.default_rng(0)
    src = np.array([0.01, -0.02, 0.005, 0.0, -0.01, 0.03] * 100)
    out = stationary_block_bootstrap(src, n_days=120, n_paths=50, block_size=10, rng=rng)
    assert out.shape == (50, 120)
    assert np.isfinite(out).all()


def test_bootstrap_preserves_fat_tails():
    src = _fat_tail_returns(3000, seed=11)
    prices, returns = bootstrap_paths(src, spot=58, n_days=252, n_paths=200,
                                      block_size=15, seed=42)
    sample_kurt = pd.Series(returns.flatten()).kurtosis()  # excess kurt
    # Gaussian excess kurt ≈ 0; our t(3) source is heavy-tailed → >> 0
    assert sample_kurt > 2.0
    assert prices.shape == (200, 253)
    assert (prices[:, 0] == 58).all()
    assert (prices > 0).all()


def test_bootstrap_drift_injection_shifts_mean():
    src = np.random.default_rng(7).normal(0.0, 0.03, size=2000)
    _, r0 = bootstrap_paths(src, spot=58, n_days=252, n_paths=400,
                            block_size=15, drift_scenario=0.0, seed=1)
    _, r1 = bootstrap_paths(src, spot=58, n_days=252, n_paths=400,
                            block_size=15, drift_scenario=0.20, seed=1)
    # Same seed → same sampled blocks; drift shifts mean by 0.20/252 per day
    expected_shift = 0.20 / 252
    assert (r1 - r0) == pytest.approx(np.full_like(r0, expected_shift), abs=1e-12)


def test_gbm_paths_iid_moments():
    p, r = gbm_paths(spot=58, sigma=0.5, n_days=10_000, n_paths=1, seed=3)
    daily_sigma = 0.5 / np.sqrt(252)
    assert r.std(ddof=1) == pytest.approx(daily_sigma, rel=0.05)
    assert p.shape == (1, 10_001)


def test_gbm_bridge_lands_at_target():
    prices, _ = gbm_bridge_paths(spot=58, target=150, sigma=0.93,
                                 n_days=252, n_paths=500, seed=5)
    terminal = prices[:, -1]
    # Bridge: terminal must exactly equal target for every path
    assert np.allclose(terminal, 150.0, rtol=1e-10)
