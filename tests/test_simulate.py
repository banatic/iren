"""Simulator: discrete↔continuous limit, conditional drift-invariance, impairment."""

from __future__ import annotations

import numpy as np
import pytest

from ilb.generators import bootstrap_paths, gbm_bridge_paths, gbm_paths
from ilb.leverage import levered_terminal
from ilb.simulate import (
    impairment_stats,
    simulate_leveraged_nav,
    terminal_conditioning,
)


def test_zero_return_nav_constant():
    r = np.zeros((10, 100))
    nav, wipe = simulate_leveraged_nav(r, leverage=2.0)
    assert np.allclose(nav, 1.0)
    assert (wipe == 0).all()


def test_simulator_converges_to_closed_form_low_vol():
    """Small daily moves → ½(β²-β) σ² drag dominates; discrete ≈ closed-form."""
    rng = np.random.default_rng(0)
    n = 500
    daily_sigma = 0.005  # 0.5% daily — small enough that O(r³) corrections are tiny
    r = rng.normal(0.0, daily_sigma, size=(8000, n))
    nav, _ = simulate_leveraged_nav(r, leverage=2.0)
    terminal_2x = nav[:, -1]
    underlying = np.exp(r.sum(axis=1))  # S_T/S_0
    qv = (r**2).sum(axis=1)
    closed = levered_terminal(underlying, qv, horizon_years=n / 252)
    # Mean ratio should be close to 1 (the closed-form is the continuous limit)
    ratio = terminal_2x / closed
    assert ratio.mean() == pytest.approx(1.0, rel=0.02)


def test_wipeout_count_flags_big_down_day():
    """Single -50%+ day on a 2x position annihilates NAV — must flag wipe-out."""
    r = np.zeros((1, 5))
    r[0, 2] = np.log(0.40)  # -60% gap → 1 + 2·(-0.60) = -0.20 → wipe-out
    nav, wipe = simulate_leveraged_nav(r, leverage=2.0)
    assert wipe[0] >= 1
    assert nav[0, -1] == 0.0  # clamped at zero


def test_terminal_conditioning_drift_invariance():
    """Spec §10: conditional 2x win-rate barely shifts when we change drift."""
    rng_seed = 99
    sigma = 0.6
    n_days = 252
    # GBM with drift = 0 vs drift = 30% — same volatility, but very different
    # arrival mass at K. The *conditional* statistic should be near-identical.
    K = 80.0
    p0, r0 = gbm_paths(spot=58, sigma=sigma, n_days=n_days, n_paths=15_000,
                       drift_scenario=0.0, seed=rng_seed)
    p1, r1 = gbm_paths(spot=58, sigma=sigma, n_days=n_days, n_paths=15_000,
                       drift_scenario=0.30, seed=rng_seed)
    a = terminal_conditioning(p0, r0, spot=58, target=K, band_pct=0.05)
    b = terminal_conditioning(p1, r1, spot=58, target=K, band_pct=0.05)
    # Arrival counts will differ a lot; conditional win-rate should not
    assert abs(a.win_rate_2x - b.win_rate_2x) < 0.10


def test_terminal_conditioning_widens_when_empty():
    p, r = gbm_paths(spot=58, sigma=0.1, n_days=20, n_paths=100, seed=1)
    res = terminal_conditioning(p, r, spot=58, target=1000.0, band_pct=0.01,
                                auto_widen=True, max_band_pct=0.4)
    assert res.actual_band_pct >= 0.01


def test_bridge_conditional_dispersion_runs():
    """Sanity: bridge paths fed through the conditional pipeline produce stats."""
    p, r = gbm_bridge_paths(spot=58, target=150, sigma=0.9, n_days=252,
                            n_paths=300, seed=4)
    res = terminal_conditioning(p, r, spot=58, target=150, band_pct=0.01)
    # All paths land at 150 → in-band by construction
    assert res.n_paths_in_band == 300
    assert np.isfinite(res.win_rate_2x)
    assert np.isfinite(res.realized_vol_mean)


def test_impairment_stats_shapes():
    p, _ = bootstrap_paths(
        np.random.default_rng(0).normal(0, 0.03, 1000),
        spot=100, n_days=252, n_paths=500, block_size=10, seed=2,
    )
    nav, wipe = simulate_leveraged_nav(np.diff(np.log(p), axis=1), leverage=2.0)
    imp = impairment_stats(nav, wipe, threshold=0.05)
    assert 0.0 <= imp.impair_fraction <= 1.0
    assert 0.0 <= imp.wipeout_fraction <= 1.0
    assert imp.p01 <= imp.p05 <= imp.median_terminal
