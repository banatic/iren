"""Path generators (spec §2.5).

Primary: **stationary block bootstrap** on empirical daily log-returns —
preserves fat tails / vol clustering / autocorrelation, sidesteps drift
estimation (which is non-identifiable on IREN's history).

Baseline: **GBM** with constant σ and a drift *scenario* (never an
estimate) — used for closed-form sanity checks and the Brownian-bridge
conditional simulator. Gaussian, so it under-fits IREN's tails.

`drift_scenario` (annual log drift) is injected as a constant per-day
shift `μ_daily = drift_scenario / 252` on the returns *after* sampling,
so it does not distort cluster structure.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def stationary_block_bootstrap(
    returns: np.ndarray,
    n_days: int,
    n_paths: int,
    block_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Stationary bootstrap (Politis & Romano 1994).

    Block lengths are geometric with mean `block_size`; start indices are
    uniform on the source array; samples wrap around. Returns an array of
    shape (n_paths, n_days) of resampled log-returns.
    """
    r = np.asarray(returns, dtype=float)
    n = r.size
    if n < 2:
        raise ValueError("need ≥ 2 source returns for the bootstrap")
    p = 1.0 / max(block_size, 1)
    out = np.empty((n_paths, n_days), dtype=float)
    for path_idx in range(n_paths):
        i = 0
        cursor = int(rng.integers(0, n))
        while i < n_days:
            out[path_idx, i] = r[cursor % n]
            # New block w.p. p; otherwise continue from next index
            if rng.random() < p:
                cursor = int(rng.integers(0, n))
            else:
                cursor += 1
            i += 1
    return out


def bootstrap_paths(
    returns: pd.Series | np.ndarray,
    spot: float,
    n_days: int,
    n_paths: int,
    block_size: int = 15,
    drift_scenario: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate (price_paths, return_paths) via block bootstrap.

    `drift_scenario` (annual log drift) shifts every sampled return by
    `drift_scenario / TRADING_DAYS` — explicit scenario, never an estimate.

    Returns:
        prices:  (n_paths, n_days+1) with prices[:, 0] = spot
        returns: (n_paths, n_days)    sampled (+ drift-shifted) log returns
    """
    src = np.asarray(returns, dtype=float)
    src = src[np.isfinite(src)]
    rng = np.random.default_rng(seed)
    sampled = stationary_block_bootstrap(src, n_days, n_paths, block_size, rng)
    if drift_scenario != 0.0:
        sampled = sampled + drift_scenario / TRADING_DAYS
    prices = np.empty((n_paths, n_days + 1), dtype=float)
    prices[:, 0] = spot
    np.cumsum(sampled, axis=1, out=prices[:, 1:])
    prices[:, 1:] = spot * np.exp(prices[:, 1:])
    return prices, sampled


def gbm_paths(
    spot: float,
    sigma: float,
    n_days: int,
    n_paths: int,
    drift_scenario: float = 0.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Toy GBM baseline: Gaussian log-returns with constant daily σ.

    `drift_scenario` is the annual *log* drift μ (not μ̂ + ½σ² — we work
    in log space). Daily mean = μ/252, daily std = σ/√252.
    """
    rng = np.random.default_rng(seed)
    daily_mu = drift_scenario / TRADING_DAYS
    daily_sigma = sigma / np.sqrt(TRADING_DAYS)
    returns = rng.normal(loc=daily_mu, scale=daily_sigma, size=(n_paths, n_days))
    prices = np.empty((n_paths, n_days + 1), dtype=float)
    prices[:, 0] = spot
    np.cumsum(returns, axis=1, out=prices[:, 1:])
    prices[:, 1:] = spot * np.exp(prices[:, 1:])
    return prices, returns


def gbm_bridge_paths(
    spot: float,
    target: float,
    sigma: float,
    n_days: int,
    n_paths: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Brownian bridge: GBM paths conditioned to land exactly at `target` at T.

    Generates n_days+1 grid points on [0, T] (T = n_days/252) so the log-price
    walk W_t is a Brownian bridge with W_0=ln(spot), W_T=ln(target). Returns
    (prices, log_returns) with the same shape contract as gbm_paths.

    Why a bridge? Drift μ is non-identifiable on IREN, so "what fraction of
    paths *reach* K" is unreliable. We sidestep that by *conditioning* on
    K-arrival and only measuring 2x vs 1x dispersion among those paths.
    """
    rng = np.random.default_rng(seed)
    T_years = n_days / TRADING_DAYS
    dt = T_years / n_days
    daily_sigma = sigma * np.sqrt(dt)
    log_s0 = np.log(spot)
    log_K = np.log(target)
    # standard BM increments
    dW = rng.normal(0.0, daily_sigma, size=(n_paths, n_days))
    W = np.cumsum(dW, axis=1)  # (n_paths, n_days), W at t = i·dt for i=1..n_days
    W_T = W[:, -1:]  # (n_paths, 1) final BM value
    # Build bridge: B_t = W_t - (t/T)·W_T  +  (t/T)·(log_K - log_s0)
    t_over_T = np.arange(1, n_days + 1) / n_days  # (n_days,)
    bridge_log_price = log_s0 + (W - t_over_T * W_T) + t_over_T * (log_K - log_s0)
    prices = np.empty((n_paths, n_days + 1), dtype=float)
    prices[:, 0] = spot
    prices[:, 1:] = np.exp(bridge_log_price)
    returns = np.diff(np.log(prices), axis=1)
    return prices, returns
