"""Closed-form leveraged-NAV identity and break-even volatility (spec §2.3–§2.4).

For continuous rebalancing at leverage β on an arbitrary continuous price
path S_t with realized integrated variance QV = ∫₀ᵀ σ_t² dt:

    L_T/L_0 = (S_T/S_0)^β · exp( -½(β²-β)·QV  -  (β-1)·c·T  -  φ·T )

where c is the daily-resettable financing rate (annual, on the borrowed
β-1 notional) and φ is the product expense ratio.

Break-even in QV (β=2): 1x and 2x tie when
    QV_be = ln(S_T/S_0) - (c+φ)·T,
i.e. σ_be(K, T) = √( max(0, [ln(K/S_0) - (c+φ)·T] / T) ).

This is a *continuous* approximation. The discrete daily simulator
(`simulate.py`) is the ground truth — gap risk and the asymmetry from
single-day β·r ≤ −1 events do not show up here. Closed-form is used for
sanity checks and for the interactive web panel.
"""

from __future__ import annotations

import numpy as np


def levered_terminal(
    s_terminal_over_spot: np.ndarray | float,
    qv: np.ndarray | float,
    horizon_years: float,
    leverage: float = 2.0,
    financing_rate: float = 0.0,
    expense_ratio: float = 0.0,
) -> np.ndarray | float:
    """L_T/L_0 given the underlying gross return, realized QV, and horizon T."""
    beta = leverage
    drag = 0.5 * (beta**2 - beta) * np.asarray(qv) + (beta - 1.0) * financing_rate * horizon_years
    drag = drag + expense_ratio * horizon_years
    return np.asarray(s_terminal_over_spot) ** beta * np.exp(-drag)


def breakeven_qv(
    target_over_spot: np.ndarray | float,
    horizon_years: float,
    financing_rate: float = 0.0,
    expense_ratio: float = 0.0,
    leverage: float = 2.0,
) -> np.ndarray | float:
    """QV at which L_T = S_T (per unit). General β:

        QV_be = (2/(β²-β)) · [ (β-1)·ln(S_T/S_0) - (β-1)·c·T - φ·T ]

    For β=2 this collapses to QV_be = ln(S_T/S_0) - (c+φ)·T.
    """
    if leverage <= 1.0:
        raise ValueError("leverage must be > 1 for breakeven to be meaningful")
    beta = leverage
    log_gross = np.log(np.asarray(target_over_spot))
    num = (beta - 1.0) * log_gross - (beta - 1.0) * financing_rate * horizon_years
    num = num - expense_ratio * horizon_years
    qv = (2.0 / (beta**2 - beta)) * num
    return np.maximum(qv, 0.0)


def breakeven_sigma(
    target: float | np.ndarray,
    spot: float,
    horizon_years: float,
    financing_rate: float = 0.0,
    expense_ratio: float = 0.0,
    leverage: float = 2.0,
) -> np.ndarray | float:
    """Annualized realized vol at which 2x ties 1x for target price K, horizon T.

    σ_be(K, T) = √( QV_be / T )
    """
    if horizon_years <= 0:
        raise ValueError("horizon_years must be > 0")
    qv = breakeven_qv(
        np.asarray(target) / spot,
        horizon_years,
        financing_rate=financing_rate,
        expense_ratio=expense_ratio,
        leverage=leverage,
    )
    return np.sqrt(qv / horizon_years)


def breakeven_map(
    targets: np.ndarray,
    horizons: np.ndarray,
    spot: float,
    sigma_base: float,
    financing_rate: float = 0.0,
    expense_ratio: float = 0.0,
    leverage: float = 2.0,
) -> np.ndarray:
    """Grid of L_T/L_0 ÷ (S_T/S_0) at the *base* regime σ for (horizons × targets).

    Values > 1 mean 2x beats 1x at that (T, K) under the base-regime
    constant-σ assumption (so QV = σ_base² · T). Pair with σ_be contours
    for the v3 main breakeven map.
    """
    horizons = np.asarray(horizons, dtype=float).reshape(-1, 1)  # (H, 1)
    targets = np.asarray(targets, dtype=float).reshape(1, -1)    # (1, K)
    gross = targets / spot
    qv = (sigma_base**2) * horizons
    beta = leverage
    drag = (
        0.5 * (beta**2 - beta) * qv
        + (beta - 1.0) * financing_rate * horizons
        + expense_ratio * horizons
    )
    levered = (gross**beta) * np.exp(-drag)
    return levered / gross  # ratio L_T/L_0 ÷ S_T/S_0
