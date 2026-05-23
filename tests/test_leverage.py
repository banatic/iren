"""Spec §10 regression tests for the closed-form breakeven math."""

from __future__ import annotations

import math

import numpy as np
import pytest

from ilb.leverage import (
    breakeven_map,
    breakeven_qv,
    breakeven_sigma,
    levered_terminal,
)

SPOT = 58.0


def test_sigma_be_regression_no_fees():
    """Spec §10: σ_be(K=150,450,700; T=1, c=φ=0) ≈ 0.975, 1.431, 1.578."""
    sig = breakeven_sigma(np.array([150.0, 450.0, 700.0]), spot=SPOT, horizon_years=1.0)
    assert sig[0] == pytest.approx(math.sqrt(math.log(150 / 58)), rel=1e-9)
    assert sig[0] == pytest.approx(0.97481, abs=1e-3)
    assert sig[1] == pytest.approx(1.43141, abs=1e-3)
    assert sig[2] == pytest.approx(1.57820, abs=1e-3)


def test_breakeven_qv_identity_beta2():
    """QV_be = ln(K/S₀) - (c+φ)·T (β=2 collapse)."""
    qv = breakeven_qv(150.0 / SPOT, horizon_years=1.0, financing_rate=0.0, expense_ratio=0.0)
    assert qv == pytest.approx(math.log(150 / 58), rel=1e-9)
    qv2 = breakeven_qv(150.0 / SPOT, horizon_years=1.0, financing_rate=0.02, expense_ratio=0.01)
    assert qv2 == pytest.approx(math.log(150 / 58) - 0.03, rel=1e-9)


def test_horizon_dominance():
    """Spec §10: at K=150, σ_be(T=0.5/1/2) ≈ 1.38 / 0.975 / 0.69 (≈ 1/√T)."""
    for T, expected in [(0.5, 1.378), (1.0, 0.975), (2.0, 0.689)]:
        sig = breakeven_sigma(150.0, spot=SPOT, horizon_years=T)
        assert sig == pytest.approx(expected, abs=2e-3)


def test_fee_overlay_reduces_qv_be():
    qv0 = breakeven_qv(150 / SPOT, 1.0)
    qv1 = breakeven_qv(150 / SPOT, 1.0, financing_rate=0.043, expense_ratio=0.0)
    assert qv1 < qv0


def test_levered_terminal_tie_at_breakeven():
    """At QV = QV_be, L_T/L_0 must equal S_T/S_0."""
    gross = 150.0 / SPOT
    qv = breakeven_qv(gross, 1.0)
    L = levered_terminal(gross, qv, horizon_years=1.0)
    assert L == pytest.approx(gross, rel=1e-9)


def test_levered_terminal_2x_beats_at_low_vol():
    """At zero realized variance, L_T/L_0 = (S_T/S_0)^2 — always beats 1x for K>S₀."""
    gross = 150 / SPOT
    L = levered_terminal(gross, qv=0.0, horizon_years=1.0)
    assert L == pytest.approx(gross**2, rel=1e-12)
    assert L > gross


def test_breakeven_map_shape_and_value():
    targets = np.array([150.0, 450.0, 700.0])
    horizons = np.array([0.25, 1.0, 3.0])
    grid = breakeven_map(targets, horizons, spot=SPOT, sigma_base=0.93)
    assert grid.shape == (3, 3)
    # At the base regime σ=0.93, QV(T=1) = 0.865 ≈ QV_be for K=150 → ratio ≈ 1
    assert grid[1, 0] == pytest.approx(np.exp(math.log(150 / 58) - 0.93**2), rel=1e-9)


def test_sigma_be_zero_when_target_below_spot_minus_fees():
    """If target < spot·exp((c+φ)T), QV_be clamps to 0 → σ_be = 0."""
    sig = breakeven_sigma(40.0, spot=SPOT, horizon_years=1.0)
    assert sig == 0.0
