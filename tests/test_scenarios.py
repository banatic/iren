"""Named paths land at K and reproduce expected QV ordering."""

from __future__ import annotations

import pytest

from ilb.scenarios import build_named_paths, monotone, up_then_chop, vol_then_up


def test_monotone_zero_qv():
    p = monotone(spot=58, target=150, n_days=252)
    assert p.prices[0] == 58
    assert p.prices[-1] == pytest.approx(150, rel=1e-9)
    # All-equal returns → variance ≈ 0
    assert p.log_returns.var() == pytest.approx(0.0, abs=1e-12)


def test_oscillating_paths_land_at_target():
    for fn in (vol_then_up, up_then_chop):
        p = fn(spot=58, target=150, n_days=252)
        assert p.prices[-1] == pytest.approx(150, rel=1e-6)
        assert p.prices[0] == 58


def test_qv_ordering_monotone_lowest():
    nps = build_named_paths(spot=58, target=150, n_days=252)
    by_name = {p.name: p for p in nps}
    assert by_name["monotone"].realized_qv < by_name["vol_then_up"].realized_qv
    assert by_name["monotone"].realized_qv < by_name["up_then_chop"].realized_qv


def test_named_paths_returns_three_archetypes():
    nps = build_named_paths(spot=58, target=200, n_days=120)
    assert len(nps) == 3
    assert {p.name for p in nps} == {"monotone", "vol_then_up", "up_then_chop"}
