"""Named deterministic paths landing at the same target (spec §2.8).

Three archetypes — same terminal K, very different intra-path variance:
  1. `monotone`     — constant daily log-return, zero realized variance
                      (the *most generous* path for 2x).
  2. `vol_then_up`  — symmetric high-variance oscillation in the first
                      half, then a clean ramp to K in the second half.
  3. `up_then_chop` — fast ramp to (near) K in the first half, then
                      high-vol chop that ends at exactly K.

In a continuous world the leveraged terminal depends only on the path's
QV, so 2 and 3 would tie at equal realized variance. The discrete daily
rebalance breaks that tie because order and gap size matter — that gap
is the *point* of this comparison.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NamedPath:
    name: str
    description: str
    prices: np.ndarray      # shape (n_days+1,)
    log_returns: np.ndarray # shape (n_days,)
    realized_qv: float      # ∑ r_t² — discrete QV proxy


def _enforce_terminal(returns: np.ndarray, target_log_return: float) -> np.ndarray:
    """Rescale-by-shift so the sum of log returns lands exactly at target_log_return.

    Adds a constant per-step adjustment so the path lands at K without
    distorting the *shape* (relative oscillation pattern is preserved).
    """
    adj = (target_log_return - returns.sum()) / returns.size
    return returns + adj


def monotone(spot: float, target: float, n_days: int) -> NamedPath:
    target_log = np.log(target / spot)
    per_day = target_log / n_days
    r = np.full(n_days, per_day, dtype=float)
    prices = spot * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return NamedPath(
        name="monotone",
        description="Straight-line ramp at constant daily log-return (zero QV beyond drift).",
        prices=prices,
        log_returns=r,
        realized_qv=float((r - r.mean()).var() * n_days),  # ≈ 0
    )


def vol_then_up(
    spot: float,
    target: float,
    n_days: int,
    osc_amp: float = 0.06,
    osc_period: int = 8,
) -> NamedPath:
    """First half: symmetric oscillation; second half: clean ramp to K."""
    half = n_days // 2
    t = np.arange(half)
    osc = osc_amp * np.sin(2 * np.pi * t / osc_period)
    ramp_len = n_days - half
    ramp = np.full(ramp_len, 0.0)  # filled by _enforce_terminal
    r = np.concatenate([osc, ramp])
    r = _enforce_terminal(r, np.log(target / spot))
    prices = spot * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return NamedPath(
        name="vol_then_up",
        description="Symmetric oscillation in the first half, then a clean ramp to K.",
        prices=prices,
        log_returns=r,
        realized_qv=float((r - r.mean()).var() * n_days),
    )


def up_then_chop(
    spot: float,
    target: float,
    n_days: int,
    chop_amp: float = 0.07,
    chop_period: int = 5,
) -> NamedPath:
    """First half: fast ramp; second half: high-vol chop ending at K."""
    half = n_days // 2
    ramp = np.full(half, 0.0)  # placeholder; constant ramp
    chop_len = n_days - half
    t = np.arange(chop_len)
    chop = chop_amp * np.sin(2 * np.pi * t / chop_period)
    r = np.concatenate([ramp, chop])
    r = _enforce_terminal(r, np.log(target / spot))
    prices = spot * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    return NamedPath(
        name="up_then_chop",
        description="Fast ramp first, then high-vol chop converging on K.",
        prices=prices,
        log_returns=r,
        realized_qv=float((r - r.mean()).var() * n_days),
    )


def build_named_paths(spot: float, target: float, n_days: int) -> list[NamedPath]:
    return [
        monotone(spot, target, n_days),
        vol_then_up(spot, target, n_days),
        up_then_chop(spot, target, n_days),
    ]
