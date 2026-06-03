"""BTC-decoupling diagnostic: is IREN's full-history σ contaminated by a regime shift?

IREN was a pure bitcoin miner (Iris Energy) at IPO and pivoted to AI cloud / HPC.
The two eras have different volatility-generating processes, so a σ_base / percentile
band estimated on the *full* history blends them and distorts the breakeven verdict
(σ_be(165) sits razor-thin against σ_base). This module makes the contamination
visible — it does NOT touch the breakeven math.

Pipeline:
  1. Align IREN and BTC daily log-returns on IREN's trading days.
  2. Rolling 60/120d OLS β_t = Cov(IREN, BTC) / Var(BTC) and Pearson ρ_t.
  3. Structural-break detection on β_t (ruptures PELT; CUSUM fallback) plus an
     optional hand-set catalyst override.
  4. Re-estimate the low/base/high σ regimes on (a) full history and (b) the
     post-break sample, reusing the existing σ pipeline so the numbers compare.

No I/O lives here (kept pure for testability): callers pass already-loaded frames.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from ilb.estimate import RegimeVols, regime_vols, rolling_sigma

logger = logging.getLogger(__name__)

TRADING_DAYS = 252

# --- Tunables (documented; tune here, not at call sites) ---------------------
# PELT penalty on the *standardized* β_60 series. Higher → fewer breaks. Picked so
# IREN's miner→cloud β decline registers as one dominant break rather than vol noise.
PELT_PENALTY = 10.0
# "rbf" keys on distributional shifts (robust to variance changes); "l2" is pure
# mean-shift. rbf is the default because the β decline coincides with a vol change.
PELT_MODEL = "rbf"
PELT_MIN_SIZE = 30          # min business days between breaks
SIGMA_WINDOW = 60           # matches the existing regime pipeline (max(windows[0], 60))
BETA_WINDOW_PRIMARY = 60    # which β series drives break detection
# Min post-split rolling-σ observations required before a detected break is allowed
# to anchor the regime split — keeps the re-estimated band out of small-sample noise.
MIN_POST_OBS = 180
# Known catalysts to overlay / override the algorithmic split. Empty by default —
# fill with date(YYYY, M, D) entries (e.g. an NVIDIA/HPC partnership announcement)
# to pin the miner→cloud boundary by hand instead of trusting the detector.
CATALYST_OVERRIDE: list[date] = []


@dataclass(frozen=True)
class DecouplingResult:
    """Everything the two diagnostic plots need."""
    beta_corr: pd.DataFrame              # cols: beta_<w>, rho_<w> for each window
    windows: tuple[int, ...]
    break_dates: list[pd.Timestamp]      # all detected breaks (drawn as vlines)
    split_date: pd.Timestamp | None      # dominant break → miner/cloud boundary
    method: str                          # "pelt" | "cusum" | "none"
    penalty: float
    full_regime: RegimeVols              # σ low/base/high on full history
    post_regime: RegimeVols              # σ low/base/high on post-split sample
    catalysts: list[pd.Timestamp]        # hand-set override dates (may be empty)
    n_obs_full: int
    n_obs_post: int


def align_log_returns(iren: pd.DataFrame, btc: pd.DataFrame) -> pd.DataFrame:
    """Inner-join BTC close onto IREN trading days → aligned daily log-returns.

    Returns a frame indexed by IREN trading days with columns `iren`, `btc`
    (weekends drop out of BTC naturally via the inner join).
    """
    a = iren[["close"]].rename(columns={"close": "iren"})
    b = btc[["close"]].rename(columns={"close": "btc"})
    j = a.join(b, how="inner").sort_index()
    out = pd.DataFrame(index=j.index)
    out["iren"] = np.log(j["iren"]).diff()
    out["btc"] = np.log(j["btc"]).diff()
    return out.dropna()


def rolling_beta_corr(ret: pd.DataFrame, windows: tuple[int, ...] = (60, 120)) -> pd.DataFrame:
    """Rolling OLS β_t = Cov/Var and Pearson ρ_t of IREN on BTC, per window."""
    cols: dict[str, pd.Series] = {}
    for w in windows:
        cov = ret["iren"].rolling(w).cov(ret["btc"])
        var = ret["btc"].rolling(w).var()
        cols[f"beta_{w}"] = cov / var
        cols[f"rho_{w}"] = ret["iren"].rolling(w).corr(ret["btc"])
    return pd.DataFrame(cols, index=ret.index)


def _cusum_break(
    s: pd.Series, threshold: float = 4.0, min_size: int = PELT_MIN_SIZE
) -> list[pd.Timestamp]:
    """Fallback: standardized-mean CUSUM, flags the single dominant level shift.

    z_t = (s_t − mean)/std; the argmax of |cumsum(z)| is the most likely single
    change point. Returns it only if the deviation clears `threshold` and lands
    away from both ends (≥ min_size). One break is enough to split the regimes.
    """
    s = s.dropna()
    if len(s) < 2 * min_size:
        return []
    std = float(s.std(ddof=0))
    if std < 1e-9:  # (near-)constant series: nothing to detect, avoid amplifying fp noise
        return []
    z = (s - s.mean()) / std
    csum = np.cumsum(z.to_numpy())
    k = int(np.argmax(np.abs(csum)))
    if abs(csum[k]) < threshold or k < min_size or k > len(s) - min_size:
        return []
    return [s.index[k]]


def detect_breaks(
    series: pd.Series,
    penalty: float = PELT_PENALTY,
    model: str = PELT_MODEL,
    min_size: int = PELT_MIN_SIZE,
) -> tuple[list[pd.Timestamp], str]:
    """Detect structural breaks in `series`. ruptures PELT if importable, else CUSUM."""
    s = series.dropna()
    if len(s) < 2 * min_size:
        return [], "none"
    std = float(s.std(ddof=0))
    if std < 1e-9:  # (near-)constant series: no structure, and standardizing fp noise
        return [], "none"  #   would manufacture spurious breaks
    x = ((s - s.mean()) / std).to_numpy().reshape(-1, 1)
    try:
        import ruptures as rpt

        algo = rpt.Pelt(model=model, min_size=min_size, jump=5).fit(x)
        idx = algo.predict(pen=penalty)  # segment END indices, last == len(x)
        dates = [s.index[i] for i in idx if 0 < i < len(s)]
        return dates, "pelt"
    except Exception as exc:  # noqa: BLE001 — any ruptures import/runtime issue → CUSUM
        logger.warning("ruptures unavailable/failed (%s); using CUSUM fallback", exc)
        return _cusum_break(s, min_size=min_size), "cusum"


def select_split(
    breaks: list[pd.Timestamp],
    iren: pd.DataFrame,
    window: int = SIGMA_WINDOW,
    min_post_obs: int = MIN_POST_OBS,
) -> pd.Timestamp | None:
    """Choose the regime-split date among the detected β breaks.

    IREN's BTC-β is regime-switching rather than a clean monotone decouple, so the
    *largest β shift* is not the economically interesting boundary (it sits in the
    2022 miner-crisis era). Since the whole point is σ contamination, we anchor the
    split on the detected break whose post-sample σ_base differs most from the
    full-history σ_base — requiring ≥ `min_post_obs` post rolling-σ observations so
    the comparison is not an artefact of a tiny tail sample.
    """
    if not breaks:
        return None
    r = iren["log_return"]
    full_base = regime_vols(rolling_sigma(r, window).dropna()).base
    best, best_gap = None, -1.0
    for b in breaks:
        post = rolling_sigma(r[r.index > b], window).dropna()
        if post.size < min_post_obs:
            continue
        gap = abs(regime_vols(post).base - full_base)
        if gap > best_gap:
            best, best_gap = b, gap
    return best


def regime_sigma_split(
    iren: pd.DataFrame, split_date: pd.Timestamp | None, window: int = SIGMA_WINDOW
) -> tuple[RegimeVols, RegimeVols, int, int]:
    """Re-estimate σ regimes on full history vs the post-split sample.

    Reuses regime_vols(rolling_sigma(...)) so the numbers are directly comparable
    to the site's σ_low/base/high. Falls back to the full regime if the post-split
    sample is too short to form a rolling-σ distribution.
    """
    r = iren["log_return"]
    full_roll = rolling_sigma(r, window).dropna()
    full = regime_vols(full_roll)
    if split_date is None:
        return full, full, int(full_roll.size), int(full_roll.size)
    post_r = r[r.index > split_date]
    post_roll = rolling_sigma(post_r, window).dropna()
    if post_roll.size < 5:
        logger.warning("post-split σ sample too short (%d); reusing full regime", post_roll.size)
        return full, full, int(full_roll.size), int(post_roll.size)
    return full, regime_vols(post_roll), int(full_roll.size), int(post_roll.size)


def analyze(
    iren: pd.DataFrame,
    btc: pd.DataFrame,
    windows: tuple[int, ...] = (60, 120),
    penalty: float = PELT_PENALTY,
    catalysts: list[date] | None = None,
) -> DecouplingResult:
    """Full diagnostic. `catalysts` (defaults to CATALYST_OVERRIDE) pins the split."""
    cats = CATALYST_OVERRIDE if catalysts is None else catalysts
    cat_ts = [pd.Timestamp(c) for c in cats]

    ret = align_log_returns(iren, btc)
    bc = rolling_beta_corr(ret, windows)
    pcol = f"beta_{BETA_WINDOW_PRIMARY}"
    primary = bc[pcol] if pcol in bc else bc.iloc[:, 0]

    breaks, method = detect_breaks(primary, penalty=penalty)
    # Catalyst override wins as the regime boundary; else the σ-relevant detected break.
    split = cat_ts[0] if cat_ts else select_split(breaks, iren, SIGMA_WINDOW)
    full, post, n_full, n_post = regime_sigma_split(iren, split, window=SIGMA_WINDOW)

    if breaks:
        logger.info("β breaks (%s, pen=%.1f): %s", method, penalty,
                    [d.date().isoformat() for d in breaks])
    logger.info("regime split @ %s | σ_base full=%.3f post=%.3f",
                split.date().isoformat() if split is not None else "none",
                full.base, post.base)

    return DecouplingResult(
        beta_corr=bc, windows=tuple(windows), break_dates=breaks, split_date=split,
        method=method, penalty=penalty, full_regime=full, post_regime=post,
        catalysts=cat_ts, n_obs_full=n_full, n_obs_post=n_post,
    )
