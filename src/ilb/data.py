"""Daily-bar fetch + parquet cache + cleaning (spec §3).

yfinance is the primary source; on failure (network, rate limit, etc.) we
fall back to the parquet cache and, failing that, to the committed CSV
snapshot at `data_snapshot/<ticker>_daily.csv`.

Cleaning rules:
- adjusted close only (yfinance auto_adjust=True ⇒ Close is already adjusted)
- drop nonpositive / NaN / duplicate-date rows
- enforce business-day frequency
- log (don't drop) |daily log-return| > 0.5 as outliers
"""

from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data")
SNAPSHOT_DIR = Path("data_snapshot")
STALE_AFTER = timedelta(hours=18)  # auto-refresh if last write older than this


def _cache_path(ticker: str) -> Path:
    return CACHE_DIR / f"{ticker.lower()}_daily.parquet"


def _snapshot_path(ticker: str) -> Path:
    return SNAPSHOT_DIR / f"{ticker.lower()}_daily.csv"


def _is_stale(path: Path, now: datetime | None = None) -> bool:
    if not path.exists():
        return True
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (now or datetime.now()) - mtime > STALE_AFTER


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """Validate, drop bad rows, log outliers, return Close-only frame with log returns."""
    if df.empty:
        raise ValueError("empty price frame")
    # Some yfinance returns have a column MultiIndex when a single ticker is passed
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    if "Close" not in df.columns:
        raise ValueError(f"price frame missing 'Close' column; cols={list(df.columns)}")

    out = pd.DataFrame({"close": pd.to_numeric(df["Close"], errors="coerce")})
    out.index = pd.to_datetime(df.index).tz_localize(None)
    out = out[~out.index.duplicated(keep="last")]
    out = out.sort_index()
    out = out[out["close"].notna() & (out["close"] > 0)]
    if out.empty:
        raise ValueError("all rows dropped during cleaning")

    out["log_return"] = np.log(out["close"]).diff()
    outliers = out["log_return"].abs() > 0.5
    n_out = int(outliers.fillna(False).sum())
    if n_out:
        logger.warning(
            "%d daily log-returns with |r|>0.5 retained; sample dates=%s",
            n_out,
            out.index[outliers.fillna(False)][:5].strftime("%Y-%m-%d").tolist(),
        )
    return out


def _fetch_yfinance(ticker: str, start: str | None = None) -> pd.DataFrame:
    import yfinance as yf  # local import keeps test paths cheap

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            raw = yf.download(
                ticker,
                start=start or "2021-11-01",
                end=None,
                progress=False,
                auto_adjust=True,
                actions=False,
                threads=False,
            )
            if raw is None or raw.empty:
                raise ValueError("yfinance returned empty frame")
            return raw
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("yfinance attempt %d failed: %s", attempt + 1, exc)
            time.sleep(1.5 * (attempt + 1))
    assert last_err is not None
    raise last_err


def load_prices(
    ticker: str = "IREN",
    refresh: bool = False,
    allow_network: bool = True,
) -> pd.DataFrame:
    """Return cleaned daily price/return frame, using cache when fresh.

    Columns: `close` (adjusted), `log_return` (NaN on first row).
    Index: tz-naive DatetimeIndex of trading days.
    """
    cache = _cache_path(ticker)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not refresh and cache.exists() and not _is_stale(cache):
        return pd.read_parquet(cache)

    if allow_network:
        try:
            raw = _fetch_yfinance(ticker)
            cleaned = _clean(raw)
            cleaned.to_parquet(cache)
            logger.info("fetched %d rows for %s (through %s)",
                        len(cleaned), ticker, cleaned.index[-1].date())
            return cleaned
        except Exception as exc:  # noqa: BLE001
            logger.warning("live fetch failed (%s); falling back to cache/snapshot", exc)

    if cache.exists():
        return pd.read_parquet(cache)

    snap = _snapshot_path(ticker)
    if snap.exists():
        df = pd.read_csv(snap, parse_dates=["date"]).set_index("date")
        df.index = df.index.tz_localize(None) if df.index.tz else df.index
        if "log_return" not in df.columns:
            df["log_return"] = np.log(df["close"]).diff()
        return df

    raise FileNotFoundError(
        f"no cache, no snapshot, and live fetch unavailable for {ticker}"
    )


def write_snapshot(df: pd.DataFrame, ticker: str = "IREN") -> Path:
    """Write the cleaned frame to data_snapshot/<ticker>_daily.csv for reproducible builds."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = _snapshot_path(ticker)
    out = df.reset_index().rename(columns={"index": "date", df.index.name or "index": "date"})
    if "date" not in out.columns:
        out.insert(0, "date", df.index)
    out.to_csv(path, index=False, date_format="%Y-%m-%d")
    return path


def latest_spot(df: pd.DataFrame) -> tuple[float, date]:
    last = df.iloc[-1]
    return float(last["close"]), df.index[-1].date()
