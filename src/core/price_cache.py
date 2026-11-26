"""
core/price_cache.py

Shared on-disk price cache for core.daily_runner (daily incremental
updates) and the backtester (one-time deep historical backfill).
Each ticker's OHLC history is stored as its own CSV under
data/price_cache/. On every call, only the date range actually missing
from the cache gets downloaded from Yahoo Finance -- everything else
is read straight from disk.

IMPORTANT: yfinance's auto_adjust=True Close is adjusted for splits and
dividends relative to the MOST RECENT DATE in that specific download
call. Naively appending a later download onto an earlier one creates a
silent price discontinuity for any ticker that split in between the
two sessions -- the old and new data end up on different adjustment
bases that don't actually connect. When extending an existing cache
forward, this module requests one extra day of overlap (the cache's
last existing date) and rescales the entire old cached series if that
overlap day's price doesn't match between old and new -- bringing
everything onto a single consistent basis before merging. A fresh/deep
backfill doesn't need this, since it's one single, internally
consistent download that fully overwrites any old data on overlapping
dates.
"""

import datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "data" / "price_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 100
DEFAULT_LOOKBACK_DAYS = 250


def _cache_path(ticker: str) -> Path:
    safe_name = ticker.replace("/", "_")
    return CACHE_DIR / f"{safe_name}.csv"


def _chunked(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def load_cached_history(ticker: str):
    path = _cache_path(ticker)
    if not path.exists():
        return None
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df if not df.empty else None


def _save_to_cache(ticker: str, df: pd.DataFrame) -> None:
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.to_csv(_cache_path(ticker))


def _download_missing(tickers_to_download: dict) -> dict:
    """tickers_to_download: {ticker: start_date (pd.Timestamp)}.
    Groups tickers needing the same start date into the same chunked
    batch download, same pattern as the old batch_download()."""
    new_data = {}
    by_start = {}
    for ticker, start in tickers_to_download.items():
        by_start.setdefault(start, []).append(ticker)

    for start, group_tickers in by_start.items():
        for chunk in _chunked(group_tickers, CHUNK_SIZE):
            raw = yf.download(
                tickers=chunk,
                start=start.strftime("%Y-%m-%d"),
                group_by="ticker",
                threads=True,
                auto_adjust=True,
                progress=False,
            )
            for ticker in chunk:
                try:
                    # Don't assume shape from chunk size -- yfinance can
                    # return a MultiIndex (ticker, field) even for a
                    # single-ticker batch. Check the actual columns
                    # instead of guessing from len(chunk).
                    if isinstance(raw.columns, pd.MultiIndex):
                        df = raw[ticker].dropna(how="all")
                    else:
                        df = raw.dropna(how="all")
                    if not df.empty:
                        new_data[ticker] = df
                except KeyError:
                    continue
    return new_data


def _reconcile_and_merge(ticker: str, cached: pd.DataFrame, new_data: pd.DataFrame) -> pd.DataFrame:
    """Merges new_data onto cached, rescaling the old cached series if
    an overlap day reveals a split/adjustment-basis discontinuity."""
    if cached is None:
        return new_data

    overlap_dates = cached.index.intersection(new_data.index)
    if len(overlap_dates) > 0:
        check_date = overlap_dates[-1]
        old_price = cached.loc[check_date, "Close"]
        new_price = new_data.loc[check_date, "Close"]
        if old_price and abs(new_price - old_price) / old_price > 0.001:
            ratio = new_price / old_price
            print(f"Price cache: adjustment-basis shift detected for {ticker} "
                  f"(likely a stock split) -- rescaling cached history by {ratio:.6f}")
            for col in ("Open", "High", "Low", "Close"):
                if col in cached.columns:
                    cached[col] = cached[col] * ratio

    merged = pd.concat([cached, new_data])
    merged = merged.sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    return merged


def get_price_history(tickers: list, min_lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                       full_backfill_days: int = None, use_cache_only: bool = False) -> dict:
    if use_cache_only:
        # Backtester comparison mode -- never touch the network, just read
        # exactly what's on disk right now. Guarantees bit-identical input
        # data across any number of runs, on any day, until a deliberate
        # cache refresh is done separately.
        result = {}
        for ticker in tickers:
            history = load_cached_history(ticker)
            if history is not None:
                result[ticker] = history
        return result
    
    """
    Returns {ticker: DataFrame}, each with at least `min_lookback_days`
    (or `full_backfill_days`, if given) of history. Only downloads
    what's actually missing -- either a forward gap since the cache's
    last date (with one overlap day requested for split reconciliation),
    or, if the existing cache doesn't reach back far enough, the whole
    requested depth as one consistent fresh download.
    """
    today = pd.Timestamp(datetime.date.today())
    requested_depth_days = full_backfill_days or min_lookback_days
    earliest_needed = today - pd.Timedelta(days=requested_depth_days)
    needed_start = {}
    needs_overlap = set()

    for ticker in tickers:
        cached = load_cached_history(ticker)
        if cached is None:
            needed_start[ticker] = earliest_needed
        else:
            cache_start = cached.index[0]
            if cache_start > earliest_needed:
                # existing cache doesn't reach back far enough -- re-pull the
                # full requested depth as one consistent download; this
                # fully overwrites old data on overlapping dates, so no
                # reconciliation is needed here
                needed_start[ticker] = earliest_needed
            else:
                gap_start = cached.index[-1] + pd.Timedelta(days=1)
                if gap_start.date() <= today.date():
                    # forward gap-fill -- request one extra day of overlap
                    # (the cache's last existing date) so a split/adjustment
                    # discontinuity can be detected and corrected
                    needed_start[ticker] = cached.index[-1]
                    needs_overlap.add(ticker)

    tickers_to_download = list(needed_start.keys())

    if tickers_to_download:
        global_start = min(needed_start.values())
        print(f"Price cache: downloading {len(tickers_to_download)} ticker(s) "
              f"from {global_start.date()}...")
        new_data = _download_missing(needed_start)
        for ticker, df in new_data.items():
            cached = load_cached_history(ticker)
            if ticker in needs_overlap and cached is not None:
                merged = _reconcile_and_merge(ticker, cached, df)
            else:
                merged = pd.concat([cached, df]) if cached is not None else df
                merged = merged.sort_index()
                merged = merged[~merged.index.duplicated(keep="last")]
            _save_to_cache(ticker, merged)

    result = {}
    for ticker in tickers:
        history = load_cached_history(ticker)
        if history is not None:
            result[ticker] = history
    return result


def get_benchmark_history(benchmarks: dict, min_lookback_days: int = DEFAULT_LOOKBACK_DAYS,
                           full_backfill_days: int = None, use_cache_only: bool = False) -> dict:
    tickers = list(benchmarks.values())
    history_by_ticker = get_price_history(tickers, min_lookback_days, full_backfill_days, use_cache_only=use_cache_only)
    return {
        track: history_by_ticker[ticker]
        for track, ticker in benchmarks.items()
        if ticker in history_by_ticker
    }