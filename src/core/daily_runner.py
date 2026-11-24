import json
import time
import datetime
from pathlib import Path

import yfinance as yf

from core.signals import add_indicators, add_relative_strength
from core.risk import add_position_size
from core.portfolio import (
    load_portfolio, position_count, sector_counts, total_invested,
    held_tickers, add_position, remove_position, save_portfolio,
)
from profiles.profile_1_trend_momentum import add_entry_signal, check_exit, trading_days_held

CHUNK_SIZE = 100
LOOKBACK = "6mo"
BENCHMARK_LOOKBACK = "2y"
REGIME_SMA_WINDOW = 200
MAX_POSITIONS = 10
CASH_RESERVE_PCT = 0.10
MIN_ADX = 20  # quality floor -- a candidate must show genuine trend strength, not just rank well today
ROTATION_SCORE_MARGIN = 1.0
MIN_HOLDING_DAYS_FOR_ROTATION = 14
SCENARIO = "zero"

BENCHMARKS = {"europe": "^STOXX50E", "global": "^GSPC"}

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def load_latest_universe() -> dict:
    files = sorted(DATA_DIR.glob("universe_*.json"))
    with open(files[-1]) as f:
        return json.load(f)


def chunked(lst, size):
    for i in range(0, len(lst), size):
        yield lst[i:i + size]


def batch_download(tickers: list) -> dict:
    data = {}
    for group in chunked(tickers, CHUNK_SIZE):
        raw = yf.download(
            tickers=group,
            period=LOOKBACK,
            group_by="ticker",
            threads=True,
            auto_adjust=True,
            progress=False,
        )
        for ticker in group:
            try:
                df = raw[ticker].dropna(how="all")
                if not df.empty:
                    data[ticker] = df
            except KeyError:
                continue
    return data


def download_benchmarks() -> dict:
    raw = yf.download(
        tickers=list(BENCHMARKS.values()),
        period=BENCHMARK_LOOKBACK,
        group_by="ticker",
        threads=True,
        auto_adjust=True,
        progress=False,
    )
    return {track: raw[ticker].dropna(how="all") for track, ticker in BENCHMARKS.items()}


def market_regime_ok(benchmark_df, window: int = REGIME_SMA_WINDOW) -> bool:
    if len(benchmark_df) < window:
        return True
    sma = benchmark_df["Close"].rolling(window=window).mean()
    return bool(benchmark_df["Close"].iloc[-1] > sma.iloc[-1])


def check_open_positions(portfolio_state: dict, price_data: dict) -> list:
    exit_candidates = []
    for position in portfolio_state["positions"]:
        ticker = position["ticker"]
        prices = price_data.get(ticker)
        if prices is None or len(prices) < 60:
            print(f"  Skipping exit check for {ticker}: no recent price data")
            continue
        df = add_indicators(prices)
        today_row = df.iloc[-1]
        result = check_exit(position, today_row)
        position["highest_close"] = result["highest_close"]
        if result["exit"]:
            exit_candidates.append({
                "ticker": ticker,
                "reason": result["reason"],
                "exit_price": round(result["exit_price"], 2),
                "shares": position["shares"],
                "proceeds": round(result["exit_price"] * position["shares"], 2),
            })
    return exit_candidates


def confirm_and_execute_exits(exit_candidates: list, portfolio_state: dict, scenario: str = SCENARIO) -> dict:
    if not exit_candidates:
        return portfolio_state

    print("\nThe following positions have hit an exit condition:")
    for e in exit_candidates:
        print(f"  {e['ticker']}: {e['reason']} at {e['exit_price']} "
              f"({e['shares']} shares, proceeds ~{e['proceeds']})")

    print("\nEnter the tickers you want to exit (comma-separated), 'all', or 'none':")
    choice = input("> ").strip().lower()

    if choice in ("", "none"):
        approved_tickers = set()
    elif choice == "all":
        approved_tickers = {e["ticker"] for e in exit_candidates}
    else:
        approved_tickers = {t.strip().upper() for t in choice.split(",")}

    for e in exit_candidates:
        if e["ticker"] in approved_tickers:
            portfolio_state = remove_position(portfolio_state, e["ticker"], e["exit_price"])
            print(f"Recorded exit: {e['ticker']}")

    save_portfolio(portfolio_state, scenario)
    return portfolio_state


def find_candidates(universe: dict, price_data: dict, benchmarks: dict, portfolio_value: float) -> list:
    europe_set = set(universe["europe"])
    sectors = universe.get("sectors", {})
    candidates = []

    regime_ok = {track: market_regime_ok(df) for track, df in benchmarks.items()}
    for track, ok in regime_ok.items():
        if not ok:
            print(f"Market regime filter: {track} benchmark below its {REGIME_SMA_WINDOW}-day SMA "
                  f"-- pausing new entries for this track today")

    for ticker, prices in price_data.items():
        if len(prices) < 60:
            continue
        track = "europe" if ticker in europe_set else "global"
        if not regime_ok[track]:
            continue
        try:
            df = add_indicators(prices)
            df = add_relative_strength(df, benchmarks[track])
            df = add_entry_signal(df)
            df = add_position_size(df, portfolio_value)
            today = df.iloc[-1]
            if today["entry_signal"] and today["adx14"] >= MIN_ADX:
                candidates.append({
                    "ticker": ticker,
                    "track": track,
                    "sector": sectors.get(ticker, "Unknown"),
                    "close": round(today["Close"], 2),
                    "stop_price": round(today["stop_price"], 2),
                    "shares": int(today["position_shares"]),
                    "cost": round(today["position_cost"], 2),
                    "adx14": today["adx14"],
                    "rel_strength_60d": today["rel_strength_60d"],
                    "atr14": today["atr14"],
                })
        except Exception as e:
            print(f"  Skipped {ticker}: {e}")
    return candidates


def zscore_params(values: list) -> tuple:
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    return mean, std


def apply_zscore(value: float, mean: float, std: float) -> float:
    if std == 0:
        return 0.0
    return (value - mean) / std


def score_candidates(candidates: list) -> tuple:
    adx_mean, adx_std = zscore_params([c["adx14"] for c in candidates])
    rs_mean, rs_std = zscore_params([c["rel_strength_60d"] for c in candidates])
    for c in candidates:
        za = apply_zscore(c["adx14"], adx_mean, adx_std)
        zr = apply_zscore(c["rel_strength_60d"], rs_mean, rs_std)
        c["score"] = 0.5 * za + 0.5 * zr
    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    return ranked, (adx_mean, adx_std, rs_mean, rs_std)


def select_shortlist(ranked_candidates: list, portfolio_state: dict, portfolio_value: float) -> tuple:
    already_held = held_tickers(portfolio_state)
    running_count = position_count(portfolio_state)
    running_sector_counts = dict(sector_counts(portfolio_state))
    running_cash = portfolio_state["cash"]
    min_cash_reserve = CASH_RESERVE_PCT * portfolio_value

    shortlist = []
    blocked = []
    for c in ranked_candidates:
        if c["ticker"] in already_held:
            continue
        if running_count >= MAX_POSITIONS:
            blocked.append(c)
            continue
        if running_cash - c["cost"] < min_cash_reserve:
            blocked.append(c)
            continue
        hypothetical_count = running_count + 1
        sector_cap = max(1, hypothetical_count // 3)
        current_sector_count = running_sector_counts.get(c["sector"], 0)
        if current_sector_count >= sector_cap:
            blocked.append(c)
            continue
        shortlist.append(c)
        running_count = hypothetical_count
        running_sector_counts[c["sector"]] = current_sector_count + 1
        running_cash -= c["cost"]
    return shortlist, blocked


def confirm_and_record(shortlist: list, portfolio_state: dict, scenario: str = SCENARIO) -> dict:
    print("\nEnter the tickers you want to take (comma-separated), 'all', or 'none':")
    choice = input("> ").strip().lower()

    if choice in ("", "none"):
        approved_tickers = set()
    elif choice == "all":
        approved_tickers = {c["ticker"] for c in shortlist}
    else:
        approved_tickers = {t.strip().upper() for t in choice.split(",")}

    today_str = datetime.date.today().isoformat()
    for c in shortlist:
        if c["ticker"] in approved_tickers:
            portfolio_state = add_position(
                portfolio_state,
                ticker=c["ticker"],
                sector=c["sector"],
                entry_date=today_str,
                entry_price=c["close"],
                shares=c["shares"],
                stop_price=c["stop_price"],
                track=c["track"],
                atr_at_entry=c["atr14"],
            )
            print(f"Recorded entry: {c['ticker']}")

    save_portfolio(portfolio_state, scenario)
    return portfolio_state


def score_held_position(position: dict, price_data: dict, benchmarks: dict, scoring_params: tuple):
    ticker = position["ticker"]
    prices = price_data.get(ticker)
    if prices is None or len(prices) < 60:
        return None
    track = position["track"]
    df = add_indicators(prices)
    df = add_relative_strength(df, benchmarks[track])
    today = df.iloc[-1]
    adx_mean, adx_std, rs_mean, rs_std = scoring_params
    za = apply_zscore(today["adx14"], adx_mean, adx_std)
    zr = apply_zscore(today["rel_strength_60d"], rs_mean, rs_std)
    score = 0.5 * za + 0.5 * zr
    return score, round(today["Close"], 2)


def evaluate_rotation(blocked_candidates: list, portfolio_state: dict, price_data: dict,
                       benchmarks: dict, scoring_params: tuple) -> dict:
    if not blocked_candidates or not portfolio_state["positions"]:
        return None

    eligible_held = []
    for position in portfolio_state["positions"]:
        if trading_days_held(position["entry_date"]) >= MIN_HOLDING_DAYS_FOR_ROTATION:
            result = score_held_position(position, price_data, benchmarks, scoring_params)
            if result is not None:
                held_score, held_close = result
                eligible_held.append((position, held_score, held_close))

    if not eligible_held:
        return None

    weakest_position, weakest_score, weakest_close = min(eligible_held, key=lambda x: x[1])
    best_blocked = blocked_candidates[0]

    if best_blocked["score"] - weakest_score >= ROTATION_SCORE_MARGIN:
        return {
            "sell": weakest_position,
            "sell_score": weakest_score,
            "sell_price": weakest_close,
            "buy": best_blocked,
        }
    return None


def confirm_and_execute_rotation(rotation: dict, portfolio_state: dict, scenario: str = SCENARIO) -> dict:
    sell = rotation["sell"]
    buy = rotation["buy"]
    margin = buy["score"] - rotation["sell_score"]
    print(f"\nRotation opportunity: {buy['ticker']} (score={buy['score']:.2f}) would replace "
          f"{sell['ticker']} (score={rotation['sell_score']:.2f}) -- margin {margin:.2f}.")
    print("Confirm this swap? (y/n):")
    choice = input("> ").strip().lower()
    if choice == "y":
        portfolio_state = remove_position(portfolio_state, sell["ticker"], rotation["sell_price"])
        today_str = datetime.date.today().isoformat()
        portfolio_state = add_position(
            portfolio_state,
            ticker=buy["ticker"],
            sector=buy["sector"],
            entry_date=today_str,
            entry_price=buy["close"],
            shares=buy["shares"],
            stop_price=buy["stop_price"],
            track=buy["track"],
            atr_at_entry=buy["atr14"],
        )
        save_portfolio(portfolio_state, scenario)
        print(f"Rotation executed: sold {sell['ticker']}, bought {buy['ticker']}")
    else:
        print("Rotation declined.")
    return portfolio_state


if __name__ == "__main__":
    universe = load_latest_universe()
    all_tickers = universe["europe"] + universe["global"]

    portfolio_state = load_portfolio(SCENARIO)
    held = held_tickers(portfolio_state)
    download_list = sorted(set(all_tickers) | held)

    print(f"Portfolio ({SCENARIO}): cash={portfolio_state['cash']:.2f}, "
          f"invested={total_invested(portfolio_state):.2f}, open positions={position_count(portfolio_state)}")

    start = time.time()
    print(f"\nDownloading price history for {len(download_list)} tickers in batches of {CHUNK_SIZE}...")
    price_data = batch_download(download_list)
    benchmarks = download_benchmarks()
    print(f"Downloaded {len(price_data)} tickers + 2 benchmarks in {time.time() - start:.1f}s")

    if portfolio_state["positions"]:
        print("\nChecking open positions for exit conditions...")
        exit_candidates = check_open_positions(portfolio_state, price_data)
        if exit_candidates:
            portfolio_state = confirm_and_execute_exits(exit_candidates, portfolio_state)
        else:
            save_portfolio(portfolio_state, SCENARIO)
            print("No exit conditions triggered today.")

    portfolio_value = portfolio_state["cash"] + total_invested(portfolio_state)

    print("\nScanning for entry signals...")
    candidates = find_candidates(universe, price_data, benchmarks, portfolio_value)
    print(f"Found {len(candidates)} raw entry candidate(s) today")

    if candidates:
        ranked, scoring_params = score_candidates(candidates)
        shortlist, blocked = select_shortlist(ranked, portfolio_state, portfolio_value)

        print(f"\nShortlist after ranking + caps ({len(shortlist)} of {len(ranked)}):\n")
        for c in shortlist:
            print(f"  {c['ticker']} ({c['sector']}, {c['track']}): score={c['score']:.2f}, "
                  f"close={c['close']}, stop={c['stop_price']}, buy {c['shares']} shares (~{c['cost']})")

        if shortlist:
            portfolio_state = confirm_and_record(shortlist, portfolio_state)

        rotation = evaluate_rotation(blocked, portfolio_state, price_data, benchmarks, scoring_params)
        if rotation:
            portfolio_state = confirm_and_execute_rotation(rotation, portfolio_state)
    else:
        print("No candidates today.")