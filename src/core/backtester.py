"""
core/backtester.py

Single-run historical replay of the trading engine. Precomputes every
ticker's indicators and entry signal once, vectorized across its full
history, then walks forward day by day reusing the same decision logic
as daily_runner: check_exit, score_candidates, select_shortlist, and a
rotation check. Human input() confirmation is replaced by "auto-approve
everything the rules select."

Applies DEGIRO's published commission schedule and AutoFX conversion
fee, and compares the result against dividend-reinvested total-return
benchmarks (SPY for global, EXW1.DE for Europe).

Current experimental changes under test:
  - time_stop_trading_days = 20 (evidence-backed from the walk-forward
    sweep; was 30)
  - min_holding_days_for_trailing_stop = 5 (confirmed win -- blocks the
    trailing stop from firing on entry-day noise; left trend_break and
    time_stop ungated)
  - rotation_score_margin = 1.5, up from 1.0 (confirmed win -- fewer,
    higher-conviction rotation swaps)
  - trailing_stop_atr_multiplier tested at 1.5 and 2.5 against the
    above two confirmed changes -- both were regressions; 2.0 (the
    original default) remains best and is locked in
  - min_composite_score gate on entries (candidates must rank at or
    above average on the same ADX/relative-strength z-score already
    used for rotation, not just clear the absolute MIN_ADX floor) --
    confirmed via clean data to NOT fix trend_break's near-zero win
    rate, kept at 0.0 (no-op threshold)
  - NEW: entry_cushion_pct tracked per trade (% distance price was
    above its 50-day SMA at entry) to test the hypothesis that
    trend_break's near-zero win rate comes from entries happening too
    close to a barely-rising SMA50 -- purely diagnostic, no behavior
    change
  - fees tracked explicitly per trade (separate from P&L)
  - position sizing AND the equity curve use true mark-to-market value
    for open positions, not cost basis
  - rotation now verifies the buy side is actually affordable (cash
    after selling the weak position) before executing either leg --
    fixes a bug where an unchecked rotation buy could drive cash
    negative and spiral into runaway position sizing
"""

import datetime
import statistics

from core.price_cache import get_price_history, get_benchmark_history
from core.signals import add_indicators, add_relative_strength
from core.daily_runner import (
    BENCHMARKS, MAX_POSITIONS, CASH_RESERVE_PCT, MIN_ADX,
    ROTATION_SCORE_MARGIN, MIN_HOLDING_DAYS_FOR_ROTATION,
    load_latest_universe, zscore_params, apply_zscore,
    score_candidates, select_shortlist,
)
from profiles.profile_1_trend_momentum import add_entry_signal, check_exit, trading_days_held, PROFILE_1_CONFIG

RISK_PCT = 0.01
REGIME_SMA_WINDOW = 200
WARMUP_BUFFER_DAYS = 400
TRADING_DAYS_PER_YEAR = 252
BENCHMARK_TOTAL_RETURN = {"europe": "EXW1.DE", "global": "SPY"}


def commission_for(ticker: str, track: str) -> float:
    if track == "global":
        return 1.00
    if ticker.endswith(".AS") or ticker.endswith(".BR"):
        return 2.00
    return 3.90


def fx_fee_for(track: str, trade_value: float) -> float:
    return trade_value * 0.0025 if track == "global" else 0.0


def _precompute_indicators(raw_history: dict, universe: dict, regime_benchmarks: dict, config: dict) -> dict:
    europe_set = set(universe["europe"])
    indicators = {}
    for ticker, prices in raw_history.items():
        if len(prices) < 60:
            continue
        track = "europe" if ticker in europe_set else "global"
        benchmark_df = regime_benchmarks.get(track)
        if benchmark_df is None:
            continue
        try:
            df = add_indicators(prices)
            df = add_relative_strength(df, benchmark_df)
            df = add_entry_signal(df, config=config)
            row_lookup = df.to_dict("index")
            indicators[ticker] = (df, track, row_lookup)
        except Exception:
            continue
    return indicators


def _precompute_regime(regime_benchmarks: dict, window: int = REGIME_SMA_WINDOW) -> dict:
    regime = {}
    for track, df in regime_benchmarks.items():
        sma = df["Close"].rolling(window=window).mean()
        regime[track] = df["Close"] > sma
    return regime


def _score_held_position(position: dict, indicators: dict, date, scoring_params: tuple):
    entry = indicators.get(position["ticker"])
    if entry is None or date not in entry[2]:
        return None
    _df, _track, row_lookup = entry
    row = row_lookup[date]
    adx_mean, adx_std, rs_mean, rs_std = scoring_params
    za = apply_zscore(row["adx14"], adx_mean, adx_std)
    zr = apply_zscore(row["rel_strength_60d"], rs_mean, rs_std)
    return 0.5 * za + 0.5 * zr, round(row["Close"], 2)


def _evaluate_rotation(blocked_candidates: list, state: dict, indicators: dict, date, as_of,
                        scoring_params: tuple, rotation_score_margin: float, min_holding_days: int):
    eligible = []
    for position in state["positions"]:
        if trading_days_held(position["entry_date"], as_of) >= min_holding_days:
            result = _score_held_position(position, indicators, date, scoring_params)
            if result is not None:
                score, close = result
                eligible.append((position, score, close))
    if not eligible:
        return None
    weakest_position, weakest_score, weakest_close = min(eligible, key=lambda x: x[1])
    best_blocked = blocked_candidates[0]
    if best_blocked["score"] - weakest_score >= rotation_score_margin:
        return {"sell": weakest_position, "sell_price": weakest_close, "buy": best_blocked}
    return None


def _mark_to_market_value(positions: list, indicators: dict, date) -> float:
    total = 0.0
    for p in positions:
        entry = indicators.get(p["ticker"])
        if entry is not None and date in entry[2]:
            total += entry[2][date]["Close"] * p["shares"]
        else:
            total += p["cost"]  # fallback if this ticker has no data that day
    return total


def _entry_cushion(position: dict):
    """% distance price was above its 50-day SMA at entry. None if the
    position predates this field (shouldn't happen in a fresh run)."""
    sma = position.get("sma50_at_entry")
    if not sma:
        return None
    return (position["entry_price"] - sma) / sma


def load_backtest_data(end_date: datetime.date, backfill_days: int, profile_config: dict = None) -> dict:
    """Expensive -- run once. Loads the universe, ensures the price cache
    covers the full requested depth, and precomputes every ticker's
    indicators/entry signals/regime series."""
    profile_config = profile_config or PROFILE_1_CONFIG
    universe = load_latest_universe()
    sectors = universe.get("sectors", {})
    all_tickers = universe["europe"] + universe["global"]

    print(f"Backtester: ensuring {backfill_days} days of cached history "
          f"(first run this far back will be slow; cached afterward)...")
    raw_history = get_price_history(all_tickers, full_backfill_days=backfill_days, use_cache_only=True)
    regime_benchmarks = get_benchmark_history(BENCHMARKS, full_backfill_days=backfill_days, use_cache_only=True)
    total_return_history = get_price_history(list(BENCHMARK_TOTAL_RETURN.values()), full_backfill_days=backfill_days, use_cache_only=True)

    indicators = _precompute_indicators(raw_history, universe, regime_benchmarks, profile_config)
    regime_series = _precompute_regime(regime_benchmarks)

    return {
        "sectors": sectors,
        "indicators": indicators,
        "regime_series": regime_series,
        "calendar": regime_benchmarks["global"].index,
        "total_return_history": total_return_history,
    }


def replay(data: dict, start_date: datetime.date, end_date: datetime.date,
           initial_cash: float = 25000.0,
           max_positions: int = MAX_POSITIONS,
           cash_reserve_pct: float = CASH_RESERVE_PCT,
           min_adx: float = MIN_ADX,
           min_composite_score: float = 0.0,
           rotation_score_margin: float = ROTATION_SCORE_MARGIN,
           min_holding_days_for_rotation: int = MIN_HOLDING_DAYS_FOR_ROTATION,
           profile_config: dict = None,
           apply_costs: bool = True) -> dict:
    """Cheap -- call many times against the same loaded data. Walks
    [start_date, end_date] with the given parameter overrides."""
    profile_config = profile_config or PROFILE_1_CONFIG
    indicators = data["indicators"]
    regime_series = data["regime_series"]
    sectors = data["sectors"]

    calendar = data["calendar"]
    calendar = calendar[(calendar.date >= start_date) & (calendar.date <= end_date)]

    state = {"cash": initial_cash, "positions": []}
    equity_curve = []
    trade_log = []

    for date in calendar:
        as_of = date.date()

        # --- exits ---
        still_open = []
        for position in state["positions"]:
            entry = indicators.get(position["ticker"])
            if entry is None or date not in entry[2]:
                still_open.append(position)
                continue
            today_row = entry[2][date]
            result = check_exit(position, today_row, config=profile_config, as_of_date=as_of)
            position["highest_close"] = result["highest_close"]
            if result["exit"]:
                exit_price = result["exit_price"]
                raw_proceeds = exit_price * position["shares"]
                proceeds = raw_proceeds
                if apply_costs:
                    proceeds -= commission_for(position["ticker"], position["track"])
                    proceeds -= fx_fee_for(position["track"], proceeds)
                state["cash"] += proceeds
                trade_log.append({
                    "date": as_of, "ticker": position["ticker"], "action": "exit",
                    "reason": result["reason"], "price": exit_price, "shares": position["shares"],
                    "pnl": proceeds - position["cost"],
                    "fees": raw_proceeds - proceeds,
                    "holding_days": trading_days_held(position["entry_date"], as_of),
                    "entry_cushion_pct": _entry_cushion(position),
                })
            else:
                still_open.append(position)
        state["positions"] = still_open

        portfolio_value = state["cash"] + _mark_to_market_value(state["positions"], indicators, date)

        regime_ok = {track: bool(series.loc[date]) if date in series.index else True
                     for track, series in regime_series.items()}

        held = {p["ticker"] for p in state["positions"]}
        candidates = []
        for ticker, (df, track, row_lookup) in indicators.items():
            if ticker in held or date not in row_lookup or not regime_ok.get(track, True):
                continue
            row = row_lookup[date]
            if not row["entry_signal"] or row["adx14"] < min_adx:
                continue
            stop_distance = row["Close"] - row["stop_price"]
            if stop_distance <= 0:
                continue
            shares = int((portfolio_value * RISK_PCT) / stop_distance)
            if shares <= 0:
                continue
            candidates.append({
                "ticker": ticker, "track": track, "sector": sectors.get(ticker, "Unknown"),
                "close": round(row["Close"], 2), "stop_price": round(row["stop_price"], 2),
                "shares": shares, "cost": round(shares * row["Close"], 2),
                "adx14": row["adx14"], "rel_strength_60d": row["rel_strength_60d"], "atr14": row["atr14"],
                "sma50_at_entry": row["sma50"],
            })

        if candidates:
            ranked, scoring_params = score_candidates(candidates)
            ranked = [c for c in ranked if c["score"] >= min_composite_score]
            shortlist, blocked = select_shortlist(ranked, state, portfolio_value,
                                                   max_positions=max_positions,
                                                   cash_reserve_pct=cash_reserve_pct)
            for c in shortlist:
                raw_cost = c["cost"]
                entry_cost = raw_cost
                if apply_costs:
                    entry_cost += commission_for(c["ticker"], c["track"])
                    entry_cost += fx_fee_for(c["track"], c["cost"])
                state["cash"] -= entry_cost
                state["positions"].append({
                    "ticker": c["ticker"], "sector": c["sector"], "entry_date": as_of.isoformat(),
                    "entry_price": c["close"], "shares": c["shares"], "cost": entry_cost,
                    "stop_price": c["stop_price"], "track": c["track"],
                    "atr_at_entry": c["atr14"], "highest_close": c["close"],
                    "sma50_at_entry": c["sma50_at_entry"],
                })
                trade_log.append({"date": as_of, "ticker": c["ticker"], "action": "entry",
                                   "price": c["close"], "shares": c["shares"], "cost": entry_cost,
                                   "fees": entry_cost - raw_cost})

            if blocked and state["positions"]:
                rotation = _evaluate_rotation(blocked, state, indicators, date, as_of, scoring_params,
                                               rotation_score_margin, min_holding_days_for_rotation)
                if rotation:
                    sell, buy = rotation["sell"], rotation["buy"]
                    raw_proceeds = rotation["sell_price"] * sell["shares"]
                    proceeds = raw_proceeds
                    if apply_costs:
                        proceeds -= commission_for(sell["ticker"], sell["track"])
                        proceeds -= fx_fee_for(sell["track"], proceeds)

                    raw_cost = buy["cost"]
                    entry_cost = raw_cost
                    if apply_costs:
                        entry_cost += commission_for(buy["ticker"], buy["track"])
                        entry_cost += fx_fee_for(buy["track"], buy["cost"])

                    cash_after_sell = state["cash"] + proceeds
                    if entry_cost <= cash_after_sell:
                        state["cash"] = cash_after_sell
                        state["positions"] = [p for p in state["positions"] if p["ticker"] != sell["ticker"]]
                        trade_log.append({"date": as_of, "ticker": sell["ticker"], "action": "exit",
                                           "reason": "rotation", "price": rotation["sell_price"],
                                           "shares": sell["shares"], "pnl": proceeds - sell["cost"],
                                           "fees": raw_proceeds - proceeds,
                                           "holding_days": trading_days_held(sell["entry_date"], as_of),
                                           "entry_cushion_pct": _entry_cushion(sell)})

                        state["cash"] -= entry_cost
                        state["positions"].append({
                            "ticker": buy["ticker"], "sector": buy["sector"], "entry_date": as_of.isoformat(),
                            "entry_price": buy["close"], "shares": buy["shares"], "cost": entry_cost,
                            "stop_price": buy["stop_price"], "track": buy["track"],
                            "atr_at_entry": buy["atr14"], "highest_close": buy["close"],
                            "sma50_at_entry": buy["sma50_at_entry"],
                        })
                        trade_log.append({"date": as_of, "ticker": buy["ticker"], "action": "entry",
                                           "price": buy["close"], "shares": buy["shares"], "cost": entry_cost,
                                           "fees": entry_cost - raw_cost})
                    # else: the proposed buy isn't affordable even after freeing up
                    # the sold position's cash -- skip this rotation entirely rather
                    # than executing a partial swap or overdrawing cash

        cash = state["cash"]
        invested_mtm = _mark_to_market_value(state["positions"], indicators, date)
        equity_curve.append({"date": as_of, "value": cash + invested_mtm, "cash": cash,
                              "invested": invested_mtm, "num_positions": len(state["positions"])})

    return {
        "equity_curve": equity_curve,
        "trade_log": trade_log,
        "final_state": state,
        "total_return_history": data["total_return_history"],
        "start_date": start_date,
        "end_date": end_date,
        "initial_cash": initial_cash,
    }


def run_backtest(start_date: datetime.date, end_date: datetime.date, **kwargs) -> dict:
    """Standalone single run -- loads data and replays in one call. For
    repeated runs (the parameter sweep), call load_backtest_data() once
    and replay() many times instead -- see core.parameter_sweep."""
    backfill_days = (datetime.date.today() - start_date).days + WARMUP_BUFFER_DAYS
    data = load_backtest_data(datetime.date.today(), backfill_days, kwargs.get("profile_config"))
    return replay(data, start_date, end_date, **kwargs)


# --- performance metrics ---

def _max_drawdown(values: list) -> float:
    peak = values[0]
    max_dd = 0.0
    for v in values:
        peak = max(peak, v)
        max_dd = min(max_dd, (v - peak) / peak)
    return max_dd


def _cagr(start_value: float, end_value: float, years: float) -> float:
    if years <= 0 or start_value <= 0:
        return 0.0
    return (end_value / start_value) ** (1 / years) - 1


def _sharpe(daily_returns: list) -> float:
    if len(daily_returns) < 2:
        return 0.0
    mean = statistics.mean(daily_returns)
    std = statistics.pstdev(daily_returns)
    if std == 0:
        return 0.0
    return (mean / std) * (TRADING_DAYS_PER_YEAR ** 0.5)


def _win_rate(trade_log: list) -> float:
    exits = [t for t in trade_log if t["action"] == "exit"]
    if not exits:
        return 0.0
    return sum(1 for t in exits if t["pnl"] > 0) / len(exits)


def _avg_holding_days(trade_log: list) -> float:
    exits = [t for t in trade_log if t["action"] == "exit"]
    if not exits:
        return 0.0
    return sum(t["holding_days"] for t in exits) / len(exits)


def _curve_summary(values: list, start_date, end_date) -> dict:
    if len(values) < 2:
        return {"cagr": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "final_value": values[-1] if values else 0.0}
    years = (end_date - start_date).days / 365.25
    daily_returns = [(values[i] / values[i - 1]) - 1 for i in range(1, len(values))]
    return {
        "cagr": _cagr(values[0], values[-1], years),
        "max_drawdown": _max_drawdown(values),
        "sharpe": _sharpe(daily_returns),
        "final_value": values[-1],
    }


def summarize(result: dict) -> dict:
    curve = result["equity_curve"]
    values = [p["value"] for p in curve]
    strategy = _curve_summary(values, result["start_date"], result["end_date"])
    strategy["num_trades"] = len([t for t in result["trade_log"] if t["action"] == "entry"])
    strategy["win_rate"] = _win_rate(result["trade_log"])
    strategy["avg_holding_days"] = _avg_holding_days(result["trade_log"])

    benchmarks = {}
    for track, ticker in BENCHMARK_TOTAL_RETURN.items():
        df = result["total_return_history"].get(ticker)
        if df is None:
            continue
        sliced = df.loc[(df.index.date >= result["start_date"]) & (df.index.date <= result["end_date"])]
        if sliced.empty:
            continue
        shares = result["initial_cash"] / sliced["Close"].iloc[0]
        bench_values = (shares * sliced["Close"]).tolist()
        benchmarks[f"{track} ({ticker}, total return)"] = _curve_summary(
            bench_values, result["start_date"], result["end_date"])

    return {"strategy": strategy, "benchmarks": benchmarks}


def print_report(summary: dict) -> None:
    s = summary["strategy"]
    print("\n=== Strategy ===")
    print(f"  CAGR: {s['cagr']*100:.2f}%  |  Max drawdown: {s['max_drawdown']*100:.2f}%  |  Sharpe: {s['sharpe']:.2f}")
    print(f"  Final value: {s['final_value']:.2f}  |  Trades: {s['num_trades']}  |  "
          f"Win rate: {s['win_rate']*100:.1f}%  |  Avg holding days: {s['avg_holding_days']:.1f}")
    for name, b in summary["benchmarks"].items():
        print(f"\n=== Benchmark: {name} ===")
        print(f"  CAGR: {b['cagr']*100:.2f}%  |  Max drawdown: {b['max_drawdown']*100:.2f}%  |  Sharpe: {b['sharpe']:.2f}")
        print(f"  Final value: {b['final_value']:.2f}")


def exit_reason_breakdown(trade_log: list) -> dict:
    breakdown = {}
    for t in trade_log:
        if t["action"] != "exit":
            continue
        b = breakdown.setdefault(t["reason"], {"count": 0, "wins": 0, "total_pnl": 0.0})
        b["count"] += 1
        b["total_pnl"] += t["pnl"]
        if t["pnl"] > 0:
            b["wins"] += 1
    for b in breakdown.values():
        b["win_rate"] = b["wins"] / b["count"] if b["count"] else 0.0
        b["avg_pnl"] = b["total_pnl"] / b["count"] if b["count"] else 0.0
    return breakdown


def capital_utilization(equity_curve: list) -> float:
    if not equity_curve:
        return 0.0
    fractions = [p["invested"] / p["value"] for p in equity_curve if p["value"] > 0]
    return sum(fractions) / len(fractions) if fractions else 0.0


def regime_blocked_fraction(data: dict, start_date, end_date) -> dict:
    result = {}
    for track, series in data["regime_series"].items():
        sliced = series[(series.index.date >= start_date) & (series.index.date <= end_date)]
        result[track] = 1 - (sliced.sum() / len(sliced)) if len(sliced) else 0.0
    return result


def total_fees_paid(trade_log: list) -> float:
    return sum(t.get("fees", 0.0) for t in trade_log)


def entry_cushion_by_exit_reason(trade_log: list) -> dict:
    """Average % distance price was above its 50-day SMA at entry,
    grouped by the reason the position eventually exited. Tests the
    hypothesis that trend_break's near-zero win rate comes from entries
    happening too close to a barely-rising SMA50 -- a thin cushion
    means ordinary noise can cross back below it almost immediately."""
    buckets = {}
    for t in trade_log:
        if t["action"] != "exit" or t.get("entry_cushion_pct") is None:
            continue
        buckets.setdefault(t["reason"], []).append(t["entry_cushion_pct"])
    return {reason: (sum(vals) / len(vals), len(vals)) for reason, vals in buckets.items()}


def print_diagnostics(result: dict, data: dict) -> None:
    print("\n=== Exit reason breakdown ===")
    breakdown = exit_reason_breakdown(result["trade_log"])
    for reason, b in sorted(breakdown.items(), key=lambda kv: -kv[1]["count"]):
        print(f"  {reason}: {b['count']} exits, win rate {b['win_rate']*100:.1f}%, avg P&L {b['avg_pnl']:.2f}")

    util = capital_utilization(result["equity_curve"])
    print(f"\nCapital utilization (avg % of portfolio invested): {util*100:.1f}%")

    blocked = regime_blocked_fraction(data, result["start_date"], result["end_date"])
    print("\nRegime filter: % of days new entries were paused")
    for track, frac in blocked.items():
        print(f"  {track}: {frac*100:.1f}%")

    print(f"\nTotal fees paid (commission + FX) over the period: {total_fees_paid(result['trade_log']):.2f}")

    print("\nAvg entry cushion above SMA50 at entry, by eventual exit reason:")
    cushions = entry_cushion_by_exit_reason(result["trade_log"])
    for reason, (avg, n) in sorted(cushions.items(), key=lambda kv: kv[1][0]):
        print(f"  {reason}: {avg*100:.2f}% (n={n})")


if __name__ == "__main__":
    end = datetime.date(2026, 11, 26)
    start = end - datetime.timedelta(days=365 * 15)
    backfill_days = (end - start).days + WARMUP_BUFFER_DAYS

    profile_config = dict(PROFILE_1_CONFIG)
    profile_config["time_stop_trading_days"] = 20
    profile_config["min_holding_days_for_trailing_stop"] = 5
    profile_config["trailing_stop_atr_multiplier"] = 2.0
    profile_config["trend_break_buffer_pct"] = -0.01  # new confirmed best -- sweep found this the standout vs 0.0 and more aggressive values

    data = load_backtest_data(end, backfill_days, profile_config=profile_config)

    result = replay(data, start, end, profile_config=profile_config, min_composite_score=0.0,
                     apply_costs=True, rotation_score_margin=1.5)
    summary = summarize(result)
    print_report(summary)
    print_diagnostics(result, data)