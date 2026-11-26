"""
core/trend_break_sweep.py

Full re-validation sweep. Every parameter conclusion drawn earlier
today was reached on a drifting, non-reproducible price cache (see
core.price_cache's use_cache_only flag) and needs re-testing under
frozen, bit-identical conditions. This sweeps, one lever at a time
(holding everything else at the current best-known config), every
parameter tested today:

  - trailing_stop_atr_multiplier       (exit-side -- shares one data load)
  - min_holding_days_for_trailing_stop (exit-side)
  - rotation_score_margin              (replay()-level kwarg, not profile_config)
  - time_stop_trading_days             (exit-side)
  - trend_break_buffer_pct             (exit-side)
  - min_entry_cushion_pct              (ENTRY-side -- affects add_entry_signal,
                                         precomputed once in load_backtest_data,
                                         so this needs its own separate load)
"""

import datetime

from core.backtester import load_backtest_data, replay, summarize, exit_reason_breakdown, WARMUP_BUFFER_DAYS
from profiles.profile_1_trend_momentum import PROFILE_1_CONFIG

END = datetime.date(2026, 11, 26)
START = END - datetime.timedelta(days=365 * 15)
BACKFILL_DAYS = (END - START).days + WARMUP_BUFFER_DAYS

# Current best-known config -- the fixed backdrop for every single-lever sweep below.
BASE_PROFILE_CONFIG = dict(PROFILE_1_CONFIG)
BASE_PROFILE_CONFIG["time_stop_trading_days"] = 20
BASE_PROFILE_CONFIG["min_holding_days_for_trailing_stop"] = 5
BASE_PROFILE_CONFIG["trailing_stop_atr_multiplier"] = 2.0
BASE_PROFILE_CONFIG["trend_break_buffer_pct"] = -0.01
BASE_ROTATION_MARGIN = 1.5


def _run(data, profile_config, rotation_score_margin):
    result = replay(data, START, END, profile_config=profile_config, min_composite_score=0.0,
                     apply_costs=True, rotation_score_margin=rotation_score_margin)
    summary = summarize(result)
    s = summary["strategy"]
    breakdown = exit_reason_breakdown(result["trade_log"])
    tb = breakdown.get("trend_break", {"count": 0, "win_rate": 0.0, "avg_pnl": 0.0})
    return s, tb


def _header():
    print(f"\n{'variant':>28} | {'CAGR':>7} | {'Sharpe':>6} | {'MaxDD':>8} | "
          f"{'trades':>6} | {'tb_n':>5} | {'tb_win%':>7} | {'tb_avgP&L':>10}")
    print("-" * 100)


def _print_row(label, s, tb):
    print(f"{label:>28} | {s['cagr']*100:>6.2f}% | {s['sharpe']:>6.2f} | "
          f"{s['max_drawdown']*100:>7.2f}% | {s['num_trades']:>6} | "
          f"{tb['count']:>5} | {tb['win_rate']*100:>6.1f}% | {tb['avg_pnl']:>10.2f}")


def sweep_exit_side_params(data):
    """Everything that only affects check_exit() -- shares one data load."""

    print("\n=== trailing_stop_atr_multiplier ===")
    _header()
    for value in [1.5, 2.0, 2.5]:
        profile_config = dict(BASE_PROFILE_CONFIG)
        profile_config["trailing_stop_atr_multiplier"] = value
        s, tb = _run(data, profile_config, BASE_ROTATION_MARGIN)
        _print_row(f"atr={value}", s, tb)

    print("\n=== min_holding_days_for_trailing_stop ===")
    _header()
    for value in [0, 5]:
        profile_config = dict(BASE_PROFILE_CONFIG)
        profile_config["min_holding_days_for_trailing_stop"] = value
        s, tb = _run(data, profile_config, BASE_ROTATION_MARGIN)
        _print_row(f"min_hold_days={value}", s, tb)

    print("\n=== rotation_score_margin ===")
    _header()
    for value in [1.0, 1.5]:
        profile_config = dict(BASE_PROFILE_CONFIG)
        s, tb = _run(data, profile_config, value)
        _print_row(f"rotation_margin={value}", s, tb)

    print("\n=== time_stop_trading_days ===")
    _header()
    for value in [20, 30]:
        profile_config = dict(BASE_PROFILE_CONFIG)
        profile_config["time_stop_trading_days"] = value
        s, tb = _run(data, profile_config, BASE_ROTATION_MARGIN)
        _print_row(f"time_stop_days={value}", s, tb)

    print("\n=== trend_break_buffer_pct ===")
    _header()
    for value in [0.02, 0.0, -0.005, -0.01, -0.015, -0.02, -0.025, -0.03]:
        profile_config = dict(BASE_PROFILE_CONFIG)
        profile_config["trend_break_buffer_pct"] = value
        s, tb = _run(data, profile_config, BASE_ROTATION_MARGIN)
        _print_row(f"tb_buffer={value}", s, tb)


def sweep_entry_side_params():
    """min_entry_cushion_pct affects add_entry_signal(), which is
    precomputed once inside load_backtest_data() -- needs its own
    separate data load per value, can't share the main one."""
    print("\n=== min_entry_cushion_pct (separate data load per value -- affects entry signal precompute) ===")
    _header()
    for value in [0.0, 0.04, 0.08]:
        profile_config = dict(BASE_PROFILE_CONFIG)
        profile_config["min_entry_cushion_pct"] = value
        data = load_backtest_data(END, BACKFILL_DAYS, profile_config=profile_config)
        s, tb = _run(data, profile_config, BASE_ROTATION_MARGIN)
        _print_row(f"min_cushion={value}", s, tb)


if __name__ == "__main__":
    print("Loading backtest data once for all exit-side sweeps (frozen cache, use_cache_only=True)...")
    data = load_backtest_data(END, BACKFILL_DAYS, profile_config=BASE_PROFILE_CONFIG)

    sweep_exit_side_params(data)
    sweep_entry_side_params()