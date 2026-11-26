"""
core/parameter_sweep.py

Walk-forward validation of every parameter this project has tuned,
either as a starting assumption (time_stop_trading_days,
min_holding_days_for_rotation, rotation_score_margin, min_adx,
cash_reserve_pct) or as a same-session discovery validated only
in-sample against a single 15-year backtest window
(trailing_stop_atr_multiplier, min_holding_days_for_trailing_stop,
trend_break_buffer_pct).

Method: rolling 3-year train / 1-year test windows, sliding forward 1
year at a time across a 15-year span. Within each window, each
parameter is swept independently (the other seven held at their
current best-known default) across a candidate grid; the value with
the best Sharpe ratio on the TRAIN period is picked, then applied on
the following TEST period, which that value was never tuned against.
That's what keeps this honest rather than overfit to the data it's
validated on.

Runs against the frozen price cache (use_cache_only=True, baked into
core.backtester.load_backtest_data) and a pinned end date, so results
are reproducible across any number of runs.
"""

import datetime

from core.backtester import load_backtest_data, replay, summarize
from core.daily_runner import MAX_POSITIONS, CASH_RESERVE_PCT, MIN_ADX, MIN_HOLDING_DAYS_FOR_ROTATION
from profiles.profile_1_trend_momentum import PROFILE_1_CONFIG

END_DATE = datetime.date(2026, 11, 26)  # pinned -- same reference date as backtester.py's confirmed baseline
TOTAL_YEARS = 15
TRAIN_YEARS = 3
TEST_YEARS = 1
STEP_YEARS = 1

# Today's confirmed-best config -- the "held constant" background for
# every single-parameter sweep below, not the stale original defaults.
BASE_PROFILE_CONFIG = dict(PROFILE_1_CONFIG)
BASE_PROFILE_CONFIG["time_stop_trading_days"] = 20
BASE_PROFILE_CONFIG["min_holding_days_for_trailing_stop"] = 5
BASE_PROFILE_CONFIG["trailing_stop_atr_multiplier"] = 2.0
BASE_PROFILE_CONFIG["trend_break_buffer_pct"] = -0.01

BASE_REPLAY_OVERRIDES = {
    "max_positions": MAX_POSITIONS,
    "cash_reserve_pct": CASH_RESERVE_PCT,
    "min_adx": MIN_ADX,
    "rotation_score_margin": 1.5,  # confirmed-best this session, was 1.0
    "min_holding_days_for_rotation": MIN_HOLDING_DAYS_FOR_ROTATION,
}

# Which bucket each parameter's value gets written into.
PROFILE_CONFIG_PARAMS = {
    "time_stop_trading_days", "trailing_stop_atr_multiplier",
    "min_holding_days_for_trailing_stop", "trend_break_buffer_pct",
}

PARAM_GRIDS = {
    "time_stop_trading_days": [20, 25, 30, 35, 40],
    "min_holding_days_for_rotation": [7, 10, 14, 18, 21],
    "rotation_score_margin": [1.0, 1.25, 1.5, 1.75, 2.0],
    "min_adx": [15, 18, 20, 22, 25],
    "cash_reserve_pct": [0.05, 0.075, 0.10, 0.125, 0.15],
    "trailing_stop_atr_multiplier": [1.5, 1.75, 2.0, 2.25, 2.5],
    "min_holding_days_for_trailing_stop": [0, 3, 5, 7, 10],
    "trend_break_buffer_pct": [0.0, -0.005, -0.01, -0.015, -0.02],
}


def _run_with_override(data, start_date, end_date, param_name, value):
    overrides = dict(BASE_REPLAY_OVERRIDES)
    overrides["profile_config"] = dict(BASE_PROFILE_CONFIG)

    if param_name in PROFILE_CONFIG_PARAMS:
        overrides["profile_config"][param_name] = value
    else:
        overrides[param_name] = value

    result = replay(data, start_date, end_date, **overrides)
    return summarize(result)["strategy"]


def _segments(end_date):
    segments = []
    train_days = 365 * TRAIN_YEARS
    test_days = 365 * TEST_YEARS
    step_days = 365 * STEP_YEARS
    total_days = 365 * TOTAL_YEARS

    cursor = end_date - datetime.timedelta(days=total_days)
    while True:
        train_start = cursor
        train_end = train_start + datetime.timedelta(days=train_days)
        test_start = train_end
        test_end = test_start + datetime.timedelta(days=test_days)
        if test_end > end_date:
            break
        segments.append((train_start, train_end, test_start, test_end))
        cursor += datetime.timedelta(days=step_days)
    return segments


def run_sweep():
    backfill_days = 365 * TOTAL_YEARS + 400
    print(f"Loading {TOTAL_YEARS} years of data once for the full sweep "
          f"(frozen cache -- no network calls)...")
    data = load_backtest_data(END_DATE, backfill_days, profile_config=BASE_PROFILE_CONFIG)

    segments = _segments(END_DATE)
    print(f"\n{len(segments)} walk-forward segment(s): "
          f"{TRAIN_YEARS}y train / {TEST_YEARS}y test, stepping {STEP_YEARS}y at a time\n")

    results = {param: [] for param in PARAM_GRIDS}

    for i, (train_start, train_end, test_start, test_end) in enumerate(segments, 1):
        print(f"Segment {i}/{len(segments)}: train {train_start}..{train_end}, test {test_start}..{test_end}")
        for param_name, grid in PARAM_GRIDS.items():
            train_scores = {value: _run_with_override(data, train_start, train_end, param_name, value)["sharpe"]
                             for value in grid}
            best_value = max(train_scores, key=train_scores.get)

            test_summary = _run_with_override(data, test_start, test_end, param_name, best_value)
            results[param_name].append({
                "segment": i, "chosen_value": best_value,
                "train_sharpe": train_scores[best_value],
                "test_cagr": test_summary["cagr"],
                "test_sharpe": test_summary["sharpe"],
                "test_max_drawdown": test_summary["max_drawdown"],
            })
            print(f"  {param_name}: chose {best_value} on train (Sharpe {train_scores[best_value]:.2f}) "
                  f"-> OOS Sharpe {test_summary['sharpe']:.2f}, CAGR {test_summary['cagr']*100:.2f}%")

    return results


def print_sweep_report(results: dict) -> None:
    print("\n=== Walk-forward sweep summary ===")
    for param_name, segment_results in results.items():
        chosen_values = [r["chosen_value"] for r in segment_results]
        avg_oos_sharpe = sum(r["test_sharpe"] for r in segment_results) / len(segment_results)
        avg_oos_cagr = sum(r["test_cagr"] for r in segment_results) / len(segment_results)
        print(f"\n{param_name}:")
        print(f"  Values chosen across segments: {chosen_values}")
        print(f"  Average OOS Sharpe: {avg_oos_sharpe:.2f}  |  Average OOS CAGR: {avg_oos_cagr*100:.2f}%")


if __name__ == "__main__":
    results = run_sweep()
    print_sweep_report(results)