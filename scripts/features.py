from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "acute_load_7",
    "chronic_load_28",
    "atl_ctl_ratio",
    "days_since_last_hard",
    "streak_length",
    "trained_today",
    "trained_hard_today",
    "trained_days_2",
    "trained_both_days_2",
    "trained_days_7",
    "trained_days_30",
    "moving_time_acute_7",
    "moving_time_chronic_28",
    "today_relative_effort",
    "dow_train_rate",
    "month_train_rate",
    "trained_last_weekend",
    "day_of_week",
    "month",
    "season",
    "days_since_last_long_ride",
    "forecast_temp_high",
    "forecast_temp_low",
    "forecast_precip_probability",
    "forecast_rain_expected",
    "forecast_wind_speed",
    "forecast_temp_high_vs_seasonal",
    "forecast_temp_low_vs_seasonal",
    "forecast_precip_probability_vs_seasonal",
    "forecast_wind_speed_vs_seasonal",
    "forecast_precip_morning",
    "forecast_precip_midday",
    "forecast_precip_evening",
]


@dataclass
class PreparedData:
    historical: pd.DataFrame
    tomorrow_features: pd.DataFrame
    hard_effort_threshold: float


def _season_from_month(month: int) -> int:
    # 0=winter, 1=spring, 2=summer, 3=fall (ordinal keeps feature count low for small dataset)
    if month in (12, 1, 2):
        return 0
    if month in (3, 4, 5):
        return 1
    if month in (6, 7, 8):
        return 2
    return 3


def _daily_activity_frame(activities: pd.DataFrame, as_of_day: date | None = None) -> pd.DataFrame:
    """Build a full daily frame through as_of_day (default: yesterday), not just the last logged activity.

    Without this, resting for a day or two with nothing logged in Strava would silently shrink the
    frame to end at the last active day, making "tomorrow" predictions land on an already-past date.
    """
    if activities.empty:
        raise ValueError("No activities returned from Strava for the configured lookback window")

    df = activities.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    df = df.dropna(subset=["date"]).copy()
    if df.empty:
        raise ValueError("Activity rows were present but no valid dates were parsed")

    for col in ("moving_time", "distance", "relative_effort"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    daily = (
        df.groupby("date", as_index=True)
        .agg(
            relative_effort=("relative_effort", "sum"),
            moving_time=("moving_time", "sum"),
            distance=("distance", "sum"),
        )
        .sort_index()
    )

    end_day = max(daily.index.max(), as_of_day) if as_of_day is not None else daily.index.max()
    full_index = pd.date_range(start=daily.index.min(), end=end_day, freq="D").date
    daily = daily.reindex(full_index, fill_value=0.0)
    daily.index.name = "date"
    daily["trained_today"] = (daily["relative_effort"] > 0).astype(int)
    return daily


def _same_weekday_rate(state: pd.DataFrame) -> pd.Series:
    """Causal training rate for each date's weekday, using only strictly earlier same-weekday dates."""
    weekday = pd.Series([d.weekday() for d in state.index], index=state.index)
    rate = pd.Series(index=state.index, dtype=float)
    for _, idx in weekday.groupby(weekday).groups.items():
        rate.loc[idx] = state.loc[idx, "trained_today"].shift(1).expanding().mean()
    return rate.fillna(0.5)


def _same_month_rate(state: pd.DataFrame) -> pd.Series:
    """Causal training rate for each date's month (across years), capturing seasonal training windows."""
    month = pd.Series([d.month for d in state.index], index=state.index)
    rate = pd.Series(index=state.index, dtype=float)
    for _, idx in month.groupby(month).groups.items():
        rate.loc[idx] = state.loc[idx, "trained_today"].shift(1).expanding().mean()
    return rate.fillna(0.5)

def _trained_last_weekend(state: pd.DataFrame) -> pd.Series:
    """Causal indicator for whether the athlete trained on the most recent weekend."""
    idx = pd.to_datetime(state.index)
    week_period = idx.to_period("W-SUN")  # groups each Mon–Sun into one period
    is_weekend = idx.weekday >= 5
    trained_on_weekend = state["trained_today"].astype(bool) & is_weekend

    # did training happen on Sat or Sun within each week?
    weekly_flag = pd.Series(trained_on_weekend.values, index=week_period).groupby(level=0).max()

    # look at the PRIOR week's weekend, not the current (possibly incomplete) one
    weekly_flag_prev = weekly_flag.shift(1)

    return  week_period.map(weekly_flag_prev).fillna(False).astype(bool)

def _compute_state_features(daily: pd.DataFrame, hard_threshold: float, long_threshold: float) -> pd.DataFrame:
    state = daily.copy()
    state["acute_load_7"] = state["relative_effort"].ewm(span=7, adjust=False).mean()
    state["chronic_load_28"] = state["relative_effort"].ewm(span=28, adjust=False).mean()
    state["atl_ctl_ratio"] = state["acute_load_7"] / state["chronic_load_28"].replace(0, np.nan)
    state["atl_ctl_ratio"] = state["atl_ctl_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    state["trained_days_2"] = state["trained_today"].rolling(2, min_periods=1).sum()
    state["trained_both_days_2"] = (state["trained_days_2"] == 2).astype(int)
    state["trained_hard_today"] = (state["relative_effort"] >= hard_threshold).astype(int)
    state["trained_days_7"] = state["trained_today"].rolling(7, min_periods=1).sum()
    state["trained_days_30"] = state["trained_today"].rolling(30, min_periods=1).sum()
    state["moving_time_acute_7"] = state["moving_time"].ewm(span=7, adjust=False).mean()
    state["moving_time_chronic_28"] = state["moving_time"].ewm(span=28, adjust=False).mean()
    state["same_weekday_rate"] = _same_weekday_rate(state)
    state["same_month_rate"] = _same_month_rate(state)
    state["trained_last_weekend"] = _trained_last_weekend(state)

    last_hard: Any = None
    last_long: Any = None
    hard_days: list[int] = []
    long_days: list[int] = []
    streak: list[int] = []
    running_streak = 0

    for idx, (day, row) in enumerate(state.iterrows()):
        if row["relative_effort"] >= hard_threshold:
            last_hard = day
        hard_days.append((day - last_hard).days if last_hard is not None else 999)

        if row["distance"] >= long_threshold and row["distance"] > 0:
            last_long = day
        long_days.append((day - last_long).days if last_long is not None else 999)

        if row["trained_today"] == 1:
            running_streak += 1
        else:
            running_streak = 0
        streak.append(running_streak)

    state["days_since_last_hard"] = hard_days
    state["days_since_last_long_ride"] = long_days
    state["streak_length"] = streak
    return state


def prepare_datasets(
    activities: pd.DataFrame,
    tomorrow_weather: dict[str, Any],
    historical_weather: pd.DataFrame | None = None,
    run_date: date | None = None,
    hard_effort_quantile: float = 0.6,
    long_ride_quantile: float = 0.75,
) -> PreparedData:
    """Build leakage-safe historical labels and tomorrow's feature row.

    historical_weather, when provided, is a DataFrame indexed by date with columns
    temp_high/temp_low/precip_probability/wind_speed (see weather_client.fetch_historical_weather).
    Dates missing from it fall back to NaN.

    run_date defaults to today. The daily frame is built through yesterday (run_date - 1 day) so a
    recent rest day with nothing logged in Strava doesn't shrink the frame and land "tomorrow"'s
    prediction on an already-past date.
    """

    yesterday = (run_date or date.today()) - timedelta(days=1)
    daily = _daily_activity_frame(activities, as_of_day=yesterday)
    nonzero_effort = daily.loc[daily["relative_effort"] > 0, "relative_effort"]
    hard_threshold = float(nonzero_effort.quantile(hard_effort_quantile)) if not nonzero_effort.empty else 0.0
    nonzero_distance = daily.loc[daily["distance"] > 0, "distance"]
    long_threshold = float(nonzero_distance.quantile(long_ride_quantile)) if not nonzero_distance.empty else 0.0
    state = _compute_state_features(daily, hard_threshold, long_threshold)

    rows: list[dict[str, Any]] = []
    all_days = list(state.index)
    for pos in range(len(all_days) - 1):
        as_of_day = all_days[pos]
        next_day = all_days[pos + 1]
        next_row = state.loc[next_day]
        today_row = state.loc[as_of_day]

        if historical_weather is not None and next_day in historical_weather.index:
            weather_row = historical_weather.loc[next_day]
            forecast_temp_high = float(weather_row["temp_high"])
            forecast_temp_low = float(weather_row["temp_low"])
            forecast_precip_probability = float(weather_row["precip_probability"])
            forecast_wind_speed = float(weather_row["wind_speed"])
            forecast_temp_high_vs_seasonal = float(weather_row["temp_high_vs_seasonal"])
            forecast_temp_low_vs_seasonal = float(weather_row["temp_low_vs_seasonal"])
            forecast_precip_probability_vs_seasonal = float(weather_row["precip_probability_vs_seasonal"])
            forecast_wind_speed_vs_seasonal = float(weather_row["wind_speed_vs_seasonal"])
            forecast_precip_morning = float(weather_row["precip_morning"])
            forecast_precip_midday = float(weather_row["precip_midday"])
            forecast_precip_evening = float(weather_row["precip_evening"])
        else:
            forecast_temp_high = forecast_temp_low = np.nan
            forecast_precip_probability = forecast_wind_speed = np.nan
            forecast_temp_high_vs_seasonal = forecast_temp_low_vs_seasonal = np.nan
            forecast_precip_probability_vs_seasonal = forecast_wind_speed_vs_seasonal = np.nan
            forecast_precip_morning = forecast_precip_midday = forecast_precip_evening = np.nan

        rows.append(
            {
                "as_of_date": as_of_day.isoformat(),
                "target_date": next_day.isoformat(),
                "acute_load_7": float(today_row["acute_load_7"]),
                "chronic_load_28": float(today_row["chronic_load_28"]),
                "atl_ctl_ratio": float(today_row["atl_ctl_ratio"]),
                "days_since_last_hard": int(today_row["days_since_last_hard"]),
                "streak_length": int(today_row["streak_length"]),
                "trained_today": int(today_row["trained_today"]),
                "trained_hard_today": int(today_row["trained_hard_today"]),
                "trained_days_2": float(today_row["trained_days_2"]),
                "trained_both_days_2": int(today_row["trained_both_days_2"]),
                "trained_days_7": float(today_row["trained_days_7"]),
                "trained_days_30": float(today_row["trained_days_30"]),
                "moving_time_acute_7": float(today_row["moving_time_acute_7"]),
                "moving_time_chronic_28": float(today_row["moving_time_chronic_28"]),
                "today_relative_effort": float(today_row["relative_effort"]),
                "dow_train_rate": float(state.loc[next_day, "same_weekday_rate"]),
                "month_train_rate": float(state.loc[next_day, "same_month_rate"]),
                "day_of_week": next_day.weekday(),
                "trained_last_weekend": bool(today_row["trained_last_weekend"]),
                "month": next_day.month,
                "season": _season_from_month(next_day.month),
                "days_since_last_long_ride": int(today_row["days_since_last_long_ride"]),
                "forecast_temp_high": forecast_temp_high,
                "forecast_temp_low": forecast_temp_low,
                "forecast_precip_probability": forecast_precip_probability,
                "forecast_rain_expected": int(forecast_precip_probability > 0) if not np.isnan(forecast_precip_probability) else 0,
                "forecast_wind_speed": forecast_wind_speed,
                "forecast_temp_high_vs_seasonal": forecast_temp_high_vs_seasonal,
                "forecast_temp_low_vs_seasonal": forecast_temp_low_vs_seasonal,
                "forecast_precip_probability_vs_seasonal": forecast_precip_probability_vs_seasonal,
                "forecast_wind_speed_vs_seasonal": forecast_wind_speed_vs_seasonal,
                "forecast_precip_morning": forecast_precip_morning,
                "forecast_precip_midday": forecast_precip_midday,
                "forecast_precip_evening": forecast_precip_evening,
                "will_train_tomorrow": int(next_row["trained_today"]),
                "next_day_relative_effort": float(next_row["relative_effort"]),
            }
        )


    historical = pd.DataFrame(rows)
    if historical.empty:
        raise ValueError("Not enough daily data to build labeled history")

    most_recent_day = all_days[-1]
    target_day = most_recent_day + timedelta(days=1)
    recent = state.loc[most_recent_day]
    target_dow_mask = [d.weekday() == target_day.weekday() for d in state.index]
    target_dow_rate = float(state.loc[target_dow_mask, "trained_today"].mean()) if any(target_dow_mask) else 0.5
    target_month_mask = [d.month == target_day.month for d in state.index]
    target_month_rate = float(state.loc[target_month_mask, "trained_today"].mean()) if any(target_month_mask) else 0.5
    tomorrow_features = pd.DataFrame(
        [
            {
                "as_of_date": most_recent_day.isoformat(),
                "target_date": target_day.isoformat(),
                "acute_load_7": float(recent["acute_load_7"]),
                "chronic_load_28": float(recent["chronic_load_28"]),
                "atl_ctl_ratio": float(recent["atl_ctl_ratio"]),
                "days_since_last_hard": int(recent["days_since_last_hard"]),
                "streak_length": int(recent["streak_length"]),
                "trained_today": int(recent["trained_today"]),
                "trained_hard_today": int(recent["trained_hard_today"]),
                "trained_days_2": float(recent["trained_days_2"]),
                "trained_both_days_2": int(recent["trained_both_days_2"]),
                "trained_days_7": float(recent["trained_days_7"]),
                "trained_days_30": float(recent["trained_days_30"]),
                "moving_time_acute_7": float(recent["moving_time_acute_7"]),
                "moving_time_chronic_28": float(recent["moving_time_chronic_28"]),
                "today_relative_effort": float(recent["relative_effort"]),
                "dow_train_rate": target_dow_rate,
                "month_train_rate": target_month_rate,
                "day_of_week": target_day.weekday(),
                "trained_last_weekend": bool(recent["trained_last_weekend"]),
                "month": target_day.month,
                "season": _season_from_month(target_day.month),
                "days_since_last_long_ride": int(recent["days_since_last_long_ride"]),
                "forecast_temp_high": float(tomorrow_weather["temp_high"]),
                "forecast_temp_low": float(tomorrow_weather["temp_low"]),
                "forecast_precip_probability": float(tomorrow_weather["precip_probability"]),
                "forecast_rain_expected": int(float(tomorrow_weather["precip_probability"]) > 0),
                "forecast_wind_speed": float(tomorrow_weather["wind_speed"]),
                "forecast_temp_high_vs_seasonal": float(tomorrow_weather["temp_high_vs_seasonal"]),
                "forecast_temp_low_vs_seasonal": float(tomorrow_weather["temp_low_vs_seasonal"]),
                "forecast_precip_probability_vs_seasonal": float(tomorrow_weather["precip_probability_vs_seasonal"]),
                "forecast_wind_speed_vs_seasonal": float(tomorrow_weather["wind_speed_vs_seasonal"]),
                "forecast_precip_morning": float(tomorrow_weather["precip_morning"]),
                "forecast_precip_midday": float(tomorrow_weather["precip_midday"]),
                "forecast_precip_evening": float(tomorrow_weather["precip_evening"]),
            }
        ]
    )

    return PreparedData(historical=historical, tomorrow_features=tomorrow_features, hard_effort_threshold=hard_threshold)
