from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "acute_load_7",
    "chronic_load_28",
    "atl_ctl_ratio",
    "days_since_last_hard",
    "streak_length",
    "day_of_week",
    "month",
    "season",
    "forecast_temp_high",
    "forecast_temp_low",
    "forecast_precip_probability",
    "forecast_wind_speed",
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


def _daily_activity_frame(activities: pd.DataFrame) -> pd.DataFrame:
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

    full_index = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="D").date
    daily = daily.reindex(full_index, fill_value=0.0)
    daily.index.name = "date"
    daily["trained_today"] = (daily["relative_effort"] > 0).astype(int)
    return daily


def _compute_state_features(daily: pd.DataFrame, hard_threshold: float) -> pd.DataFrame:
    state = daily.copy()
    state["acute_load_7"] = state["relative_effort"].ewm(span=7, adjust=False).mean()
    state["chronic_load_28"] = state["relative_effort"].ewm(span=28, adjust=False).mean()
    state["atl_ctl_ratio"] = state["acute_load_7"] / state["chronic_load_28"].replace(0, np.nan)
    state["atl_ctl_ratio"] = state["atl_ctl_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    last_hard: Any = None
    hard_days: list[int] = []
    streak: list[int] = []
    running_streak = 0

    for idx, (day, row) in enumerate(state.iterrows()):
        if row["relative_effort"] >= hard_threshold:
            last_hard = day
        hard_days.append((day - last_hard).days if last_hard is not None else 999)

        if row["trained_today"] == 1:
            running_streak += 1
        else:
            running_streak = 0
        streak.append(running_streak)

    state["days_since_last_hard"] = hard_days
    state["streak_length"] = streak
    return state


def prepare_datasets(activities: pd.DataFrame, tomorrow_weather: dict[str, Any]) -> PreparedData:
    """Build leakage-safe historical labels and tomorrow's feature row.

    Historical rows intentionally leave forecast-weather fields as NaN because this pipeline
    does not yet persist a daily archived forecast feed. Using observed weather hindsight here
    would leak information unavailable at prediction time.
    """

    daily = _daily_activity_frame(activities)
    nonzero_effort = daily.loc[daily["relative_effort"] > 0, "relative_effort"]
    hard_threshold = float(nonzero_effort.quantile(0.75)) if not nonzero_effort.empty else 0.0
    state = _compute_state_features(daily, hard_threshold)

    rows: list[dict[str, Any]] = []
    all_days = list(state.index)
    for pos in range(len(all_days) - 1):
        as_of_day = all_days[pos]
        next_day = all_days[pos + 1]
        next_row = state.loc[next_day]
        today_row = state.loc[as_of_day]

        rows.append(
            {
                "as_of_date": as_of_day.isoformat(),
                "target_date": next_day.isoformat(),
                "acute_load_7": float(today_row["acute_load_7"]),
                "chronic_load_28": float(today_row["chronic_load_28"]),
                "atl_ctl_ratio": float(today_row["atl_ctl_ratio"]),
                "days_since_last_hard": int(today_row["days_since_last_hard"]),
                "streak_length": int(today_row["streak_length"]),
                "day_of_week": next_day.weekday(),
                "month": next_day.month,
                "season": _season_from_month(next_day.month),
                "forecast_temp_high": np.nan,
                "forecast_temp_low": np.nan,
                "forecast_precip_probability": np.nan,
                "forecast_wind_speed": np.nan,
                "will_train_tomorrow": int(next_row["trained_today"]),
                "next_day_relative_effort": (
                    float(next_row["relative_effort"]) if next_row["trained_today"] == 1 else np.nan
                ),
            }
        )

    historical = pd.DataFrame(rows)
    if historical.empty:
        raise ValueError("Not enough daily data to build labeled history")

    most_recent_day = all_days[-1]
    target_day = most_recent_day + timedelta(days=1)
    recent = state.loc[most_recent_day]
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
                "day_of_week": target_day.weekday(),
                "month": target_day.month,
                "season": _season_from_month(target_day.month),
                "forecast_temp_high": float(tomorrow_weather["temp_high"]),
                "forecast_temp_low": float(tomorrow_weather["temp_low"]),
                "forecast_precip_probability": float(tomorrow_weather["precip_probability"]),
                "forecast_wind_speed": float(tomorrow_weather["wind_speed"]),
            }
        ]
    )

    return PreparedData(historical=historical, tomorrow_features=tomorrow_features, hard_effort_threshold=hard_threshold)
