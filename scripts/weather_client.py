from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_LAT = 37.7749
DEFAULT_LON = -122.4194

DAILY_FIELDS = "temperature_2m_max,temperature_2m_min,precipitation_hours,wind_speed_10m_max"
SEASONAL_COLUMNS = ["temp_high", "temp_low", "precip_probability", "wind_speed"]
SEASONAL_YEARS_BACK = 5
SEASONAL_WINDOW_DAYS = 30
# morning/midday/evening precip buckets, as [start_hour, end_hour) local time
PRECIP_BUCKETS = {"precip_morning": (5, 9), "precip_midday": (9, 16), "precip_evening": (16, 21)}


def _daily_frame(daily: dict[str, Any]) -> pd.DataFrame:
    precip_hours = daily.get("precipitation_hours") or []
    frame = pd.DataFrame(
        {
            "date": daily.get("time") or [],
            "temp_high": daily.get("temperature_2m_max") or [],
            "temp_low": daily.get("temperature_2m_min") or [],
            "precip_probability": [h / 24 * 100 if h is not None else None for h in precip_hours],
            "wind_speed": daily.get("wind_speed_10m_max") or [],
        }
    )
    frame["date"] = pd.to_datetime(frame["date"]).dt.date
    return frame.set_index("date")


def _precip_bucket_frame(hourly: dict[str, Any]) -> pd.DataFrame:
    """Per-day fraction of hours with rain in the morning (5-9), midday (9-16), and evening (16-21) windows."""
    times = pd.to_datetime(hourly.get("time") or [])
    precip = pd.Series(hourly.get("precipitation") or [], index=times, dtype=float).fillna(0.0)
    wet = (precip > 0).astype(int)
    day = times.date

    columns = {}
    for name, (start_hr, end_hr) in PRECIP_BUCKETS.items():
        mask = (times.hour >= start_hr) & (times.hour < end_hr)
        columns[name] = wet[mask].groupby(day[mask]).mean() * 100
    frame = pd.DataFrame(columns)
    frame.index.name = "date"
    return frame


def _fetch_archive_daily(lat: float, lon: float, start_date: date, end_date: date) -> pd.DataFrame:
    """Raw daily archive weather (no seasonal columns), used both for real data and climatology baselines."""
    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": DAILY_FIELDS,
            "timezone": "auto",
        },
        timeout=30,
    )
    response.raise_for_status()
    return _daily_frame(response.json().get("daily") or {})


def _seasonal_baseline(climatology: pd.DataFrame, target_day: date, window_days: int = SEASONAL_WINDOW_DAYS) -> pd.Series:
    """Mean of each seasonal column across climatology rows within `window_days` of target_day's day-of-year."""
    if climatology.empty:
        return pd.Series({col: float("nan") for col in SEASONAL_COLUMNS})

    doy = pd.Series([d.timetuple().tm_yday for d in climatology.index], index=climatology.index)
    target_doy = target_day.timetuple().tm_yday
    circular_diff = (doy - target_doy).abs()
    circular_diff = np.minimum(circular_diff, 365 - circular_diff)
    window = climatology.loc[circular_diff <= window_days]
    if window.empty:
        window = climatology
    return window[SEASONAL_COLUMNS].mean()


def _apply_seasonal_anomalies(frame: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Add `<col>_vs_seasonal` columns: each value minus the historical average for that day-of-year."""
    if frame.empty:
        return frame

    earliest_day = min(frame.index)
    climatology_end = earliest_day - timedelta(days=1)
    climatology_start = date(climatology_end.year - SEASONAL_YEARS_BACK, climatology_end.month, climatology_end.day)
    climatology = _fetch_archive_daily(lat, lon, climatology_start, climatology_end)

    baselines = {day: _seasonal_baseline(climatology, day) for day in set(frame.index)}
    for col in SEASONAL_COLUMNS:
        frame[f"{col}_vs_seasonal"] = [frame.at[day, col] - baselines[day][col] for day in frame.index]
    return frame


def fetch_tomorrow_forecast() -> dict[str, Any]:
    """Fetch tomorrow's Open-Meteo forecast for configured/default coordinates.

    precip_probability is derived from precipitation_hours (predicted hours of rain / 24 * 100)
    to match the same "fraction of day raining" concept used for historical training rows.
    """
    lat = float(os.getenv("FORECAST_LAT", DEFAULT_LAT))
    lon = float(os.getenv("FORECAST_LON", DEFAULT_LON))
    tomorrow = date.today() + timedelta(days=1)

    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": DAILY_FIELDS,
            "hourly": "precipitation",
            "timezone": "auto",
            "forecast_days": 3,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    frame = _daily_frame(payload.get("daily") or {})
    if tomorrow not in frame.index:
        raise RuntimeError("Tomorrow forecast missing from Open-Meteo response")
    frame = _apply_seasonal_anomalies(frame, lat, lon)

    buckets = _precip_bucket_frame(payload.get("hourly") or {})
    bucket_row = buckets.loc[tomorrow] if tomorrow in buckets.index else pd.Series(
        {name: float("nan") for name in PRECIP_BUCKETS}
    )

    result = {
        "forecast_date": tomorrow.isoformat(),
        "latitude": lat,
        "longitude": lon,
        **frame.loc[tomorrow].to_dict(),
        **bucket_row.to_dict(),
    }
    return result


def fetch_historical_weather(start_date: date, end_date: date) -> pd.DataFrame:
    """Backfill daily weather for a date range in a single call (used for training features).

    The archive API has no forecast-style precipitation probability, so precip_probability
    here is approximated as the fraction of the day with recorded rain (precipitation_hours / 24).
    Also adds `<col>_vs_seasonal` anomaly columns and morning/midday/evening precip fraction columns.
    """
    lat = float(os.getenv("FORECAST_LAT", DEFAULT_LAT))
    lon = float(os.getenv("FORECAST_LON", DEFAULT_LON))

    response = requests.get(
        OPEN_METEO_ARCHIVE_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "daily": DAILY_FIELDS,
            "hourly": "precipitation",
            "timezone": "auto",
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    frame = _daily_frame(payload.get("daily") or {})
    frame = _apply_seasonal_anomalies(frame, lat, lon)
    frame = frame.join(_precip_bucket_frame(payload.get("hourly") or {}))
    return frame

