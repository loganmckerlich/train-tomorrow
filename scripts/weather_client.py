from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import pandas as pd
import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DEFAULT_LAT = 37.7749
DEFAULT_LON = -122.4194


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
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_hours,wind_speed_10m_max",
            "timezone": "auto",
            "forecast_days": 3,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    daily = payload.get("daily") or {}
    dates = daily.get("time") or []
    try:
        idx = dates.index(tomorrow.isoformat())
    except ValueError as exc:
        raise RuntimeError("Tomorrow forecast missing from Open-Meteo response") from exc

    precip_hours = (daily.get("precipitation_hours") or [None])[idx]

    return {
        "forecast_date": tomorrow.isoformat(),
        "latitude": lat,
        "longitude": lon,
        "temp_high": float((daily.get("temperature_2m_max") or [None])[idx]),
        "temp_low": float((daily.get("temperature_2m_min") or [None])[idx]),
        "precip_probability": float(precip_hours) / 24 * 100 if precip_hours is not None else float("nan"),
        "wind_speed": float((daily.get("wind_speed_10m_max") or [None])[idx]),
    }


def fetch_historical_weather(start_date: date, end_date: date) -> pd.DataFrame:
    """Backfill daily weather for a date range in a single call (used for training features).

    The archive API has no forecast-style precipitation probability, so precip_probability
    here is approximated as the fraction of the day with recorded rain (precipitation_hours / 24).
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
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_hours,wind_speed_10m_max",
            "timezone": "auto",
        },
        timeout=30,
    )
    response.raise_for_status()
    daily = response.json().get("daily") or {}

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

