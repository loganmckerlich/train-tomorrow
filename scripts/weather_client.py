from __future__ import annotations

import os
from datetime import date, timedelta
from typing import Any

import requests

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_LAT = 37.7749
DEFAULT_LON = -122.4194


def fetch_tomorrow_forecast() -> dict[str, Any]:
    """Fetch tomorrow's Open-Meteo forecast for configured/default coordinates."""
    lat = float(os.getenv("FORECAST_LAT", DEFAULT_LAT))
    lon = float(os.getenv("FORECAST_LON", DEFAULT_LON))
    tomorrow = date.today() + timedelta(days=1)

    response = requests.get(
        OPEN_METEO_URL,
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,wind_speed_10m_max",
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

    return {
        "forecast_date": tomorrow.isoformat(),
        "latitude": lat,
        "longitude": lon,
        "temp_high": float((daily.get("temperature_2m_max") or [None])[idx]),
        "temp_low": float((daily.get("temperature_2m_min") or [None])[idx]),
        "precip_probability": float((daily.get("precipitation_probability_max") or [None])[idx]),
        "wind_speed": float((daily.get("wind_speed_10m_max") or [None])[idx]),
    }
