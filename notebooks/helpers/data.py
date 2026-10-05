import os

import pandas as pd
import yaml

from . import ROOT

HISTORICAL_CSV = ROOT / "notebooks" / "local_data" / "historical.csv"


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def load_historical(use_live: bool = False, days_back: int = 365) -> pd.DataFrame:
    """Cached CSV by default; use_live pulls Strava + Open-Meteo (needs .env secrets) and refreshes the cache."""
    if not use_live:
        return pd.read_csv(HISTORICAL_CSV)

    from features import prepare_datasets
    from strava_client import fetch_activities_dataframe, out_of_range_dates
    from weather_client import DEFAULT_LAT, DEFAULT_LON, fetch_historical_weather, fetch_tomorrow_forecast

    activities = fetch_activities_dataframe(
        days_back=days_back, activity_types=load_params().get("strava", {}).get("activity_types")
    )
    weather = fetch_tomorrow_forecast()
    activity_dates = pd.to_datetime(activities["date"], errors="coerce").dt.date.dropna()
    historical_weather = (
        fetch_historical_weather(activity_dates.min(), activity_dates.max()) if not activity_dates.empty else None
    )
    if historical_weather is not None:
        home_lat = float(os.getenv("FORECAST_LAT", DEFAULT_LAT))
        home_lon = float(os.getenv("FORECAST_LON", DEFAULT_LON))
        stale_dates = out_of_range_dates(activities, home_lat, home_lon)
        historical_weather = historical_weather.drop(index=list(stale_dates), errors="ignore")
    historical = prepare_datasets(
        activities=activities, tomorrow_weather=weather, historical_weather=historical_weather
    ).historical
    HISTORICAL_CSV.parent.mkdir(parents=True, exist_ok=True)
    historical.to_csv(HISTORICAL_CSV, index=False)
    return historical
