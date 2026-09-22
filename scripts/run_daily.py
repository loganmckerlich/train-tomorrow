from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import yaml

from blurb import generate_blurb, generate_blurb_llm, summarize_top_contributors
from features import prepare_datasets
from model import feature_contributions, predict_tomorrow, train_and_save_models
from strava_client import fetch_activities_dataframe, out_of_range_dates
from weather_client import DEFAULT_LAT, DEFAULT_LON, fetch_historical_weather, fetch_tomorrow_forecast

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
LATEST_JSON_PATH = ROOT_DIR / "data" / "latest.json"
PARAMS_PATH = ROOT_DIR / "params.yaml"


def load_params(path: Path = PARAMS_PATH) -> dict:
    if not path.exists():
        return {}
    with path.open() as fp:
        return yaml.safe_load(fp) or {}


def run_pipeline() -> dict[str, object]:
    activities = fetch_activities_dataframe(days_back=90)
    weather = fetch_tomorrow_forecast()

    activity_dates = pd.to_datetime(activities["date"], errors="coerce").dt.date.dropna()
    historical_weather = (
        fetch_historical_weather(activity_dates.min(), activity_dates.max())
        if not activity_dates.empty
        else None
    )
    if historical_weather is not None:
        home_lat = float(os.getenv("FORECAST_LAT", DEFAULT_LAT))
        home_lon = float(os.getenv("FORECAST_LON", DEFAULT_LON))
        stale_dates = out_of_range_dates(activities, home_lat, home_lon)
        historical_weather = historical_weather.drop(index=list(stale_dates), errors="ignore")

    prepared = prepare_datasets(
        activities=activities,
        tomorrow_weather=weather,
        historical_weather=historical_weather,
    )


    # Retraining each run keeps v1 simple; incremental retraining can be added later.
    models = train_and_save_models(prepared.historical, MODELS_DIR)

    prediction = predict_tomorrow(models, prepared.tomorrow_features)
    contribs = feature_contributions(models.classifier, prepared.tomorrow_features)
    top_contributors = summarize_top_contributors(contribs, top_n=3)
    blurb = generate_blurb(
        will_train=prediction["will_train"],
        probability=prediction["probability"],
        predicted_effort=prediction["predicted_effort"],
        top_contributors=top_contributors,
    )
    if load_params().get("blurb", {}).get("use_llm", True):
        blurb = generate_blurb_llm(
            will_train=prediction["will_train"],
            probability=prediction["probability"],
            predicted_effort=prediction["predicted_effort"],
            top_contributors=top_contributors,
            fallback_blurb=blurb,
        )

    return {
        "date": str(prepared.tomorrow_features.iloc[0]["target_date"]),
        "will_train": bool(prediction["will_train"]),
        "probability": round(float(prediction["probability"]), 4),
        "predicted_effort": (
            round(float(prediction["predicted_effort"]), 2)
            if prediction["predicted_effort"] is not None
            else None
        ),
        "top_contributors": top_contributors,
        "blurb": blurb,
    }


def write_latest(payload: dict[str, object], output_path: Path = LATEST_JSON_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
        fp.write("\n")


def main() -> None:
    payload = run_pipeline()
    write_latest(payload)
    print(f"[run_daily] wrote {LATEST_JSON_PATH}")


if __name__ == "__main__":
    main()
