from __future__ import annotations

import json
from pathlib import Path

from blurb import generate_blurb, summarize_top_contributors
from features import prepare_datasets
from model import feature_contributions, predict_tomorrow, train_and_save_models
from strava_client import fetch_activities_dataframe
from weather_client import fetch_tomorrow_forecast

ROOT_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT_DIR / "models"
LATEST_JSON_PATH = ROOT_DIR / "data" / "latest.json"


def run_pipeline() -> dict[str, object]:
    activities = fetch_activities_dataframe(days_back=90)
    weather = fetch_tomorrow_forecast()

    prepared = prepare_datasets(activities=activities, tomorrow_weather=weather)

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
