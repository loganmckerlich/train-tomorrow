from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_PATH = ROOT_DIR / "data" / "latest_prediction.json"


@dataclass
class FeatureContributor:
    feature: str
    signed_contribution: float
    phrase: str


@dataclass
class DailyPrediction:
    prediction_date: str
    will_train_tomorrow: bool
    train_probability: float
    predicted_workload_tss: float | None
    top_contributors: list[FeatureContributor]
    challenge_blurb: str


def fetch_strava_activities(strava_access_token: str) -> list[dict[str, Any]]:
    """Fetch recent activities from Strava API.

    TODO(v1):
    - call Strava API with personal token from repo secret
    - request recent activities needed for rolling ATL/CTL/TSB-style features
    - return normalized activity records
    """
    if not strava_access_token:
        raise ValueError("STRAVA_ACCESS_TOKEN is required")
    return []


def fetch_tomorrow_weather(weather_api_key: str) -> dict[str, Any]:
    """Fetch tomorrow weather forecast (temp/precip/wind).

    TODO(v1):
    - call selected free weather API using WEATHER_API_KEY
    - return normalized forecast values for feature engineering
    """
    if not weather_api_key:
        raise ValueError("WEATHER_API_KEY is required")
    return {}


def engineer_features(
    activities: list[dict[str, Any]],
    weather_forecast: dict[str, Any],
    prediction_day: date,
) -> dict[str, float | int]:
    """Build tomorrow features for train/no-train and workload models.

    TODO(v1):
    - rolling load trajectory (ATL/CTL/TSB-style)
    - day-of-week preference pattern
    - days since last hard effort
    - current streak length
    - season/daylight
    - forecasted weather inputs (temperature, precipitation, wind)
    """
    _ = activities, weather_forecast
    return {
        "day_of_week": prediction_day.weekday(),
        "days_since_last_hard": 0,
        "streak_length": 0,
    }


def score_models(tomorrow_features: dict[str, float | int]) -> tuple[bool, float, float | None]:
    """Run separate XGBoost classifier and conditional regressor.

    TODO(v1):
    - load classifier artifact from /models
    - load separate regressor artifact from /models
    - score classifier probability: P(train tomorrow)
    - if predicted to train, score conditional workload/TSS regression
    """
    _ = tomorrow_features
    train_probability = 0.5
    will_train = train_probability >= 0.5
    predicted_workload_tss = 55.0 if will_train else None
    return will_train, train_probability, predicted_workload_tss


def compute_feature_contributors(
    tomorrow_features: dict[str, float | int],
) -> list[FeatureContributor]:
    """Compute top signed feature contributors via XGBoost pred_contribs.

    TODO(v1):
    - use booster.predict(..., pred_contribs=True)
    - rank signed contributor magnitudes
    - map top 2-3 contributors to phrase bank entries
    - keep this independent of any standalone shap dependency
    """
    _ = tomorrow_features
    return [
        FeatureContributor(
            feature="streak_length",
            signed_contribution=0.21,
            phrase="You're on a streak, so momentum is nudging you back out there.",
        ),
        FeatureContributor(
            feature="day_of_week",
            signed_contribution=-0.11,
            phrase="This weekday is usually quieter in your training pattern.",
        ),
    ]


def generate_challenge_blurb(
    will_train_tomorrow: bool,
    train_probability: float,
    predicted_workload_tss: float | None,
    top_contributors: list[FeatureContributor],
) -> str:
    """Template-based v1 blurb generation with an LLM extension point.

    TODO(v1+): optional LLM polish hook can be added here later while preserving
    deterministic template fallback behavior.
    """
    if not will_train_tomorrow:
        return (
            f"Tomorrow looks like a recharge day ({train_probability:.0%} confidence). "
            "Bank it now so your next hard session lands stronger."
        )

    top_phrase = top_contributors[0].phrase if top_contributors else "Your recent pattern supports a solid session."
    workload_fragment = (
        f"around {predicted_workload_tss:.0f} TSS" if predicted_workload_tss is not None else "a moderate load"
    )
    return (
        f"You're likely to train tomorrow ({train_probability:.0%}). "
        f"Aim for {workload_fragment}. {top_phrase}"
    )


def compute_prediction_for_user() -> DailyPrediction:
    """Core logic entry point, isolated for future migration to per-user jobs."""
    strava_access_token = os.getenv("STRAVA_ACCESS_TOKEN", "")
    weather_api_key = os.getenv("WEATHER_API_KEY", "")

    activities = fetch_strava_activities(strava_access_token)
    weather_forecast = fetch_tomorrow_weather(weather_api_key)

    prediction_day = date.today()
    tomorrow_features = engineer_features(
        activities=activities,
        weather_forecast=weather_forecast,
        prediction_day=prediction_day,
    )

    will_train, train_probability, predicted_workload_tss = score_models(tomorrow_features)
    top_contributors = compute_feature_contributors(tomorrow_features)
    challenge_blurb = generate_challenge_blurb(
        will_train_tomorrow=will_train,
        train_probability=train_probability,
        predicted_workload_tss=predicted_workload_tss,
        top_contributors=top_contributors,
    )

    return DailyPrediction(
        prediction_date=prediction_day.isoformat(),
        will_train_tomorrow=will_train,
        train_probability=train_probability,
        predicted_workload_tss=predicted_workload_tss,
        top_contributors=top_contributors,
        challenge_blurb=challenge_blurb,
    )


def write_prediction_json(prediction: DailyPrediction, output_path: Path = DEFAULT_OUTPUT_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output = asdict(prediction)
    output["top_contributors"] = [asdict(contributor) for contributor in prediction.top_contributors]

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(output, fp, indent=2)
        fp.write("\n")


def main() -> None:
    prediction = compute_prediction_for_user()
    write_prediction_json(prediction)


if __name__ == "__main__":
    main()
