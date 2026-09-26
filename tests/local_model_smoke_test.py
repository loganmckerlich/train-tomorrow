from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from _bootstrap import bootstrap_scripts_path

bootstrap_scripts_path()

import pandas as pd

from blurb import generate_blurb
from explain import build_explanation, feature_contributions, summarize_top_contributors
from features import prepare_datasets
from model import predict_tomorrow, train_and_save_models
from run_daily import build_waterfall

logger = logging.getLogger(__name__)


def _synthetic_activities(days: int = 120) -> pd.DataFrame:
    start = pd.Timestamp.today().normalize() - pd.to_timedelta(days, unit="D")
    rows: list[dict[str, object]] = []
    for i in range(days):
        day = start + pd.to_timedelta(i, unit="D")
        if i % 2 == 0 or i % 5 == 0:
            rows.append(
                {
                    "date": day.date().isoformat(),
                    "type": "Run",
                    "moving_time": 1800 + i * 12,
                    "distance": 4500 + i * 35,
                    "relative_effort": float(20 + (i % 14) * 3),
                    "average_watts": float("nan"),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    activities = _synthetic_activities()
    weather = {
        "temp_high": 18.0,
        "temp_low": 11.0,
        "precip_probability": 25.0,
        "wind_speed": 10.0,
        "temp_high_vs_seasonal": 1.5,
        "temp_low_vs_seasonal": 0.5,
        "precip_probability_vs_seasonal": -3.0,
        "wind_speed_vs_seasonal": 2.0,
        "precip_morning": 10.0,
        "precip_midday": 5.0,
        "precip_evening": 0.0,
    }
    prepared = prepare_datasets(activities=activities, tomorrow_weather=weather)

    with tempfile.TemporaryDirectory() as tmpdir:
        models = train_and_save_models(prepared.historical, Path(tmpdir))
        prediction = predict_tomorrow(models, prepared.tomorrow_features)
        explanation = build_explanation(models.classifier, prepared.tomorrow_features, prepared.historical, top_n=3)
        contributions = feature_contributions(models.classifier, prepared.tomorrow_features)

    top_contributors = explanation["top_contributors"]
    waterfall = build_waterfall(
        summarize_top_contributors(contributions, top_n=len(contributions)),
        explanation["baseline_probability"],
        prediction["probability"],
    )
    blurb = generate_blurb(
        will_train=prediction["will_train"],
        probability=prediction["probability"],
        predicted_effort=prediction["predicted_effort"],
        top_contributors=top_contributors,
    )

    assert top_contributors
    assert 0.0 < explanation["baseline_probability"] < 1.0
    assert isinstance(explanation["other_contribution"], float)
    assert waterfall["steps"]
    assert abs(float(waterfall["final_probability"]) - float(prediction["probability"])) < 1e-4
    for contributor in top_contributors:
        plot = contributor["plot"]
        assert plot["kind"] in {"categorical", "continuous"}
        if plot["kind"] == "continuous":
            assert plot["current_value"] is None or isinstance(plot["current_value"], float)
            assert isinstance(plot["current_shap"], float)
            assert plot["points"]
        else:
            assert plot["current_value"] is None or isinstance(plot["current_value"], int)
            assert plot["current_label"] is None or isinstance(plot["current_label"], str)
            assert plot["categories"]
            assert all(isinstance(category["label"], str) for category in plot["categories"])

    logger.info("Local modeling smoke test passed.")
    logger.info(
        "%s",
        {
            "probability": round(float(prediction["probability"]), 4),
            "predicted_effort": prediction["predicted_effort"],
            "top_contributor": top_contributors[0]["feature"] if top_contributors else None,
            "blurb_preview": blurb,
        },
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
