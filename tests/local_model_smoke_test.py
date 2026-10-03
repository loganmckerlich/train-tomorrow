from __future__ import annotations

import logging
import math
import tempfile
from datetime import date, timedelta
from pathlib import Path

from _bootstrap import bootstrap_scripts_path

bootstrap_scripts_path()

import pandas as pd

from blurb import generate_blurb
from explain import PHRASE_BANK, _interaction_contributions, _rank_contributors, build_explanation
from features import prepare_datasets
from model import predict_tomorrow, train_and_save_models

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


def _check_interaction_explanation() -> None:
    interaction_values = type("InteractionValuesStub", (), {})()
    interaction_values.dict_values = {(0,): 2.0, (1,): -1.0, (0, 1): 0.8}
    main_effects, interactions = _interaction_contributions(interaction_values, ["load", "weather"])

    assert main_effects == {"load": 2.0, "weather": -1.0}
    assert interactions == {("load", "weather"): 0.8}
    assert math.isclose(sum(main_effects.values()) + sum(interactions.values()), 1.8)

    ranked = _rank_contributors(
        {"load": 0.1, "weather": -0.2},
        {("load", "weather"): 0.8},
    )
    assert ranked[0]["kind"] == "interaction"
    assert ranked[0]["features"] == ["load", "weather"]
    assert "interaction between" in ranked[0]["phrase"]


def main() -> None:
    _check_interaction_explanation()
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
    run_date = date.today()
    prepared = prepare_datasets(activities=activities, tomorrow_weather=weather, run_date=run_date)
    assert prepared.tomorrow_features.iloc[0]["as_of_date"] == run_date.isoformat()
    assert prepared.tomorrow_features.iloc[0]["target_date"] == (run_date + timedelta(days=1)).isoformat()
    mileage_features = {"mileage_acute_7", "mileage_chronic_28"}
    assert mileage_features <= set(prepared.historical.columns)
    assert mileage_features <= set(prepared.tomorrow_features.columns)
    assert mileage_features <= PHRASE_BANK.keys()

    with tempfile.TemporaryDirectory() as tmpdir:
        models = train_and_save_models(prepared.historical, Path(tmpdir))
        assert models.validation_predictions
        assert all(
            set(prediction) == {"date", "probability", "actual_will_train"}
            for prediction in models.validation_predictions
        )
        prediction = predict_tomorrow(models, prepared.tomorrow_features)
        explanation = build_explanation(models.classifier, prepared.tomorrow_features, prepared.historical, top_n=3)

    top_contributors = explanation["top_contributors"]
    blurb = generate_blurb(
        will_train=prediction["will_train"],
        probability=prediction["probability"],
        predicted_effort=prediction["predicted_effort"],
        top_contributors=top_contributors,
    )

    assert top_contributors
    assert 0.0 < explanation["baseline_probability"] < 1.0
    total_log_odds = (
        explanation["baseline_log_odds"]
        + sum(item["signed_contribution"] for item in top_contributors)
        + explanation["other_contribution"]
    )
    assert abs(1 / (1 + math.exp(-total_log_odds)) - prediction["probability"]) < 1e-3
    assert isinstance(explanation["other_contribution"], float)
    for contributor in top_contributors:
        assert contributor["kind"] in {"feature", "interaction"}
        if contributor["kind"] == "interaction":
            assert len(contributor["features"]) == 2
            assert "×" in contributor["feature"]
            continue
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
