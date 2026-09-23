from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))  # allow importing sibling scripts modules

import pandas as pd

from blurb import generate_blurb, summarize_top_contributors
from features import prepare_datasets
from model import attach_feature_plots, feature_contributions, predict_tomorrow, train_and_save_models

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
    }
    prepared = prepare_datasets(activities=activities, tomorrow_weather=weather)

    with tempfile.TemporaryDirectory() as tmpdir:
        models = train_and_save_models(prepared.historical, Path(tmpdir))
        prediction = predict_tomorrow(models, prepared.tomorrow_features)
        contributions = feature_contributions(models.classifier, prepared.tomorrow_features)

    top_contributors = summarize_top_contributors(contributions, top_n=3)
    top_contributors = attach_feature_plots(
        models.classifier, top_contributors, prepared.historical, prepared.tomorrow_features
    )
    blurb = generate_blurb(
        will_train=prediction["will_train"],
        probability=prediction["probability"],
        predicted_effort=prediction["predicted_effort"],
        top_contributors=top_contributors,
    )

    assert top_contributors
    for contributor in top_contributors:
        plot = contributor["plot"]
        assert plot["kind"] in {"categorical", "continuous"}
        if plot["kind"] == "continuous":
            assert plot["current_value"] is None or isinstance(plot["current_value"], float)
            assert isinstance(plot["current_shap"], float)
            assert plot["points"]
        else:
            assert plot["current_value"] is None or isinstance(plot["current_value"], int)
            assert plot["categories"]

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
