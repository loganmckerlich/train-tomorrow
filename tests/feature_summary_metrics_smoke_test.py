from __future__ import annotations

import json
import tempfile
from pathlib import Path

from _bootstrap import bootstrap_scripts_path

bootstrap_scripts_path()

import numpy as np
import pandas as pd

from explain import build_feature_summary
from model import train_and_save_models

FEATURES = ["load", "day_of_week", "trained_today"]


def _dataset(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "load": rng.normal(size=n),
            "day_of_week": rng.integers(0, 7, n),
            "trained_today": rng.integers(0, 2, n),
        }
    )
    df["will_train_tomorrow"] = (df["load"] + rng.normal(size=n) > 0).astype(int)
    df["next_day_relative_effort"] = rng.uniform(10, 100, n)
    days = pd.date_range("2025-01-01", periods=n, freq="D")
    df["as_of_date"] = days.date
    df["target_date"] = (days + pd.Timedelta(days=1)).date
    return df


def main() -> None:
    df = _dataset()
    with tempfile.TemporaryDirectory() as tmp:
        models = train_and_save_models(df, Path(tmp), features=FEATURES, validation_period_days=100)

    clf = models.metrics["classifier"]
    assert clf["n_holdout"] == 100 and clf["n_train"] == 300
    assert clf["auc"] is not None and clf["auc"] > 0.5
    assert clf["roc_curve"] and clf["calibration"]
    assert models.metrics["regressor"]["mae"] is not None

    summary = build_feature_summary(models.classifier, df, df.tail(1), features=FEATURES)
    assert {item["feature"] for item in summary} == set(FEATURES)
    importances = [item["importance"] for item in summary]
    assert importances == sorted(importances, reverse=True)

    by_name = {item["feature"]: item for item in summary}
    assert by_name["load"]["kind"] == "continuous" and len(by_name["load"]["density"]) > 1
    assert by_name["day_of_week"]["kind"] == "categorical" and by_name["day_of_week"]["current_label"]
    assert by_name["trained_today"]["kind"] == "categorical"
    assert 0 <= by_name["load"]["current_percentile"] <= 1

    json.dumps({"feature_summary": summary, "model_metrics": models.metrics}, allow_nan=False)
    print("feature summary / metrics smoke test passed")


if __name__ == "__main__":
    main()
