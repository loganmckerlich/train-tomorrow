from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from explain import build_explanation
from shapiq.interaction_values import InteractionValues


def main() -> None:
    rows = list(itertools.product(range(2), repeat=4)) * 24
    features = ["day_of_week", "trained_days_7", "load", "weather"]
    values = pd.DataFrame(rows, columns=features)
    labels = ((values["day_of_week"] == 1) & (values["trained_days_7"] == 1)).astype(int)
    model = xgb.XGBClassifier(
        n_estimators=20,
        max_depth=2,
        learning_rate=0.4,
        n_jobs=1,
        random_state=7,
        eval_metric="logloss",
    )
    model.fit(values, labels)

    explanation = build_explanation(
        model,
        values.iloc[[0]],
        top_n=6,
        features=features,
        render_plots=False,
    )
    interaction_values = explanation["interaction_values"]
    assert isinstance(interaction_values, InteractionValues)
    assert interaction_values.index == "k-SII"
    assert interaction_values.min_order == 0
    assert interaction_values.max_order == 2
    assert all(len(item["indices"]) <= 2 for item in explanation["waterfall_steps"])
    assert all(
        item["signed_contribution"] == interaction_values[tuple(item["indices"])]
        for item in explanation["waterfall_steps"]
    )
    assert all(
        any(item is waterfall_item for waterfall_item in explanation["waterfall_steps"])
        for item in explanation["top_contributors"]
    )
    assert any(item["is_interaction"] for item in explanation["top_contributors"])
    assert all(
        "interaction" in item["phrase"]
        for item in explanation["top_contributors"]
        if item["is_interaction"]
    )
    assert np.isclose(
        explanation["baseline_log_odds"] + explanation["attribution_sum"],
        explanation["margin"],
        rtol=1e-5,
        atol=1e-5,
    )
    assert np.isclose(
        explanation["probability"],
        model.predict_proba(values.iloc[[0]])[0, 1],
        rtol=1e-5,
        atol=1e-5,
    )
    assert explanation["consistent"]
    print("shapiq explanation smoke test passed")


if __name__ == "__main__":
    main()
