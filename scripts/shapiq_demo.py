from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd
import xgboost as xgb

from explain import build_explanation
from features import FEATURE_COLUMNS, prepare_datasets


def _synthetic_activities(days: int = 120) -> pd.DataFrame:
    start = pd.Timestamp.today().normalize() - pd.to_timedelta(days, unit="D")
    return pd.DataFrame(
        [
            {
                "date": (start + pd.to_timedelta(i, unit="D")).date().isoformat(),
                "type": "Run",
                "moving_time": 1800 + i * 12,
                "distance": 4500 + i * 35,
                "relative_effort": float(20 + (i % 14) * 3),
                "average_watts": float("nan"),
            }
            for i in range(days)
            if i % 2 == 0 or i % 5 == 0
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local order-two shapiq explanation demo.")
    parser.add_argument("--output-dir", type=Path, default=Path("shapiq_demo_output"))
    args = parser.parse_args()

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
    prepared = prepare_datasets(
        activities=_synthetic_activities(),
        tomorrow_weather=weather,
        run_date=date.today(),
    )
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=24,
        max_depth=2,
        n_jobs=1,
        random_state=42,
    )
    model.fit(
        prepared.historical[FEATURE_COLUMNS],
        prepared.historical["will_train_tomorrow"],
    )
    explanation = build_explanation(
        model,
        prepared.tomorrow_features,
        top_n=5,
        output_dir=args.output_dir,
    )

    print(f"baseline_log_odds: {explanation['baseline_log_odds']:.6f}")
    print(f"sum_attributions: {explanation['attribution_sum']:.6f}")
    print(f"margin: {explanation['margin']:.6f}")
    print(f"probability: {explanation['probability']:.6f}")
    print(f"predict_proba: {explanation['model_probability']:.6f}")
    print(f"reconciliation: {'PASS' if explanation['consistent'] else 'FAIL'}")
    for contributor in explanation["top_contributors"]:
        print(f"{contributor['phrase']}: {contributor['signed_contribution']:+.4f} log-odds")
    print(f"plots saved to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
