"""Model explainability: SHAP contributions, phrasing, and effect plots for the frontend.

Single entrypoint is `build_explanation`; everything else here is a building block for it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from features import FEATURE_COLUMNS

CATEGORICAL_FEATURES = {"day_of_week", "month", "season"}
DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON_LABELS = ["Winter", "Spring", "Summer", "Fall"]

CONTINUOUS_BIN_COUNT = 24  # caps historical scatter points; ceiling is coarser resolution, raise if plots look chunky

PHRASE_BANK: dict[str, str] = {
    "acute_load_7": "recent training load in your legs",
    "chronic_load_28": "longer-term fitness base",
    "atl_ctl_ratio": "how peaky your load balance looks",
    "days_since_last_hard": "time since your last hard effort",
    "streak_length": "your current streak momentum",
    "trained_today": "whether you already trained today",
    "trained_hard_today": "whether today's session was a hard one",
    "trained_days_2": "how much you've trained the past 2 days",
    "trained_both_days_2": "back-to-back training the last 2 days",
    "trained_days_7": "how often you've trained this past week",
    "trained_days_30": "your training frequency over the past month",
    "moving_time_acute_7": "your recent training volume",
    "moving_time_chronic_28": "your training volume base over the past month",
    "today_relative_effort": "how hard today's session was",
    "dow_train_rate": "your usual habit on this day of the week",
    "month_train_rate": "how you usually train this time of year",
    "day_of_week": "your usual day-of-week rhythm",
    "month": "the time of year",
    "season": "seasonal daylight vibes",
    "days_since_last_long_ride": "time since your last really long session",
    "forecast_temp_high": "the daytime temperature forecast",
    "forecast_temp_low": "the overnight low",
    "forecast_precip_probability": "rain in the forecast",
    "forecast_rain_expected": "whether rain is expected at all",
    "forecast_wind_speed": "the wind forecast",
}


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def summarize_top_contributors(contributions: dict[str, float], top_n: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:top_n]
    output: list[dict[str, Any]] = []
    for feature, value in ranked:
        output.append(
            {
                "feature": feature,
                "signed_contribution": float(value),
                "direction": "helping" if value >= 0 else "hurting",
                "phrase": PHRASE_BANK.get(feature, feature.replace("_", " ")),
            }
        )
    return output


def feature_contributions(model: xgb.XGBModel, feature_row: pd.DataFrame) -> dict[str, float]:
    return explain_prediction(model, feature_row)["contributions"]


def explain_prediction(model: xgb.XGBModel, feature_row: pd.DataFrame) -> dict[str, Any]:
    matrix = xgb.DMatrix(feature_row[FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS)
    contribs = model.get_booster().predict(matrix, pred_contribs=True)[0]
    contributions = {
        name: float(value)
        for name, value in zip(FEATURE_COLUMNS, contribs[:-1], strict=True)
    }
    baseline_log_odds = float(contribs[-1])
    total_log_odds = baseline_log_odds + float(np.sum(contribs[:-1]))
    return {
        "contributions": contributions,
        "baseline_log_odds": baseline_log_odds,
        "baseline_probability": _sigmoid(baseline_log_odds),
        "probability": _sigmoid(total_log_odds),
    }


def _feature_contributions_frame(
    model: xgb.XGBModel, feature_rows: pd.DataFrame, columns: list[str] | None = None
) -> pd.DataFrame:
    matrix = xgb.DMatrix(feature_rows[FEATURE_COLUMNS], feature_names=FEATURE_COLUMNS)
    contribs = model.get_booster().predict(matrix, pred_contribs=True)
    frame = pd.DataFrame(contribs[:, :-1], columns=FEATURE_COLUMNS, index=feature_rows.index)
    return frame if columns is None else frame[columns]


def _category_label(feature: str, value: float) -> str:
    numeric = int(value) if float(value).is_integer() else round(float(value), 4)
    if feature == "day_of_week":
        return DAY_LABELS[int(numeric)] if 0 <= int(numeric) < len(DAY_LABELS) else str(numeric)
    if feature == "month":
        month_index = int(numeric) - 1
        return MONTH_LABELS[month_index] if 0 <= month_index < len(MONTH_LABELS) else str(numeric)
    if feature == "season":
        return SEASON_LABELS[int(numeric)] if 0 <= int(numeric) < len(SEASON_LABELS) else str(numeric)
    return str(numeric)


def _binned_continuous_points(values: pd.Series, shap_values: pd.Series) -> list[dict[str, float]]:
    """Average SHAP into quantile buckets so plot size doesn't grow with history length."""
    combined = pd.DataFrame({"value": values, "shap": shap_values}).dropna()
    if combined.empty:
        return []
    if len(combined) > CONTINUOUS_BIN_COUNT and combined["value"].nunique() > 1:
        bins = min(CONTINUOUS_BIN_COUNT, combined["value"].nunique())
        combined["bucket"] = pd.qcut(combined["value"], q=bins, duplicates="drop")
        combined = combined.groupby("bucket", observed=True)[["value", "shap"]].mean()
    return [
        {"feature_value": round(float(value), 4), "shap_value": round(float(shap), 4)}
        for value, shap in zip(combined["value"], combined["shap"], strict=True)
    ]


def attach_feature_plots(
    model: xgb.XGBModel,
    top_contributors: list[dict[str, Any]],
    historical: pd.DataFrame,
    feature_row: pd.DataFrame,
) -> list[dict[str, Any]]:
    current = feature_row.iloc[0]
    requested_features = [contributor["feature"] for contributor in top_contributors]
    historical_contribs = _feature_contributions_frame(
        model, historical[FEATURE_COLUMNS], columns=requested_features
    )

    enriched: list[dict[str, Any]] = []
    for contributor in top_contributors:
        feature = contributor["feature"]
        values = pd.to_numeric(historical[feature], errors="coerce")
        current_value = pd.to_numeric(pd.Series([current.get(feature)]), errors="coerce").iloc[0]
        shap_values = pd.to_numeric(historical_contribs[feature], errors="coerce")

        if feature in CATEGORICAL_FEATURES:
            categories = (
                pd.DataFrame({"value": values, "shap": shap_values})
                .dropna()
                .groupby("value", sort=True)["shap"]
                .mean()
                .items()
            )
            plot: dict[str, Any] = {
                "kind": "categorical",
                "current_value": int(current_value) if not pd.isna(current_value) else None,
                "current_label": _category_label(feature, float(current_value)) if not pd.isna(current_value) else None,
                "categories": [
                    {
                        "value": int(value) if float(value).is_integer() else round(float(value), 4),
                        "label": _category_label(feature, float(value)),
                        "mean_shap": round(float(mean_shap), 4),
                    }
                    for value, mean_shap in categories
                ],
            }
        else:
            plot = {
                "kind": "continuous",
                "current_value": round(float(current_value), 4) if not pd.isna(current_value) else None,
                "current_shap": round(float(contributor["signed_contribution"]), 4),
                "points": _binned_continuous_points(values, shap_values),
            }

        enriched.append(
            {
                **contributor,
                "plot": plot,
            }
        )
    return enriched


def build_explanation(
    model: xgb.XGBModel,
    feature_row: pd.DataFrame,
    historical: pd.DataFrame,
    top_n: int = 3,
) -> dict[str, Any]:
    """Rank SHAP contributions, attach effect plots to the top N, and summarize the rest.

    Single source of truth for both the contributor list and the waterfall chart: the frontend
    builds the waterfall directly from `top_contributors` + `baseline_probability` +
    `other_contribution`, instead of a separately-computed duplicate structure.
    """
    explanation = explain_prediction(model, feature_row)
    ranked = summarize_top_contributors(explanation["contributions"], top_n=len(explanation["contributions"]))
    top_contributors = attach_feature_plots(model, ranked[:top_n], historical, feature_row)

    remainder = ranked[top_n:]
    other_contribution = sum(item["signed_contribution"] for item in remainder)
    if abs(other_contribution) <= 1e-9:
        other_contribution = 0.0

    return {
        "baseline_probability": round(float(explanation["baseline_probability"]), 4),
        "top_contributors": top_contributors,
        "other_contribution": round(float(other_contribution), 4),
    }
