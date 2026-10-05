"""Model explainability: SHAP-IQ contributions, phrasing, and effect plots for the frontend.

Single entrypoint is `build_explanation`; everything else here is a building block for it.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from shapiq import TreeExplainer
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
    "mileage_acute_7": "your recent mileage volume",
    "mileage_chronic_28": "your longer-term mileage base",
    "today_relative_effort": "how hard today's session was",
    "dow_train_rate": "regular train rate for this day of the week",
    "month_train_rate": "regular train rate for this month",
    "trained_last_weekend": "whether you trained this past weekend",
    "day_of_week": "the day of the week",
    "month": "the month of year",
    "season": "seasonal vibes",
    "days_since_last_long_ride": "time since your last really long session",
    "forecast_temp_high": "the temp high forecast",
    "forecast_temp_low": "the temp low forecast",
    "forecast_precip_probability": "rain in the forecast",
    "forecast_rain_expected": "whether rain is expected at all",
    "forecast_wind_speed": "the wind forecast",
    "forecast_temp_high_vs_seasonal": "how the temp high compares to normal for this time of year",
    "forecast_temp_low_vs_seasonal": "how the temp low compares to normal for this time of year",
    "forecast_precip_probability_vs_seasonal": "how much rainier or drier than usual it is",
    "forecast_wind_speed_vs_seasonal": "how much windier or calmer than usual it is",
    "forecast_precip_morning": "rain chances during your morning window",
    "forecast_precip_midday": "rain chances during the midday window",
    "forecast_precip_evening": "rain chances during your evening window",
}


def _feature_list(features: list[str] | None) -> list[str]:
    return FEATURE_COLUMNS if features is None else features


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


def _interaction_contributions(
    interaction_values: Any, features: list[str]
) -> tuple[dict[str, float], dict[tuple[str, str], float]]:
    values = interaction_values.dict_values
    main_effects = {feature: float(values.get((index,), 0.0)) for index, feature in enumerate(features)}
    interactions = {
        (features[interaction[0]], features[interaction[1]]): float(value)
        for interaction, value in values.items()
        if len(interaction) == 2
    }
    return main_effects, interactions


def _rank_contributors(
    contributions: dict[str, float], interactions: dict[tuple[str, str], float]
) -> list[dict[str, Any]]:
    ranked = [
        {
            "feature": feature,
            "kind": "feature",
            "signed_contribution": value,
            "direction": "helping" if value >= 0 else "hurting",
            "phrase": PHRASE_BANK.get(feature, feature.replace("_", " ")),
        }
        for feature, value in contributions.items()
    ]
    ranked.extend(
        {
            "feature": f"{feature_a} × {feature_b}",
            "features": [feature_a, feature_b],
            "kind": "interaction",
            "signed_contribution": value,
            "direction": "helping" if value >= 0 else "hurting",
            "phrase": (
                f"interaction between {PHRASE_BANK.get(feature_a, feature_a.replace('_', ' '))} "
                f"and {PHRASE_BANK.get(feature_b, feature_b.replace('_', ' '))}"
            ),
        }
        for (feature_a, feature_b), value in interactions.items()
    )
    ranked.sort(key=lambda item: abs(item["signed_contribution"]), reverse=True)
    return ranked


def feature_contributions(
    model: xgb.XGBModel, feature_row: pd.DataFrame, features: list[str] | None = None
) -> dict[str, float]:
    return explain_prediction(model, feature_row, features)["contributions"]


def explain_prediction(
    model: xgb.XGBModel, feature_row: pd.DataFrame, features: list[str] | None = None
) -> dict[str, Any]:
    features = _feature_list(features)
    explainer = TreeExplainer(model=model, class_index=1, index="k-SII", min_order=1, max_order=2)
    interaction_values = explainer.explain(feature_row[features].iloc[0].to_numpy(dtype=float))
    contributions, interactions = _interaction_contributions(interaction_values, features)
    baseline_log_odds = float(interaction_values.baseline_value)
    total_log_odds = baseline_log_odds + sum(contributions.values()) + sum(interactions.values())
    return {
        "contributions": contributions,
        "interactions": interactions,
        "baseline_log_odds": baseline_log_odds,
        "baseline_probability": _sigmoid(baseline_log_odds),
        "probability": _sigmoid(total_log_odds),
    }


def _feature_contributions_frame(
    explainer: TreeExplainer,
    feature_rows: pd.DataFrame,
    features: list[str] | None = None,
) -> pd.DataFrame:
    features = _feature_list(features)
    explanations = explainer.explain_X(feature_rows[features].to_numpy(dtype=float), verbose=False)
    return pd.DataFrame(
        [_interaction_contributions(explanation, features)[0] for explanation in explanations],
        columns=features,
        index=feature_rows.index,
    )


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
    top_contributors: list[dict[str, Any]],
    historical: pd.DataFrame,
    feature_row: pd.DataFrame,
    historical_contribs: pd.DataFrame,
    features: list[str] | None = None,
) -> list[dict[str, Any]]:
    features = _feature_list(features)
    current = feature_row.iloc[0]

    enriched: list[dict[str, Any]] = []
    for contributor in top_contributors:
        if contributor["kind"] == "interaction":
            enriched.append(contributor)
            continue
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
    features: list[str] | None = None,
) -> dict[str, Any]:
    """Rank SHAP-IQ effects, attach feature plots to the top N, and summarize the rest.

    Single source of truth for both the contributor list and the waterfall chart: the frontend
    builds the waterfall directly from `top_contributors` + `baseline_probability` +
    `other_contribution`, instead of a separately-computed duplicate structure.
    """
    features = _feature_list(features)
    explainer = TreeExplainer(model=model, class_index=1, index="k-SII", min_order=1, max_order=2)
    interaction_values = explainer.explain(feature_row[features].iloc[0].to_numpy(dtype=float))
    contributions, interactions = _interaction_contributions(interaction_values, features)
    ranked = _rank_contributors(contributions, interactions)
    selected = ranked[:top_n]
    if any(contributor["kind"] == "feature" for contributor in selected):
        historical_contribs = _feature_contributions_frame(explainer, historical, features=features)
        top_contributors = attach_feature_plots(
            selected, historical, feature_row, historical_contribs, features=features
        )
    else:
        top_contributors = selected

    remainder = ranked[top_n:]
    other_contribution = sum(item["signed_contribution"] for item in remainder)
    if abs(other_contribution) <= 1e-9:
        other_contribution = 0.0

    return {
        "baseline_log_odds": float(interaction_values.baseline_value),
        "baseline_probability": round(_sigmoid(float(interaction_values.baseline_value)), 4),
        "top_contributors": top_contributors,
        "other_contribution": round(float(other_contribution), 4),
    }


DENSITY_POINTS = 40


def _density_curve(values: pd.Series) -> list[dict[str, float]]:
    """Gaussian KDE (Silverman bandwidth) over the 1st-99th percentile, scaled to max 1 for violin width."""
    lo, hi = values.quantile(0.01), values.quantile(0.99)
    std = float(values.std())
    if hi <= lo or pd.isna(std) or std == 0:
        return []
    xs = np.linspace(lo, hi, DENSITY_POINTS)
    arr = values.to_numpy(dtype=float)
    bandwidth = 1.06 * std * len(arr) ** -0.2
    density = np.exp(-0.5 * ((xs[:, None] - arr[None, :]) / bandwidth) ** 2).sum(axis=1)
    density = density / density.max()
    return [{"x": round(float(x), 4), "density": round(float(d), 4)} for x, d in zip(xs, density, strict=True)]


def build_feature_summary(
    model: xgb.XGBModel,
    historical: pd.DataFrame,
    feature_row: pd.DataFrame,
    features: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Per-feature describe stats, KDE and runtime value, sorted by XGBoost feature_importances_ (not SHAP)."""
    features = _feature_list(features)
    importances = dict(zip(features, model.feature_importances_, strict=True))
    current = feature_row.iloc[0]
    summary: list[dict[str, Any]] = []
    for feature in features:
        values = pd.to_numeric(historical[feature], errors="coerce").dropna()
        if values.empty:
            continue
        current_value = pd.to_numeric(pd.Series([current.get(feature)]), errors="coerce").iloc[0]
        current_value = None if pd.isna(current_value) else float(current_value)
        entry: dict[str, Any] = {
            "feature": feature,
            "label": PHRASE_BANK.get(feature, feature),
            "importance": round(float(importances[feature]), 4),
            "current_value": None if current_value is None else round(current_value, 4),
            "current_percentile": None if current_value is None else round(float((values <= current_value).mean()), 4),
        }
        if feature in CATEGORICAL_FEATURES or values.nunique() <= 2:
            entry["kind"] = "categorical"
            entry["current_label"] = None if current_value is None else _category_label(feature, current_value)
            entry["categories"] = [
                {"value": float(v), "label": _category_label(feature, float(v)), "share": round(float(s), 4)}
                for v, s in values.value_counts(normalize=True).sort_index().items()
            ]
        else:
            q = values.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            entry["kind"] = "continuous"
            entry["stats"] = {
                "count": int(values.count()),
                "mean": round(float(values.mean()), 4),
                "std": round(float(values.std()), 4),
                "min": round(float(values.min()), 4),
                "p05": round(float(q[0.05]), 4),
                "p25": round(float(q[0.25]), 4),
                "median": round(float(q[0.5]), 4),
                "p75": round(float(q[0.75]), 4),
                "p95": round(float(q[0.95]), 4),
                "max": round(float(values.max()), 4),
            }
            entry["density"] = _density_curve(values)
        summary.append(entry)
    return sorted(summary, key=lambda item: item["importance"], reverse=True)
