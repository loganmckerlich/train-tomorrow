"""Model explainability: Shapley contributions and plots for the frontend.

Single entrypoint is `build_explanation`; everything else here is a building block for it.
"""

from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb
from shapiq import TreeExplainer
from shapiq.interaction_values import InteractionValues

from features import FEATURE_COLUMNS

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


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp_term = math.exp(-value)
        return 1.0 / (1.0 + exp_term)
    exp_term = math.exp(value)
    return exp_term / (1.0 + exp_term)


def _feature_list(features: list[str] | None) -> list[str]:
    return FEATURE_COLUMNS if features is None else features


def _attributions(interaction_values: InteractionValues, features: list[str]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for indices, value in interaction_values.dict_values.items():
        if not indices:
            continue
        if len(indices) > 2:
            raise ValueError("Tree explanations must not contain interactions above order 2")
        names = [features[index] for index in indices]
        is_interaction = len(names) == 2
        phrase = " × ".join(PHRASE_BANK.get(name, name.replace("_", " ")) for name in names)
        if is_interaction:
            phrase += " (interaction)"
        output.append(
            {
                "indices": list(indices),
                "feature": " × ".join(names),
                "phrase": phrase,
                "is_interaction": is_interaction,
                "signed_contribution": float(value),
                "direction": "helping" if value >= 0 else "hurting",
            }
        )
    return sorted(output, key=lambda item: abs(item["signed_contribution"]), reverse=True)


def render_explanation_plots(
    interaction_values: InteractionValues,
    features: list[str],
    attributions: list[dict[str, Any]],
    baseline_probability: float,
    probability: float,
    top_n: int,
    output_dir: Path | None = None,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    def save_plot(name: str, figure: Any) -> str:
        buffer = io.BytesIO()
        figure.savefig(buffer, format="png", bbox_inches="tight", dpi=140)
        image = buffer.getvalue()
        if output_dir is not None:
            (output_dir / f"{name}.png").write_bytes(image)
        plt.close(figure)
        return "data:image/png;base64," + base64.b64encode(image).decode("ascii")

    waterfall_ax = interaction_values.plot_waterfall(
        feature_names=features, show=False, max_display=10
    )
    if waterfall_ax is None:
        raise RuntimeError("shapiq did not return a waterfall plot")
    waterfall_fig = waterfall_ax.figure
    waterfall_ax.set_xlabel("Attribution (log-odds)")
    baseline_ax, prediction_ax = waterfall_fig.axes[1:3]
    baseline_ticks = baseline_ax.get_xticks()
    prediction_ticks = prediction_ax.get_xticks()
    baseline_ax.set_xticks(baseline_ticks, [f"Baseline p={baseline_probability:.1%}", ""])
    prediction_ax.set_xticks(prediction_ticks, [f"Prediction p={probability:.1%}", ""])
    plots = {"waterfall": save_plot("waterfall", waterfall_fig)}

    top = attributions[:top_n]
    figure, ax = plt.subplots(figsize=(9, max(2.5, len(top) * 0.55)))
    ax.barh(
        [item["phrase"] for item in top],
        [item["signed_contribution"] for item in top],
        color=["#059669" if item["signed_contribution"] >= 0 else "#e11d48" for item in top],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Attribution (log-odds)")
    ax.set_title(f"Top {len(top)} contributions")
    figure.tight_layout()
    plots["top_n"] = save_plot("top_n", figure)

    force_fig = interaction_values.plot_force(feature_names=features, show=False)
    if force_fig is None:
        raise RuntimeError("shapiq did not return a force plot")
    force_ax = force_fig.axes[0]
    baseline_log_odds = float(interaction_values.baseline_value)
    margin = baseline_log_odds + math.fsum(item["signed_contribution"] for item in attributions)
    for label in force_ax.texts:
        x_position, y_position = label.get_position()
        if math.isclose(y_position, 0.25) and math.isclose(x_position, baseline_log_odds):
            label.set_text(f"Baseline p={baseline_probability:.1%}")
        elif math.isclose(y_position, 0.25) and math.isclose(x_position, margin):
            label.set_text(f"Prediction p={probability:.1%}")
    force_ax.set_xlabel("Attribution (log-odds)")
    plots["force"] = save_plot("force", force_fig)

    network_result = interaction_values.plot_network(feature_names=features, show=False)
    if network_result is None:
        raise RuntimeError("shapiq did not return a network plot")
    network_fig, network_ax = network_result
    network_ax.set_title("Feature attributions and interactions (log-odds)")
    plots["network"] = save_plot("network", network_fig)
    return plots


def build_explanation(
    model: xgb.XGBClassifier,
    feature_row: pd.DataFrame,
    top_n: int = 3,
    features: list[str] | None = None,
    output_dir: Path | None = None,
    render_plots: bool = True,
) -> dict[str, Any]:
    features = _feature_list(features)
    row = feature_row[features]
    # XGBoost classifiers return raw-margin (log-odds) contributions, not probability deltas.
    interaction_values = TreeExplainer(
        model=model, index="k-SII", min_order=0, max_order=2, class_index=1
    ).explain(row.to_numpy()[0])
    if interaction_values.max_order > 2:
        raise ValueError("Tree explanations must not contain interactions above order 2")

    attributions = _attributions(interaction_values, features)
    attribution_sum = math.fsum(item["signed_contribution"] for item in attributions)
    margin = float(model.predict(row, output_margin=True)[0])
    model_probability = float(model.predict_proba(row)[0, 1])
    native_contributions = model.get_booster().predict(
        xgb.DMatrix(row, feature_names=features), pred_contribs=True
    )[0]
    native_baseline = float(native_contributions[-1])
    if not np.isclose(attribution_sum, float(np.sum(native_contributions[:-1])), rtol=1e-5, atol=1e-5):
        raise ValueError("shapiq interactions do not reconcile with XGBoost feature contributions")
    # Keep the InteractionValues baseline aligned with XGBoost's raw-margin empty-set value.
    interaction_values.baseline_value = native_baseline
    interaction_values.interactions[()] = native_baseline
    baseline_log_odds = native_baseline
    baseline_probability = _sigmoid(baseline_log_odds)
    probability = _sigmoid(margin)
    consistent = np.isclose(
        baseline_log_odds + attribution_sum, margin, rtol=1e-5, atol=1e-5
    ) and np.isclose(probability, model_probability, rtol=1e-5, atol=1e-5)
    if not consistent:
        raise ValueError("shapiq attributions do not reconcile with the XGBoost prediction")

    if top_n < 0:
        raise ValueError("top_n must be non-negative")
    plots = (
        render_explanation_plots(
            interaction_values,
            features,
            attributions,
            baseline_probability,
            probability,
            top_n,
            output_dir=output_dir,
        )
        if render_plots
        else {}
    )
    return {
        "interaction_values": interaction_values,
        "baseline_log_odds": baseline_log_odds,
        "baseline_probability": baseline_probability,
        "attribution_sum": attribution_sum,
        "margin": margin,
        "probability": probability,
        "model_probability": model_probability,
        "consistent": bool(consistent),
        "waterfall_steps": attributions,
        "top_contributors": attributions[:top_n],
        "plots": plots,
    }
