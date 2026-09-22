from __future__ import annotations

import random
from datetime import date
from typing import Any

PHRASE_BANK: dict[str, str] = {
    "acute_load_7": "recent training load in your legs",
    "chronic_load_28": "longer-term fitness base",
    "atl_ctl_ratio": "how peaky your load balance looks",
    "days_since_last_hard": "time since your last hard effort",
    "streak_length": "your current streak momentum",
    "day_of_week": "your usual day-of-week rhythm",
    "month": "the time of year",
    "season": "seasonal daylight vibes",
    "forecast_temp_high": "the daytime temperature forecast",
    "forecast_temp_low": "the overnight low",
    "forecast_precip_probability": "rain in the forecast",
    "forecast_wind_speed": "the wind forecast",
}


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


def generate_blurb(
    will_train: bool,
    probability: float,
    predicted_effort: float | None,
    top_contributors: list[dict[str, Any]],
) -> str:
    positives = [item["phrase"] for item in top_contributors if item["signed_contribution"] >= 0]
    negatives = [item["phrase"] for item in top_contributors if item["signed_contribution"] < 0]

    pos_text = positives[0] if positives else "your routine momentum"
    neg_text = negatives[0] if negatives else "a bit of friction in the setup"
    effort_text = f"~{predicted_effort:.0f} relative-effort points" if predicted_effort is not None else "a lighter day"

    train_templates = [
        "Looks like a go ({p:.0%}). {pos} is helping, even with {neg}. Aim for {effort}.",
        "Tomorrow trends green at {p:.0%}. {pos} is on your side; {neg} is the speed bump. Think {effort}.",
        "Model says likely session ({p:.0%}). Credit {pos}. Watch out for {neg}. Target: {effort}.",
        "You’re favored to train ({p:.0%}) — nice. {pos} is pushing forward; {neg} keeps it honest. Shoot for {effort}.",
        "Momentum check: {p:.0%} to train. {pos} is a tailwind, {neg} is a headwind. Plan on {effort}.",
    ]
    rest_templates = [
        "Probably a recovery day ({p:.0%} train chance). {neg} is dragging more than {pos} can offset.",
        "Forecast says easier day ({p:.0%} to train). {neg} stands out, so bank rest and reload.",
        "Tomorrow leans rest ({p:.0%} train probability). {pos} helps, but {neg} is winning this round.",
        "Not impossible ({p:.0%}), but likely recovery. {neg} is the blocker right now.",
        "Call it a reset day ({p:.0%} chance to train). {neg} outweighs the boost from {pos}.",
    ]

    template_pool = train_templates if will_train else rest_templates
    selector = random.Random(date.today().toordinal() + len(top_contributors))
    template = selector.choice(template_pool)
    return template.format(p=probability, pos=pos_text, neg=neg_text, effort=effort_text)


def generate_blurb_llm() -> str:
    """UNUSED STUB: optional future LLM polish step for narrative output."""
    return "LLM polish not enabled in this pipeline."
