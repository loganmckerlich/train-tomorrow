from __future__ import annotations

import os
import random
from datetime import date
from typing import Any

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
        "You're favored to train ({p:.0%}) — nice. {pos} is pushing forward; {neg} keeps it honest. Shoot for {effort}.",
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

def generate_gemini_prompt(
    will_train: bool,
    probability: float,
    predicted_effort: float | None,
    top_contributors: list[dict[str, Any]],
    tone: str,
) -> str:
    factors = "\n".join(
        f"- {c['feature']} ({c['phrase']}): signed_contribution={c['signed_contribution']:+.3f} "
        f"({c['direction']} the training call)"
        for c in top_contributors
    )
    effort_line = f"predicted_effort: {predicted_effort:.1f} (relative-effort units)" if predicted_effort is not None else "predicted_effort: n/a (rest day)"
    prompt = (
        "I have a model that predicts whether an athlete will train tomorrow based on various factors.\n"+
        "I want you to make a clean, concise summary of the prediction based on the data provided.\n"+
        "Tell the athlete what we think will occur tommorow based on the prediction along with why based on the top contributing factors.\n"+
        f"Use a {tone} tone\n\n"+
        "HERE IS YOUR DATA INPUT:\n"+
        f"prediction: {'train' if will_train else 'rest'}\n"+
        f"probability_to_train: {probability:.1%}\n"+
        f"{effort_line}\n"+
        f"top_contributing_factors (feature, meaning, signed contribution toward training):\n{factors}"
    )
    return prompt

def generate_blurb_llm(
    will_train: bool,
    probability: float,
    predicted_effort: float | None,
    top_contributors: list[dict[str, Any]],
    fallback_blurb: str,
    tone: str = "sassy",
) -> str:
    """Have Gemini write the summary directly from the raw prediction data (no template scaffold).

    Falls back to fallback_blurb (the templated generate_blurb output) if GEMINI_API_KEY is
    missing or the API call fails, so the daily pipeline never breaks on an LLM issue.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback_blurb

    try:
        from google import genai

        prompt = generate_gemini_prompt(
            will_train=will_train,
            probability=probability,
            predicted_effort=predicted_effort,
            top_contributors=top_contributors,
            tone=tone,
        )
        client = genai.Client(api_key=api_key)
        chat = client.chats.create(model="gemini-3.6-flash")
        response = chat.send_message(prompt)
        polished = (response.text or "").strip()
        return polished or fallback_blurb
    except Exception as e:
        print(f"Error generating blurb with LLM: {e}")
        return fallback_blurb
