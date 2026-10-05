from __future__ import annotations

import logging
import os
import random
import time
from datetime import date
from typing import Any

logger = logging.getLogger(__name__)


def describe_values(contributor: dict[str, Any]) -> str:
    """e.g. 'value=0 (typical 0.3, 12th percentile)'; empty when no context is attached."""
    parts = []
    for ctx in contributor.get("value_context") or []:
        if ctx.get("current_value") is None:
            continue
        unit = f" {ctx['unit']}" if ctx.get("unit") else ""
        shown = ctx.get("current_label") or f"{ctx['current_value']}{unit}"
        text = f"value={shown} (typical {ctx['typical_value']}{unit}, {ctx['percentile']:.0%} of history is at or below)"
        parts.append(text if len(contributor["value_context"]) == 1 else f"{ctx['feature']} {text}")
    return "; ".join(parts)


def describe_definitions(contributor: dict[str, Any]) -> str:
    ctxs = contributor.get("value_context") or []
    return "; ".join(
        (f"{c['feature']}: {c['definition']}" if len(ctxs) > 1 else c["definition"]) for c in ctxs if c.get("definition")
    ) or "n/a"


def _phrase_with_value(contributor: dict[str, Any]) -> str:
    values = describe_values(contributor)
    return f"{contributor['phrase']} [{values}]" if values else contributor["phrase"]


def generate_blurb(
    will_train: bool,
    probability: float,
    predicted_effort: float | None,
    top_contributors: list[dict[str, Any]],
) -> str:
    positives = [_phrase_with_value(item) for item in top_contributors if item["signed_contribution"] >= 0]
    negatives = [_phrase_with_value(item) for item in top_contributors if item["signed_contribution"] < 0]

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
        f"- {c['feature']} (variable meaning: {c['phrase']}): {describe_values(c) or 'value n/a'}; "
        f"definition: {describe_definitions(c)}; "
        f"signed_contribution={c['signed_contribution']:+.3f} ({c['direction']} the training call)"
        for c in top_contributors
    )
    effort_line = f"predicted_effort: {predicted_effort:.1f} (relative-effort units)" if predicted_effort is not None else "predicted_effort: n/a (rest day)"
    prompt = (
        "I have a model that predicts whether an athlete will train tomorrow based on various factors.\n"+
        "I want you to make a clean, concise summary of the prediction based on the data provided.\n"+
        "Tell the athlete what we think will occur tomorrow based on the prediction along with why based on the top contributing factors.\n"+
        "Each factor's meaning only names the variable, not its state (value=0 for 'rain in the forecast' means NO rain). "+
        "Describe tomorrow's actual condition from the value, units and definition, and never claim anything they contradict. "+
        "A positive contribution from a low value means the low value is helping.\n"+
        f"Use a {tone} tone\n\n"+
        "HERE IS YOUR DATA INPUT:\n"+
        f"prediction: {'train' if will_train else 'rest'}\n"+
        f"probability_to_train: {probability:.1%}\n"+
        f"{effort_line}\n"+
        f"top_contributing_factors (feature, meaning, signed contribution toward training):\n{factors}"
    )
    logger.info(f"Generated Gemini prompt:\n{prompt}")
    return prompt

def generate_blurb_llm(
    will_train: bool,
    probability: float,
    predicted_effort: float | None,
    top_contributors: list[dict[str, Any]],
    fallback_blurb: str,
    tone: str = "sassy",
    gemini_model: str = "gemini-3.6-flash",
    max_retries: int = 3,
) -> str:
    """Have Gemini write the summary directly from the raw prediction data (no template scaffold).

    Falls back to fallback_blurb (the templated generate_blurb output) if GEMINI_API_KEY is
    missing or the API call fails after retries, so the daily pipeline never breaks on an LLM issue.
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
        total_attempts = max(1, max_retries)
        for attempt in range(total_attempts):
            try:
                response = client.models.generate_content(
                    model=gemini_model,
                    contents=prompt,
                )
                polished = (response.text or "").strip()
                if polished:
                    return polished
                break
            except Exception as e:
                error_text = str(e).upper()
                if ("503" in error_text or "UNAVAILABLE" in error_text) and attempt < total_attempts - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        "Gemini overloaded (attempt %d/%d), retrying in %ds",
                        attempt + 1,
                        total_attempts,
                        wait,
                    )
                    time.sleep(wait)
                    continue
                logger.warning("Error generating blurb with LLM: %s", e)
                break
        return fallback_blurb
    except Exception as e:
        logger.warning("Error generating blurb with LLM: %s", e)
        return fallback_blurb
