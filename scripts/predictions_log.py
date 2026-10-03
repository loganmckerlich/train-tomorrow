"""Append-only log of daily predictions plus their eventual real-world outcome."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from scipy.stats import fisher_exact

LIVE_DEPLOYMENT_DATE = "2026-09-27"
BUCKET_COUNT = 3
SIGNIFICANCE_LEVEL = 0.05


def _empty_bucket_counts() -> list[dict]:
    return [
        {
            "lower_bound": index / BUCKET_COUNT,
            "upper_bound": (index + 1) / BUCKET_COUNT,
            "false_negative": {"n": 0, "events": 0, "rate": None},
            "false_positive": {"n": 0, "events": 0, "rate": None},
        }
        for index in range(BUCKET_COUNT)
    ]


def _bucket_index(probability: float) -> int | None:
    if not math.isfinite(probability) or probability < 0 or probability > 1:
        return None
    return min(int(probability * BUCKET_COUNT), BUCKET_COUNT - 1)


def _bucket_counts(records: list[dict]) -> list[dict]:
    buckets = _empty_bucket_counts()
    for record in records:
        probability = record.get("probability")
        actual = record.get("actual_will_train")
        if not isinstance(probability, (int, float)) or not isinstance(actual, bool):
            continue
        index = _bucket_index(float(probability))
        if index is None:
            continue
        direction = "false_negative" if probability < 0.5 else "false_positive"
        counts = buckets[index][direction]
        counts["n"] += 1
        counts["events"] += int(actual if direction == "false_negative" else not actual)

    for bucket in buckets:
        for direction in ("false_negative", "false_positive"):
            counts = bucket[direction]
            counts["rate"] = counts["events"] / counts["n"] if counts["n"] else None
    return buckets


def load_or_create_baseline(path: Path, validation_predictions: list[dict]) -> dict:
    """Keep the first pre-deployment validation-set rates as a fixed baseline."""
    if path.exists():
        with path.open(encoding="utf-8") as fp:
            existing = json.load(fp)
        if len(existing.get("buckets", [])) == BUCKET_COUNT:
            return existing

    pre_deployment = [
        {
            **record,
            "actual_will_train": bool(record["actual_will_train"]),
        }
        for record in validation_predictions
        if isinstance(record.get("date"), str)
        and record["date"] < LIVE_DEPLOYMENT_DATE
        and record.get("actual_will_train") is not None
    ]
    if not pre_deployment:
        raise ValueError("No pre-deployment validation predictions are available for the frozen baseline")
    baseline = {
        "deployment_date": LIVE_DEPLOYMENT_DATE,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "buckets": _bucket_counts(pre_deployment),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(baseline, fp, indent=2)
        fp.write("\n")
    return baseline


def build_impact_summary(entries: list[dict], baseline: dict) -> dict:
    live_entries = [
        entry
        for entry in entries
        if entry.get("date", "") >= LIVE_DEPLOYMENT_DATE
        and entry.get("actual_will_train") is not None
    ]
    live_buckets = _bucket_counts(live_entries)
    comparisons: list[dict] = []
    rolling: list[dict] = []

    for bucket_index, (baseline_bucket, live_bucket) in enumerate(zip(baseline["buckets"], live_buckets)):
        for direction in ("false_negative", "false_positive"):
            baseline_counts = baseline_bucket[direction]
            live_counts = live_bucket[direction]
            p_value = None
            if baseline_counts["n"] and live_counts["n"]:
                table = [
                    [live_counts["events"], live_counts["n"] - live_counts["events"]],
                    [baseline_counts["events"], baseline_counts["n"] - baseline_counts["events"]],
                ]
                p_value = float(fisher_exact(table, alternative="greater").pvalue)
            comparisons.append(
                {
                    "lower_bound": live_bucket["lower_bound"],
                    "upper_bound": live_bucket["upper_bound"],
                    "direction": direction,
                    "baseline_n": baseline_counts["n"],
                    "baseline_rate": baseline_counts["rate"],
                    "live_n": live_counts["n"],
                    "live_rate": live_counts["rate"],
                    "p_value": p_value,
                    "significant": p_value is not None and p_value < SIGNIFICANCE_LEVEL,
                }
            )

            relevant = sorted(
                (
                    entry
                    for entry in live_entries
                    if isinstance(entry.get("probability"), (int, float))
                    and isinstance(entry.get("actual_will_train"), bool)
                    and _bucket_index(float(entry["probability"])) == bucket_index
                    and ("false_negative" if entry["probability"] < 0.5 else "false_positive") == direction
                ),
                key=lambda entry: entry["date"],
            )
            events = 0
            points: list[dict] = []
            for entry in relevant:
                actual = bool(entry["actual_will_train"])
                events += int(actual if direction == "false_negative" else not actual)
                points.append(
                    {
                        "date": entry["date"],
                        "n": len(points) + 1,
                        "rate": events / (len(points) + 1),
                    }
                )
            rolling.append(
                {
                    "lower_bound": live_bucket["lower_bound"],
                    "upper_bound": live_bucket["upper_bound"],
                    "direction": direction,
                    "baseline_rate": baseline_counts["rate"],
                    "points": points,
                }
            )

    return {
        "deployment_date": LIVE_DEPLOYMENT_DATE,
        "significance_level": SIGNIFICANCE_LEVEL,
        "baseline_created_at": baseline.get("created_at"),
        "comparisons": comparisons,
        "rolling": rolling,
    }

def load_entries(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fp:
        return [json.loads(line) for line in fp if line.strip()]


def backfill_outcomes(entries: list[dict], historical: pd.DataFrame) -> list[dict]:
    """Fill in actual outcomes for past predictions using the freshly rebuilt historical dataset."""
    outcomes = historical.set_index("target_date")[["will_train_tomorrow", "next_day_relative_effort"]]
    now = datetime.now(timezone.utc).isoformat()
    for entry in entries:
        if entry.get("actual_will_train") is not None:
            continue
        target_date = entry.get("date")
        if target_date not in outcomes.index:
            continue
        row = outcomes.loc[target_date]
        entry["actual_will_train"] = bool(row["will_train_tomorrow"])
        entry["actual_effort"] = float(row["next_day_relative_effort"])
        entry["checked_at"] = now
    return entries


def upsert_entry(entries: list[dict], payload: dict, as_of_date: str) -> list[dict]:
    """Replace any existing entry for the same target date (handles manual re-runs) and append the latest one.

    Only keeps the fields backfill_outcomes/build_impact_summary/the UI actually need -
    explanation/plot data and the aggregate calibration snapshot are latest.json-only, not history.
    """
    target_date = payload["date"]
    entries = [entry for entry in entries if entry.get("date") != target_date]
    entries.append(
        {
            "date": payload["date"],
            "generated_at": payload.get("generated_at"),
            "will_train": payload["will_train"],
            "probability": payload["probability"],
            "predicted_effort": payload["predicted_effort"],
            "blurb": payload.get("blurb"),
            "as_of_date": as_of_date,
            "actual_will_train": None,
            "actual_effort": None,
            "logged_at": datetime.now(timezone.utc).isoformat(),
            "checked_at": None,
        }
    )
    return entries


def save_entries(entries: list[dict], path: Path) -> None:
    entries = sorted(entries, key=lambda entry: entry["date"])
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for entry in entries:
            fp.write(json.dumps(entry) + "\n")
