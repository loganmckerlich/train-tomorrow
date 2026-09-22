"""Append-only log of daily predictions plus their eventual real-world outcome.

Lets a later analysis pass measure whether the false-negative rate (predicted
"won't train" but the user trained anyway) drops once the app is in daily use.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


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
    """Replace any existing entry for the same target date (handles manual re-runs) and append the latest one."""
    target_date = payload["date"]
    entries = [entry for entry in entries if entry.get("date") != target_date]
    entries.append(
        {
            **payload,
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
