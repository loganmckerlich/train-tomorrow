from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))  # allow importing sibling scripts modules

from predictions_log import (
    LIVE_DEPLOYMENT_DATE,
    backfill_outcomes,
    build_impact_summary,
    load_entries,
    load_or_create_baseline,
    save_entries,
    upsert_entry,
)

import pandas as pd

logger = logging.getLogger(__name__)


def main() -> None:
    _check_impact_tracking()
    historical = pd.DataFrame(
        [
            {"target_date": "2026-01-02", "will_train_tomorrow": 1, "next_day_relative_effort": 42.0},
            {"target_date": "2026-01-03", "will_train_tomorrow": 0, "next_day_relative_effort": 0.0},
        ]
    )
    entries = [
        {
            "date": "2026-01-02",
            "will_train": False,
            "probability": 0.2,
            "predicted_effort": None,
            "actual_will_train": None,
            "actual_effort": None,
            "checked_at": None,
        },
        {
            "date": "2026-01-01",
            "will_train": True,
            "probability": 0.9,
            "predicted_effort": 30.0,
            "actual_will_train": True,
            "actual_effort": 25.0,
            "checked_at": "2026-01-02T00:00:00+00:00",
        },
    ]

    entries = backfill_outcomes(entries, historical)
    resolved = next(entry for entry in entries if entry["date"] == "2026-01-02")
    assert resolved["actual_will_train"] is True
    assert resolved["actual_effort"] == 42.0
    assert resolved["checked_at"] is not None

    already_resolved = next(entry for entry in entries if entry["date"] == "2026-01-01")
    assert already_resolved["checked_at"] == "2026-01-02T00:00:00+00:00"  # untouched, not overwritten

    payload = {
        "date": "2026-01-04",
        "will_train": True,
        "probability": 0.7,
        "predicted_effort": 20.0,
        "top_contributors": [],
        "blurb": "test",
    }
    entries = upsert_entry(entries, payload, as_of_date="2026-01-03")
    entries = upsert_entry(entries, payload, as_of_date="2026-01-03")  # simulate a re-run same day
    matches = [entry for entry in entries if entry["date"] == "2026-01-04"]
    assert len(matches) == 1

    with tempfile.TemporaryDirectory() as tmpdir:
        log_path = Path(tmpdir) / "predictions_history.jsonl"
        save_entries(entries, log_path)
        reloaded = load_entries(log_path)
        assert len(reloaded) == len(entries)

    logger.info("Predictions log smoke test passed.")


def _check_impact_tracking() -> None:
    baseline_predictions = [
        {"date": f"2026-09-{day:02}", "probability": 0.1, "actual_will_train": False}
        for day in range(1, 5)
    ] + [
        {"date": f"2026-09-{day:02}", "probability": 0.9, "actual_will_train": True}
        for day in range(1, 5)
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        baseline_path = Path(tmpdir) / "baseline_rates.json"
        baseline = load_or_create_baseline(baseline_path, baseline_predictions)
        assert baseline["deployment_date"] == LIVE_DEPLOYMENT_DATE
        assert baseline["buckets"][0]["false_negative"] == {"n": 4, "events": 0, "rate": 0.0}
        assert baseline["buckets"][4]["false_positive"] == {"n": 4, "events": 0, "rate": 0.0}

        changed_baseline = load_or_create_baseline(baseline_path, [])
        assert changed_baseline == baseline

    live_entries = [
        {"date": f"2026-09-{day:02}", "probability": 0.1, "actual_will_train": True}
        for day in range(22, 25)
    ] + [
        {"date": f"2026-09-{day:02}", "probability": 0.9, "actual_will_train": False}
        for day in range(22, 25)
    ]
    live_entries.append({"date": "2026-09-21", "probability": 0.1, "actual_will_train": True})
    summary = build_impact_summary(live_entries, baseline)
    false_negative = next(
        row for row in summary["comparisons"] if row["direction"] == "false_negative" and row["lower_bound"] == 0
    )
    false_positive = next(
        row for row in summary["comparisons"] if row["direction"] == "false_positive" and row["lower_bound"] == 0.8
    )
    assert false_negative["baseline_n"] == false_positive["baseline_n"] == 4
    assert false_negative["live_n"] == false_positive["live_n"] == 3
    assert false_negative["live_rate"] == false_positive["live_rate"] == 1.0
    assert round(false_negative["p_value"], 6) == round(false_positive["p_value"], 6) == 0.028571
    assert false_negative["significant"] and false_positive["significant"]
    false_negative_rolling = next(
        row for row in summary["rolling"] if row["direction"] == "false_negative" and row["lower_bound"] == 0
    )
    assert [point["rate"] for point in false_negative_rolling["points"]] == [1.0, 1.0, 1.0]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
