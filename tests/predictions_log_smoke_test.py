from __future__ import annotations

import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))  # allow importing sibling scripts modules

from predictions_log import backfill_outcomes, load_entries, save_entries, upsert_entry

import pandas as pd

logger = logging.getLogger(__name__)


def main() -> None:
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    main()
