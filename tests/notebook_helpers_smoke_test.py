from __future__ import annotations

import sys
from pathlib import Path

from _bootstrap import bootstrap_scripts_path

bootstrap_scripts_path()
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "notebooks"))

import numpy as np
import pandas as pd

from helpers.modeling import holdout_report, time_holdout_split


def test_holdout_split_is_chronological_and_report_runs() -> None:
    rng = np.random.default_rng(0)
    n = 200
    frame = pd.DataFrame(
        {
            "as_of_date": pd.date_range("2024-01-01", periods=n)[::-1],  # shuffled order must be re-sorted
            "a": rng.normal(size=n),
            "will_train_tomorrow": rng.integers(0, 2, size=n),
        }
    )
    split = time_holdout_split(frame, ["a"], holdout_frac=0.25)
    X_search, X_hold, _, _ = split
    assert len(X_hold) == 50 and X_search["as_of_date"].max() < X_hold["as_of_date"].min()

    report = holdout_report(split, ["a"], {"n_estimators": 20, "max_depth": 2}, 90)
    assert set(report) == {"train_auc", "holdout_auc"}
