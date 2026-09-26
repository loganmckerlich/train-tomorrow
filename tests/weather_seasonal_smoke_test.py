from __future__ import annotations

from datetime import date

from _bootstrap import bootstrap_scripts_path

bootstrap_scripts_path()

import pandas as pd

import weather_client


def test_seasonal_baseline_prefers_leap_day_match() -> None:
    climatology = pd.DataFrame(
        {
            "temp_high": [10.0, 20.0],
            "temp_low": [1.0, 2.0],
            "precip_probability": [30.0, 40.0],
            "wind_speed": [5.0, 6.0],
        },
        index=[date(2020, 2, 29), date(2021, 3, 1)],
    )

    baseline = weather_client._seasonal_baseline(climatology, date(2024, 2, 29), window_days=0)
    assert float(baseline["temp_high"]) == 10.0


def test_apply_seasonal_anomalies_clamps_non_leap_start_date() -> None:
    frame = pd.DataFrame(
        {
            "temp_high": [15.0],
            "temp_low": [7.0],
            "precip_probability": [20.0],
            "wind_speed": [8.0],
        },
        index=[date(2024, 3, 1)],
    )

    original_fetch = weather_client._fetch_archive_daily

    def fake_fetch_archive_daily(lat: float, lon: float, start_date: date, end_date: date) -> pd.DataFrame:
        assert start_date == date(2019, 2, 28)
        assert end_date == date(2024, 2, 29)
        return pd.DataFrame(
            {
                "temp_high": [12.0],
                "temp_low": [5.0],
                "precip_probability": [10.0],
                "wind_speed": [6.0],
            },
            index=[date(2020, 3, 1)],
        )

    try:
        weather_client._fetch_archive_daily = fake_fetch_archive_daily
        result = weather_client._apply_seasonal_anomalies(frame.copy(), 0.0, 0.0)
    finally:
        weather_client._fetch_archive_daily = original_fetch

    assert float(result.at[date(2024, 3, 1), "temp_high_vs_seasonal"]) == 3.0


def main() -> None:
    test_seasonal_baseline_prefers_leap_day_match()
    test_apply_seasonal_anomalies_clamps_non_leap_start_date()


if __name__ == "__main__":
    main()
