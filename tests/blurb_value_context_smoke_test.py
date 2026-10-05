from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))  # allow importing sibling scripts modules

import pandas as pd

from blurb import generate_blurb, generate_gemini_prompt
from explain import _value_context


def main() -> None:
    historical = pd.DataFrame({"forecast_precip_probability": [0.0, 0.2, 0.6, 0.8, 0.0]})
    ctx = _value_context("forecast_precip_probability", historical, pd.Series({"forecast_precip_probability": 0.0}))
    assert ctx["current_value"] == 0.0 and ctx["typical_value"] == 0.2

    contributor = {
        "feature": "forecast_precip_probability",
        "phrase": "rain in the forecast",
        "signed_contribution": 0.5,
        "direction": "helping",
        "value_context": [ctx],
    }
    prompt = generate_gemini_prompt(True, 0.7, 40.0, [contributor], "sassy")
    assert "value=0.0 % of hours" in prompt and "means NO rain" in prompt
    assert "value=0.0" in generate_blurb(True, 0.7, 40.0, [contributor])

    miles = _value_context("mileage_acute_7", pd.DataFrame({"mileage_acute_7": [1609.344]}), pd.Series({"mileage_acute_7": 1609.344}))
    assert miles["current_value"] == 1.0 and miles["unit"] == "miles/day"
    days = _value_context("trained_days_30", pd.DataFrame({"trained_days_30": [16.0]}), pd.Series({"trained_days_30": 16.0}))
    assert "ACTIVE DAYS" in days["definition"]
    temp = _value_context("forecast_temp_high", pd.DataFrame({"forecast_temp_high": [100.0]}), pd.Series({"forecast_temp_high": 100.0}))
    assert temp["current_value"] == 212.0 and temp["unit"] == "°F"
    print("blurb value context smoke test passed")


if __name__ == "__main__":
    main()
