# train-tomorrow

Daily pipeline that predicts:

1. **Will I train tomorrow?** (binary)
2. **If so, how hard?** (relative-effort regression)

It also generates a short Strava-style blurb using top XGBoost feature contributions.

## Architecture

```text
Strava OAuth refresh-token exchange
        +
Open-Meteo tomorrow forecast
        |
        v
scripts/run_daily.py
  - feature engineering + leakage-safe labels
  - XGBoost classifier/regressor training
  - pred_contribs extraction (no shap package)
  - blurb generation
        |
        v
data/latest.json + models/*.json
        |
        v
frontend (Next.js + Tailwind)
```

## Repository layout

- `/scripts/strava_client.py` – refresh-token exchange, optional secret rotation, activity pull
- `/scripts/weather_client.py` – tomorrow forecast from Open-Meteo
- `/scripts/features.py` – rolling-load features + labeled dataset prep
- `/scripts/model.py` – train/score XGBoost models + feature contributions
- `/scripts/blurb.py` – phrase bank + template blurb generation
- `/scripts/run_daily.py` – end-to-end orchestration
- `/tests` – smoke tests for the modeling + predictions-log pipeline
- `/notebooks` – local exploration notebooks
- `/data/latest.json` – latest prediction payload
- `/models` – serialized classifier/regressor
- `/frontend` – Next.js app

## Python setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export STRAVA_CLIENT_ID=...
export STRAVA_CLIENT_SECRET=...
export STRAVA_REFRESH_TOKEN=...
# optional location override for weather forecast:
export FORECAST_LAT=37.7749
export FORECAST_LON=-122.4194

python scripts/run_daily.py
```

## Local modeling exploration

```bash
source .venv/bin/activate
pip install -r requirements-dev.txt
python tests/local_model_smoke_test.py
jupyter lab notebooks/modeling_exploration.ipynb
```

The smoke test uses synthetic activities and writes temporary model artifacts only.

## Prediction impact tracking

The daily pipeline saves classifier predictions from its chronological validation split as a fixed baseline in
`data/baseline_rates.json` on its first run with this feature. Rates are counted in five 20-point probability buckets;
only pre-deployment validation targets are included. The live comparison uses resolved prediction-history entries
from the first `daily-predict.yml` run on **2026-09-22** onward and is recomputed as outcomes are backfilled.
The baseline file is committed by the daily workflow and is never recalculated once present, even though the
production model continues to retrain on all available data.

False-negative rates are compared in predicted-low buckets and false-positive rates in predicted-high buckets,
using one-sided Fisher exact tests. The UI reports both sample sizes and p-values and shows cumulative live rates
against the fixed baseline. This is a correlational proxy, not a controlled experiment: it assumes stationary
baseline rates, may be underpowered with small daily samples, and cannot determine whether the prediction was viewed
before training. Treat early results as provisional, not causal evidence.

## Frontend setup

```bash
cd frontend
npm install
npm run dev
```

Optional override for frontend data source:

```bash
export NEXT_PUBLIC_PREDICTION_URL="https://raw.githubusercontent.com/loganmckerlich/train-tomorrow/master/data/latest.json"
```

## GitHub Actions

`.github/workflows/daily-predict.yml` runs daily and on manual dispatch.
The workflow job targets the `production - predictions` environment, so configure the listed secrets/variables there.

Required repository secrets:

- `STRAVA_CLIENT_ID`
- `STRAVA_CLIENT_SECRET`
- `STRAVA_REFRESH_TOKEN`

Optional repository/environment variables:

- `FORECAST_LAT`
- `FORECAST_LON`

Notes:
- Open-Meteo does not require an API key.
- If Strava rotates refresh tokens, the workflow attempts to update `STRAVA_REFRESH_TOKEN` via `gh secret set` using `${{ github.token }}`.
