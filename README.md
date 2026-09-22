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
