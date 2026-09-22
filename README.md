# train-tomorrow

A personal fitness-tech side project that refreshes daily and answers:

1. **Will I train tomorrow?** (binary classification)
2. **If so, how hard?** (conditional workload/TSS regression)

It also generates a short, upbeat challenge blurb based on top model contributors.

## v1 architecture (single user)

```text
Strava API + Weather API
        |
        v
scripts/daily_predict.py
  - feature engineering
  - XGBoost classifier (train/no-train)
  - XGBoost regressor (conditional workload)
  - pred_contribs-based contributors + phrase mapping
  - template blurb generation
        |
        v
data/latest_prediction.json  <-- committed daily by GitHub Actions
        |
        v
frontend (Next.js on Vercel)
  - renders prediction, challenge, blurb
```

## Repository layout

- `/scripts` – prediction pipeline + feature engineering/model stubs
- `/models` – serialized model artifacts
- `/data` – latest prediction JSON output
- `/frontend` – Next.js frontend for Vercel deployment
- `/.github/workflows` – scheduled daily prediction workflow
- `requirements.txt` – Python dependencies for the prediction pipeline

## Local setup

### Python pipeline

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export STRAVA_ACCESS_TOKEN=your_token
export WEATHER_API_KEY=your_key
python scripts/daily_predict.py
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

## GitHub Actions

`.github/workflows/daily-predict.yml` runs once daily (`cron`) and on manual dispatch. It expects:

- `STRAVA_ACCESS_TOKEN`
- `WEATHER_API_KEY`

as repository secrets.

## Notes

- v1 intentionally avoids the standalone `shap` package and uses XGBoost `pred_contribs=True` for feature attribution.
- Core prediction logic is isolated in `compute_prediction_for_user()` for easier migration to per-user batch jobs later.
