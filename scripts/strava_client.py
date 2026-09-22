from __future__ import annotations

import math
import os
import shutil
import subprocess
from datetime import date, datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests

STRAVA_TOKEN_URL = "https://www.strava.com/oauth/token"
STRAVA_ACTIVITIES_URL = "https://www.strava.com/api/v3/athlete/activities"
EARTH_RADIUS_MILES = 3958.8


def exchange_refresh_token() -> dict[str, Any]:
    """Exchange the configured refresh token for a short-lived access token."""
    client_id = os.getenv("STRAVA_CLIENT_ID")
    client_secret = os.getenv("STRAVA_CLIENT_SECRET")
    refresh_token = os.getenv("STRAVA_REFRESH_TOKEN")
    if not client_id or not client_secret or not refresh_token:
        raise ValueError("STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, and STRAVA_REFRESH_TOKEN are required")

    response = requests.post(
        STRAVA_TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        },
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    new_refresh_token = payload.get("refresh_token")
    if isinstance(new_refresh_token, str) and new_refresh_token and new_refresh_token != refresh_token:
        update_refresh_token_secret(new_refresh_token)

    return payload


def update_refresh_token_secret(new_refresh_token: str) -> None:
    """Rotate STRAVA_REFRESH_TOKEN repository secret via gh CLI when possible."""
    repo = os.getenv("GITHUB_REPOSITORY")
    gh_token = os.getenv("GH_TOKEN") or os.getenv("GITHUB_TOKEN")
    gh_path = shutil.which("gh")
    if not repo or not gh_token or not gh_path:
        print("[strava] Skipping STRAVA_REFRESH_TOKEN rotation (gh/repo/token unavailable).")
        return

    env = os.environ.copy()
    env["GH_TOKEN"] = gh_token
    cmd = [gh_path, "secret", "set", "STRAVA_REFRESH_TOKEN", "--repo", repo, "--body", new_refresh_token]
    try:
        subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        print("[strava] Rotated STRAVA_REFRESH_TOKEN secret.")
    except subprocess.CalledProcessError as exc:
        print(f"[strava] Failed to rotate STRAVA_REFRESH_TOKEN secret: {exc.stderr.strip()}")


def fetch_recent_activities(access_token: str, days_back: int = 90) -> pd.DataFrame:
    """Pull authenticated athlete activities from the last N days as a DataFrame."""
    after_ts = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
    headers = {"Authorization": " ".join(["Bearer", access_token])}

    records: list[dict[str, Any]] = []
    page = 1
    while True:
        response = requests.get(
            STRAVA_ACTIVITIES_URL,
            headers=headers,
            params={"after": after_ts, "per_page": 200, "page": page},
            timeout=30,
        )
        response.raise_for_status()
        activities = response.json()
        if not activities:
            break

        for activity in activities:
            start_date_raw = activity.get("start_date_local") or activity.get("start_date")
            if not start_date_raw:
                continue
            start_dt = pd.to_datetime(start_date_raw, utc=True, errors="coerce")
            if pd.isna(start_dt):
                continue

            relative_effort = activity.get("suffer_score")
            if relative_effort is None:
                relative_effort = activity.get("relative_effort")

            start_latlng = activity.get("start_latlng") or []

            records.append(
                {
                    "date": start_dt.date().isoformat(),
                    "type": activity.get("type", "Unknown"),
                    "moving_time": float(activity.get("moving_time") or 0),
                    "distance": float(activity.get("distance") or 0),
                    "relative_effort": float(relative_effort or 0),
                    "average_watts": (
                        float(activity["average_watts"])
                        if activity.get("average_watts") is not None
                        else float("nan")
                    ),
                    "start_lat": float(start_latlng[0]) if len(start_latlng) == 2 else float("nan"),
                    "start_lng": float(start_latlng[1]) if len(start_latlng) == 2 else float("nan"),
                }
            )
        page += 1

    #TODO proper logging
    print(f"Fetched {len(records)} activities from Strava. Used {page-1} pages to get {days_back} days")

    columns = ["date", "type", "moving_time", "distance", "relative_effort", "average_watts", "start_lat", "start_lng"]
    return pd.DataFrame(records, columns=columns)


def fetch_activities_dataframe(days_back: int = 90) -> pd.DataFrame:
    token_payload = exchange_refresh_token()
    access_token = token_payload.get("access_token")
    if not isinstance(access_token, str) or not access_token:
        raise RuntimeError("Strava OAuth response did not include access_token")
    return fetch_recent_activities(access_token=access_token, days_back=days_back)


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_MILES * math.asin(math.sqrt(a))


def out_of_range_dates(
    activities: pd.DataFrame, home_lat: float, home_lon: float, radius_miles: float = 50.0
) -> set[date]:
    """Dates whose GPS-tagged activities all started more than radius_miles from home coordinates.

    Days with no GPS-tagged activity (rest days, indoor trainer rides) are left out of the
    result and treated as in-range, since there's no location signal to say otherwise.
    """
    located = activities.dropna(subset=["start_lat", "start_lng"]).copy()
    if located.empty:
        return set()

    located["date"] = pd.to_datetime(located["date"], errors="coerce").dt.date
    located = located.dropna(subset=["date"])
    distance = located.apply(
        lambda row: _haversine_miles(row["start_lat"], row["start_lng"], home_lat, home_lon), axis=1
    )
    nearby_dates = set(located.loc[distance <= radius_miles, "date"])
    return set(located["date"]) - nearby_dates
