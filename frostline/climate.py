"""Frost climatology from open weather data (Open-Meteo, no API key).

Everything is cached to data/*.csv, so after the first run FrostLine works with no signal.
"""
from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FROST_C = 0.0  # a "frost night" is tmin <= this


@dataclass
class Place:
    name: str
    lat: float
    lon: float


@dataclass
class FrostDates:
    """Day-of-year-free dates for the target year, plus how many years backed them."""
    last_spring_median: date
    last_spring_safe: date      # p90: only ~1 in 10 years sees frost after this
    first_fall_median: date
    first_fall_early: date      # p10: only ~1 in 10 years sees frost before this
    years: int
    frost_free: bool = False


def _get_json(url: str, timeout: int = 20) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 (fixed https hosts)
        return json.load(r)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def geocode(query: str) -> Place:
    cache = DATA_DIR / f"place_{_slug(query)}.json"
    if cache.exists():
        return Place(**json.loads(cache.read_text()))
    q = urllib.parse.urlencode({"name": query, "count": 1})
    res = _get_json(f"https://geocoding-api.open-meteo.com/v1/search?{q}").get("results")
    if not res:
        raise ValueError(f"Could not find a place called {query!r}")
    r = res[0]
    place = Place(", ".join(x for x in (r["name"], r.get("admin1"), r.get("country")) if x), r["latitude"], r["longitude"])
    DATA_DIR.mkdir(exist_ok=True)
    cache.write_text(json.dumps(place.__dict__))
    return place


def load_history(place: Place, years: int = 15, today: date | None = None) -> pd.DataFrame:
    """Daily min temperature, cached on disk. Columns: date (datetime64), tmin."""
    today = today or date.today()
    cache = DATA_DIR / f"tmin_{_slug(place.name)}.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["date"])
    start = date(today.year - years, 1, 1)
    end = date(today.year - 1, 12, 31)
    q = urllib.parse.urlencode({
        "latitude": place.lat, "longitude": place.lon, "start_date": start, "end_date": end,
        "daily": "temperature_2m_min", "timezone": "auto",
    })
    d = _get_json(f"https://archive-api.open-meteo.com/v1/archive?{q}", timeout=60)["daily"]
    df = pd.DataFrame({"date": pd.to_datetime(d["time"]), "tmin": d["temperature_2m_min"]}).dropna()
    DATA_DIR.mkdir(exist_ok=True)
    df.to_csv(cache, index=False)
    return df


def recent_observed(place: Place, days: int = 10) -> pd.DataFrame | None:
    """Last few days of real tmin (needs signal). Returns None when offline."""
    q = urllib.parse.urlencode({
        "latitude": place.lat, "longitude": place.lon, "past_days": days, "forecast_days": 1,
        "daily": "temperature_2m_min", "timezone": "auto",
    })
    try:
        d = _get_json(f"https://api.open-meteo.com/v1/forecast?{q}", timeout=8)["daily"]
    except Exception:
        return None
    return pd.DataFrame({"date": pd.to_datetime(d["time"]), "tmin": d["temperature_2m_min"]}).dropna()


def _season_shift(lat: float) -> int:
    """Shift the calendar by half a year in the southern hemisphere so 'spring' always comes first."""
    return 182 if lat < 0 else 0


def frost_dates(hist: pd.DataFrame, lat: float, year: int) -> FrostDates:
    shift = _season_shift(lat)
    h = hist.copy()
    shifted = h["date"] - pd.Timedelta(days=shift)
    h["sy"] = shifted.dt.year
    h["doy"] = shifted.dt.dayofyear
    frost = h[h["tmin"] <= FROST_C]
    n_years = h["sy"].nunique()
    if len(frost) < n_years * 2:  # fewer than ~2 frost nights a year: effectively frost-free
        d = date(year, 1, 1)
        return FrostDates(d, d, d, d, n_years, frost_free=True)

    mid = 182  # split the (shifted) year into spring half and fall half
    last, first = [], []
    for _, g in frost.groupby("sy"):
        spring, fall = g[g["doy"] <= mid], g[g["doy"] > mid]
        if len(spring):
            last.append(spring["doy"].max())
        if len(fall):
            first.append(fall["doy"].min())
    last, first = np.array(last), np.array(first)

    def to_date(doy: float) -> date:
        return (pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(round(doy)) - 1 + shift)).date()

    return FrostDates(
        last_spring_median=to_date(np.percentile(last, 50)),
        last_spring_safe=to_date(np.percentile(last, 90)),
        first_fall_median=to_date(np.percentile(first, 50)) if len(first) else to_date(365),
        first_fall_early=to_date(np.percentile(first, 10)) if len(first) else to_date(365),
        years=n_years,
    )
