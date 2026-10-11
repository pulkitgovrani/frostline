from datetime import date

import numpy as np
import pandas as pd

from frostline import climate, planner


def synthetic(lat=45.0, years=10):
    d = pd.date_range("2010-01-01", periods=365 * years + 3, freq="D")
    doy = d.dayofyear.values
    t = 8 - 17 * np.cos(2 * np.pi * (doy - 15) / 365.25)  # coldest mid-Jan, hottest mid-Jul
    if lat < 0:
        t = 8 - 17 * np.cos(2 * np.pi * (doy - 197) / 365.25)
    return pd.DataFrame({"date": d, "tmin": t})


def test_frost_dates_northern_order():
    fd = climate.frost_dates(synthetic(), 45.0, 2026)
    assert fd.last_spring_median <= fd.last_spring_safe
    assert fd.last_spring_median.month in (3, 4, 5)
    assert fd.first_fall_early <= fd.first_fall_median and fd.first_fall_median.month in (9, 10, 11)


def test_frost_dates_southern_hemisphere():
    fd = climate.frost_dates(synthetic(-40.0), -40.0, 2026)
    assert fd.last_spring_median.month in (8, 9, 10, 11)
    assert fd.first_fall_median.month in (3, 4, 5, 6)


def test_frost_free():
    h = synthetic(); h["tmin"] = h["tmin"] + 40
    assert climate.frost_dates(h, 1.0, 2026).frost_free
    fd = climate.frost_dates(h, 1.0, 2026)
    assert planner.tasks_for_week(fd, date(2026, 5, 1)) == []
    cool = planner.tasks_for_week(fd, date(2026, 1, 15), h)
    warm = planner.tasks_for_week(fd, date(2026, 7, 15), h)
    assert any(t.crop == "Spinach" for t in cool) and not any(t.crop == "Spinach" for t in warm)
    assert any(t.crop == "Tomatoes" for t in warm)


def test_tender_crops_wait_for_safe_date():
    fd = climate.frost_dates(synthetic(), 45.0, 2026)
    before = planner.tasks_for_week(fd, fd.last_spring_median - pd.Timedelta(days=30))
    assert not any(t.crop == "Tomatoes" and t.action == "transplant outside" for t in before)
    during = planner.tasks_for_week(fd, fd.last_spring_safe + pd.Timedelta(days=7))
    assert any(t.crop == "Tomatoes" and t.action == "transplant outside" for t in during)


def test_hardy_crops_sow_early():
    fd = climate.frost_dates(synthetic(), 45.0, 2026)
    t = planner.tasks_for_week(fd, fd.last_spring_median - pd.Timedelta(days=21))
    assert any(x.crop == "Peas" for x in t)
