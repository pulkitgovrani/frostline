"""Deterministic, explainable planting rules relative to frost dates.

The LLM never decides *when* to plant. This module does; the LLM only narrates it.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .climate import FrostDates

CROPS = json.loads((Path(__file__).with_name("crops.json")).read_text())


@dataclass
class Task:
    crop: str
    action: str       # "sow outdoors" | "start indoors" | "transplant outside" | "plant garlic"
    window: tuple[date, date]
    tip: str
    risky: bool = False  # window opens before the "safe" last-frost date
    why: str = ""


def _w(anchor: date, lo: float, hi: float) -> tuple[date, date]:
    return anchor + timedelta(weeks=lo), anchor + timedelta(weeks=hi)


def _overlaps(win: tuple[date, date], start: date, days: int = 7) -> bool:
    return win[0] < start + timedelta(days=days) and win[1] >= start


def _frost_free_tasks(hist: pd.DataFrame, start: date) -> list[Task]:
    """No frost to anchor on, so use the local cool/warm cycle: cool-season crops go in when this
    week is among the cooler ~40%'s weeks, warm-season crops in the warmer ~40%."""
    h = hist.assign(woy=hist["date"].dt.dayofyear // 7)
    weekly = h.groupby("woy")["tmin"].mean()
    w = pd.Timestamp(start).dayofyear // 7
    t = float(weekly.get(w, weekly.mean()))
    rank = float((weekly <= t).mean())  # 0 = coolest week of the year, 1 = warmest
    end = start + timedelta(days=6)
    out = []
    for c in CROPS:
        if "fall_only" in c:
            continue
        kind = c["hardy"]
        if (kind == "hardy" and rank <= 0.4) or (kind == "tender" and rank >= 0.6) or kind == "half":
            season = "cool" if kind == "hardy" else "warm"
            action = "sow outdoors" if "direct_sow" in c else "start indoors"
            out.append(Task(c["name"], action, (start, end), c["tip"],
                            why=f"{season}-season crop; this week's usual night low here is {t:.0f}C ({rank:.0%} of the way up the yearly range)"))
    return out


def tasks_for_week(fd: FrostDates, start: date, hist: pd.DataFrame | None = None) -> list[Task]:
    if fd.frost_free:
        return _frost_free_tasks(hist, start) if hist is not None else []
    out: list[Task] = []
    lf, safe = fd.last_spring_median, fd.last_spring_safe
    for c in CROPS:
        tender = c["hardy"] == "tender"
        anchor = safe if tender else lf
        if "direct_sow" in c:
            win = _w(anchor, *c["direct_sow"])
            if _overlaps(win, start):
                out.append(Task(c["name"], "sow outdoors", win, c["tip"], risky=win[0] < safe and tender,
                                why=f"{'safe' if tender else 'median'} last frost is {anchor:%b %d}"))
        if "indoor" in c:
            win = _w(lf, *c["indoor"])
            if _overlaps(win, start):
                out.append(Task(c["name"], "start indoors", win, c["tip"], why=f"{abs(c['indoor'][1])}-{abs(c['indoor'][0])} weeks before last frost"))
        if "transplant" in c:
            win = _w(anchor, *c["transplant"])
            if _overlaps(win, start):
                out.append(Task(c["name"], "transplant outside", win, c["tip"], why=f"after the safe last-frost date ({safe:%b %d})"))
        if c.get("fall_sow"):
            # sow late enough to avoid bolting in summer heat, early enough to mature before hard cold
            latest = fd.first_fall_median - timedelta(days=c["dtm"] - 14)
            win = (latest - timedelta(weeks=3), latest)
            if win[0] > safe + timedelta(weeks=8) and _overlaps(win, start):
                out.append(Task(c["name"], "sow outdoors (autumn crop)", win, c["tip"],
                                why=f"needs ~{c['dtm']} days; first fall frost is around {fd.first_fall_median:%b %d}"))
        if "fall_only" in c:
            win = _w(fd.first_fall_median, *c["fall_only"])
            if _overlaps(win, start):
                out.append(Task(c["name"], "plant", win, c["tip"], why="roots in autumn, harvest next summer"))
    out.sort(key=lambda t: (t.risky, t.window[1]))  # most urgent (closing soonest) first among non-risky
    return out


def frost_watch(fd: FrostDates, today: date) -> str | None:
    """Plain-language frost-season status for today."""
    if fd.frost_free:
        return "Your climate is essentially frost-free, so timing depends on heat and rain more than frost."
    if today < fd.last_spring_median:
        return f"Frost season isn't over: median last frost is {fd.last_spring_median:%b %d}. Keep tender plants inside."
    if today < fd.last_spring_safe:
        return f"Past the median last frost, but frost can still bite until about {fd.last_spring_safe:%b %d} (1-in-10 risk). Keep covers handy."
    if today < fd.first_fall_early:
        return f"Frost-free growing season. First fall frost is typically around {fd.first_fall_median:%b %d}."
    return f"Frost is due any time now (median first fall frost {fd.first_fall_median:%b %d}). Cover or harvest tender crops."
