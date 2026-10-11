from __future__ import annotations

from dataclasses import asdict
from datetime import date

import pandas as pd

from . import climate, llm, model, planner


def build_plan(place_query: str, today: date | None = None, use_llm: bool = True, backend: str | None = None) -> dict:
    today = today or date.today()
    place = climate.geocode(place_query)
    hist = climate.load_history(place, today=today)
    fd = climate.frost_dates(hist, place.lat, today.year)
    tasks = planner.tasks_for_week(fd, today, hist)

    tonight = None
    if not fd.frost_free:
        recent = climate.recent_observed(place, 10)
        if recent is not None and len(recent) >= 8:
            m = model.train(hist, place.lat, backend)
            row = model.row_for(pd.Timestamp(today), recent["tmin"].tolist()[-8:-1], place.lat)
            tonight = {"probability": round(m.prob(row), 3), "backend": m.backend,
                       "backtest_brier": round(m.brier, 4), "baseline_brier": round(m.brier_climatology, 4)}

    plan = {
        "place": place.name, "date": today.isoformat(), "years_of_data": fd.years,
        "frost_free_climate": fd.frost_free,
        "last_spring_frost": {"median": fd.last_spring_median.isoformat(), "safe": fd.last_spring_safe.isoformat()},
        "first_fall_frost": {"median": fd.first_fall_median.isoformat(), "early": fd.first_fall_early.isoformat()},
        "frost_watch": planner.frost_watch(fd, today),
        "tonight_frost_risk": tonight,
        "tasks": [{**asdict(t), "window": [t.window[0].isoformat(), t.window[1].isoformat()]} for t in tasks],
    }
    # nothing to narrate -> skip the LLM entirely; small models pad with invented chores
    note = llm.field_note(plan) if use_llm and tasks else None
    plan["field_note"] = note or llm.fallback_note(plan)
    plan["note_source"] = f"local LLM ({llm.MODEL})" if note else ("template (nothing urgent this week)" if use_llm and not tasks else "template")
    return plan
