from __future__ import annotations

import argparse
from datetime import date

from .core import build_plan


def main() -> None:
    ap = argparse.ArgumentParser(prog="frostline", description="This week's garden jobs from your own frost data.")
    ap.add_argument("place", help='e.g. "Minneapolis" or "Pune"')
    ap.add_argument("--date", type=date.fromisoformat, default=None, help="pretend today is YYYY-MM-DD")
    ap.add_argument("--no-llm", action="store_true", help="skip the local LLM field note")
    ap.add_argument("--backend", choices=["sklearn", "tabpfn"], default=None)
    a = ap.parse_args()
    p = build_plan(a.place, a.date, use_llm=not a.no_llm, backend=a.backend)

    print(f"\nFrostLine for {p['place']} ({p['date']}), from {p['years_of_data']} years of weather data")
    if not p["frost_free_climate"]:
        l, f = p["last_spring_frost"], p["first_fall_frost"]
        print(f"  Last spring frost: {l['median']} (safe after {l['safe']})")
        print(f"  First fall frost:  {f['median']} (could come as early as {f['early']})")
    print(f"  {p['frost_watch']}")
    t = p["tonight_frost_risk"]
    if t:
        print(f"  Tonight's frost risk: {t['probability']:.0%}  [{t['backend']}; backtest Brier {t['backtest_brier']} vs {t['baseline_brier']} climatology]")
    print("\nThis week, outside:")
    for k in p["tasks"] or []:
        flag = "  (risky: frost may still come)" if k["risky"] else ""
        print(f"  - {k['action'].capitalize()}: {k['crop']} [{k['window'][0]} to {k['window'][1]}]{flag}\n      {k['tip']}")
    if not p["tasks"]:
        print("  Nothing urgent. Walk the beds.")
    print(f"\nField note ({p['note_source']}):\n{p['field_note']}\n")


if __name__ == "__main__":
    main()
