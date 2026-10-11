# FrostLine

Live demo: https://frostline-h8z9.onrender.com (free tier, first load may take ~50s; no local LLM there, so the note is templated)

This week's garden jobs, computed from your own town's frost history, with a local open-weight LLM writing the field note.
Weather history is fetched once (Open-Meteo, no key) and cached in `data/`, so afterwards it works without signal.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
ollama pull gemma4:e4b            # any Ollama model works: FROSTLINE_MODEL=llama3.2 ...
.venv/bin/python -m frostline.cli "Minneapolis"          # CLI
.venv/bin/python -m frostline 8765                       # web UI on http://localhost:8765
.venv/bin/python -m pytest -q
```

## How it works
- `climate.py`: 15 years of daily minimum temperature -> median and 1-in-10 safe last-spring / first-fall frost dates (hemisphere-aware).
- `planner.py`: transparent rules for ~17 crops anchored on those dates. Tender crops wait for the *safe* date. Frost-free climates use the local cool/warm cycle instead.
- `model.py`: tonight's frost probability from tabular features. Default is scikit-learn gradient boosting; set `TABPFN_TOKEN` to use TabPFN. It is backtested on the last 2 held-out years against a day-of-year climatology baseline.
- `llm.py`: Ollama writes the narration. It never picks dates; if it is not running (or there are no tasks) a template is used.
