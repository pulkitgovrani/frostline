---
title: FrostLine: a garden planner that runs on frost data and a local LLM
tags: devchallenge, hf26challenge, opensource, python
---

*This is a submission for the Hacktoberfest Open-Source AI Challenge: Week 1 ("Touch Grass").*

## What I Built
FrostLine tells you what to do in the garden **this week**. You type your town. It pulls 15 years of nightly lows, works out your median and 1-in-10 "safe" last spring frost and first fall frost, and lists what to sow, start indoors or transplant now, plus tonight's frost risk. A local LLM turns the plan into a short field note that always ends by sending you outside. It is for anyone with a patch of soil, a balcony or an allotment who is tired of generic "plant in spring" advice.

## Demo
<!-- TODO: add a screen recording of the web UI (python -m frostline 8765) -->

## Code
{% embed https://github.com/pulkitgovrani/frostline %}

## How I Built It
- **Open-weight LLM:** Gemma (`gemma4:e4b`) through Ollama, running locally. The LLM only narrates. Planting dates come from code, because small models invent dates. I caught mine inventing chores (compost, gloves) when the plan was empty, so empty weeks now skip the LLM and use a template.
- **Rules, not vibes:** `planner.py` anchors ~17 crops on the frost dates. Tender crops wait for the 1-in-10 safe date; hardy ones use the median. Frost-free places (I tested Pune) use the local cool/warm cycle instead.
- **Frost model:** a classifier predicts tonight's frost from day of year, latitude and recent lows. On two held-out years it scored a Brier of 0.054 vs 0.068 for a plain day-of-year baseline (Minneapolis), so it does beat climatology.
- Open-Meteo data is cached on disk, so after the first run for a town it works with no signal.

## Why Does Open Innovation Matter?
- **Offline:** the garden has no wifi. Weights and cached data live on the device.
- **Your location stays yours:** your town and garden never reach a server I don't control.
- **Swappable:** `FROSTLINE_MODEL=llama3.2` changes the narrator with no code change, and the frost model can swap sklearn for TabPFN.
- **Free to run:** no API bill for a weekend hobby.

## My Agent Session
<!-- optional: DevRelay link -->

## Prize Categories
<!-- Only list a category you really qualify for. TabPFN is supported by the code but needs a Prior Labs token; I have not run it. -->
