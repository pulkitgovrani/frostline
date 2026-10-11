"""Tonight's frost probability from tabular weather features.

Backends: TabPFN (tabular foundation model, pip install tabpfn, needs a free Prior Labs
license token) or scikit-learn gradient boosting (default, fully offline).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURES = ["doy_sin", "doy_cos", "lat", "t1", "t2", "t3", "mean7"]


def build_features(hist: pd.DataFrame, lat: float) -> tuple[pd.DataFrame, pd.Series]:
    """Row for day D uses temperatures up to D-1 and predicts frost on D."""
    h = hist.sort_values("date").reset_index(drop=True)
    t = h["tmin"]
    doy = h["date"].dt.dayofyear
    X = pd.DataFrame({
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "lat": lat, "t1": t.shift(1), "t2": t.shift(2), "t3": t.shift(3),
        "mean7": t.shift(1).rolling(7).mean(),
    })
    y = (t <= 0).astype(int)
    ok = X.notna().all(axis=1) & (h["date"].diff().dt.days == 1)
    return X[ok][FEATURES], y[ok]


def row_for(day: pd.Timestamp, recent_tmin: list[float], lat: float) -> pd.DataFrame:
    """Feature row for `day` from the 7+ most recent daily minima (oldest first)."""
    doy = day.dayofyear
    r = recent_tmin
    return pd.DataFrame([{
        "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
        "lat": lat, "t1": r[-1], "t2": r[-2], "t3": r[-3], "mean7": float(np.mean(r[-7:])),
    }])[FEATURES]


def _make(backend: str):
    if backend == "tabpfn":
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(device="cpu")
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.08, max_iter=150, random_state=0)


def _cap(X, y, backend: str, max_rows: int):
    """TabPFN is happiest with a few thousand rows; subsample for it only."""
    if backend == "tabpfn" and len(X) > max_rows:
        X = X.sample(max_rows, random_state=0)
        y = y.loc[X.index]
    return X, y


@dataclass
class FrostModel:
    clf: object
    backend: str
    brier: float
    brier_climatology: float

    def prob(self, X: pd.DataFrame) -> float:
        p = self.clf.predict_proba(X)[:, list(self.clf.classes_).index(1)] if 1 in self.clf.classes_ else [0.0]
        return float(p[0])


def train(hist: pd.DataFrame, lat: float, backend: str | None = None, max_rows: int = 3000) -> FrostModel:
    backend = backend or ("tabpfn" if os.environ.get("TABPFN_TOKEN") else "sklearn")
    X, y = build_features(hist, lat)
    # honest backtest: train on all but the last 2 years, score on the last 2
    cut = X.index.max() - 730
    tr, te = X.index <= cut, X.index > cut
    Xtr, ytr = X[tr], y[tr]
    Xtr, ytr = _cap(Xtr, ytr, backend, max_rows)
    clf = _make(backend)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(X[te])[:, 1]
    brier = float(np.mean((p - y[te]) ** 2))
    # baseline: same-day-of-year frost frequency from the training years only
    doy = hist.loc[X.index, "date"].dt.dayofyear
    freq = y[tr].groupby(doy[tr]).mean()
    base = doy[te].map(freq).fillna(y[tr].mean())
    brier_c = float(np.mean((base - y[te]) ** 2))
    full = _make(backend)
    full.fit(*_cap(X, y, backend, max_rows))
    return FrostModel(full, backend, brier, brier_c)
