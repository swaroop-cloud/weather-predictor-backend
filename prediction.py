"""
Turns 8 days of raw daily weather (oldest first, today last) into the
exact 19-feature vector the models were trained on, then predicts
tomorrow's TMAX/TMIN and extreme-weather risk.

Feature convention (matches training):
  days[7] = "today" (the day features are computed FOR)
  days[6] = yesterday   -> *_lag1
  days[5] = 2 days ago  -> *_lag2
  days[4] = 3 days ago  -> *_lag3
  days[0:7] (7 days ending yesterday) -> rolling means/sums
  Target = tomorrow = today + 1 day
"""

import math
from datetime import timedelta

import pandas as pd

from model_loader import get_models


def build_features(days: list, snwd_yesterday: float = 0.0) -> dict:
    if len(days) != 8:
        raise ValueError("Exactly 8 consecutive days are required (oldest first, today last).")

    today = days[7]
    doy = pd.Timestamp(today["date"]).dayofyear
    month = pd.Timestamp(today["date"]).month

    tmax_vals = [d["tmax"] for d in days]
    tmin_vals = [d["tmin"] for d in days]
    prcp_vals = [d["prcp"] for d in days]

    features = {
        "doy_sin": math.sin(2 * math.pi * doy / 365.25),
        "doy_cos": math.cos(2 * math.pi * doy / 365.25),
        "month": month,
        "tmax_lag1": tmax_vals[6],
        "tmax_lag2": tmax_vals[5],
        "tmax_lag3": tmax_vals[4],
        "tmin_lag1": tmin_vals[6],
        "tmin_lag2": tmin_vals[5],
        "tmin_lag3": tmin_vals[4],
        "prcp_lag1": prcp_vals[6],
        "prcp_lag2": prcp_vals[5],
        "prcp_lag3": prcp_vals[4],
        "tmax_roll3": sum(tmax_vals[4:7]) / 3,
        "tmax_roll7": sum(tmax_vals[0:7]) / 7,
        "tmin_roll3": sum(tmin_vals[4:7]) / 3,
        "tmin_roll7": sum(tmin_vals[0:7]) / 7,
        "prcp_roll7": sum(prcp_vals[0:7]),
        "range_lag1": tmax_vals[6] - tmin_vals[6],
        "snwd_lag1": snwd_yesterday,
    }
    return features


def predict_tomorrow(days: list, snwd_yesterday: float = 0.0) -> dict:
    models = get_models()
    feats = build_features(days, snwd_yesterday)

    row = pd.DataFrame([feats])[models["features"]]

    tmax_pred = float(models["tmax"].predict(row)[0])
    tmin_pred = float(models["tmin"].predict(row)[0])

    heat_proba = float(models["heat"].predict_proba(row)[0][1])
    freeze_proba = float(models["freeze"].predict_proba(row)[0][1])
    rain_proba = float(models["rain"].predict_proba(row)[0][1])

    today_date = pd.Timestamp(days[7]["date"])
    tomorrow_date = (today_date + timedelta(days=1)).strftime("%Y-%m-%d")

    return {
        "prediction_date": tomorrow_date,
        "temperature": {
            "tmax_f": round(tmax_pred, 1),
            "tmin_f": round(tmin_pred, 1),
        },
        "risks": [
            {"label": "Extreme heat", "detail": "TMAX \u2265 90\u00b0F", "probability": round(heat_proba, 4)},
            {"label": "Freeze", "detail": "TMIN \u2264 32\u00b0F", "probability": round(freeze_proba, 4)},
            {"label": "Heavy rain", "detail": "PRCP \u2265 1.0 in", "probability": round(rain_proba, 4)},
        ],
        "note": "Heavy rain risk is not reliably predictable from this feature set (test ROC-AUC ~0.49) \u2014 shown for completeness, not as a trustworthy signal.",
    }
