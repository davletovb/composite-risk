# composite_risk_app.py – v4.0 (Scenario Lab) — Completed
"""Risk‑probability dashboard with pluggable ML models **and** a Scenario Lab.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
from fredapi import Fred
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBClassifier
from scipy.optimize import minimize

try:
    from lightgbm import LGBMClassifier  # type: ignore
except ImportError:
    LGBMClassifier = None  # type: ignore
try:
    from catboost import CatBoostClassifier  # type: ignore
except ImportError:
    CatBoostClassifier = None  # type: ignore

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
CACHE_DIR = Path(".cache"); CACHE_DIR.mkdir(exist_ok=True)
FRED_API_KEY = os.getenv("FRED_API_KEY")
START_DATE = "1960-01-01"
DEFAULT_HORIZON = 12

FRED_SERIES = {
    "DGS10": "10‑Year Treasury Constant Maturity Rate",
    "TB3MS": "3‑Month Treasury Bill Secondary Market Rate",
    "BAA": "Moody's Seasoned Baa Corporate Bond Yield",
    "AAA": "Moody's Seasoned Aaa Corporate Bond Yield",
    "UNRATE": "Unemployment Rate",
    "VIXCLS": "CBOE VIX Close",
    "USRECD": "NBER Recession Indicator",
    "SP500": "S&P 500 Index (Daily Close)",
    "USSLIND": "Philly Fed Leading Index",
    "NFCI": "Chicago Fed NFCI",
    "ANFCI": "Chicago Fed Adjusted NFCI",
}

st.set_page_config(page_title="Composite Risk Gauge", layout="wide")

# -----------------------------------------------------------------------------
# Data helpers
# -----------------------------------------------------------------------------

def load_fred():
    fred = Fred(api_key=FRED_API_KEY)
    df = pd.DataFrame({name: fred.get_series(sid, START_DATE) for sid, name in FRED_SERIES.items()})
    df.index = pd.to_datetime(df.index)
    return df.resample("M").last().ffill()


def engineer_features(raw: pd.DataFrame, horizon: int):
    feats = pd.DataFrame(index=raw.index)
    feats["term_spread"] = raw["10‑Year Treasury Constant Maturity Rate"] - raw["3‑Month Treasury Bill Secondary Market Rate"]
    feats["credit_spread"] = raw["Moody's Seasoned Baa Corporate Bond Yield"] - raw["Moody's Seasoned Aaa Corporate Bond Yield"]
    feats["vix"] = raw["CBOE VIX Close"]
    rolling_min = raw["Unemployment Rate"].rolling(12, min_periods=1).min()
    feats["sahm_rule"] = raw["Unemployment Rate"] - rolling_min
    feats["lei_6m_pct"] = raw["Philly Fed Leading Index"].pct_change(6)
    feats["nfci"] = raw["Chicago Fed NFCI"]
    feats["an_fci"] = raw["Chicago Fed Adjusted NFCI"]
    base = ["term_spread", "credit_spread", "vix", "sahm_rule", "lei_6m_pct", "nfci"]
    for col in base:
        feats[f"{col}_chg6"] = feats[col] - feats[col].shift(6)
    z = (feats - feats.expanding(60).mean()) / feats.expanding(60).std(ddof=0)
    feats = pd.concat([feats, z.add_prefix("z_")], axis=1)
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats.dropna(inplace=True)
    fut_ret = raw["S&P 500 Index (Daily Close)"].resample("M").last().pct_change(horizon)
    rec = raw["NBER Recession Indicator"].shift(-horizon).rolling(horizon).max()
    feats["target"] = ((rec > 0) | (fut_ret <= -0.20)).astype(int)
    return feats

# -----------------------------------------------------------------------------
# Scenario builders
# -----------------------------------------------------------------------------

def build_dotcom(base):
    s = base.copy()
    s["term_spread"] += 1
    s["credit_spread"] += 1.5
    s["vix"] += 10
    s["nfci"] += 0.5
    return s

def build_covid(base):
    s = base.copy()
    s["term_spread"] += 0.8
    s["credit_spread"] += 3
    s["vix"] += 40
    s["nfci"] += 1.2
    return s

SCENARIOS = {"Dot‑com 2001": build_dotcom, "Covid‑flash 2020": build_covid}

# -----------------------------------------------------------------------------
# Model builders
# -----------------------------------------------------------------------------

def build_models(X, y, names):
    models = {}
    if "Logit" in names:
        models["Logit"] = Pipeline([("std", StandardScaler()), ("clf", LogisticRegression(max_iter=600, solver="liblinear", penalty="l1"))]).fit(X, y)
    if "Random Forest" in names:
        models["Random Forest"] = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42).fit(X, y)
    if "Gradient Boosting" in names:
        models["Gradient Boosting"] = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42).fit(X, y)
    if "XGBoost" in names:
        models["XGBoost"] = XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8, colsample_bytree=0.8, eval_metric="logloss", random_state=42).fit(X, y)
    if "LightGBM" in names and LGBMClassifier:
        models["LightGBM"] = LGBMClassifier(n_estimators=400, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, objective="binary").fit(X, y)
    if "CatBoost" in names and CatBoostClassifier:
        models["CatBoost"] = CatBoostClassifier(iterations=400, learning_rate=0.03, depth=4, loss_function="Logloss", verbose=False).fit(X, y)
    return models

# -----------------------------------------------------------------------------
# Probability helpers
# -----------------------------------------------------------------------------

def prob(models, X, sel):
    if sel == "Ensemble":
        return np.mean([m.predict_proba(X)[:, 1] for m in models.values()], axis=0)
    return models[sel].predict_proba(X)[:, 1]

# -----------------------------------------------------------------------------
# Reverse stress
# -----------------------------------------------------------------------------

def reverse_stress(base_row, model, threshold):
    x0 = np.zeros(len(base_row))
    bounds = [(-3, 3)] * len(base_row)
    def obj(d):
        p = model.predict_proba(pd.DataFrame([base_row + d]))[0, 1]
        return np.sum(np.abs(d)) + 100 * max(0, threshold - p) ** 2
    res = minimize(obj, x0, bounds=bounds)
    return base_row + res.x, model.predict_proba(pd.DataFrame([base_row + res.x]))[0, 1]

# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------

def main():
    raw = load_fred()
    horizon = st.sidebar.slider("Forecast horizon (m)", 6, 18, DEFAULT_HORIZON, 3)
    feats = engineer_features(raw, horizon)
    X, y = feats.drop(columns=["target"]), feats["target"]
    split = feats.index[-1] - relativedelta(years=5)
    X_train, y_train = X[X.index < split], y[y.index < split]

    model_opts = ["Logit", "Random Forest", "Gradient Boosting", "XGBoost"]
    if LGBMClassifier: model_opts.append("LightGBM")
    if CatBoostClassifier: model_opts.append("CatBoost")
    model_opts.append("Ensemble")

    model_sel = st.sidebar.selectbox("Model", model_opts, index=len(model_opts) - 1)

    choices = model_opts[:-1] if model_sel == "Ensemble" else [model_sel]
    models = build_models(X_train, y_train, choices)
    latest = X.iloc[[-1]]
    p_now = prob(models, latest, model_sel)[0]

    st.title("🛡️ Composite Risk Gauge v4.0")
    st.metric("Current probability", f"{p_now:.1%}")

    # Scenario Lab
    st.header("Scenario Lab")
    scen_name = st.selectbox("Choose scenario", list(SCENARIOS.keys()) + ["Monte‑Carlo", "Reverse stress‑test"])
    if scen_name in SCENARIOS:
        shocked = SCENARIOS[scen_name](latest.iloc[0])
        p_s = prob(models, pd.DataFrame([shocked]), model_sel)[0]
        st.write(f"Probability under **{scen_name}** shock: **{p_s:.1%}** vs baseline {p_now:.1%}")
        st.dataframe(shocked.T.rename({0: "Shocked"}))
    elif scen_name == "Monte‑Carlo":
        n = st.number_input("Simulations", 100, 3000, 1000)
        cov = X_train.cov()
        draws = np.random.multivariate_normal(np.zeros(len(X.columns)), cov.values, size=n)
        sims = latest.values + draws
        p_mc = prob(models, pd.DataFrame(sims, columns=X.columns), model_sel)
        st.write(f"5‑95th percentile: {np.percentile(p_mc,5):.1%} – {np.percentile(p_mc,95):.1%}")
        st.line_chart(pd.Series(sorted(p_mc)))
    else:
        thr = st.slider("Target prob", 0.5, 0.9, 0.6, 0.05)
        shocked, p_new = reverse_stress(latest.iloc[0].values, models[choices[0]], thr)
        st.write(f"Min shock (L1 norm) to hit {thr:.0%}: prob={p_new:.1%}")
        st.dataframe(pd.Series(shocked, index=X.columns, name="Shock"))

if __name__ == "__main__":
    main()
