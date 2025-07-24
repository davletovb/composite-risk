# composite_risk_app.py – v3.1 (multi‑model)
"""Composite early‑warning dashboard with **pluggable ML models**.

New in v3.1
============
1. **Model picker** in the sidebar – choose *Logit*, *XGBoost*, *Random Forest*,
   *Gradient Boosting*, or an **Ensemble** that averages all available models.
2. Training pipeline automatically builds whichever models are selected and
   caches them in .cache/.
3. Probability history & feature‑importance adapts to the chosen model.

Installation (adds scipy & scikit‑learn GB/forest dependencies already present):
```bash
pip install --upgrade streamlit fredapi joblib scikit‑learn xgboost shap pandas numpy python-dateutil
export FRED_API_KEY=YOUR_KEY
streamlit run composite_risk_app.py
```
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBClassifier

try:
    import shap  # optional
except ImportError:
    shap = None

# ────────────────────────────────────────────────────────────────────────────────
# Config & constants
# ────────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path(".cache"); CACHE_DIR.mkdir(exist_ok=True)
MODEL_PATH = CACHE_DIR / "risk_models.pkl"
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

# ────────────────────────────────────────────────────────────────────────────────
# Data helpers
# ────────────────────────────────────────────────────────────────────────────────

def load_fred(series_map: dict[str, str]) -> pd.DataFrame:
    fred = Fred(api_key=FRED_API_KEY)
    data, skipped = {}, []
    for sid, name in series_map.items():
        try:
            data[name] = fred.get_series(sid, observation_start=START_DATE)
        except ValueError:
            skipped.append(sid)
    if skipped:
        st.warning("Skipped missing series: " + ", ".join(skipped))
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    return df.resample("M").last().ffill()


def engineer_features(raw: pd.DataFrame, horizon: int) -> pd.DataFrame:
    feats = pd.DataFrame(index=raw.index)
    feats["term_spread"] = raw[FRED_SERIES["DGS10"]] - raw[FRED_SERIES["TB3MS"]]
    feats["credit_spread"] = raw[FRED_SERIES["BAA"]] - raw[FRED_SERIES["AAA"]]
    feats["vix"] = raw[FRED_SERIES["VIXCLS"]]
    rolling_min_u = raw[FRED_SERIES["UNRATE"]].rolling(12, min_periods=1).min()
    feats["sahm_rule"] = raw[FRED_SERIES["UNRATE"]] - rolling_min_u
    feats["lei_6m_pct"] = raw[FRED_SERIES["USSLIND"]].pct_change(6)
    feats["nfci"] = raw[FRED_SERIES["NFCI"]]
    feats["an_fci"] = raw[FRED_SERIES["ANFCI"]]

    base_cols = ["term_spread", "credit_spread", "vix", "sahm_rule", "lei_6m_pct", "nfci"]
    for col in base_cols:
        feats[f"{col}_chg6"] = feats[col] - feats[col].shift(6)

    z = (feats - feats.expanding(60).mean()) / feats.expanding(60).std(ddof=0)
    z.columns = [f"z_{c}" for c in z.columns]
    feats = pd.concat([feats, z], axis=1)
    feats.replace([np.inf, -np.inf], np.nan, inplace=True)
    feats.dropna(inplace=True)

    # Target
    future_sp = raw[FRED_SERIES["SP500"]].resample("M").last()
    fut_ret = future_sp.pct_change(horizon)
    rec_flag = raw[FRED_SERIES["USRECD"]].shift(-horizon).rolling(horizon).max()
    feats["target"] = ((rec_flag > 0) | (fut_ret <= -0.20)).astype(int)
    return feats

# ────────────────────────────────────────────────────────────────────────────────
# Model factory
# ────────────────────────────────────────────────────────────────────────────────

def build_models(X_train: pd.DataFrame, y_train: pd.Series, choices: list[str]):
    models: dict[str, object] = {}

    if "Logit" in choices:
        logit = Pipeline([
            ("std", StandardScaler()),
            ("clf", LogisticRegression(max_iter=600, solver="liblinear", penalty="l1")),
        ]).fit(X_train, y_train)
        models["Logit"] = logit

    if "Random Forest" in choices:
        rf = RandomForestClassifier(n_estimators=400, max_depth=6, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        models["Random Forest"] = rf

    if "Gradient Boosting" in choices:
        gb = GradientBoostingClassifier(n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42)
        gb.fit(X_train, y_train)
        models["Gradient Boosting"] = gb

    if "XGBoost" in choices:
        xgb = XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=4,
        ).fit(X_train, y_train)
        models["XGBoost"] = xgb

    return models


def predict_prob(models: dict[str, object], X: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "Ensemble":
        probs = np.column_stack([m.predict_proba(X)[:, 1] for m in models.values()])
        return probs.mean(axis=1)
    return list(models.values())[0].predict_proba(X)[:, 1]

# ────────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────

def main():
    st.title("🛡️ Composite Market‑Risk Gauge (v3.1 – multi‑model)")

    # Sidebar controls
    horizon = st.sidebar.slider("Look‑ahead horizon (months)", 6, 18, DEFAULT_HORIZON, 3)
    model_choice = st.sidebar.selectbox(
        "Model", ["Logit", "XGBoost", "Random Forest", "Gradient Boosting", "Ensemble"], index=4
    )
    alert_thr = st.sidebar.slider("Alert threshold", 0.1, 0.9, 0.45, 0.05)

    raw = load_fred(FRED_SERIES)
    feats = engineer_features(raw, horizon)
    X, y = feats.drop(columns=["target"]), feats["target"]
    cutoff = feats.index[-1] - relativedelta(years=5)
    X_tr, y_tr = X[X.index < cutoff], y[y.index < cutoff]

    # Use cache keyed on model choice & horizon
    cache_key = f"{model_choice}_{horizon}.pkl"
    cache_file = CACHE_DIR / cache_key
    if cache_file.exists():
        models = joblib.load(cache_file)
    else:
        choices = [model_choice] if model_choice != "Ensemble" else ["Logit", "XGBoost", "Random Forest", "Gradient Boosting"]
        models = build_models(X_tr, y_tr, choices)
        joblib.dump(models, cache_file)

    latest_prob = float(predict_prob(models, X.iloc[-1:], model_choice)[0])

    st.metric("Current probability", f"{latest_prob:.1%}")
    st.progress(int(latest_prob * 100))
    if latest_prob >= alert_thr:
        st.error(f"⚠️ Prob {latest_prob:.1%} ≥ {alert_thr:.0%}")

    # History tab
    tab_hist, tab_feat, tab_imp = st.tabs(["Probability history", "Feature snapshot", "Importance"])
    with tab_hist:
        hist = predict_prob(models, X, model_choice)
        st.line_chart(pd.Series(hist, index=feats.index))

    with tab_feat:
        st.dataframe(X.iloc[-1:].T)

    with tab_imp:
        st.write(f"Feature importance – {model_choice}")
        if model_choice == "XGBoost" and shap is not None:
            explainer = shap.TreeExplainer(models["XGBoost"], feature_perturbation="interventional")
            shap_vals = explainer.shap_values(X_tr)
            shap_sum = pd.DataFrame({
                "feature": X.columns,
                "importance": np.abs(shap_vals).mean(axis=0),
            }).sort_values("importance", ascending=False)
            st.bar_chart(shap_sum.set_index("feature"))
        elif model_choice in ["Random Forest", "Gradient Boosting"]:
            imp = models[model_choice].feature_importances_
            st.bar_chart(pd.Series(imp, index=X.columns))
        elif model_choice == "Logit":
            coef = models["Logit"].named_steps["clf"].coef_[0]
            st.bar_chart(pd.Series(np.abs(coef), index=X.columns))
        elif model_choice == "Ensemble":
            st.write("Importance varies per model – view individual model tabs for detail.")

    st.caption("Data: FRED.  Code MIT‑licensed.")

if __name__ == "__main__":
    main()
