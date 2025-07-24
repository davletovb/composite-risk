# composite_risk_app.py – v3.0
"""Streamlit dashboard for a composite early‑warning market‑risk indicator
(ISM‑free build with interactive controls, feature importance & CSV export).

What’s new in **v3.0**
======================
1. **Interactive sidebar** – choose look‑ahead horizon (6‒18 m) and
   probability‑alert threshold.
2. **Feature importance tab** – shows *mean absolute* SHAP values for the
   XGBoost model (falls back to built‑in feature gain if `shap` isn’t
   installed).
3. **Download‑as‑CSV** – grab the full probability history for your own
   back‑tests.
4. **Cleaner error handling** – any missing FRED series are skipped with a
   single warning; NaNs/±∞ are sanitised before model fit.

Usage
-----
```bash
pip install --upgrade streamlit fredapi joblib scikit‑learn xgboost pandas numpy python-dateutil shap
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
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBClassifier

try:
    import shap  # optional
except ImportError:  # pragma: no cover
    shap = None

# ────────────────────────────────────────────────────────────────────────────────
# 1  Config & constants
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
# 2  Utility helpers
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


def train_models(df: pd.DataFrame) -> dict[str, object]:
    X, y = df.drop(columns=["target"]), df["target"]
    cutoff = df.index[-1] - relativedelta(years=5)
    X_train, y_train = X[X.index < cutoff], y[y.index < cutoff]

    logit = Pipeline([
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=500, penalty="l1", solver="liblinear")),
    ]).fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
        objective="binary:logistic",
    ).fit(X_train, y_train)

    return {"logit": logit, "xgb": xgb}


def get_probs(models: dict[str, object], X: pd.DataFrame) -> np.ndarray:
    return 0.5 * (
        models["logit"].predict_proba(X)[:, 1] + models["xgb"].predict_proba(X)[:, 1]
    )

# ────────────────────────────────────────────────────────────────────────────────
# 3  Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────

def main():
    st.title("🛡️ Composite Market‑Risk Gauge (v3.0)")

    # Sidebar controls
    horizon = st.sidebar.slider("Look‑ahead horizon (months)", 6, 18, DEFAULT_HORIZON, 3)
    alert_threshold = st.sidebar.slider("Alert threshold", 0.1, 0.9, 0.45, 0.05)

    raw = load_fred(FRED_SERIES)
    feats = engineer_features(raw, horizon)
    models = train_models(feats)

    X_latest = feats.drop(columns=["target"]).iloc[-1:]
    latest_prob = float(get_probs(models, X_latest)[0])

    # Top metric
    st.metric("Current ensemble probability", f"{latest_prob:.1%}", delta=None)
    st.progress(int(latest_prob * 100))

    # Alert banner
    if latest_prob >= alert_threshold:
        st.error(f"⚠️ Risk probability {latest_prob:.1%} ≥ alert threshold {alert_threshold:.0%}")

    # Tabs
    tab_hist, tab_feats, tab_imp = st.tabs(["Probability history", "Feature snapshot", "Feature importance"])

    with tab_hist:
        X_all = feats.drop(columns=["target"])
        hist = get_probs(models, X_all)
        st.line_chart(pd.Series(hist, index=feats.index, name="Prob"))
        csv = pd.Series(hist, index=feats.index, name="probability").to_csv().encode()
        st.download_button("Download CSV", csv, "risk_probability.csv", "text/csv")

    with tab_feats:
        st.dataframe(X_latest.T, use_container_width=True)

    with tab_imp:
        st.write("Feature importance – XGBoost model")
        if shap is not None:
            explainer = shap.TreeExplainer(models["xgb"], feature_perturbation="interventional")
            shap_vals = explainer.shap_values(feats.drop(columns=["target"]))
            shap_sum = pd.DataFrame({
                "feature": feats.drop(columns=["target"]).columns,
                "importance": np.abs(shap_vals).mean(axis=0),
            }).sort_values("importance", ascending=False)
            st.bar_chart(shap_sum.set_index("feature"))
        else:
            imp = models["xgb"].get_booster().get_score(importance_type="gain")
            imp_df = (
                pd.Series(imp).sort_values(ascending=False).rename("gain").to_frame()
            )
            st.bar_chart(imp_df)

    st.caption("Data: St. Louis Fed FRED.  Code MIT‑licensed.")

if __name__ == "__main__":
    main()
