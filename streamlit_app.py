# composite_risk_app.py – v2
"""Streamlit dashboard for a composite early‑warning market‑risk indicator (enhanced).

Adds:
    • Short‑leading indicators: Philly Fed Leading Index (USSLIND), Chicago Fed
      NFCI / Adjusted‑NFCI, ISM New Orders – Inventories spread.
    • Gradient‑Boosted Trees (XGBoost) ensemble with the original L1‑Logit.
    • Ensemble probability (mean of the two models) plotted & displayed.

Run with:
    pip install streamlit fredapi joblib scikit‑learn xgboost pandas numpy python-dateutil
    export FRED_API_KEY=YOUR_KEY
    streamlit run composite_risk_app.py

Environment:
    FRED_API_KEY – get yours at <https://fred.stlouisfed.org>
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
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
from xgboost import XGBClassifier

# ────────────────────────────────────────────────────────────────────────────────
# 1  Config & constants
# ────────────────────────────────────────────────────────────────────────────────

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
MODEL_PATH = CACHE_DIR / "risk_models.pkl"
FRED_API_KEY = os.getenv("FRED_API_KEY")
START_DATE = "1960-01-01"
EVENT_HORIZON_MONTHS = 12  # 12‑month look‑ahead window

# Base & new series pulled from FRED
FRED_SERIES = {
    # Term‑structure & credit
    "DGS10": "10‑Year Treasury Constant Maturity Rate",
    "TB3MS": "3‑Month Treasury Bill Secondary Market Rate",
    "BAA10YM": "Moody's Seasoned Baa Corporate Bond Yield",
    "AAA10YM": "Moody's Seasoned Aaa Corporate Bond Yield",
    # Labor & vol
    "UNRATE": "Unemployment Rate",
    "VIXCLS": "CBOE VIX Close",
    # Recession flag & equity index
    "USRECD": "NBER Recession Indicator",
    "SP500": "S&P 500 Index (Daily Close)",
    # NEW Short‑leading & financial‑conditions
    "USSLIND": "Philly Fed Leading Index",
    "NFCI": "Chicago Fed NFCI",
    "ANFCI": "Chicago Fed Adjusted NFCI",
    "NAPMNOI": "ISM New Orders Index",       # Manufacturing New Orders
    "NAPMII": "ISM Inventories Index",       # Manufacturing Inventories
}

st.set_page_config(page_title="Composite Risk Gauge", layout="wide")

# ────────────────────────────────────────────────────────────────────────────────
# 2  Data loading helpers
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading data from FRED…", ttl=43200)
def load_fred(series: dict[str, str], start: str = START_DATE) -> pd.DataFrame:
    fred = Fred(api_key=FRED_API_KEY)
    df = pd.DataFrame({sid: fred.get_series(sid, observation_start=start) for sid in series})
    df.index = pd.to_datetime(df.index)
    df = df.resample("M").last()
    df.rename(columns=series, inplace=True)
    return df


def compute_term_spread(df: pd.DataFrame) -> pd.Series:
    return df["10‑Year Treasury Constant Maturity Rate"] - df["3‑Month Treasury Bill Secondary Market Rate"]


def compute_baa_aaa_spread(df: pd.DataFrame) -> pd.Series:
    return df["Moody's Seasoned Baa Corporate Bond Yield"] - df["Moody's Seasoned Aaa Corporate Bond Yield"]


@st.cache_data(show_spinner="Engineering features…", ttl=43200)
def build_feature_table() -> pd.DataFrame:
    raw = load_fred(FRED_SERIES)

    # Forward‑fill for missing monthly observations (esp. VIX prior to 1990)
    raw = raw.ffill()

    feats = pd.DataFrame(index=raw.index)
    feats["term_spread"] = compute_term_spread(raw)
    feats["credit_spread"] = compute_baa_aaa_spread(raw)
    feats["vix"] = raw["CBOE VIX Close"]

    # Sahm rule ΔU
    rolling_min_u = raw["Unemployment Rate"].rolling(window=12, min_periods=1).min()
    feats["sahm_rule"] = raw["Unemployment Rate"] - rolling_min_u

    # NEW indicators
    feats["lei_6m_pct"] = raw["Philly Fed Leading Index"].pct_change(6)
    feats["nfci"] = raw["Chicago Fed NFCI"]
    feats["an_fci"] = raw["Chicago Fed Adjusted NFCI"]
    feats["ism_spread"] = raw["ISM New Orders Index"] - raw["ISM Inventories Index"]

    # Momentum (6‑month change) for selected cols
    momentum_cols = [
        "term_spread",
        "credit_spread",
        "vix",
        "sahm_rule",
        "lei_6m_pct",
        "nfci",
        "ism_spread",
    ]
    for col in momentum_cols:
        feats[f"{col}_chg6"] = feats[col] - feats[col].shift(6)

    # Z‑scores with expanding window (avoid look‑ahead)
    z = (feats - feats.expanding(min_periods=60).mean()) / (
        feats.expanding(min_periods=60).std(ddof=0)
    )
    z.columns = [f"z_{c}" for c in z.columns]

    feats = pd.concat([feats, z], axis=1).dropna()

    # Target variable
    future_sp = raw["S&P 500 Index (Daily Close)"].resample("M").last()
    future_returns = future_sp.pct_change(EVENT_HORIZON_MONTHS)
    recession_forward = raw["NBER Recession Indicator"].shift(-EVENT_HORIZON_MONTHS).rolling(
        EVENT_HORIZON_MONTHS
    ).max()
    y = ((recession_forward > 0) | (future_returns <= -0.20)).astype(int)
    feats["target"] = y

    return feats.dropna()

# ────────────────────────────────────────────────────────────────────────────────
# 3  Model training & caching
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def train_or_load_models(df: pd.DataFrame):
    """Returns a dict with 'logit' and 'xgb' models."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]

    train_cutoff = datetime.utcnow() - relativedelta(years=5)  # keep 5 yrs OOS
    X_train = X[X.index < train_cutoff]
    y_train = y[y.index < train_cutoff]

    # Logistic regression (interpretable baseline)
    logit_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=400, penalty="l1", solver="liblinear")),
    ])
    logit_pipe.fit(X_train, y_train)

    # XGBoost classifier (non‑linear lift)
    xgb_model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=350,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
    )
    xgb_model.fit(X_train, y_train)

    # Quick in‑sample metrics
    logit_auc = roc_auc_score(y_train, logit_pipe.predict_proba(X_train)[:, 1])
    xgb_auc = roc_auc_score(y_train, xgb_model.predict_proba(X_train)[:, 1])
    st.info(f"Models trained – Logit AUC {logit_auc:.2f} | XGB AUC {xgb_auc:.2f}.  Cached to {MODEL_PATH}.")

    models = {"logit": logit_pipe, "xgb": xgb_model}
    joblib.dump(models, MODEL_PATH)
    return models

# ────────────────────────────────────────────────────────────────────────────────
# 4  Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────

def main():
    st.title("🛡️ Composite Market‑Risk Gauge (v2)")
    st.caption("Probability that the next 12 months contain an NBER recession **or** ≥ 20 % S&P draw‑down.")

    feats = build_feature_table()
    models = train_or_load_models(feats)

    X_latest = feats.drop(columns=["target"]).iloc[-1:]
    prob_logit = float(models["logit"].predict_proba(X_latest)[0, 1])
    prob_xgb = float(models["xgb"].predict_proba(X_latest)[0, 1])
    prob = 0.5 * (prob_logit + prob_xgb)

    st.metric("Current ensemble probability", f"{prob:.1%}")
    st.progress(min(int(prob * 100), 100))

    # Historical ensemble
    X_all = feats.drop(columns=["target"])
    hist_prob = 0.5 * (
        models["logit"].predict_proba(X_all)[:, 1] + models["xgb"].predict_proba(X_all)[:, 1]
    )
    st.subheader("Historical back‑test (ensemble)")
    st.line_chart(pd.Series(hist_prob, index=feats.index, name="Ensemble risk prob"))

    st.subheader("Latest feature snapshot")
    st.dataframe(X_latest.T.rename(columns={X_latest.index[-1]: "Latest"}))

    with st.expander("Feature engineering & target definition"):
        st.markdown(
            "* **term_spread** = 10‑y − 3‑m Treasury.\n"
            "* **credit_spread** = Baa − Aaa IG yield.\n"
            "* **sahm_rule** = U‑rate minus 12‑m low.\n"
            "* **lei_6m_pct** = 6‑month %Δ Philly Fed Leading Index.\n"
            "* **nfci / an_fci** = Chicago Fed financial‑conditions.\n"
            "* **ism_spread** = ISM New Orders − Inventories.\n"
            "* Each level and 6‑month change is z‑scored via expanding‑window mean/std to prevent look‑ahead.\n"
            "* **Target** = 1 if within next 12 m an NBER recession starts *or* real S&P draw‑down ≤ −20 %."
        )

    st.subheader("Next steps ➜")
    st.markdown(
        "‑ Wire a GitHub Actions workflow to refresh monthly and warn when probability > 0.45.\n"
        "‑ Back‑test a regime‑switching 60/40 vs. defensive tilt using this ensemble.\n"
        "‑ Experiment with SHAP values for the XGBoost model to see which variables drive changes over time."
    )


if __name__ == "__main__":
    main()

