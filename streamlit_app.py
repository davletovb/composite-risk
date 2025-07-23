# composite_risk_app.py
"""Streamlit dashboard for a composite early‑warning market‑risk indicator.

The app pulls fresh macro/market data from FRED, engineers features, trains (or
loads) a logistic‑regression‑based ensemble, and outputs a 12‑month recession /
20%‑draw‑down probability.  Replace or augment the model with XGBoost for
non‑linear lift if desired.

Run with:
    streamlit run composite_risk_app.py

Environment variables required:
    FRED_API_KEY  – get yours free at https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
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

# -----------------------------------------------------------------------------
# 1. Configuration & constants
# -----------------------------------------------------------------------------

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

MODEL_PATH = CACHE_DIR / "risk_model.pkl"
FRED_API_KEY = os.getenv("FRED_API_KEY")
START_DATE = "1960-01-01"
EVENT_HORIZON_MONTHS = 12  # 12‑month look‑ahead window

FRED_SERIES = {
    "DGS10": "10‑Year Treasury Constant Maturity Rate",
    "TB3MS": "3‑Month Treasury Bill Secondary Market Rate",
    "BAA10YM": "Moody's Seasoned Baa Corporate Bond Yield",
    "AAA10YM": "Moody's Seasoned Aaa Corporate Bond Yield",
    "UNRATE": "Unemployment Rate",
    "CIVPART": "Labor Force Participation Rate",
    "VIXCLS": "CBOE VIX Close",
    "USRECD": "NBER Recession Indicator",
    "SP500": "S&P 500 Index (Daily Close)",
    # Add additional series here as desired
}

st.set_page_config(page_title="Composite Risk Gauge", layout="wide")

# -----------------------------------------------------------------------------
# 2. Data loading helpers
# -----------------------------------------------------------------------------

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


@st.cache_data(show_spinner="Fetching and engineering data…", ttl=43200)
def build_feature_table() -> pd.DataFrame:
    raw = load_fred(FRED_SERIES)

    features = pd.DataFrame(index=raw.index)
    features["term_spread"] = compute_term_spread(raw)
    features["credit_spread"] = compute_baa_aaa_spread(raw)
    features["vix"] = raw["CBOE VIX Close"].fillna(method="ffill")
    # Sahm Rule ΔU
    rolling_min_u = raw["Unemployment Rate"].rolling(window=12, min_periods=1).min()
    features["sahm_rule"] = raw["Unemployment Rate"] - rolling_min_u

    # Momentum terms (6‑month change)
    for col in ["term_spread", "credit_spread", "vix", "sahm_rule"]:
        features[f"{col}_chg6"] = features[col] - features[col].shift(6)

    # Simple levels standardization (z‑score w/ expanding window to avoid look‑ahead)
    z = (features - features.expanding(min_periods=60).mean()) / (
        features.expanding(min_periods=60).std(ddof=0)
    )
    z.columns = [f"z_{c}" for c in z.columns]
    features = pd.concat([features, z], axis=1).dropna()

    # Target variable y_t: 1 if recession OR 20% draw‑down over next 12 m
    future_window = raw["S&P 500 Index (Daily Close)"].resample("M").last()
    future_returns = future_window.pct_change(EVENT_HORIZON_MONTHS)
    recession_forward = raw["NBER Recession Indicator"].shift(-EVENT_HORIZON_MONTHS).rolling(
        EVENT_HORIZON_MONTHS
    ).max()
    y = ((recession_forward > 0) | (future_returns <= -0.20)).astype(int)
    features["target"] = y

    return features.dropna()

# -----------------------------------------------------------------------------
# 3. Model training & evaluation
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def train_or_load_model(df: pd.DataFrame) -> Pipeline:
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]

    train_cutoff = datetime.utcnow() - relativedelta(years=5)  # keep last 5 yrs for out‑of‑sample
    X_train = X[X.index < train_cutoff]
    y_train = y[y.index < train_cutoff]

    # Simple, interpretable pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=200, penalty="l1", solver="liblinear")),
    ])
    pipe.fit(X_train, y_train)

    # Evaluate quick AUC
    auc = roc_auc_score(y_train, pipe.predict_proba(X_train)[:, 1])
    st.info(f"Model trained (AUC in‑sample ≈ {auc:.2f}).  Saved to {MODEL_PATH}.")

    joblib.dump(pipe, MODEL_PATH)
    return pipe

# -----------------------------------------------------------------------------
# 4. Streamlit UI
# -----------------------------------------------------------------------------

def main():
    st.title("🛡️  Composite Market‑Risk Gauge")
    st.caption("Probability that the next 12 months contain an NBER recession **or** ≥ 20 % S&P 500 draw‑down.")

    features = build_feature_table()
    model = train_or_load_model(features)

    X_latest = features.drop(columns=["target"]).iloc[-1:]
    prob = float(model.predict_proba(X_latest)[0, 1])

    st.metric("Current risk probability", f"{prob:.1%}")

    # Gauge ‑ simple visual with st.progress
    st.progress(min(int(prob * 100), 100))

    st.subheader("Historical back‑test")
    hist_probs = pd.Series(
        model.predict_proba(features.drop(columns=["target"]))[:, 1], index=features.index
    )
    st.line_chart(hist_probs.rename("Risk probability"))

    st.subheader("Feature values – latest snapshot")
    st.dataframe(X_latest.T.rename(columns={X_latest.index[-1]: "latest"}))

    with st.expander("📈  Feature engineering & target definition"):
        st.markdown(
            "* **term_spread**  = 10 y Treasury minus 3 m T‑bill.\n"
            "* **credit_spread**  = Baa ‑ Aaa corporate yield spread.\n"
            "* **sahm_rule**  = Unemployment rate less its 12‑m low.\n"
            "* **vix**  = CBOE volatility index.\n"
            "* Momentum terms are 6‑month changes.  Each level & change is z‑scored.\n"
            "* **target** = 1 if either condition holds in the next 12 m: NBER recession starts, or real S&P 500 draw‑down ≤ −20 %."
        )

    st.subheader("Next steps")
    st.markdown(
        "‑ Swap **LogisticRegression** for **XGBoostClassifier** to capture non‑linearities.\n"
        "‑ Schedule job via **GitHub Actions** or **Dagster** to refresh monthly & send Slack/email alerts when probability > 0.45.\n"
        "‑ Add additional macro factors (LEI, NFCI, ISM New Orders) – simply extend `FRED_SERIES` and adjust engineering.\n"
    )


if __name__ == "__main__":
    main()

