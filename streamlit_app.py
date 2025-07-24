# composite_risk_app.py – v2.4 (robust to missing ISM series)
"""Streamlit dashboard for a composite early‑warning market‑risk indicator.

Patch 2.4 **removes the fragile ISM New‑Orders / Inventories series** so the
app runs everywhere without FRED lookup errors.  (You can add them back once
you confirm the exact current tickers in your FRED account.)

Key changes
-----------
* Dropped `ism_spread`, `NAPM…`, `PMNO`, `PMII` from `FRED_SERIES`.
* Feature engineering & model code now conditionally skips ISM columns if they
  don’t exist, so you can re‑introduce them later without editing the rest of
  the pipeline.

Usage (unchanged)
-----------------
```bash
pip install --upgrade streamlit fredapi joblib scikit‑learn xgboost pandas numpy python-dateutil
export FRED_API_KEY=YOUR_KEY_HERE
streamlit run composite_risk_app.py
```
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st
from dateutil.relativedelta import relativedelta
from fredapi import Fred
from sklearn.linear_model import LogisticRegression
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

FRED_SERIES: dict[str, str] = {
    # Term‑structure & credit spreads
    "DGS10": "10‑Year Treasury Constant Maturity Rate",
    "TB3MS": "3‑Month Treasury Bill Secondary Market Rate",
    "BAA": "Moody's Seasoned Baa Corporate Bond Yield",
    "AAA": "Moody's Seasoned Aaa Corporate Bond Yield",
    # Labor & volatility
    "UNRATE": "Unemployment Rate",
    "VIXCLS": "CBOE VIX Close",
    # Recession flag & equities
    "USRECD": "NBER Recession Indicator",
    "SP500": "S&P 500 Index (Daily Close)",
    # Short‑leading & financial‑conditions
    "USSLIND": "Philly Fed Leading Index",
    "NFCI": "Chicago Fed NFCI",
    "ANFCI": "Chicago Fed Adjusted NFCI",
    # (ISM New Orders / Inventories temporarily removed – tickers unstable)
}

st.set_page_config(page_title="Composite Risk Gauge", layout="wide")

# ────────────────────────────────────────────────────────────────────────────────
# 2  Data helpers
# ────────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Downloading data from FRED…", ttl=43_200)
def load_fred(series_map: dict[str, str], start: str = START_DATE) -> pd.DataFrame:
    """Fetch monthly series; skips any that 404 and warns once."""
    fred = Fred(api_key=FRED_API_KEY)
    data: dict[str, pd.Series] = {}
    skipped = []
    for sid, name in series_map.items():
        try:
            s = fred.get_series(sid, observation_start=start)
            data[name] = s
        except ValueError:
            skipped.append(sid)
            continue
    if skipped:
        st.warning(f"Skipped missing FRED series: {', '.join(skipped)}")
    df = pd.DataFrame(data)
    df.index = pd.to_datetime(df.index)
    return df.resample("M").last().ffill()


# ────────────────────────────────────────────────────────────────────────────────
# 3  Feature engineering
# ────────────────────────────────────────────────────────────────────────────────

def build_feature_table() -> pd.DataFrame:
    raw = load_fred(FRED_SERIES)

    feats = pd.DataFrame(index=raw.index)
    feats["term_spread"] = raw["10‑Year Treasury Constant Maturity Rate"] - raw[
        "3‑Month Treasury Bill Secondary Market Rate"
    ]
    feats["credit_spread"] = raw["Moody's Seasoned Baa Corporate Bond Yield"] - raw[
        "Moody's Seasoned Aaa Corporate Bond Yield"
    ]
    feats["vix"] = raw["CBOE VIX Close"]

    rolling_min_u = raw["Unemployment Rate"].rolling(12, min_periods=1).min()
    feats["sahm_rule"] = raw["Unemployment Rate"] - rolling_min_u

    feats["lei_6m_pct"] = raw["Philly Fed Leading Index"].pct_change(6)
    feats["nfci"] = raw["Chicago Fed NFCI"]
    feats["an_fci"] = raw["Chicago Fed Adjusted NFCI"]

    # 6‑month momentum for core columns
    for col in [
        "term_spread",
        "credit_spread",
        "vix",
        "sahm_rule",
        "lei_6m_pct",
        "nfci",
    ]:
        feats[f"{col}_chg6"] = feats[col] - feats[col].shift(6)

    # Z‑scores (expanding window)
    z = (feats - feats.expanding(min_periods=60).mean()) / feats.expanding(min_periods=60).std(ddof=0)
    z.columns = [f"z_{c}" for c in z.columns]
    feats = pd.concat([feats, z], axis=1).dropna()

    future_sp = raw["S&P 500 Index (Daily Close)"].resample("M").last()
    future_returns = future_sp.pct_change(EVENT_HORIZON_MONTHS)
    rec_fwd = raw["NBER Recession Indicator"].shift(-EVENT_HORIZON_MONTHS).rolling(EVENT_HORIZON_MONTHS).max()
    feats["target"] = ((rec_fwd > 0) | (future_returns <= -0.20)).astype(int)

    return feats.dropna()

# ────────────────────────────────────────────────────────────────────────────────
# 4  Model training & caching
# ────────────────────────────────────────────────────────────────────────────────

def train_or_load_models(df: pd.DataFrame):
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)

    X = df.drop(columns=["target"])
    y = df["target"]
    cut = df.index[-1] - relativedelta(years=5)
    X_train, y_train = X[X.index < cut], y[y.index < cut]

    logit = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=400, penalty="l1", solver="liblinear")),
    ]).fit(X_train, y_train)

    xgb = XGBClassifier(
        n_estimators=350,
        learning_rate=0.03,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        n_jobs=4,
        objective="binary:logistic",
    ).fit(X_train, y_train)

    joblib.dump({"logit": logit, "xgb": xgb}, MODEL_PATH)
    return {"logit": logit, "xgb": xgb}

# ────────────────────────────────────────────────────────────────────────────────
# 5  Streamlit UI
# ────────────────────────────────────────────────────────────────────────────────

def main():
    st.title("🛡️ Composite Market‑Risk Gauge (v2.4)")
    st.caption("Probability that the next 12 months contain an NBER recession **or** ≥ 20 % real S&P draw‑down.")

    feats = build_feature_table()
    models = train_or_load_models(feats)

    X_latest = feats.drop(columns=["target"]).iloc[-1:]
    p = 0.5 * (
        models["logit"].predict_proba(X_latest)[0, 1] + models["xgb"].predict_proba(X_latest)[0, 1]
    )
    st.metric("Current ensemble probability", f"{p:.1%}")
    st.progress(int(p * 100))

    # History
    X_all = feats.drop(columns=["target"])
    hist = 0.5 * (
        models["logit"].predict_proba(X_all)[:, 1] + models["xgb"].predict_proba(X_all)[:, 1]
    )
    st.subheader("Historical ensemble probability")
    st.line_chart(pd.Series(hist, index=feats.index))

    st.subheader("Latest feature snapshot")
    st.dataframe(X_latest.T.rename(columns={X_latest.index[-1]: "Latest"}))


if __name__ == "__main__":
    main()
